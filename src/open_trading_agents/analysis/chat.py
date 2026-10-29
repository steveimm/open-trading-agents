from __future__ import annotations

import asyncio
import sys
import uuid
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from open_trading_agents.analysis.models import AgentEvent, AgentSpec, EventSink, HarnessConfig, HarnessSession
from open_trading_agents.analysis.prompts import ROLES, system_prompt
from open_trading_agents.dataflows.files import locked
from open_trading_agents.harnesses import open_session
from open_trading_agents.persistence.events import EventRecorder
from open_trading_agents.persistence.storage import RunStore

if TYPE_CHECKING:
    from open_trading_agents.analysis.workflow import SessionFactory


class ReportChat:
    def __init__(
        self, directory: Path, config: HarnessConfig, language: str, session_factory: SessionFactory = open_session,
        *, allow_execution: bool = True, web_research: bool = True, max_tool_rounds: int = 20,
    ) -> None:
        """Prepare a saved report conversation without starting model work.

        Args:
            directory: Completed research run directory.
            config: Model configuration for follow-up answers.
            language: Preferred response language.
            session_factory: Native session constructor.
            allow_execution: Enable native calculation tools.
            web_research: Include web search and page retrieval tools.
            max_tool_rounds: Per-turn tool budget.
        """
        self.directory, self.config, self.language = directory, config, language
        self.workspace = directory / "chat"
        self.session_factory = session_factory
        self.allow_execution, self.web_research, self.max_tool_rounds = allow_execution, web_research, max_tool_rounds
        self.system_prompt = system_prompt(ROLES["chat"], language, allow_execution=allow_execution, web_research=web_research)
        self.session: HarnessSession | None = None
        self.context_sent = False
        self.sink: EventSink | None = None
        self.store: RunStore | None = None
        self.recorder: EventRecorder | None = None
        self._resources = ExitStack()
        self._prompt_lock = asyncio.Lock()
        self._turn_id: str | None = None
        if (self.workspace / "run.sqlite").is_file():
            reader = RunStore(self.workspace)
            try:
                self.system_prompt = reader.metadata("system_prompt", self.system_prompt)
            finally:
                reader.close()

    def _open(self) -> None:
        """Lock the conversation and restore its model identity and context state."""
        if self.store is not None:
            if self.recorder is not None and self.recorder.error is not None:
                raise RuntimeError("Chat history could not be saved. Close the interface before retrying.") from self.recorder.error
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        try:
            if not self._resources.enter_context(locked(self.workspace / "execution", wait=False)):
                raise RuntimeError("This chat is already active in another process.")
            self.store = RunStore(self.workspace)
            identity = {key: getattr(self.config, key) for key in ("kind", "provider", "model", "effort")}
            identity.update(language=self.language, allow_execution=self.allow_execution,
                            web_research=self.web_research, max_tool_rounds=self.max_tool_rounds)
            if self.store.metadata("configuration", identity) != identity:
                raise ValueError("Chat settings differ from the saved conversation. Reopen the run with its original settings.")
            self.store.set_metadata("configuration", identity)
            self.system_prompt = self.store.metadata("system_prompt", self.system_prompt)
            self.store.set_metadata("system_prompt", self.system_prompt)
            self.context_sent = self.store.metadata("context_sent", False)
            self.store.append(self._interrupted_events(self.store))
            self.recorder = EventRecorder(self.workspace)
        except BaseException:
            if self.store is not None:
                self.store.close()
                self.store = None
            self._resources.close()
            raise

    async def replay(self, emit: EventSink) -> None:
        """Restore saved messages and partial output without calling the model.

        Args:
            emit: Observer receiving the saved conversation events.
        """
        if not (self.workspace / "run.sqlite").is_file():
            return
        reader = RunStore(self.workspace)
        try:
            seen = set()
            for index, event in enumerate(reader.iter_events(), 1):
                seen.add(event.event_id)
                await emit(event)
                if index % 256 == 0:
                    await asyncio.sleep(0)
            for event in self._interrupted_events(reader, seen):
                await emit(event)
        finally:
            reader.close()

    @staticmethod
    def _interrupted_events(reader: RunStore, seen: set[str] | None = None) -> list[AgentEvent]:
        """Recover a committed question whose streamed events were interrupted.

        Args:
            reader: Saved conversation store.
            seen: Event identifiers already read during replay.

        Returns:
            Missing question and interruption events with stable identifiers.
        """
        pending = reader.metadata("pending_turn")
        if not pending:
            return []
        if seen is None:
            seen = {event.event_id for event in reader.iter_events()}
        turn_id = pending["id"]
        data = {"conversation_turn_id": turn_id}
        recovered = []
        if f"{turn_id}:prompt" not in seen:
            recovered.append(AgentEvent("prompt", ROLES["chat"].name, "chat", pending["question"], data, event_id=f"{turn_id}:prompt"))
        if not {f"{turn_id}:end", f"{turn_id}:error"} & seen:
            recovered.append(AgentEvent("error", ROLES["chat"].name, "chat", "The previous reply was interrupted. You can continue this conversation.",
                                        data, event_id=f"{turn_id}:error"))
        return recovered

    async def _record(self, event: AgentEvent) -> None:
        """Save a normalized event and update durable native-session metadata.

        Args:
            event: User message, model output, or lifecycle event.
        """
        assert self.store is not None and self.recorder is not None
        if self._turn_id is not None:
            event = replace(event, data={**event.data, "conversation_turn_id": self._turn_id})
        if event.kind == "session" and event.data.get("session_id"):
            self.store.save_session(ROLES["chat"].name, event.data["session_id"])
        if event.data.get("prompt_accepted"):
            self.context_sent = True
            self.store.set_metadata("context_sent", True)
        self.recorder.append(event)
        if self.sink is not None:
            await self.sink(event)

    async def _emit(self, event: AgentEvent) -> None:
        """Record native events without displaying the injected report as a user question.

        Args:
            event: Native harness event.
        """
        if event.kind != "prompt":
            await self._record(event)

    async def ask(self, question: str, emit: EventSink) -> None:
        """Save a question and continue the native conversation with its original report context.

        Args:
            question: User's follow-up question.
            emit: Observer for streaming responses.
        """
        async with self._prompt_lock:
            self.sink = emit
            self._turn_id = uuid.uuid4().hex
            try:
                self._open()
                assert self.store is not None
                self.store.set_metadata("pending_turn", {"id": self._turn_id, "question": question})
                await self._record(AgentEvent("prompt", ROLES["chat"].name, "chat", question, event_id=f"{self._turn_id}:prompt"))
                if self.session is None:
                    tools = ROLES["chat"].tools + (("search_web", "fetch_webpage") if self.web_research else ())
                    spec = AgentSpec(
                        agent=ROLES["chat"].name, stage="chat", workspace=self.workspace, session_dir=self.workspace / "sessions",
                        session_id=self.store.session(ROLES["chat"].name), system_prompt=self.system_prompt,
                        allow_execution=self.allow_execution, tools=tools, max_tool_rounds=self.max_tool_rounds,
                        tool_server_command=(
                            sys.executable, "-m", "open_trading_agents.dataflows.tools",
                            "--context", str(self.directory / "tool-context.json"), "--tools", ",".join(tools),
                            "--artifact-dir", str(self.workspace / "data"),
                        ),
                    )
                    self.session = await self.session_factory(self.config, spec, self._emit)
                    if self.session.session_id:
                        self.store.save_session(ROLES["chat"].name, self.session.session_id)
                prompt = question
                if not self.context_sent:
                    report = await asyncio.to_thread((self.directory / "reports" / "complete_report.md").read_text, encoding="utf-8")
                    prompt = f"Research report:\n<report>\n{report}\n</report>\n\nUser question:\n{question}"
                result = await self.session.prompt(prompt)
                self.store.save_session(ROLES["chat"].name, result.session_id)
                self.context_sent = True
                self.store.set_metadata("context_sent", True)
                if result.stop_reason not in ("stop", "end_turn", "completed") or not result.text.strip():
                    raise RuntimeError("The chat response did not complete. Please try again.")
                await self._record(AgentEvent("agent_end", ROLES["chat"].name, "chat", data={"output": result.text}, event_id=f"{self._turn_id}:end"))
            except BaseException as exc:
                text = "Reply interrupted. You can continue this conversation." if isinstance(exc, asyncio.CancelledError) else str(exc)
                event = AgentEvent("error", ROLES["chat"].name, "chat", text, event_id=f"{self._turn_id}:error")
                if self.recorder is not None:
                    await self._record(event)
                else:
                    await emit(event)
                raise
            finally:
                try:
                    if self.recorder is not None:
                        await asyncio.to_thread(self.recorder.flush)
                        assert self.store is not None
                        self.store.set_metadata("pending_turn", None)
                finally:
                    self._turn_id = None

    async def close(self) -> None:
        """Flush the transcript and release native resources and the conversation lock."""
        try:
            if self.session is not None:
                session, self.session = self.session, None
                await session.close()
        finally:
            try:
                if self.recorder is not None:
                    recorder, self.recorder = self.recorder, None
                    await asyncio.to_thread(recorder.close)
            finally:
                if self.store is not None:
                    self.store.close()
                    self.store = None
                self._resources.close()
