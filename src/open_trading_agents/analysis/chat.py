from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from open_trading_agents.analysis.models import AgentEvent, AgentSpec, EventSink, HarnessConfig, HarnessSession
from open_trading_agents.analysis.prompts import ROLES, system_prompt
from open_trading_agents.harnesses import open_session

if TYPE_CHECKING:
    from open_trading_agents.analysis.workflow import SessionFactory


class ReportChat:
    def __init__(
        self, directory: Path, config: HarnessConfig, language: str, session_factory: SessionFactory = open_session,
        *, allow_execution: bool = True, web_research: bool = True, max_tool_rounds: int = 20,
    ) -> None:
        """Prepare a report conversation using the selected native harness.

        Args:
            directory: Completed research run directory.
            config: Model configuration for follow-up answers.
            language: Preferred response language.
            session_factory: Native session constructor.
            allow_execution: Enable the same native calculation tools as the analysts.
            web_research: Include the analysts’ web search and page retrieval tools.
            max_tool_rounds: Per-turn tool budget.
        """
        self.directory, self.config, self.language = directory, config, language
        self.session_factory = session_factory
        self.allow_execution, self.web_research, self.max_tool_rounds = allow_execution, web_research, max_tool_rounds
        self.system_prompt = system_prompt(ROLES["chat"], language)
        self.session: HarnessSession | None = None
        self.context_sent = False
        self.sink: EventSink | None = None

    async def _emit(self, event: AgentEvent) -> None:
        """Forward model events while keeping the full context out of displayed questions.

        Args:
            event: Native harness event.
        """
        if self.sink is not None and event.kind != "prompt":
            await self.sink(event)

    async def ask(self, question: str, emit: EventSink) -> None:
        """Append a question to the conversation, supplying the report on its first turn.

        Args:
            question: User's follow-up question.
            emit: Observer for streaming responses.
        """
        self.sink = emit
        await emit(AgentEvent("prompt", "Report Assistant", "chat", question))
        if self.session is None:
            workspace = self.directory / "chat"
            tools = ROLES["chat"].tools + (("search_web", "fetch_webpage") if self.web_research else ())
            spec = AgentSpec(
                agent=ROLES["chat"].name, stage="chat", workspace=workspace, session_dir=workspace / "sessions",
                system_prompt=self.system_prompt, allow_execution=self.allow_execution, tools=tools,
                max_tool_rounds=self.max_tool_rounds,
                tool_server_command=(
                    sys.executable, "-m", "open_trading_agents.dataflows.tools",
                    "--context", str(self.directory / "tool-context.json"), "--tools", ",".join(tools),
                    "--artifact-dir", str(workspace / "data"),
                ),
            )
            self.session = await self.session_factory(self.config, spec, self._emit)
        prompt = question
        if not self.context_sent:
            report = await asyncio.to_thread((self.directory / "reports" / "complete_report.md").read_text, encoding="utf-8")
            prompt = f"Research report:\n<report>\n{report}\n</report>\n\nUser question:\n{question}"
        result = await self.session.prompt(prompt)
        self.context_sent = True
        if result.stop_reason not in ("stop", "end_turn", "completed") or not result.text.strip():
            raise RuntimeError("The chat response did not complete. Please try again.")
        await emit(AgentEvent("agent_end", "Report Assistant", "chat", data={"output": result.text}))

    async def close(self) -> None:
        """Release the native conversation when the interface closes."""
        if self.session is not None:
            await self.session.close()
            self.session = None
