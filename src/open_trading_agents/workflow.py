from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import uuid
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from open_trading_agents.config import Portfolio, Settings, validate_date
from open_trading_agents.dataflows.config import default_data_config
from open_trading_agents.dataflows.files import locked, replace_file
from open_trading_agents.dataflows.symbols import crypto_base, normalize_symbol, safe_ticker_component
from open_trading_agents.decisions import SCHEMAS, parse_decision, render_decision
from open_trading_agents.domain import AgentEvent, AgentSpec, EventSink, HarnessConfig, HarnessSession
from open_trading_agents.events import EventRecorder
from open_trading_agents.harnesses import open_session
from open_trading_agents.prompts import ROLES, system_prompt
from open_trading_agents.reporting import write_reports, write_transcript
from open_trading_agents.storage import RunStore
from open_trading_agents.tools import FinancialTools, ToolContext

SessionFactory = Callable[[HarnessConfig, AgentSpec, EventSink], Awaitable[HarnessSession]]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    directory: Path
    rating: str
    outputs: dict[str, str]


class Workflow:
    def __init__(self, ticker: str, trade_date: str, settings: Settings, portfolio: Portfolio | None = None,
                 run_id: str | None = None, session_factory: SessionFactory = open_session) -> None:
        """Prepare a resumable financial research run without contacting any provider.

        Args:
            ticker: Instrument symbol or supported market alias.
            trade_date: Analysis information cutoff.
            settings: Validated application configuration.
            portfolio: Optional known holdings and cash.
            run_id: Existing run identifier to resume, or None to create one.
            session_factory: Native session constructor, injectable for offline tests.
        """
        self.ticker = safe_ticker_component(normalize_symbol(ticker))
        self.trade_date = validate_date(trade_date)
        settings = settings.model_copy(update={"state_dir": settings.state_dir.expanduser().resolve()})
        self.settings, self.portfolio = settings, portfolio
        self.asset_type = "crypto" if crypto_base(self.ticker) else "stock"
        self.run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        safe_ticker_component(self.run_id, max_len=100)
        self.directory = settings.state_dir / "runs" / self.run_id
        self.store = RunStore(self.directory)
        self.sessions: dict[str, HarnessSession] = {}
        self.session_factory = session_factory
        self.outputs: dict[str, str] = {}
        self.rating = "REVIEW"
        self.recorder: EventRecorder | None = None
        self.sink: EventSink | None = None
        self.active_steps: dict[str, str] = {}
        self.conversation_turns: dict[str, str] = {}
        self.screening_futures: set[Future[tuple[list[bool], str]]] = set()
        self.data_config = default_data_config()
        self.data_config["data_cache_dir"] = str(settings.state_dir / "cache")
        for key, value in settings.data.items():
            if isinstance(value, dict) and isinstance(self.data_config.get(key), dict):
                self.data_config[key].update(value)
            else:
                self.data_config[key] = value
        self.tool_context = ToolContext(self.ticker, self.trade_date, self.asset_type, self.data_config)
        self.context_file = self.directory / "tool-context.json"
        identity = {"ticker": self.ticker, "trade_date": self.trade_date, "settings": settings.model_dump(mode="json"),
                    "portfolio": portfolio.model_dump() if portfolio is not None else None}
        previous = self.store.metadata("input")
        if previous and previous != identity:
            self.store.close()
            raise ValueError("run inputs differ from the saved run; resume its original configuration or start a new run")
        if not self.context_file.exists():
            context_json = json.dumps({"ticker": self.ticker, "trade_date": self.trade_date,
                                       "asset_type": self.asset_type, "config": self.data_config})
            replace_file(self.context_file, lambda temporary: Path(temporary).write_text(context_json))
        self.store.set_metadata("input", identity)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Allow only one process to drive this run's native sessions.

        Yields:
            Control while holding the run's exclusive writer lock.

        Raises:
            RuntimeError: Another process is already running this workflow.
        """
        with locked(self.directory / "execution", wait=False) as acquired:
            if not acquired:
                self.store.close()
                raise RuntimeError(f"run {self.run_id} is already active in another process")
            yield

    @classmethod
    def resume(cls, directory: Path, session_factory: SessionFactory = open_session) -> Workflow:
        """Reconstruct a run using its original configuration and explicit sessions.

        Args:
            directory: Existing run directory.
            session_factory: Native session constructor.

        Returns:
            A workflow ready to continue incomplete steps.
        """
        if not (directory / "run.sqlite").is_file():
            raise ValueError(f"no saved run in {directory}")
        store = RunStore(directory)
        inputs = store.metadata("input")
        store.close()
        if not inputs:
            raise ValueError("saved run is missing its inputs")
        portfolio = Portfolio.model_validate(inputs["portfolio"]) if inputs["portfolio"] is not None else None
        return cls(inputs["ticker"], inputs["trade_date"], Settings.model_validate(inputs["settings"]), portfolio,
                   directory.name, session_factory)

    async def emit(self, event: AgentEvent) -> None:
        """Persist events independently of the UI and remember native session IDs.

        Args:
            event: A workflow or harness event.
        """
        conversation_turn = self.conversation_turns.get(event.agent)
        if conversation_turn:
            event = replace(event, data={**event.data, "conversation_turn_id": conversation_turn})
        if event.kind == "session" and event.data.get("session_id"):
            self.store.save_session(event.agent, event.data["session_id"])
        if event.data.get("prompt_accepted") and event.agent in self.active_steps:
            self.store.set_metadata(f"delivered:{self.active_steps[event.agent]}", True)
        if self.recorder:
            self.recorder.append(event)
        if self.sink:
            await self.sink(event)

    async def _event(self, kind: str, stage: str, agent: str = "Workflow", text: str = "", **data: Any) -> None:
        """Emit a lifecycle event with a consistent role and stage.

        Args:
            kind: Event kind.
            stage: Stage identifier.
            agent: Logical speaker.
            text: Optional readable content.
            data: Structured event details.
        """
        await self.emit(AgentEvent(kind, agent, stage, text, data))

    async def _call(self, key: str, role_key: str, prompt: str, session_name: str | None = None) -> str:
        """Complete one step while retaining the role's native conversation.

        Args:
            key: Unique step identifier within the run.
            role_key: Role definition key.
            prompt: New evidence and instructions for this turn.
            session_name: Optional distinct identity for independent reflections.

        Returns:
            The settled output, including output recovered from completed steps.
        """
        saved = self.store.step(key)
        if saved and saved["status"] == "complete":
            return saved["output"]
        role = ROLES[role_key]
        name = session_name or role.name
        prior_session = self.store.session(name)
        await self._event("agent_start", role.stage, name)
        if name not in self.sessions:
            workspace = self.directory / "agents" / "".join(c if c.isalnum() else "_" for c in name)
            tools = ("calculate", *role.tools)
            if role_key in ("market", "social", "news", "fundamentals") and self.settings.web_research:
                tools = (*tools, "search_web", "fetch_webpage")
            spec = AgentSpec(name, role.stage, system_prompt(role, self.settings.output_language), workspace,
                             workspace / "sessions", self.store.session(name), tools,
                             (sys.executable, "-m", "open_trading_agents.tools", "--context", str(self.context_file),
                              "--tools", ",".join(tools)) if tools else (),
                             max_tool_rounds=self.settings.max_tool_rounds, allow_execution=self.settings.allow_execution)
            self.sessions[name] = await self.session_factory(self.settings.model_settings(role.deep).runtime(), spec, self.emit)
            if self.sessions[name].session_id:
                self.store.save_session(name, self.sessions[name].session_id)
        if saved and prior_session and self.store.metadata(f"delivered:{key}", False):
            supplied = "Continue the unfinished task in this conversation. Reuse completed tool evidence and return its requested final output."
        else:
            supplied = saved["prompt"] if saved else prompt
        self.store.begin_step(key, name, prompt)
        self.active_steps[name] = key
        try:
            result = await self.sessions[name].prompt(supplied)
            if result.stop_reason not in ("stop", "end_turn", "completed") or not result.text.strip():
                raise RuntimeError(f"{name} did not produce a complete response ({result.stop_reason})")
        except BaseException as exc:
            await self._event("error", role.stage, name, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.active_steps.pop(name, None)
        self.store.save_session(name, result.session_id)
        self.store.finish_step(key, result.text)
        await self._event("agent_end", role.stage, name, output=result.text)
        return result.text

    async def _release_session(self, name: str) -> None:
        """Release an idle role process while retaining its native session reference.

        Args:
            name: Logical role whose stage has finished.
        """
        session = self.sessions.pop(name, None)
        if session is not None:
            await session.close()

    def _instrument(self) -> str:
        """Describe the immutable instrument and information cutoff.

        Returns:
            A common input block for new role sessions.
        """
        return f"Instrument: {self.ticker}\nAsset type: {self.asset_type}\nAnalysis date: {self.trade_date}\nPython for calculation scripts: {sys.executable}\nOnly assert a company identity supported by the evidence."

    async def _analyst(self, key: str, semaphore: asyncio.Semaphore) -> None:
        """Run one initial analyst with an isolated session and assigned tools.

        Args:
            key: Analyst role key.
            semaphore: Limit on concurrent initial model rollouts.
        """
        async with semaphore:
            evidence = ""
            if key == "social" and not self.store.step("analyst:social"):
                evidence = self.store.metadata("sentiment_evidence", "")
                if not evidence:
                    await self._event("tool_start", "analysis", ROLES[key].name, tool="get_social_sentiment", tool_call_id="prefetch-social")
                    try:
                        loop = asyncio.get_running_loop()

                        def screen(posts: list[str]) -> tuple[list[bool], str]:
                            """Request optional screening without blocking the application event loop.

                            Args:
                                posts: Source posts supplied by the data adapter.

                            Returns:
                                Relevance decisions and the source summary note.
                            """
                            future = asyncio.run_coroutine_threadsafe(self._screen_posts(posts), loop)
                            self.screening_futures.add(future)
                            try:
                                return future.result()
                            finally:
                                self.screening_futures.discard(future)

                        tools = FinancialTools(self.tool_context, screen=screen if self.settings.social_screening else None)
                        evidence = await asyncio.to_thread(tools.get_social_sentiment)
                    except Exception as exc:
                        evidence = f"DATA_UNAVAILABLE: sentiment sources could not be retrieved ({type(exc).__name__})."
                    self.store.set_metadata("sentiment_evidence", evidence)
                    await self._event("tool_result", "analysis", ROLES[key].name, evidence,
                                      tool="get_social_sentiment", tool_call_id="prefetch-social")
            text = await self._call(f"analyst:{key}", key, f"{self._instrument()}\n\nProduce your research report.\n\n{evidence}")
            self.outputs[ROLES[key].name] = text
            await self._release_session(ROLES[key].name)

    async def _screen_posts(self, posts: list[str]) -> tuple[list[bool], str]:
        """Screen a source batch through a normal harness session, preserving data on failure.

        Args:
            posts: Untrusted source post texts in stable order.

        Returns:
            A keep flag per post and an explicitly model-derived stance summary.
        """
        if not posts:
            return [], "No posts available for screening."
        body = json.dumps({"ticker": self.ticker, "posts": posts}, ensure_ascii=False)
        digest = hashlib.sha256(body.encode()).hexdigest()[:16]
        session_name = f"Sentiment Screening {digest}"
        try:
            response = await self._call(f"screen:{digest}", "screening", body, session_name)
            parsed = parse_decision("screening", response)
            indexes = [item.index for item in parsed.items]
            if sorted(indexes) != list(range(len(posts))):
                raise ValueError("screening did not return exactly one result per source post")
            ordered = sorted(parsed.items, key=lambda item: item.index)
            keep = [item.relevance >= 0.3 for item in ordered]
            stances = [item.stance if item.confidence >= 0.5 else "unclear" for item, kept in zip(ordered, keep, strict=True) if kept]
            counts = ", ".join(f"{stances.count(stance)} {stance}" for stance in ("bullish", "bearish", "neutral", "unclear"))
            return keep, f"Model screening retained {sum(keep)} of {len(posts)} posts for {self.ticker}: {counts}."
        except Exception as exc:
            await self._event("status", "analysis", "Sentiment Analyst", f"Optional social screening unavailable: {type(exc).__name__}")
            return [True] * len(posts), "Model screening unavailable; all source posts were retained."
        finally:
            await self._release_session(session_name)

    async def _debate(self, stage: str, keys: tuple[str, ...], rounds: int, evidence: str) -> str:
        """Rotate speakers while appending only new public contributions to each native session.

        Args:
            stage: Research or risk stage.
            keys: Speaker role keys in fixed order.
            rounds: Number of complete rotations.
            evidence: Initial common evidence for each participant.

        Returns:
            The complete public discussion, with speaker and round attribution.
        """
        await self._event("stage_start", stage)
        await self.emit(AgentEvent(
            "conversation_input", "Input", stage, f"{self._instrument()}\n\n{evidence}",
            data={"participants": [ROLES[key].name for key in keys], "total_rounds": rounds},
            event_id=f"{self.run_id}:{stage}:input",
        ))
        discussion: list[str] = []
        seen = dict.fromkeys(keys, 0)
        for round_number in range(1, rounds + 1):
            for turn_number, key in enumerate(keys, 1):
                role = ROLES[key]
                step_key = f"{stage}:{round_number}:{key}"
                additions = "\n\n".join(discussion[seen[key]:]) or "This is the opening turn."
                prefix = f"{self._instrument()}\n\n{evidence}\n\n" if round_number == 1 else ""
                prompt = f"{prefix}Round {round_number} of {rounds}.\nNew public contributions since your previous turn:\n\n{additions}\n\nGive your next argument."
                self.conversation_turns[role.name] = step_key
                try:
                    saved = self.store.step(step_key)
                    if not saved or saved["status"] != "complete":
                        await self._event("turn_start", stage, role.name, round=round_number, turn=turn_number,
                                          total_rounds=rounds, turns_per_round=len(keys))
                    text = await self._call(step_key, key, prompt)
                    completed = self.store.step(step_key)
                    await self.emit(AgentEvent(
                        "conversation_output", role.name, stage, text,
                        data={"round": round_number, "turn": turn_number, "total_rounds": rounds,
                              "turns_per_round": len(keys)},
                        timestamp=completed["updated"], event_id=f"{self.run_id}:{step_key}:output",
                    ))
                finally:
                    self.conversation_turns.pop(role.name, None)
                discussion.append(f"### Round {round_number}: {role.name}\n\n{text}")
                seen[key] = len(discussion)
                self.outputs["Research Debate" if stage == "research" else "Risk Debate"] = "\n\n".join(discussion)
        for key in keys:
            await self._release_session(ROLES[key].name)
        await self._event("stage_end", stage)
        return "\n\n".join(discussion)

    async def _decision(self, key: str, evidence: str) -> str:
        """Request a validated decision, allowing one native-session formatting repair.

        Args:
            key: Decision-making role key.
            evidence: Reports, debate, portfolio and lessons for this decision.

        Returns:
            Rendered decision or clearly marked unvalidated output.
        """
        role = ROLES[key]
        await self._event("stage_start", role.stage)
        text = await self._call(key, key, f"{self._instrument()}\n\n{evidence}")
        parsed = None
        for attempt in range(2):
            try:
                parsed = parse_decision(key, text)
                break
            except ValueError:
                if attempt == 0:
                    schema = json.dumps(SCHEMAS[key].model_json_schema())
                    text = await self._call(f"{key}:format", key, f"Reformat your decision as one JSON object matching this schema. Preserve your substantive conclusion and use null for unavailable optional fields.\n{schema}")
        if parsed is not None:
            rendered = render_decision(parsed)
            if key == "portfolio_manager":
                self.rating = parsed.rating
        else:
            rendered = f"Decision format could not be validated. Human review required.\n\n{text}"
            await self._event("error", role.stage, role.name, "Decision requires review because its schema could not be validated.")
        self.outputs[role.name] = rendered
        await self._event("report", role.stage, role.name, rendered)
        await self._release_session(role.name)
        await self._event("stage_end", role.stage)
        return rendered

    async def _settle_memory(self) -> list[str]:
        """Resolve eligible outcomes through harness-owned reflection sessions.

        Returns:
            Notices for decisions that could not be settled.
        """
        if not self.settings.memory_enabled:
            return []
        from open_trading_agents.memory import MemoryLog

        memory = MemoryLog(self.settings.state_dir / "memory.sqlite")

        async def reflect(entry: dict[str, Any]) -> str:
            """Turn one measured outcome into a short lesson through the configured harness.

            Args:
                entry: Decision and observed outcome evidence.

            Returns:
                The completed reflection text.
            """
            identifier = str(entry["id"])
            name = f"Reflection {identifier}"
            try:
                return await self._call(f"reflection:{identifier}", "reflection", json.dumps(entry, ensure_ascii=False), name)
            finally:
                await self._release_session(name)

        try:
            notes = await memory.settle(self.data_config, reflect)
            for note in notes:
                await self._event("status", "analysis", "Memory", note)
            return notes
        except Exception as exc:
            note = f"Prior outcomes remain pending: {type(exc).__name__}: {exc}"
            await self._event("error", "analysis", "Memory", note)
            return [note]

    async def settle_pending(self, emit: EventSink | None = None) -> list[str]:
        """Run a dedicated settlement pass without starting a new investment analysis.

        Args:
            emit: Optional live event observer.

        Returns:
            Failure notices for outcomes left pending.
        """
        with self._exclusive():
            return await self._settle_pending(emit)

    async def _settle_pending(self, emit: EventSink | None) -> list[str]:
        """Settle decisions while the caller holds the run lock.

        Args:
            emit: Optional live event observer.

        Returns:
            Notices for unresolved decisions.
        """
        self.sink = emit
        self.recorder = EventRecorder(self.directory)
        self.store.set_metadata("kind", "settlement")
        try:
            notes = await self._settle_memory()
            self.store.set_metadata("status", "complete" if not notes else "partial")
            return notes
        finally:
            for future in tuple(self.screening_futures):
                future.cancel()
            await asyncio.gather(*(session.close() for session in self.sessions.values()), return_exceptions=True)
            try:
                await asyncio.to_thread(self.recorder.close)
            except Exception:
                self.store.set_metadata("status", "failed")
                raise
            finally:
                self.store.close()

    async def run(self, emit: EventSink | None = None) -> RunResult:
        """Execute all required stages, preserving completed steps on cancellation or failure.

        Args:
            emit: Optional observer for live TUI or headless events.

        Returns:
            The final rating and saved artifact location.
        """
        with self._exclusive():
            return await self._run(emit)

    async def _run(self, emit: EventSink | None) -> RunResult:
        """Execute the staged pipeline under the run's exclusive lock.

        Args:
            emit: Optional live event observer.

        Returns:
            The completed run result.
        """
        self.sink = emit
        if emit:
            for index, event in enumerate(self.store.iter_events(), 1):
                await emit(event)
                if index % 256 == 0:
                    await asyncio.sleep(0)
        self.recorder = EventRecorder(self.directory)
        self.store.set_metadata("status", "running")
        try:
            await self._event("stage_start", "analysis")
            selected = [key for key in self.settings.analysts if not (self.asset_type == "crypto" and key == "fundamentals")]
            if not selected:
                raise ValueError("crypto analysis needs a market, news, or sentiment analyst")
            semaphore = asyncio.Semaphore(self.settings.analyst_concurrency)
            results = await asyncio.gather(*(self._analyst(key, semaphore) for key in selected), self._settle_memory(),
                                           return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            self.outputs = {ROLES[key].name: self.outputs[ROLES[key].name] for key in selected}
            await self._event("stage_end", "analysis")
            evidence = "\n\n".join(f"## {name}\n\n{text}" for name, text in self.outputs.items())
            research = await self._debate("research", ("bull", "bear"), self.settings.research_rounds, evidence)
            self.outputs["Research Debate"] = research
            plan = await self._decision("research_manager", f"Analyst reports:\n{evidence}\n\nResearch debate:\n{research}")
            portfolio = self.portfolio.render(self.ticker) if self.portfolio is not None else "Portfolio not supplied. Holdings and cash are unknown."
            proposal = await self._decision("trader", f"{portfolio}\n\nResearch plan:\n{plan}\n\nAnalyst reports:\n{evidence}")
            risk = await self._debate("risk", ("aggressive", "conservative", "neutral"), self.settings.risk_rounds,
                                      f"{portfolio}\n\nAnalyst reports:\n{evidence}\n\nTrader proposal:\n{proposal}")
            self.outputs["Risk Debate"] = risk
            lessons = ""
            if self.settings.memory_enabled:
                from open_trading_agents.memory import MemoryLog

                memory = MemoryLog(self.settings.state_dir / "memory.sqlite")
                lessons = memory.past_context(self.ticker, as_of=self.trade_date)
            final = await self._decision("portfolio_manager", f"{portfolio}\n\nResearch plan:\n{plan}\n\nTrader proposal:\n{proposal}\n\nRisk debate:\n{risk}\n\nResolved prior lessons:\n{lessons or 'None available.'}")
            if self.settings.memory_enabled:
                memory.record(self.ticker, self.trade_date, self.rating, final, self.run_id,
                              holding_days=self.settings.holding_period_days, benchmark=self.settings.benchmark_ticker)
            await self._event("stage_start", "reports")
            await self._event("report", "reports", "Portfolio Manager", final)
            self.store.set_metadata("rating", self.rating)
            self.store.set_metadata("outputs", self.outputs)
        except BaseException as exc:
            self.store.set_metadata("status", "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed")
            await self._event("error", "reports", text=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            for future in tuple(self.screening_futures):
                future.cancel()
            await asyncio.gather(*(session.close() for session in self.sessions.values()), return_exceptions=True)
            report_settings = {
                "harness": self.settings.harness.kind, "quick_model": self.settings.model_settings().model,
                "deep_model": self.settings.model_settings(True).model, "analysts": self.settings.analysts,
                "research_rounds": self.settings.research_rounds, "risk_rounds": self.settings.risk_rounds,
                "data_vendors": self.data_config["data_vendors"],
            }
            try:
                await asyncio.to_thread(self.recorder.close)
                await asyncio.to_thread(self._export_reports, report_settings)
                if self.store.metadata("status") == "running":
                    completed = [AgentEvent("stage_end", "Workflow", "reports"),
                                 AgentEvent("run_end", "Workflow", "reports", data={"rating": self.rating,
                                                                                   "directory": str(self.directory)})]
                    self.store.append(completed)
                    await asyncio.to_thread(write_transcript, self.directory / "transcript.jsonl", completed, True)
                    self.store.set_metadata("status", "complete")
                    if self.sink:
                        for event in completed:
                            await self.sink(event)
            except Exception:
                self.store.set_metadata("status", "failed")
                raise
            finally:
                self.store.close()
        return RunResult(self.run_id, self.directory, self.rating, self.outputs)

    def _export_reports(self, settings: dict[str, Any]) -> None:
        """Load recorded events and render reports away from the live UI loop.

        Args:
            settings: Non-secret run configuration exported with the reports.
        """
        reader = RunStore(self.directory)
        try:
            events = reader.events()
        finally:
            reader.close()
        write_reports(self.directory, self.ticker, self.trade_date, self.outputs, self.rating, settings, events)
