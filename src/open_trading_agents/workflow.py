from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from open_trading_agents.config import Portfolio, Settings, validate_date
from open_trading_agents.dataflows.config import default_data_config
from open_trading_agents.dataflows.symbols import crypto_base, normalize_symbol, safe_ticker_component
from open_trading_agents.decisions import SCHEMAS, parse_decision, render_decision
from open_trading_agents.domain import AgentEvent, AgentSpec, EventSink, HarnessConfig, HarnessSession
from open_trading_agents.events import EventRecorder
from open_trading_agents.harnesses import open_session
from open_trading_agents.prompts import ROLES, system_prompt
from open_trading_agents.reporting import write_reports
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
        self.data_config = default_data_config()
        self.data_config["data_cache_dir"] = str(settings.state_dir / "cache")
        for key, value in settings.data.items():
            if isinstance(value, dict) and isinstance(self.data_config.get(key), dict):
                self.data_config[key].update(value)
            else:
                self.data_config[key] = value
        self.tool_context = ToolContext(self.ticker, self.trade_date, self.asset_type, self.data_config)
        self.context_file = self.directory / "tool-context.json"
        self.context_file.write_text(json.dumps({"ticker": self.ticker, "trade_date": self.trade_date,
                                                "asset_type": self.asset_type, "config": self.data_config}))
        identity = {"ticker": self.ticker, "trade_date": self.trade_date, "settings": settings.model_dump(mode="json"),
                    "portfolio": portfolio.model_dump() if portfolio is not None else None}
        previous = self.store.metadata("input")
        if previous and previous != identity:
            raise ValueError("run inputs differ from the saved run; resume its original configuration or start a new run")
        self.store.set_metadata("input", identity)

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
        if event.kind == "session" and event.data.get("session_id"):
            self.store.save_session(event.agent, event.data["session_id"])
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
        await self._event("agent_start", role.stage, name)
        if name not in self.sessions:
            workspace = self.directory / "agents" / "".join(c if c.isalnum() else "_" for c in name)
            spec = AgentSpec(name, role.stage, system_prompt(role, self.settings.output_language), workspace,
                             workspace / "sessions", self.store.session(name), role.tools,
                             (sys.executable, "-m", "open_trading_agents.tools", "--context", str(self.context_file),
                              "--tools", ",".join(role.tools)) if role.tools else (),
                             max_tool_rounds=self.settings.max_tool_rounds)
            self.sessions[name] = await self.session_factory(self.settings.model_settings(role.deep).runtime(), spec, self.emit)
            if self.sessions[name].session_id:
                self.store.save_session(name, self.sessions[name].session_id)
        if saved and self.store.session(name):
            supplied = "Continue the unfinished task in this conversation. Reuse completed tool evidence and return its requested final output."
        else:
            supplied = prompt
        self.store.begin_step(key, name, prompt)
        result = await self.sessions[name].prompt(supplied)
        if result.stop_reason not in ("stop", "end_turn", "completed") or not result.text.strip():
            raise RuntimeError(f"{name} did not produce a complete response ({result.stop_reason})")
        self.store.save_session(name, result.session_id)
        self.store.finish_step(key, result.text)
        await self._event("agent_end", role.stage, name, output=result.text)
        return result.text

    def _instrument(self) -> str:
        """Describe the immutable instrument and information cutoff.

        Returns:
            A common input block for new role sessions.
        """
        return f"Instrument: {self.ticker}\nAsset type: {self.asset_type}\nAnalysis date: {self.trade_date}\nOnly assert a company identity supported by the evidence."

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
                        evidence = await asyncio.to_thread(FinancialTools(self.tool_context).get_social_sentiment)
                    except Exception as exc:
                        evidence = f"DATA_UNAVAILABLE: sentiment sources could not be retrieved ({type(exc).__name__})."
                    self.store.set_metadata("sentiment_evidence", evidence)
                    await self._event("tool_result", "analysis", ROLES[key].name, evidence,
                                      tool="get_social_sentiment", tool_call_id="prefetch-social")
            text = await self._call(f"analyst:{key}", key, f"{self._instrument()}\n\nProduce your research report.\n\n{evidence}")
            self.outputs[ROLES[key].name] = text

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
        discussion: list[str] = []
        seen = dict.fromkeys(keys, 0)
        for round_number in range(1, rounds + 1):
            for turn_number, key in enumerate(keys, 1):
                role = ROLES[key]
                await self._event("turn_start", stage, role.name, round=round_number, turn=turn_number,
                                  total_rounds=rounds, turns_per_round=len(keys))
                additions = "\n\n".join(discussion[seen[key]:]) or "This is the opening turn."
                prefix = f"{self._instrument()}\n\n{evidence}\n\n" if round_number == 1 else ""
                prompt = f"{prefix}Round {round_number} of {rounds}.\nNew public contributions since your previous turn:\n\n{additions}\n\nGive your next argument."
                text = await self._call(f"{stage}:{round_number}:{key}", key, prompt)
                discussion.append(f"### Round {round_number}: {role.name}\n\n{text}")
                seen[key] = len(discussion)
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
        await self._event("stage_end", role.stage)
        return rendered

    async def _settle_memory(self) -> None:
        """Resolve eligible historical outcomes through harness-owned reflection sessions."""
        if not self.settings.memory_enabled:
            return
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
            return await self._call(f"reflection:{identifier}", "reflection", json.dumps(entry, ensure_ascii=False),
                                    f"Reflection {identifier}")

        try:
            notes = await memory.settle(self.data_config, reflect)
            for note in notes:
                await self._event("status", "analysis", "Memory", note)
        except Exception as exc:
            await self._event("error", "analysis", "Memory", f"Prior outcomes remain pending: {type(exc).__name__}: {exc}")

    async def run(self, emit: EventSink | None = None) -> RunResult:
        """Execute all required stages, preserving completed steps on cancellation or failure.

        Args:
            emit: Optional observer for live TUI or headless events.

        Returns:
            The final rating and saved artifact location.
        """
        self.sink = emit
        if emit:
            for event in self.store.events():
                await emit(event)
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
            await self._event("stage_end", "reports")
            await self._event("run_end", "reports", rating=self.rating, directory=str(self.directory))
        except BaseException as exc:
            self.store.set_metadata("status", "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed")
            await self._event("error", "reports", text=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            await asyncio.gather(*(session.close() for session in self.sessions.values()), return_exceptions=True)
            await asyncio.to_thread(self.recorder.close)
            report_settings = {
                "harness": self.settings.harness.kind, "quick_model": self.settings.model_settings().model,
                "deep_model": self.settings.model_settings(True).model, "analysts": self.settings.analysts,
                "research_rounds": self.settings.research_rounds, "risk_rounds": self.settings.risk_rounds,
                "data_vendors": self.data_config["data_vendors"],
            }
            try:
                await asyncio.to_thread(write_reports, self.directory, self.ticker, self.trade_date,
                                        self.outputs, self.rating, report_settings, self.store.events())
                if self.store.metadata("status") == "running":
                    self.store.set_metadata("status", "complete")
            except Exception:
                self.store.set_metadata("status", "failed")
                raise
            finally:
                self.store.close()
        return RunResult(self.run_id, self.directory, self.rating, self.outputs)
