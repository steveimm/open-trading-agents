import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from open_trading_agents.analysis.decisions import SCHEMAS
from open_trading_agents.analysis.models import AgentEvent, AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.analysis.workflow import Workflow
from open_trading_agents.config import Settings
from open_trading_agents.dataflows.files import locked
from open_trading_agents.persistence.events import EventRecorder
from open_trading_agents.persistence.storage import RunStore


class FakeHarness:
    def __init__(self, spec: AgentSpec, emit: EventSink, calls: list[tuple[str, str]], failure: list[str]) -> None:
        """Retain a synthetic session identity and record newly supplied turns."""
        self.spec, self.emit, self.calls, self.failure = spec, emit, calls, failure
        self.session_id = spec.session_id or f"native:{spec.agent}"

    async def prompt(self, text: str) -> AgentResult:
        """Return role-appropriate fixtures after yielding to concurrent analysts."""
        self.calls.append((self.spec.agent, text))
        await self.emit(AgentEvent("prompt", self.spec.agent, self.spec.stage, text))
        await self.emit(AgentEvent("status", self.spec.agent, self.spec.stage, data={"prompt_accepted": True}))
        await asyncio.sleep(0)
        if self.failure and self.failure[0] == self.spec.agent:
            self.failure.pop()
            raise RuntimeError("simulated interrupted provider")
        outputs = {
            "Research Manager": '{"recommendation":"Overweight","rationale":"Evidence","strategic_actions":"Scale in"}',
            "Trader": '{"action":"Buy","reasoning":"Evidence","entry_price":123.45}',
            "Portfolio Manager": '{"rating":"Overweight","executive_summary":"Scale in","investment_thesis":"Evidence"}',
        }
        answer = outputs.get(self.spec.agent, f"{self.spec.agent} report based on synthetic evidence")
        await self.emit(AgentEvent("text", self.spec.agent, self.spec.stage, answer))
        return AgentResult(answer, self.session_id)

    async def cancel(self) -> None:
        """Cancel the synthetic session without external effects."""

    async def close(self) -> None:
        """Close the synthetic session without external effects."""


async def test_debates_reuse_native_sessions_and_append_only_new_arguments(tmp_path: Path) -> None:
    """Preserve the stage sequence and avoid reconstructing prior model history."""
    calls = []
    opened = []

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FakeHarness:
        """Record each independent native role session."""
        opened.append(spec.agent)
        if spec.tools:
            position = spec.tool_server_command.index("--artifact-dir")
            assert Path(spec.tool_server_command[position + 1]) == spec.workspace / "data"
        return FakeHarness(spec, emit, calls, [])

    settings = Settings(state_dir=tmp_path, analysts=("market", "news"), research_rounds=2, memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    result = await workflow.run()
    assert result.rating == "Overweight"
    assert opened.count("Bull Researcher") == opened.count("Bear Researcher") == 1
    bull_prompts = [prompt for name, prompt in calls if name == "Bull Researcher"]
    assert len(bull_prompts) == 2
    assert "Market Analyst report" in bull_prompts[0]
    assert "Market Analyst report" not in bull_prompts[1]
    assert "Bear Researcher report" in bull_prompts[1]
    assert [name for name, _ in calls[-4:]] == ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"]
    assert (result.directory / "reports" / "complete_report.html").is_file()
    assert (result.directory / "transcript.jsonl").is_file()


@pytest.mark.parametrize("invalid_first_response", [False, True])
async def test_managers_receive_the_schema_initially_and_repair_only_invalid_output(tmp_path: Path, invalid_first_response: bool) -> None:
    """Provide field types before generation and reuse the native session for a bounded format correction."""
    calls: list[tuple[str, str]] = []
    opened: list[str] = []
    seen_schemas: dict[str, dict] = {}

    class DecisionHarness(FakeHarness):
        async def prompt(self, text: str) -> AgentResult:
            """Return a deliberately nested rationale once to reproduce a schema failure."""
            previous = [prompt for name, prompt in calls if name == self.spec.agent]
            if self.spec.agent == "Research Manager" and invalid_first_response and not previous:
                calls.append((self.spec.agent, text))
                await self.emit(AgentEvent("status", self.spec.agent, self.spec.stage, data={"prompt_accepted": True}))
                return AgentResult('{"recommendation":"Overweight","rationale":{"evidence":"Verified"},"strategic_actions":["Scale in"]}', self.session_id)
            if self.spec.agent == "Research Manager" and previous:
                assert "rationale: Input should be a valid string" in text
                assert "strategic_actions: Input should be a valid string" in text
                assert "without changing its substantive conclusion" in text
            return await super().prompt(text)

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> DecisionHarness:
        """Inspect the initial native instructions before the first role prompt is submitted."""
        opened.append(spec.agent)
        for name, key in (("Research Manager", "research_manager"), ("Trader", "trader"), ("Portfolio Manager", "portfolio_manager")):
            if spec.agent == name:
                seen_schemas[key] = json.loads(spec.system_prompt.split("JSON schema:\n", 1)[1])
                assert seen_schemas[key] == SCHEMAS[key].model_json_schema()
        return DecisionHarness(spec, emit, calls, [])

    result = await Workflow("NVDA", "2026-01-02", Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False),
                            session_factory=factory).run()
    assert result.rating == "Overweight"
    assert set(seen_schemas) == {"research_manager", "trader", "portfolio_manager"}
    assert opened.count("Research Manager") == 1
    assert len([prompt for name, prompt in calls if name == "Research Manager"]) == (2 if invalid_first_response else 1)


async def test_resume_keeps_finished_work_and_continues_pending_native_session(tmp_path: Path) -> None:
    """Avoid replaying completed reports or the entire input of an interrupted turn."""
    calls = []
    failure = ["Neutral Analyst"]

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FakeHarness:
        """Restore the native identifier passed by the completion ledger."""
        return FakeHarness(spec, emit, calls, failure)

    settings = Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    with pytest.raises(RuntimeError, match="interrupted provider"):
        await workflow.run()
    first_calls = len(calls)
    resumed = Workflow.resume(workflow.directory, session_factory=factory)
    result = await resumed.run()
    assert result.rating == "Overweight"
    assert [name for name, _ in calls[first_calls:]] == ["Neutral Analyst", "Portfolio Manager"]
    assert calls[first_calls][1].startswith("Continue the unfinished task")


def test_rejected_resume_does_not_mutate_saved_tool_context(tmp_path: Path) -> None:
    """Protect instrument bindings when a caller attempts an incompatible resume."""
    settings = Settings(state_dir=tmp_path, memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, run_id="fixed")
    original = workflow.context_file.read_bytes()
    workflow.store.close()
    with pytest.raises(ValueError, match="inputs differ"):
        Workflow("AAPL", "2026-01-02", settings, run_id="fixed")
    assert workflow.context_file.read_bytes() == original


def test_workflows_do_not_share_mutable_data_overrides(tmp_path: Path) -> None:
    """Keep source lists and vendor mappings independent across runs and caller settings.

    Args:
        tmp_path: Isolated run storage.
    """
    settings = Settings(
        state_dir=tmp_path,
        data={"global_news_queries": ["earnings"], "tool_vendors": {"get_news": "alpha_vantage"}},
    )
    first = Workflow("NVDA", "2026-01-02", settings)
    second = Workflow("AAPL", "2026-01-02", settings)

    try:
        first.data_config["global_news_queries"].append("rates")
        first.data_config["tool_vendors"]["get_news"] = "yfinance"

        assert settings.data["global_news_queries"] == second.data_config["global_news_queries"] == ["earnings"]
        assert settings.data["tool_vendors"] == second.data_config["tool_vendors"] == {"get_news": "alpha_vantage"}
        assert second.data_config["data_vendors"]["technical_indicators"] == "yfinance"
    finally:
        first.store.close()
        second.store.close()


async def test_unacknowledged_pending_task_keeps_original_input(tmp_path: Path) -> None:
    """A crash before native acceptance must not replace the task with an empty continuation."""
    calls = []

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FakeHarness:
        """Create a new synthetic native session."""
        return FakeHarness(spec, emit, calls, [])

    settings = Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    workflow.store.begin_step("analyst:market", "Market Analyst", "Original pending input with essential evidence")
    workflow.store.close()
    resumed = Workflow.resume(workflow.directory, session_factory=factory)
    await resumed.run()
    assert calls[0][1] == "Original pending input with essential evidence"


async def test_same_run_cannot_be_driven_twice(tmp_path: Path) -> None:
    """Reject a second runner before it opens or submits to a native session."""
    settings = Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings)
    with locked(workflow.directory / "execution", wait=False) as acquired:
        assert acquired
        with pytest.raises(RuntimeError, match="already active"):
            await workflow.run()


def test_relative_storage_produces_absolute_subprocess_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve paths before harnesses change into their isolated workspaces."""
    monkeypatch.chdir(tmp_path)
    workflow = Workflow("NVDA", "2026-01-02", Settings(state_dir=Path("relative"), memory_enabled=False))
    assert workflow.directory.is_absolute()
    assert workflow.context_file.is_absolute()
    workflow.store.close()


async def test_persistence_failure_never_reports_success_and_closes_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed transcript flush cannot leave completed status or a live database handle."""
    calls = []

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FakeHarness:
        """Supply complete role outputs before the simulated disk failure."""
        return FakeHarness(spec, emit, calls, [])

    close = EventRecorder.close

    def fail_close(recorder: EventRecorder) -> None:
        """Drain the real writer before simulating a failed persistence operation."""
        close(recorder)
        raise RuntimeError("simulated flush failure")

    monkeypatch.setattr(EventRecorder, "close", fail_close)
    workflow = Workflow("NVDA", "2026-01-02", Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False),
                        session_factory=factory)
    with pytest.raises(RuntimeError, match="flush failure"):
        await workflow.run()
    with pytest.raises(sqlite3.ProgrammingError):
        workflow.store.db.execute("SELECT 1")
    saved = RunStore(workflow.directory)
    assert saved.metadata("status") == "failed"
    assert not any(event.kind == "run_end" for event in saved.events())
    saved.close()


async def test_large_replay_yields_before_starting_more_model_work(tmp_path: Path) -> None:
    """Keep the UI event loop available while restoring a long saved transcript."""
    calls = []
    ticks = 0
    seen = 0

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FakeHarness:
        """Use offline role outputs after replay finishes."""
        return FakeHarness(spec, emit, calls, [])

    async def heartbeat() -> None:
        """Represent terminal timers and input callbacks during transcript replay."""
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    async def observe(event: AgentEvent) -> None:
        """Check that replay does not monopolize the loop before its next batch."""
        nonlocal seen
        seen += 1
        if seen == 512:
            assert ticks > 0

    settings = Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    workflow.store.append([AgentEvent("text", "Market Analyst", "analysis", "recorded") for _ in range(1024)])
    timer = asyncio.create_task(heartbeat())
    try:
        await workflow.run(observe)
    finally:
        timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)


async def test_finished_stages_release_idle_cli_processes(tmp_path: Path) -> None:
    """Bound native-process lifetime to its stage while retaining debaters across rounds."""
    calls = []
    live = set()

    class CountedHarness(FakeHarness):
        async def close(self) -> None:
            """Record release of the synthetic native process."""
            live.discard(self.spec.agent)

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> CountedHarness:
        """Assert that completed stages do not accumulate idle native clients."""
        live.add(spec.agent)
        assert len(live) <= 3
        return CountedHarness(spec, emit, calls, [])

    workflow = Workflow("NVDA", "2026-01-02", Settings(state_dir=tmp_path, analysts=("market", "news"),
                                                       research_rounds=2, memory_enabled=False), session_factory=factory)
    await workflow.run()
    assert live == set()


async def test_debate_chat_input_is_shared_without_changing_native_prompts(tmp_path: Path) -> None:
    """Expose one shared input and stable turns while retaining exact per-role submissions."""
    calls: list[tuple[str, str]] = []
    events: list[AgentEvent] = []

    class NativeTurnHarness(FakeHarness):
        async def prompt(self, text: str) -> AgentResult:
            """Emit a native turn identifier that presentation grouping must preserve."""
            native = AgentEvent("status", self.spec.agent, self.spec.stage, data={"turn_id": "native-turn-id"})
            await self.emit(native)
            assert native.data == {"turn_id": "native-turn-id"}
            return await super().prompt(text)

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> NativeTurnHarness:
        """Keep the existing native session behavior for the presentation-only change."""
        return NativeTurnHarness(spec, emit, calls, [])

    async def observe(event: AgentEvent) -> None:
        """Collect the normalized event contract seen by the chat view."""
        events.append(event)

    settings = Settings(state_dir=tmp_path, analysts=("market", "news"), research_rounds=2, memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    await workflow.run(observe)
    shared = [event for event in events if event.kind == "conversation_input"]
    assert [event.stage for event in shared] == ["research", "risk"]
    assert shared[0].agent == "Input"
    assert shared[0].data == {"participants": ["Bull Researcher", "Bear Researcher"], "total_rounds": 2}
    assert "Instrument: NVDA" in shared[0].text and "Market Analyst report" in shared[0].text
    assert "Trader proposal" in shared[1].text and "Portfolio not supplied" in shared[1].text
    turns = [event for event in events if event.kind == "turn_start" and event.stage == "research"]
    assert [event.data["conversation_turn_id"] for event in turns] == [
        "research:1:bull", "research:1:bear", "research:2:bull", "research:2:bear"
    ]
    finished = [event for event in events if event.kind == "conversation_output" and event.stage == "research"]
    assert [event.data["conversation_turn_id"] for event in finished] == [event.data["conversation_turn_id"] for event in turns]
    assert all(event.text == f"{event.agent} report based on synthetic evidence" for event in finished)
    for event in events:
        if event.stage in {"research", "risk"} and event.kind in {"prompt", "text", "agent_start", "agent_end"}:
            assert event.data["conversation_turn_id"].startswith(f"{event.stage}:")
        if event.stage in {"research", "risk"} and event.data.get("turn_id"):
            assert event.data["turn_id"] == "native-turn-id"
            assert event.data["conversation_turn_id"] != event.data["turn_id"]
    assert [(event.agent, event.text) for event in events if event.kind == "prompt"] == calls
    bear_first = next(text for agent, text in calls if agent == "Bear Researcher")
    assert "Market Analyst report" in bear_first and "Bull Researcher report" in bear_first


async def test_debate_resume_reuses_chat_identity_without_restarting_completed_turns(tmp_path: Path) -> None:
    """Keep completed bubbles once while allowing a failed turn to resume visibly."""
    calls: list[tuple[str, str]] = []
    failure = ["Bear Researcher"]

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FakeHarness:
        """Fail one debate attempt after the native harness accepted its input."""
        return FakeHarness(spec, emit, calls, failure)

    settings = Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    with pytest.raises(RuntimeError, match="interrupted provider"):
        await workflow.run()
    resumed = Workflow.resume(workflow.directory, session_factory=factory)
    await resumed.run()
    saved = RunStore(workflow.directory)
    try:
        recorded = saved.events()
    finally:
        saved.close()
    shared = [event for event in recorded if event.kind == "conversation_input" and event.stage == "research"]
    assert len(shared) == 1
    assert shared[0].event_id == f"{workflow.run_id}:research:input"
    turns = [event for event in recorded if event.kind == "turn_start" and event.stage == "research"]
    assert sum(event.agent == "Bull Researcher" for event in turns) == 1
    bear = [event for event in turns if event.agent == "Bear Researcher"]
    assert len(bear) == 2
    assert bear[0].event_id != bear[1].event_id
    assert bear[0].data["conversation_turn_id"] == bear[1].data["conversation_turn_id"] == "research:1:bear"
    assert sum(agent == "Bull Researcher" for agent, _ in calls) == 1
    complete_bull = [event for event in recorded if event.kind == "conversation_output" and event.agent == "Bull Researcher"]
    assert len(complete_bull) == 1


async def test_debate_reconstructs_completed_public_output_when_stream_events_were_not_saved(tmp_path: Path) -> None:
    """A durable completed step supplies its final bubble after an interrupted event flush."""
    calls: list[tuple[str, str]] = []

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FakeHarness:
        """Run only the remaining roles, without regenerating saved debate work."""
        return FakeHarness(spec, emit, calls, [])

    settings = Settings(state_dir=tmp_path, analysts=("market",), memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    workflow.store.begin_step("research:1:bull", "Bull Researcher", "Previously delivered original task")
    workflow.store.finish_step("research:1:bull", "The saved final bull argument")
    finished_at = workflow.store.step("research:1:bull")["updated"]
    workflow.store.close()

    resumed = Workflow.resume(workflow.directory, session_factory=factory)
    await resumed.run()
    assert not any(agent == "Bull Researcher" for agent, _ in calls)
    saved = RunStore(workflow.directory)
    try:
        outputs = [event for event in saved.events() if event.kind == "conversation_output" and event.agent == "Bull Researcher"]
    finally:
        saved.close()
    assert len(outputs) == 1
    assert outputs[0].text == "The saved final bull argument"
    assert outputs[0].timestamp == finished_at
    assert outputs[0].data["conversation_turn_id"] == "research:1:bull"
