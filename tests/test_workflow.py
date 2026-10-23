import asyncio
from pathlib import Path

import pytest

from open_trading_agents.config import Settings
from open_trading_agents.domain import AgentEvent, AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.workflow import Workflow


class FakeHarness:
    def __init__(self, spec: AgentSpec, emit: EventSink, calls: list[tuple[str, str]], failure: list[str]) -> None:
        """Retain a synthetic session identity and record newly supplied turns."""
        self.spec, self.emit, self.calls, self.failure = spec, emit, calls, failure
        self.session_id = spec.session_id or f"native:{spec.agent}"

    async def prompt(self, text: str) -> AgentResult:
        """Return role-appropriate fixtures after yielding to concurrent analysts."""
        self.calls.append((self.spec.agent, text))
        await self.emit(AgentEvent("prompt", self.spec.agent, self.spec.stage, text))
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
