import json
from collections import Counter
from pathlib import Path

import pytest

from open_trading_agents import __version__
from open_trading_agents.analysis.models import AgentEvent, AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.analysis.workflow import Workflow
from open_trading_agents.config import HarnessSettings, Settings
from open_trading_agents.persistence.storage import RunStore


async def test_reports_preserve_observed_models_across_resume_and_mixed_tiers(tmp_path: Path) -> None:
    """Keep native defaults, model changes, unknowns, and tier selectors distinct in saved reports."""
    calls: Counter[str] = Counter()

    class FixtureSession:
        def __init__(self, config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> None:
            """Retain the selected harness and native role identity."""
            self.config, self.spec, self.emit = config, spec, emit
            self.session_id = spec.session_id or f"native-{spec.agent}"

        async def prompt(self, text: str) -> AgentResult:
            """Report model metadata and interrupt one role before its first completion."""
            agent = self.spec.agent
            calls[agent] += 1
            if agent != "Trader":
                model = f"native-{self.config.kind}"
                if agent == "Neutral Analyst" and calls[agent] > 1:
                    model = "native-replacement"
                event = AgentEvent("model", agent, self.spec.stage, data={
                    "harness": self.config.kind, "provider": "native-provider", "model": model,
                    "api_key": "excluded-from-report-settings", "base_url": "http://private-endpoint",
                })
                await self.emit(event)
                await self.emit(event)
            await self.emit(AgentEvent("status", agent, self.spec.stage, data={"prompt_accepted": True}))
            if agent == "Neutral Analyst" and calls[agent] == 1:
                raise RuntimeError("interrupted fixture")
            output = json.dumps({
                "recommendation": "Hold", "rationale": "Evidence", "strategic_actions": "Wait",
                "action": "Hold", "reasoning": "Evidence", "rating": "Hold",
                "executive_summary": "Summary", "investment_thesis": "Evidence",
            })
            return AgentResult(output, self.session_id)

        async def close(self) -> None:
            """Release this offline session."""

        async def cancel(self) -> None:
            """Cancel this offline session."""

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> FixtureSession:
        """Create an offline session without inferring model identity from configuration."""
        return FixtureSession(config, spec, emit)

    settings = Settings(
        state_dir=tmp_path, analysts=("market",), memory_enabled=False,
        harness=HarnessSettings(kind="claude"),
        quick=HarnessSettings(kind="pi", model="quick-selector", provider="quick-provider"),
        deep=HarnessSettings(kind="codex", model="deep-selector", provider="deep-provider"),
        agents={"trader": HarnessSettings(kind="claude", model="trader-selector", command=("private-command",))},
        data={"tool_vendors": {"get_news": "alpha_vantage"}},
    )
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory)
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        await workflow.run()
    resumed = Workflow.resume(workflow.directory, session_factory=factory)
    result = await resumed.run()
    report_settings = json.loads((result.directory / "reports" / "settings.json").read_text())
    identities = report_settings["runtime_agents"]
    assert calls["Market Analyst"] == 1
    assert identities.count({"agent": "Market Analyst", "harness": "pi", "provider": "native-provider", "model": "native-pi"}) == 1
    assert {"agent": "Research Manager", "harness": "codex", "provider": "native-provider", "model": "native-codex"} in identities
    assert {"agent": "Trader", "harness": "claude", "provider": None, "model": None} in identities
    configured = {row["role"]: row for row in report_settings["configured_agents"]}
    assert configured["trader"]["harness"] == "claude" and configured["trader"]["model"] == "trader-selector"
    assert configured["bull"]["model"] == "quick-selector"
    assert configured["research_manager"]["model"] == "deep-selector"
    assert "private-command" not in json.dumps(report_settings)
    assert "trader-selector (Trader)" in (result.directory / "reports" / "complete_report.md").read_text()
    assert [row["model"] for row in identities if row["agent"] == "Neutral Analyst"] == ["native-pi", "native-replacement"]
    assert report_settings["quick_harness"] == "pi" and report_settings["deep_harness"] == "codex"
    assert report_settings["quick_provider"] == "quick-provider" and report_settings["deep_provider"] == "deep-provider"
    assert report_settings["quick_model"] == "quick-selector" and report_settings["deep_model"] == "deep-selector"
    assert report_settings["version"] == __version__
    assert report_settings["tool_vendors"] == {"get_news": "alpha_vantage"}
    assert "get news: alpha_vantage" in (result.directory / "reports" / "complete_report.html").read_text()
    assert "excluded-from-report-settings" not in json.dumps(report_settings)
    assert "http://private-endpoint" not in json.dumps(report_settings)
    store = RunStore(result.directory)
    try:
        assert store.metadata("runtime_agents") == identities
        assert store.metadata("report_settings") == report_settings
    finally:
        store.close()
