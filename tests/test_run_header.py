import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Static

from open_trading_agents import cli, tui
from open_trading_agents.config import HarnessSettings, Settings
from open_trading_agents.domain import AgentEvent, HarnessConfig
from open_trading_agents.tui import TradingApp
from open_trading_agents.workflow import RunResult


@pytest.mark.parametrize("size", [(170, 43), (80, 24)])
async def test_run_header_shows_selected_tiers_without_stage_duplication_or_hidden_values(size: tuple[int, int]) -> None:
    """Keep the chosen models and effort visible on wide and compact terminals without exposing native process settings."""
    working = HarnessConfig(kind="pi", provider="local-provider", model="working-model-with-a-long-name", effort="low",
                            command=("private-command",), environment={"TOKEN": "private-environment-value"})
    decision = HarnessConfig(kind="codex", model="decision-model-with-a-long-name", effort="high",
                             extra_args=("private-argument",))
    app = TradingApp(ticker="NVDA", working_config=working, decision_config=decision)
    async with app.run_test(size=size) as pilot:
        await app.receive(AgentEvent("stage_start", "Workflow", "analysis"))
        app.flush_events()
        app.refresh_clocks()
        await pilot.pause()
        header = str(app.query_one("#run-status", Static).render())
        config = app.query_one("#model-config", Static)
        details = str(config.render())
        assert "Open Trading Agents · NVDA" in header and "Total " in header
        assert "Running" not in header and "Analysis" not in header
        assert "working" in details.lower() and "manager" in details.lower()
        for value in ("pi", "local-provider", working.model, "low", "codex", decision.model, "high"):
            assert value in details
        for value in ("private-command", "private-environment-value", "private-argument"):
            assert value not in details + str(config.tooltip)
        assert "…" not in details and "..." not in details
        assert config.display and config.region.bottom <= app.query_one("#stage-tabs").region.y
        assert app.query_one("#shortcuts", Static).region.bottom <= size[1]
        assert app.screen.max_scroll_x == app.screen.max_scroll_y == 0


async def test_shared_native_defaults_are_shown_once_without_inventing_model_identity() -> None:
    """A shared profile needs one configuration row and honest native-default labels."""
    profile = HarnessConfig(kind="pi")
    app = TradingApp(ticker="NVDA", working_config=profile, decision_config=profile)
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        details = str(app.query_one("#model-config", Static).render())
        assert "Agents" in details
        assert details.count("pi") == 1
        assert "native" in details.lower()
        assert "None" not in details
        assert "\n" not in details


def test_cli_forwards_final_working_and_decision_settings_to_the_tui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Display the actual per-run tier choices rather than the shared configuration they override."""
    settings = Settings(
        harness=HarnessSettings(kind="pi", model="unused-shared-model"),
        quick=HarnessSettings(kind="pi", provider="local", model="selected-worker", effort="low"),
        deep=HarnessSettings(kind="claude", model="selected-manager", effort="high"),
    )
    result = RunResult("fixture", tmp_path, "Hold", {})
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "usage.json").write_text(json.dumps({"totals": {"input_tokens": 0, "output_tokens": 0}}))
    workflow = SimpleNamespace(settings=settings, ticker="NVDA", asset_type="stock", directory=tmp_path,
                               run_id="fixture", run=object())
    received: dict[str, object] = {}

    def create_app(run: object, **kwargs: object) -> SimpleNamespace:
        """Capture UI inputs without starting a screen, native model, or workflow."""
        assert run is workflow.run
        received.update(kwargs)
        return SimpleNamespace(run=lambda: None, error=None, result=result)

    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True), stdout=SimpleNamespace(isatty=lambda: True)))
    monkeypatch.setattr(tui, "TradingApp", create_app)
    assert cli.run_workflow(workflow, headless=False, events=False) is result
    assert received["ticker"] == "NVDA"
    assert received["working_config"] == settings.model_settings().runtime()
    assert received["decision_config"] == settings.model_settings(True).runtime()
