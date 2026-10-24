import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Static

from open_trading_agents import cli
from open_trading_agents.analysis.models import AgentEvent, HarnessConfig
from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.analysis.workflow import RunResult
from open_trading_agents.config import HarnessSettings, Settings
from open_trading_agents.ui import app as tui
from open_trading_agents.ui.app import TradingApp


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
        for value in ("pi", working.model, "low", "codex", decision.model, "high"):
            assert value in details
        for value in ("local-provider", "private-command", "private-environment-value", "private-argument"):
            assert value not in details + str(config.tooltip)
        assert "…" not in details and "..." not in details
        assert config.display and config.region.bottom <= app.query_one("#stage-tabs").region.y
        assert app.query_one("#shortcuts", Static).region.bottom <= size[1]
        assert app.screen.max_scroll_x == app.screen.max_scroll_y == 0


async def test_shared_harness_defaults_are_shown_once_without_inventing_model_identity() -> None:
    """A shared profile needs one configuration row and honest harness-default labels."""
    profile = HarnessConfig(kind="pi")
    app = TradingApp(ticker="NVDA", working_config=profile, decision_config=profile)
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        details = str(app.query_one("#model-config", Static).render())
        assert "Agents" in details
        assert details.count("pi") == 1
        assert "Harness default" in details
        assert "None" not in details
        assert "\n" not in details


@pytest.mark.parametrize("size", [(170, 43), (80, 24)])
async def test_individual_profiles_follow_visible_agent_without_expanding_the_header(size: tuple[int, int]) -> None:
    """Show the selected role's effective settings across panes and stages.

    Args:
        size: Wide or compact terminal dimensions.
    """
    profiles = {
        role.name: HarnessConfig(kind="pi", provider="local", model=f"{key}-model", effort="low",
                                 command=("private-command",), environment={"TOKEN": "private-environment-value"})
        for key, role in ROLES.items()
    }
    profiles["News Analyst"] = HarnessConfig(kind="claude", model="news-model", effort="high", extra_args=("private-argument",))
    profiles["Bear Researcher"] = HarnessConfig(kind="codex", model="bear-model", effort="high")
    app = TradingApp(ticker="NVDA", selected_analysts=("market", "news"), agent_configs=profiles)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        config = app.query_one("#model-config", Static)

        def assert_header(expected: str) -> None:
            """Check visible identity and keep all panes and controls inside the terminal.

            Args:
                expected: Exact public configuration of the currently inspected role.
            """
            assert str(config.render()) == expected
            assert config.display and config.region.height <= 2
            assert config.region.bottom <= app.query_one("#stage-tabs").region.y
            assert app.query_one("#shortcuts", Static).region.bottom <= size[1]
            assert app.screen.max_scroll_x == app.screen.max_scroll_y == 0

        assert_header("Market Analyst · pi · market-model · low")
        tooltip = str(config.tooltip)
        for name, profile in profiles.items():
            assert name in tooltip and profile.model in tooltip
        for value in ("local", "private-command", "private-environment-value", "private-argument"):
            assert value not in tooltip

        await pilot.press("right")
        assert_header("News Analyst · claude · news-model · high")
        await pilot.press("f")
        assert app.fullscreen
        assert_header("News Analyst · claude · news-model · high")
        await pilot.press("f", "tab")
        assert_header("Bull Researcher · pi · bull-model · low")

        await app.receive(AgentEvent("turn_start", "Bear Researcher", "research", data={"round": 1, "turn": 2, "total_rounds": 1}))
        app.flush_events()
        await pilot.pause()
        assert_header("Bear Researcher · codex · bear-model · high")
        await pilot.press("left", "left", "f")
        assert app._debate_fullscreen["research"] is app.prompt_views["research"]["Bull Researcher"]
        assert_header("Bull Researcher · pi · bull-model · low")
        await pilot.press("f", "right", "right")
        assert_header("Bear Researcher · codex · bear-model · high")

        for name, key in (("Research Manager", "research_manager"), ("Trader", "trader"), ("Aggressive Analyst", "aggressive")):
            await pilot.press("tab")
            assert_header(f"{name} · pi · {key}-model · low")
        await app.receive(AgentEvent("turn_start", "Conservative Analyst", "risk", data={"round": 1, "turn": 2, "total_rounds": 1}))
        app.flush_events()
        await pilot.pause()
        assert_header("Conservative Analyst · pi · conservative-model · low")
        for _ in range(2):
            await pilot.press("tab")
            assert_header("Portfolio Manager · pi · portfolio_manager-model · low")


@pytest.mark.parametrize("individual", [False, True])
def test_cli_forwards_final_working_and_decision_settings_to_the_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, individual: bool,
) -> None:
    """Forward effective group or role settings so the header describes the configured run.

    Args:
        tmp_path: Isolated run output location.
        monkeypatch: Scoped terminal and UI replacements.
        individual: Whether one role uses its own harness selection.
    """
    settings = Settings(
        harness=HarnessSettings(kind="pi", model="unused-shared-model"),
        quick=HarnessSettings(kind="pi", provider="local", model="selected-worker", effort="low"),
        deep=HarnessSettings(kind="claude", model="selected-manager", effort="high"),
        output_language="Korean",
    )
    if individual:
        settings.agents["bull"] = HarnessSettings(kind="codex", model="selected-bull", effort="medium")
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
    assert received["output_language"] == settings.output_language
    if individual:
        profiles = received["agent_configs"]
        assert isinstance(profiles, dict)
        assert set(profiles) == {role.name for role in ROLES.values()}
        assert profiles["Bull Researcher"] == HarnessConfig(kind="codex", model="selected-bull", effort="medium")
        assert profiles["Market Analyst"] == settings.model_settings().runtime()
        assert profiles["Market Analyst"].provider == "local"
        assert profiles["Research Manager"] == settings.model_settings(True).runtime()
    else:
        assert received["agent_configs"] is None
