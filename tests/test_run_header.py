import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual.widgets import Static, Tooltip

from open_trading_agents import cli
from open_trading_agents.analysis.models import AgentEvent, HarnessConfig
from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.analysis.workflow import RunResult
from open_trading_agents.config import HarnessSettings, Settings
from open_trading_agents.ui import app as tui
from open_trading_agents.ui.app import TradingApp


@pytest.mark.parametrize("size,preset", [((170, 43), "astra"), ((80, 24), "astra"), ((80, 24), "research-" + "a" * 55)])
async def test_analyst_panels_show_profile_labels_and_individual_hover_details(size: tuple[int, int], preset: str) -> None:
    """Keep profile labels inside their panels and reveal only that agent's configuration on hover.

    Args:
        size: Wide or compact terminal dimensions.
        preset: Selected preset name, including a maximum-length name.
    """
    profiles = {
        "Market Analyst": HarnessConfig(kind="pi", provider="private-provider", model="market-model", effort="low",
                                        command=("private-command",), environment={"TOKEN": "private-environment-value"}),
        "News Analyst": HarnessConfig(kind="claude-code", model="claude-news-model", effort="high", extra_args=("private-argument",)),
    }
    app = TradingApp(ticker="NVDA", selected_analysts=("market", "news"), agent_configs=profiles,
                     agent_presets={"Market Analyst": preset})
    app.TOOLTIP_DELAY = 0.01

    async with app.run_test(size=size, tooltips=True) as pilot:
        await app.receive(AgentEvent("text", "Market Analyst", "analysis", "Market evidence"))
        app.flush_events()
        await pilot.pause()
        assert not app.query("#model-config")
        run_header = str(app.query_one("#run-status", Static).render())
        assert "Open Trading Agents · NVDA" in run_header and "Total" in run_header
        market = app.query_one("#agent-status-0", Static)
        news = app.query_one("#agent-status-1", Static)
        market_title = str(market.render()).splitlines()[0]
        news_title = str(news.render()).splitlines()[0]
        assert re.fullmatch(r"Market(?: Analyst)? · .+ · \d{2}:\d{2}", market_title)
        displayed_preset = market_title.split(" · ")[1]
        assert displayed_preset and preset.startswith(displayed_preset)
        assert Text(market_title).cell_len <= market.content_region.width
        if len(preset) == 64:
            assert len(displayed_preset) < len(preset)
        assert re.fullmatch(r"News(?: Analyst)? · claude-news-model · \d{2}:\d{2}", news_title)
        assert "market-model" not in market_title and "low" not in market_title
        assert preset not in "".join(segment.text for segment in app.agent_views["Market Analyst"].render_line(0))

        await pilot.hover(market, offset=(2, 0))
        await pilot.pause(0.03)
        tooltip = app.screen.query_one(Tooltip)
        details = str(tooltip.render())
        assert tooltip.display
        for value in ("Market Analyst", f"Preset: {preset}", "Harness: pi", "Model: market-model", "Effort: low"):
            assert value in details
        for value in ("News Analyst", "claude-news-model", "private-provider", "private-command", "private-environment-value"):
            assert value not in details

        await pilot.hover(market, offset=(2, 1))
        await pilot.pause(0.03)
        assert not tooltip.display
        await pilot.hover(news, offset=(2, 0))
        await pilot.pause(0.03)
        details = str(tooltip.render())
        assert tooltip.display and "News Analyst" in details and "Harness: claude-code" in details
        assert "Model: claude-news-model" in details and "Effort: high" in details
        assert "Market Analyst" not in details and "Preset:" not in details and "private-argument" not in details

        await pilot.hover("#run-status")
        await pilot.press("right", "f")
        assert app.fullscreen and news.is_on_screen
        assert "claude-news-model" in str(news.render()).splitlines()[0]
        assert not market.is_on_screen
        assert app.screen.max_scroll_x == app.screen.max_scroll_y == 0
        assert app.query_one("#shortcuts", Static).region.bottom <= size[1]
        await pilot.press("f")
        assert market.is_on_screen and news.is_on_screen


async def test_missing_model_uses_harness_default_in_its_panel() -> None:
    """Keep unspecified model choices explicit without inventing an observed model identity."""
    app = TradingApp(selected_analysts=("market",), agent_configs={"Market Analyst": HarnessConfig(kind="pi")})
    app.TOOLTIP_DELAY = 0.01
    async with app.run_test(size=(120, 32), tooltips=True) as pilot:
        await pilot.pause()
        assert not app.query("#model-config")
        status = app.query_one("#agent-status-0", Static)
        assert "Market Analyst · Harness default ·" in str(status.render())
        await pilot.hover(status, offset=(2, 0))
        await pilot.pause(0.03)
        tooltip = app.screen.query_one(Tooltip)
        assert tooltip.display
        details = str(tooltip.render())
        assert "Model: Harness default" in details and "Effort: Harness default" in details
        assert "None" not in details and "Preset:" not in details


@pytest.mark.parametrize("size", [(170, 43), (80, 24)])
async def test_debate_and_manager_panels_keep_their_own_labels_and_hover_details(size: tuple[int, int]) -> None:
    """Retain per-role identities in prompt panes and shared transcripts through fullscreen changes.

    Args:
        size: Wide or compact terminal dimensions.
    """
    profiles = {role.name: HarnessConfig(kind="pi", model=f"{key}-model", effort="low") for key, role in ROLES.items()}
    profiles["Bear Researcher"] = HarnessConfig(kind="codex", model="bear-model", effort="high")
    app = TradingApp(agent_configs=profiles, agent_presets={"Bull Researcher": "astra", "Research Manager": "decider"})
    app.TOOLTIP_DELAY = 0.01

    async with app.run_test(size=size, tooltips=True) as pilot:
        await pilot.press("tab")
        assert not app.query("#model-config")
        bull = app.prompt_views["research"]["Bull Researcher"]
        bear = app.prompt_views["research"]["Bear Researcher"]
        assert bull.border_title == "Bull Researcher · astra"
        assert bear.border_title == "Bear Researcher · bear-model"
        assert bull.tooltip is None and bear.tooltip is None

        await pilot.hover(bull, offset=(2, 0))
        await pilot.pause(0.03)
        tooltip = app.screen.query_one(Tooltip)
        assert tooltip.display and "Preset: astra" in str(tooltip.render())
        assert "Bear Researcher" not in str(tooltip.render())
        await pilot.hover(bull, offset=(2, 2))
        await pilot.pause(0.03)
        assert not tooltip.display
        await pilot.hover(bear, offset=(2, 0))
        await pilot.pause(0.03)
        assert tooltip.display and "Harness: codex" in str(tooltip.render())
        assert "Bull Researcher" not in str(tooltip.render()) and "Preset:" not in str(tooltip.render())
        await pilot.hover("#run-status")

        for turn, name in enumerate(("Bull Researcher", "Bear Researcher"), 1):
            data = {"round": 1, "turn": turn, "total_rounds": 1, "conversation_turn_id": f"research:1:{turn}"}
            await app.receive(AgentEvent("turn_start", name, "research", data=data))
            await app.receive(AgentEvent("text", name, "research", "Evidence", data))
            await app.receive(AgentEvent("agent_end", name, "research", data=data))
        app.flush_events()
        await pilot.pause()
        chat = app.stage_views["research"]
        headers = [segment.text for row in range(chat.size.height) for segment in chat.render_line(row)]
        rendered = "".join(headers)
        assert "Bull Researcher · astra" in rendered and "Bear Researcher · bear-model" in rendered
        await pilot.press("left", "left", "f")
        assert app._debate_fullscreen["research"] is bull
        assert bull.border_title == "Bull Researcher · astra"
        await pilot.hover(bull, offset=(2, 0))
        await pilot.pause(0.03)
        assert tooltip.display and "Model: bull-model" in str(tooltip.render())
        await pilot.hover("#run-status")
        await pilot.press("f", "right", "right", "f")
        assert app._debate_fullscreen["research"] is chat
        assert not bull.is_on_screen and not bear.is_on_screen
        assert "Bear Researcher · bear-model" in "".join(segment.text for row in range(chat.size.height) for segment in chat.render_line(row))
        assert app.screen.max_scroll_x == app.screen.max_scroll_y == 0
        assert app.query_one("#shortcuts", Static).region.bottom <= size[1]

        await pilot.press("tab")
        await app.receive(AgentEvent("text", "Research Manager", "research_decision", "Decision evidence"))
        app.flush_events()
        await pilot.pause()
        manager = app.stage_views["research_decision"]
        assert "Research Manager · decider" in "".join(segment.text for segment in manager.render_line(0))
        assert "Model: research_manager-model" in app._agent_presentations["Research Manager"].details


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
        deep=HarnessSettings(kind="claude-code", model="selected-manager", effort="high"),
        agent_presets={"market": "worker-profile", "portfolio_manager": "manager-profile"},
        output_language="Korean",
    )
    if individual:
        settings.agents["bull"] = HarnessSettings(kind="codex", model="selected-bull", effort="medium")
        settings.agents["chat"] = HarnessSettings(kind="pi", model="selected-chat", effort="high")
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
    assert "working_config" not in received and "decision_config" not in received
    assert received["output_language"] == settings.output_language
    assert received["report_chat"].config == settings.model_settings(role="chat").runtime()
    assert received["report_chat"].config != settings.model_settings(role="portfolio_manager").runtime()
    profiles = received["agent_configs"]
    assert isinstance(profiles, dict)
    assert set(profiles) == {role.name for role in ROLES.values()}
    assert profiles["Market Analyst"] == settings.model_settings().runtime()
    assert profiles["Market Analyst"].provider == "local"
    assert profiles["Research Manager"] == settings.model_settings(True).runtime()
    assert received["agent_presets"] == {"Market Analyst": "worker-profile", "Portfolio Manager": "manager-profile"}
    if individual:
        assert profiles["Bull Researcher"] == HarnessConfig(kind="codex", model="selected-bull", effort="medium")
    else:
        assert profiles["Bull Researcher"] == settings.model_settings().runtime()
