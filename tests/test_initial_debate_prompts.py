import pytest

from open_trading_agents.analysis.models import STAGES, AgentEvent
from open_trading_agents.analysis.prompts import ROLES, system_prompt
from open_trading_agents.ui.app import TradingApp


@pytest.mark.parametrize("stage,role_key", [("research", "bear"), ("risk", "neutral")])
async def test_debate_prompts_are_ready_before_turns_and_native_events_remain_authoritative(stage: str, role_key: str) -> None:
    """Inspect every prompt before work starts, preserving scroll on matching events and applying recorded overrides."""
    app = TradingApp(output_language="Korean")
    roles = ("bull", "bear", "aggressive", "conservative", "neutral")
    assert sum(len(panes) for panes in app.prompt_views.values()) == len(roles)
    for key in roles:
        role = ROLES[key]
        pane = app.prompt_views[role.stage][role.name]
        assert pane.document.plain() == system_prompt(role, "Korean")
        assert "Write the report in Korean." in pane.document.plain()
        assert not pane.following
    assert all(activity.status == "waiting" for activity in app.stages.values())
    assert not app.current_speakers and not app.usage and not app._pending
    assert all(not view.bubbles for view in app.stage_views.values())

    async with app.run_test(size=(120, 36)) as pilot:
        for _ in range([key for key, _ in STAGES].index(stage)):
            await pilot.press("tab")
        await pilot.pause()
        role = ROLES[role_key]
        pane = app.prompt_views[stage][role.name]
        assert pane.is_on_screen and pane.max_scroll_y > 0
        assert pane.scroll_y == 0
        pane.navigate("down")
        await pilot.pause()
        position = pane.scroll_y
        document = pane.document
        assert position > 0

        await app.receive(AgentEvent("system", role.name, stage, system_prompt(role, "Korean")))
        app.flush_events()
        await pilot.pause()
        assert pane.document is document and pane.scroll_y == position

        recorded = f"Recorded instructions for {role.name}.\n" + "Use the original run's evidence.\n" * 80
        await app.receive(AgentEvent("system_prompt", role.name, stage, recorded))
        app.flush_events()
        await pilot.pause()
        assert pane.document.plain() == recorded
        assert pane.document is not document and pane.scroll_y == 0
        assert not pane.following
        assert "Recorded instructions" in "".join(segment.text for segment in pane.render_line(0))
