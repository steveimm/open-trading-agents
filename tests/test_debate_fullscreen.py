import pytest
from textual.widgets import Static, Tooltip

from open_trading_agents.analysis.models import STAGES, AgentEvent, HarnessConfig
from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.ui.app import TradingApp


@pytest.mark.parametrize("stage,size", [("research", (160, 44)), ("risk", (80, 24))])
async def test_debate_transcript_fullscreen_retains_scrollback_and_collapsed_turns(stage: str, size: tuple[int, int]) -> None:
    """Expand the conversation alone and restore prompt panes without moving the reader or discarding content."""
    profiles = {role.name: HarnessConfig(kind="pi", model=f"{key}-model") for key, role in ROLES.items()}
    app = TradingApp(agent_configs=profiles)
    app.TOOLTIP_DELAY = 0.01
    async with app.run_test(size=size, tooltips=True) as pilot:
        for _ in range([key for key, _ in STAGES].index(stage)):
            await pilot.press("tab")
        first_agent, second_agent = list(app.prompt_views[stage])[:2]
        await app.receive(AgentEvent("conversation_input", "Workflow", stage, "Shared evidence"))
        for turn, agent, text in ((1, first_agent, "Earlier contribution"),
                                  (2, second_agent, "\n".join(f"Evidence line {number}" for number in range(150)))):
            metadata = {"conversation_turn_id": f"{stage}:1:{turn}", "round": 1, "turn": turn, "total_rounds": 1}
            await app.receive(AgentEvent("turn_start", agent, stage, data=metadata))
            await app.receive(AgentEvent("text", agent, stage, text, {**metadata, "message_id": f"reply-{turn}"}))
            if turn == 1:
                await app.receive(AgentEvent("agent_end", agent, stage, data={**metadata, "output": text}))
        app.flush_events()
        await pilot.pause()
        chat = app.stage_views[stage]
        earlier = next(bubble for bubble in chat.bubbles if bubble.agent == first_agent)
        await pilot.press("home", "]", "]", "space")
        assert earlier.collapsed
        chat.scroll_to(y=20, animate=False, immediate=True)
        await pilot.pause()
        previous_line = "".join(segment.text for segment in chat.render_line(1)).strip(" │")
        previous_width = chat.size.width
        assert not chat.following and app._active_view() is chat

        title_offset = chat.scrollable_content_region.offset - chat.region.offset
        await pilot.hover(chat, offset=(title_offset.x + 4, title_offset.y))
        await pilot.pause(0.03)
        tooltip = app.screen.query_one(Tooltip)
        assert tooltip.display and profiles[second_agent].model in str(tooltip.render())
        assert profiles[first_agent].model not in str(tooltip.render())

        await pilot.press("f")
        await pilot.pause()
        assert app._debate_fullscreen[stage] is chat
        assert not app.query_one(f"#prompts-{stage}").display
        assert all(not pane.is_on_screen for pane in app.prompt_views[stage].values())
        assert chat.is_on_screen and chat.size.width > previous_width
        assert chat.region.x == 0 and chat.region.right == size[0]
        assert "".join(segment.text for segment in chat.render_line(1)).strip(" │") == previous_line
        assert earlier.collapsed and not chat.following
        title_offset = chat.scrollable_content_region.offset - chat.region.offset
        await pilot.hover(chat, offset=(title_offset.x + 4, title_offset.y))
        await pilot.pause(0.03)
        assert tooltip.display and profiles[second_agent].model in str(tooltip.render())
        assert not app.check_action("navigate", ("left",)) and not app.check_action("navigate", ("right",))
        await pilot.press("left", "right", "enter", "escape")
        assert app._active_view() is chat and chat.has_focus
        assert app._debate_fullscreen[stage] is chat

        await app.receive(AgentEvent("text", second_agent, stage, "\nNew streamed evidence", {"message_id": "reply-2"}))
        app.flush_events()
        await pilot.pause()
        assert "New streamed evidence" in chat.plain()
        await pilot.press("f")
        await pilot.pause()
        assert app._debate_fullscreen[stage] is None
        assert app.query_one(f"#prompts-{stage}").display
        assert all(pane.is_on_screen for pane in app.prompt_views[stage].values())
        assert chat.size.width == previous_width
        assert "".join(segment.text for segment in chat.render_line(1)).strip(" │") == previous_line
        assert earlier.collapsed and not chat.following
        assert "Earlier contribution" in chat.plain() and "Earlier contribution" not in chat.visible_plain()
        assert app.screen.max_scroll_x == app.screen.max_scroll_y == 0
        assert app.query_one("#shortcuts", Static).region.bottom <= size[1]


async def test_debate_fullscreen_states_are_independent_when_switching_stages() -> None:
    """Keep each debate's selected fullscreen pane while single-pane stages retain their normal controls."""
    app = TradingApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("tab", "f")
        research = app.stage_views["research"]
        assert app._debate_fullscreen["research"] is research
        await pilot.press("tab")
        assert app.active_stage == "research_decision"
        assert not app.check_action("fullscreen", ())
        assert "fullscreen" not in str(app.query_one("#shortcuts", Static).render())
        await pilot.press("f", "tab", "tab")
        assert app.active_stage == "risk" and app._debate_fullscreen["risk"] is None
        await pilot.press("left", "f")
        prompt = list(app.prompt_views["risk"].values())[-1]
        assert app._debate_fullscreen["risk"] is prompt
        await pilot.press("tab", "tab", "tab", "tab")
        await pilot.pause()
        assert app.active_stage == "research" and app._active_view() is research
        assert app._debate_fullscreen["research"] is research
        assert not app.query_one("#prompts-research").display
        await pilot.press("f")
        assert app._debate_fullscreen["research"] is None
        assert app._debate_fullscreen["risk"] is prompt
