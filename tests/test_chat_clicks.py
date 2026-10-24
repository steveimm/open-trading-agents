import pytest

from open_trading_agents.analysis.models import STAGES, AgentEvent
from open_trading_agents.ui.app import TradingApp
from open_trading_agents.ui.chat import ChatView


def content_position(view: ChatView, x: int, y: int) -> tuple[int, int]:
    """Locate a rendered cell within the bordered widget for a real pointer click.

    Args:
        view: Conversation viewport receiving the click.
        x: Column in the rendered content area.
        y: Row in the rendered content area.

    Returns:
        Widget-relative mouse coordinates including its border and padding.
    """
    region = view.scrollable_content_region
    return region.x - view.region.x + x, region.y - view.region.y + y


@pytest.mark.parametrize("stage,agent", [("research", "Bull Researcher"), ("risk", "Aggressive Analyst")])
async def test_bordered_debate_headers_toggle_at_their_visible_mouse_positions(stage: str, agent: str) -> None:
    """Click natural, pinned, and nested disclosures accurately after a compact resize."""
    app = TradingApp()
    async with app.run_test(size=(140, 40)) as pilot:
        for _ in range([key for key, _ in STAGES].index(stage)):
            await pilot.press("tab")
        evidence = "\n".join(f"Shared input {number}" for number in range(80))
        for event in (
            AgentEvent("conversation_input", "Workflow", stage, evidence),
            AgentEvent("turn_start", agent, stage, data={"round": 1, "turn": 1, "total_rounds": 1}),
            AgentEvent("reasoning", agent, stage, "Check the dated evidence.", {"message_id": "reasoning"}),
            AgentEvent("tool_start", agent, stage, '{"command":"python3 checks.py"}', {"tool": "bash", "tool_call_id": "script"}),
            AgentEvent("tool_result", agent, stage, "Verified result 25", {"tool_call_id": "script"}),
            AgentEvent("text", agent, stage, "Public argument\n" * 80, {"message_id": "reply"}),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        view = app.stage_views[stage]
        selector = f"#{view.id}"
        section = view.bubbles[0].sections[0]
        view.navigate("home")
        await pilot.pause()
        assert "▾ Input" in "".join(segment.text for segment in view.render_line(0))
        await pilot.click(selector, offset=content_position(view, 2, 0))
        assert section.collapsed
        assert "Shared input 0" not in view.visible_plain()
        await pilot.click(selector, offset=content_position(view, 2, 0))
        assert not section.collapsed

        content_x, content_y = content_position(view, 2, 0)
        await pilot.click(selector, offset=(content_x, 0))
        await pilot.click(selector, offset=(0, content_y))
        assert not section.collapsed
        assert view.show_vertical_scrollbar
        scrollbar_x = view.scrollable_content_region.right - view.region.x
        await pilot.click(selector, offset=(scrollbar_x, content_y))
        assert not section.collapsed

        await pilot.resize_terminal(80, 24)
        view.scroll_to(y=20, animate=False, immediate=True)
        await pilot.pause()
        assert "▾ Input" in "".join(segment.text for segment in view.render_line(0))
        await pilot.click(selector, offset=content_position(view, 2, 0))
        assert section.collapsed and view.scroll_y == 0
        assert evidence in view.plain()
        await pilot.click(selector, offset=content_position(view, 2, 0))
        assert not section.collapsed

        bubble = next(bubble for bubble in view.bubbles if bubble.agent == agent)
        reasoning, tool = bubble.sections[:2]
        view.scroll_to(y=view._detail_rows[bubble], animate=False, immediate=True)
        await pilot.pause()
        reasoning_row = view._detail_rows[reasoning] - int(view.scroll_y)
        await pilot.click(selector, offset=content_position(view, 2, reasoning_row))
        assert not reasoning.collapsed
        assert "Check the dated evidence." in view.visible_plain()
        tool_row = view._detail_rows[tool] - int(view.scroll_y)
        await pilot.click(selector, offset=content_position(view, 2, tool_row))
        assert not tool.collapsed
        assert "python3 checks.py" in view.visible_plain()
        assert "Verified result 25" in view.visible_plain()

        view.scroll_to(y=view._detail_rows[bubble] + 1, animate=False, immediate=True)
        await pilot.pause()
        assert agent in "".join(segment.text for segment in view.render_line(0))
        await pilot.click(selector, offset=content_position(view, 2, 0))
        assert bubble.collapsed and not reasoning.collapsed and not tool.collapsed
        assert "Public argument" not in view.visible_plain()
        bubble_row = view._detail_rows[bubble] - int(view.scroll_y)
        await pilot.click(selector, offset=content_position(view, 2, bubble_row))
        assert not bubble.collapsed
        view.scroll_to(y=view._detail_rows[bubble], animate=False, immediate=True)
        assert "Public argument" in view.visible_plain()
