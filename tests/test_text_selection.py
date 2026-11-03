import pytest
from textual.widgets import Tabs

from open_trading_agents.analysis.models import AgentEvent
from open_trading_agents.ui.app import TradingApp
from open_trading_agents.ui.transcript import Transcript


@pytest.mark.parametrize("pane", ["analyst", "report", "system"])
async def test_drag_selects_unicode_text_without_borders(pane: str) -> None:
    """Highlight dragged text while retaining header clicks and streaming state.

    Args:
        pane: Custom viewport to exercise through real mouse events.
    """
    app = TradingApp(selected_analysts=("market",))
    async with app.run_test(size=(120, 35)) as pilot:
        content = "Prices 한국어 123\nSecond line 456"
        if pane == "analyst":
            await app.receive(AgentEvent("text", "Market Analyst", "analysis", content, {"message_id": "reply"}))
            app.flush_events()
            view = app.agent_views["Market Analyst"]
            x, y = 2, 1
        else:
            if pane == "report":
                app.query_one(Tabs).active = "tab-reports"
                view = app.report_view
            else:
                app.query_one(Tabs).active = "tab-research"
                view = app.prompt_views["research"]["Bull Researcher"]
            await pilot.pause()
            document = Transcript()
            document.append(content, "#91d5ac")
            view.set_document(document)
            view.sync()
            x, y = 0, 0
        view.navigate("home")
        await pilot.pause()
        region = view.scrollable_content_region
        start = (region.x - view.region.x + x, region.y - view.region.y + y)
        end = (start[0] + 10, start[1] + 1)
        await pilot.mouse_down(view, offset=start)
        await pilot.hover(view, offset=end)
        await pilot.mouse_up(view, offset=end)
        await pilot.pause()

        assert view.text_selection is not None
        assert app.screen.get_selected_text() == "Prices 한국어 123\nSecond line"
        assert not view.following
        highlighted = view.render_line(y)
        selection_style = app.screen.get_component_rich_style("screen--selection")
        assert any(segment.style and segment.style.bgcolor == selection_style.bgcolor for segment in highlighted)
        assert app.is_running

        if pane == "analyst":
            await app.receive(AgentEvent("text", "Market Analyst", "analysis", "\nNew output", {"message_id": "reply"}))
            app.flush_events()
            await pilot.pause()
            assert app.screen.get_selected_text() == "Prices 한국어 123\nSecond line"
            await pilot.click(view, offset=(start[0], start[1] - 1))
            assert view.bubbles[0].collapsed


async def test_scrolled_tool_selection_excludes_hidden_rows_and_survives_scroll() -> None:
    """Highlight a reverse drag within an expanded tool while a turn title is pinned."""
    app = TradingApp(selected_analysts=("market",))
    async with app.run_test(size=(100, 25)) as pilot:
        view = app.agent_views["Market Analyst"]
        for event in (
            AgentEvent("text", "Market Analyst", "analysis", "Introduction\n" * 30, {"message_id": "reply"}),
            AgentEvent("tool_start", "Market Analyst", "analysis", "", {"tool": "bash", "tool_call_id": "script"}),
            AgentEvent("tool_result", "Market Analyst", "analysis", "Prices 한국어 123\nSecond line 456\n" * 20,
                       {"tool_call_id": "script"}),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        tool = view.bubbles[0].sections[-1]
        view._toggle(tool)
        part_index = next(index for index, part in enumerate(view._parts) if part.section is tool and part.kind == "body")
        body_row = view._offsets[part_index]
        view.scroll_to(y=body_row, animate=False, immediate=True)
        await pilot.pause()
        assert "Market Analyst" in view.render_line(0).text
        row = next(row for row in range(1, 10) if "Prices 한국어" in view.render_line(row).text)
        region = view.scrollable_content_region
        start = (region.x - view.region.x + 9, region.y - view.region.y + row)
        end = (start[0] + 4, start[1])
        await pilot.mouse_down(view, offset=end)
        await pilot.hover(view, offset=start)
        await pilot.mouse_up(view, offset=start)
        await pilot.pause()
        assert app.screen.get_selected_text() == "한국어"
        view.navigate("down")
        await pilot.pause()
        assert app.screen.get_selected_text() == "한국어"
        await pilot.resize_terminal(80, 25)
        assert view.text_selection is None
