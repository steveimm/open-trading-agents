from __future__ import annotations

import time

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.scrollbar import ScrollTo

from open_trading_agents.analysis.models import AgentEvent
from open_trading_agents.ui.chat import ChatView
from open_trading_agents.ui.transcript import Transcript, TranscriptIndex, TranscriptView


class ChatTestApp(App[None]):
    def __init__(self, view: ChatView | TranscriptView) -> None:
        """Mount one viewport for component interaction tests."""
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        """Provide the isolated conversation viewport."""
        yield self.view


async def test_debate_bubbles_keep_shared_input_once_without_per_turn_inputs() -> None:
    """Present each public contribution once without repeated prompt displays."""
    view = ChatView(id="chat", debate=True)
    app = ChatTestApp(view)
    async with app.run_test(size=(100, 34)) as pilot:
        view.consume(AgentEvent("conversation_input", "Input", "research", "Shared analyst evidence"))
        for index, name in enumerate(("Bull Researcher", "Bear Researcher"), 1):
            turn = f"research:1:{index}"
            metadata = {"conversation_turn_id": turn, "round": 1, "turn": index, "total_rounds": 2}
            view.consume(AgentEvent("turn_start", name, "research", data=metadata))
            view.consume(AgentEvent("system", name, "research", f"{name} instructions", metadata))
            view.consume(AgentEvent("prompt", name, "research", "Shared analyst evidence and prior arguments", metadata))
            view.consume(AgentEvent("reasoning", name, "research", f"Private reasoning {index}", {**metadata, "message_id": f"m{index}"}))
            view.consume(AgentEvent("text", name, "research", f"Public argument {index}", {**metadata, "message_id": f"m{index}"}))
            view.consume(AgentEvent("text", name, "research", f"Public argument {index}", {**metadata, "message_id": f"m{index}", "final": True}))
            view.consume(AgentEvent("agent_end", name, "research", data={**metadata, "output": f"Public argument {index}"}))
            view.consume(AgentEvent("conversation_output", name, "research", f"Public argument {index}", metadata))
        view.sync()
        await pilot.pause()
        displayed = view.visible_plain()
        assert displayed.count("Shared analyst evidence") == 1
        assert displayed.count("Public argument 1") == displayed.count("Public argument 2") == 1
        assert "Private reasoning" not in displayed
        assert "prior arguments" not in displayed
        assert "prior arguments" not in view.plain()
        assert all(section.kind != "prompt" for bubble in view.bubbles if bubble.kind == "agent" for section in bubble.sections)
        assert "Private reasoning 1" in view.plain()
        assert len(view.systems) == 2
        assert len(view.bubbles) == 3
        assert all(bubble.status == "complete" for bubble in view.bubbles)
        view.toggle_system()
        assert "Bull Researcher instructions" in view.visible_plain()
        view.toggle_system()
        assert "Bull Researcher instructions" not in view.visible_plain()


async def test_bubble_border_cells_use_the_content_background_in_both_focus_states() -> None:
    """Avoid inherited terminal backgrounds that form colored bars beside bubble text."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 30)) as pilot:
        view.consume(AgentEvent("conversation_input", "Input", "research", "Input evidence\nsecond line"))
        view.consume(AgentEvent("turn_start", "Bull Researcher", "research", data={"round": 1, "turn": 1}))
        view.consume(AgentEvent("reasoning", "Bull Researcher", "research", "Verify the values", {"message_id": "one"}))
        view.consume(AgentEvent("text", "Bull Researcher", "research", "Public argument", {"message_id": "one"}))
        view.sync()
        view.navigate("home")
        for focused in (False, True):
            if focused:
                view.focus()
            else:
                view.app.set_focus(None)
            await pilot.pause()
            background = view.rich_style.bgcolor
            assert background is not None
            for index, part in enumerate(view._parts):
                if part.kind not in {"body", "detail"} or part.bubble is None:
                    continue
                row = view._offsets[index]
                segments = list(view.render_line(row))
                assert segments[0].text == "│ "
                assert segments[-1].text == " │"
                assert segments[0].style.bgcolor == background
                assert segments[-1].style.bgcolor == background
                assert all(segment.style and segment.style.bgcolor == background for segment in segments)


async def test_reasoning_and_tool_details_toggle_with_keyboard_and_mouse() -> None:
    """Reveal grouped tool input/results and reasoning through explicit controls."""
    view = ChatView(id="chat")
    app = ChatTestApp(view)
    async with app.run_test(size=(100, 30)) as pilot:
        events = (
            AgentEvent("agent_start", "Market Analyst", "analysis"),
            AgentEvent("reasoning", "Market Analyst", "analysis", "Check the arithmetic", {"message_id": "one"}),
            AgentEvent("tool_start", "Market Analyst", "analysis", '{"command":"python calc.py"}', {"tool": "bash", "tool_call_id": "calc"}),
            AgentEvent("tool_result", "Market Analyst", "analysis", "2", {"tool_call_id": "calc", "delta": True}),
            AgentEvent("tool_result", "Market Analyst", "analysis", "5", {"tool_call_id": "calc", "delta": True}),
            AgentEvent("tool_result", "Market Analyst", "analysis", "25", {"tool_call_id": "calc", "final": True}),
            AgentEvent("text", "Market Analyst", "analysis", "The change is 25%.", {"message_id": "two"}),
        )
        for event in events:
            view.consume(event)
        view.sync()
        await pilot.pause()
        assert "Check the arithmetic" not in view.visible_plain()
        assert "python calc.py" not in view.visible_plain()
        view.navigate("home")
        view.select_detail(1)
        assert view._selected is view.bubbles[0]
        view.select_detail(1)
        view.toggle_selected()
        assert "Check the arithmetic" in view.visible_plain()
        view.select_detail(1)
        view.toggle_selected()
        assert "python calc.py" in view.visible_plain()
        assert view.visible_plain().count("Result\n25") == 1
        section = view.bubbles[0].sections[0]
        row = view._detail_rows[section] - int(view.scroll_y)
        await pilot.click("#chat", offset=(4, row))
        assert section.collapsed
        assert not view.following


async def test_bottom_navigation_resumes_follow_without_resize_or_append_reset() -> None:
    """Return to live via ordinary downward scrolling while preserving intentional scrollback."""
    document = Transcript()
    document.append("\n".join(f"line {index}" for index in range(100)))
    view = TranscriptView(document, id="view")
    app = ChatTestApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.sync()
        await pilot.pause()
        view.navigate("home")
        assert not view.following
        document.append("\nnew line")
        view.sync()
        assert not view.following and view.scroll_y == 0
        await pilot.resize_terminal(100, 30)
        assert not view.following
        for _ in range(10):
            view.navigate("pagedown")
        assert view.following
        document.append("\nlatest")
        view.sync()
        assert view.scroll_y == view.max_scroll_y
        view.navigate("home")
        view._on_scroll_to(ScrollTo(y=view.max_scroll_y, animate=False))
        assert view.following
        view._on_scroll_to(ScrollTo(y=0, animate=False))
        assert not view.following
        view.scroll_to(y=max(0, view.max_scroll_y - 1), animate=False, immediate=True)
        view._on_mouse_scroll_down(events.MouseScrollDown(view, 3, 3, 0, 1, 0, False, False, False))
        assert view.following
        view._on_mouse_scroll_up(events.MouseScrollUp(view, 3, 3, 0, -1, 0, False, False, False))
        assert not view.following


async def test_legacy_inputs_and_durable_outputs_reconcile_without_duplicate_bubbles() -> None:
    """Recover old saved debates and interrupted event streams without overwriting later turns."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(90, 30)):
        for number in (1, 2):
            view.consume(AgentEvent("turn_start", "Bull Researcher", "research", data={"round": number, "turn": 1, "total_rounds": 2}))
            view.consume(AgentEvent("prompt", "Bull Researcher", "research", f"Legacy evidence {number}"))
            view.consume(AgentEvent("text", "Bull Researcher", "research", f"Argument {number}", {"message_id": str(number)}))
            view.consume(AgentEvent("agent_end", "Bull Researcher", "research", data={"output": f"Argument {number}"}))
        view.consume(AgentEvent("conversation_input", "Input", "research", "Authoritative shared evidence"))
        for number in (1, 2):
            view.consume(AgentEvent("conversation_output", "Bull Researcher", "research", f"Argument {number}",
                                    {"conversation_turn_id": f"research:{number}:bull", "round": number, "turn": 1, "total_rounds": 2}))
        view.sync()
        assert len(view.bubbles) == 3
        assert view.visible_plain().count("Argument 1") == view.visible_plain().count("Argument 2") == 1
        assert view.visible_plain().count("Authoritative shared evidence") == 1


async def test_large_collapsed_results_preserve_source_and_bounded_rendering() -> None:
    """Keep megabyte-sized results complete while the visible tool preview stays bounded."""
    view = ChatView(id="chat")
    large = "HEAD" + "x" * 1_000_000 + "TAIL"
    async with ChatTestApp(view).run_test(size=(120, 38)) as pilot:
        view.consume(AgentEvent("tool_result", "Market Analyst", "analysis", large,
                                {"tool": "calculate", "tool_call_id": "large"}))
        view.consume(AgentEvent("text", "Market Analyst", "analysis", "Final " + "日本語\n" * 300, {"message_id": "final"}))
        view.sync()
        await pilot.pause()
        assert large in view.plain()
        assert "HEAD" not in view.visible_plain()
        tool = view.bubbles[0].sections[0]
        tool.collapsed = False
        view._dirty = True
        view.sync()
        assert "...[omitted 992008 chars]..." in view.visible_plain()
        assert "TAIL" in view.visible_plain()
        assert len(view._cache) <= 256
        view.navigate("home")
        top = view.scroll_y
        for number in range(200):
            view.consume(AgentEvent("text", "Market Analyst", "analysis", f"fragment {number}\n", {"message_id": "final"}))
        started = time.monotonic()
        view.sync()
        assert time.monotonic() - started < 1
        assert view.scroll_y == top and not view.following


async def test_late_systems_and_shared_input_do_not_displace_scrollback() -> None:
    """Keep the same logical output visible when earlier collapsed context appears."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 24)) as pilot:
        body = "\n".join(f"Evidence line {number}" for number in range(200))
        view.consume(AgentEvent("text", "Bull Researcher", "research", body, {"message_id": "one"}))
        view.sync()
        await pilot.pause()
        view.navigate("home")
        view.navigate("pagedown")
        before = "".join(segment.text for segment in view.render_line(0))
        body_before = "".join(segment.text for segment in view.render_line(1))
        view.consume(AgentEvent("system", "Bear Researcher", "research", "New participant's system prompt"))
        view.consume(AgentEvent("conversation_input", "Input", "research", "Shared evidence\n" * 20))
        view.sync()
        await pilot.pause()
        after = "".join(segment.text for segment in view.render_line(0))
        assert before == after
        assert body_before == "".join(segment.text for segment in view.render_line(1))
        assert not view.following


async def test_native_and_durable_final_output_repair_only_the_current_contribution() -> None:
    """Complete interrupted text and reconcile tool snapshots without duplicating public replies."""
    view = ChatView(id="chat", debate=True)
    turn = {"conversation_turn_id": "research:1:bull", "round": 1, "turn": 1, "total_rounds": 1}
    async with ChatTestApp(view).run_test(size=(100, 30)):
        for event in (
            AgentEvent("turn_start", "Bull Researcher", "research", data=turn),
            AgentEvent("tool_start", "Bull Researcher", "research", "{}", {**turn, "tool_call_id": "tool", "tool": "calculate"}),
            AgentEvent("tool_start", "Bull Researcher", "research", '{"expression":', {**turn, "tool_call_id": "tool", "delta": True}),
            AgentEvent("tool_start", "Bull Researcher", "research", '"1+1"}', {**turn, "tool_call_id": "tool", "delta": True}),
            AgentEvent("tool_result", "Bull Researcher", "research", "preliminary", {**turn, "tool_call_id": "tool", "snapshot": True}),
            AgentEvent("tool_result", "Bull Researcher", "research", "corrected", {**turn, "tool_call_id": "tool", "final": True}),
            AgentEvent("text", "Bull Researcher", "research", "Partial", {**turn, "message_id": "one"}),
            AgentEvent("conversation_output", "Bull Researcher", "research", "Complete public reply", turn),
        ):
            view.consume(event)
        view.sync()
        assert len(view.bubbles) == 1
        assert view.plain().count("Complete public reply") == 1
        assert "Partial" not in view.plain()
        assert "preliminary" not in view.plain()
        assert '{"expression":"1+1"}' in view.plain()


async def test_pending_legacy_turn_reopens_its_existing_bubble() -> None:
    """Keep interrupted text when a resumed workflow assigns a stable turn identity."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 30)):
        metadata = {"round": 2, "turn": 1, "total_rounds": 3}
        view.consume(AgentEvent("turn_start", "Bull Researcher", "research", data=metadata))
        view.consume(AgentEvent("text", "Bull Researcher", "research", "Earlier partial output", {"message_id": "old"}))
        view.consume(AgentEvent("error", "Bull Researcher", "research", "Interrupted"))
        view.consume(AgentEvent("turn_start", "Bull Researcher", "research", data={**metadata, "conversation_turn_id": "research:2:bull"}))
        assert len(view.bubbles) == 1
        assert "Earlier partial output" in view.plain()
        assert view.bubbles[0].status == "streaming"
        view.finish("stopped")
        assert view.bubbles[0].status == "stopped"


async def test_reused_native_ids_do_not_mutate_previous_contributions() -> None:
    """Scope transport identifiers to their contribution and distinguish later tool executions."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 30)):
        for number in (1, 2):
            metadata = {"conversation_turn_id": f"research:{number}:bull", "round": number, "turn": 1, "total_rounds": 2}
            view.consume(AgentEvent("turn_start", "Bull Researcher", "research", data=metadata))
            for tool_number in (1, 2):
                view.consume(AgentEvent("tool_start", "Bull Researcher", "research", "{}",
                                        {**metadata, "tool": "calculate", "tool_call_id": "reused"}))
                view.consume(AgentEvent("tool_result", "Bull Researcher", "research", f"value-{number}-{tool_number}",
                                        {**metadata, "tool_call_id": "reused", "final": True}))
            view.consume(AgentEvent("text", "Bull Researcher", "research", f"argument-{number}",
                                    {**metadata, "message_id": "reused-message"}))
            view.consume(AgentEvent("agent_end", "Bull Researcher", "research", data={**metadata, "output": f"argument-{number}"}))
        first, second = view.bubbles
        assert [section.source for section in first.sections if section.kind == "text"] == ["argument-1"]
        assert [section.source for section in second.sections if section.kind == "text"] == ["argument-2"]
        assert "value-1-1" in first.sections[0].source and "value-1-2" in first.sections[1].source
        assert "value-2-1" in second.sections[0].source and "value-2-2" in second.sections[1].source


async def test_long_bubble_titles_stay_visible_and_switch_at_contribution_boundaries() -> None:
    """Keep the visible speaker identifiable while scrolling, resizing and following new output."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 24)) as pilot:
        view.consume(AgentEvent("system", "Bull Researcher", "research", "System guidance\n" * 40))
        for turn, name in enumerate(("Bull Researcher", "Bear Researcher"), 1):
            data = {"round": 1, "turn": turn, "total_rounds": 2, "conversation_turn_id": f"research:1:{turn}"}
            view.consume(AgentEvent("turn_start", name, "research", data=data))
            view.consume(AgentEvent("text", name, "research", "\n".join(f"{name} line {number}" for number in range(80)),
                                    {**data, "message_id": name}))
        view.sync()
        await pilot.pause()
        view.navigate("home")
        assert view._sticky_header() is None
        assert "System" in "".join(segment.text for segment in view.render_line(0))
        await pilot.click("#chat", offset=(4, 0))
        view.scroll_to(y=5, animate=False, immediate=True)
        assert view._sticky_header() is None
        assert "System guidance" in "".join(segment.text for segment in view.render_line(0))
        view.navigate("home")
        await pilot.click("#chat", offset=(4, 0))
        bull_body = next(i for i, part in enumerate(view._parts)
                         if part.kind == "body" and part.bubble is not None and part.bubble.agent == "Bull Researcher")
        view.scroll_to(y=view._offsets[bull_body] + 10, animate=False, immediate=True)
        first = "".join(segment.text for segment in view.render_line(0))
        assert "Bull Researcher" in first and "Round 1/2" in first and "Turn 1" in first
        assert view._activity_frame in first and "streaming" not in first
        body = "".join(segment.text for segment in view.render_line(1))
        assert "line 11" in body
        data = {"conversation_turn_id": "research:1:1", "round": 1, "turn": 1, "total_rounds": 2}
        view.consume(AgentEvent("agent_end", "Bull Researcher", "research", data=data))
        view.sync()
        assert "✓" in "".join(segment.text for segment in view.render_line(0))
        await pilot.resize_terminal(80, 20)
        await pilot.pause()
        assert "Bull Researcher" in "".join(segment.text for segment in view.render_line(0))
        assert "line 11" in "".join(segment.text for segment in view.render_line(1))
        gap = next(i for i, part in enumerate(view._parts)
                   if part.kind == "gap" and part.bubble is not None and part.bubble.agent == "Bull Researcher")
        view.scroll_to(y=view._offsets[gap], animate=False, immediate=True)
        assert view._sticky_header() is None
        assert "Bull Researcher" not in "".join(segment.text for segment in view.render_line(0))
        bear_title = next(i for i, part in enumerate(view._parts) if part.kind == "top" and part.bubble.agent == "Bear Researcher")
        view.scroll_to(y=view._offsets[bear_title], animate=False, immediate=True)
        assert view._sticky_header() is None
        assert "Turn 2" in "".join(segment.text for segment in view.render_line(0))
        assert "Round 1/2" not in "".join(segment.text for segment in view.render_line(1))
        view.navigate("end")
        view.consume(AgentEvent("text", "Bear Researcher", "research", "\nLATEST",
                                {"conversation_turn_id": "research:1:2", "message_id": "Bear Researcher"}))
        view.sync()
        assert view.following and view.scroll_y == view.max_scroll_y
        assert "Bear Researcher" in "".join(segment.text for segment in view.render_line(0))
        assert any("LATEST" in "".join(segment.text for segment in view.render_line(row)) for row in range(view.size.height))
        assert len(view._cache) <= 256


async def test_sticky_input_clicks_and_covered_details_use_visible_header_targets() -> None:
    """Collapse a pinned input safely and never click through a pinned role title."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 24)) as pilot:
        evidence = "\n".join(f"Shared input {number}" for number in range(60))
        view.consume(AgentEvent("conversation_input", "Input", "research", evidence))
        view.consume(AgentEvent("text", "Bull Researcher", "research", "Bull text\n" * 40, {"message_id": "bull"}))
        view.consume(AgentEvent("reasoning", "Bull Researcher", "research", "Hidden arithmetic", {"message_id": "reason"}))
        view.consume(AgentEvent("text", "Bear Researcher", "research", "Bear text\n" * 80, {"message_id": "bear"}))
        view.sync()
        await pilot.pause()
        view.navigate("home")
        view.scroll_to(y=20, animate=False, immediate=True)
        assert "Input" in "".join(segment.text for segment in view.render_line(0))
        input_section = view.bubbles[0].sections[0]
        await pilot.click("#chat", offset=(view.size.width - 1, 0))
        assert not input_section.collapsed
        view.scroll_to(y=20, animate=False, immediate=True)
        view.select_detail(-1)
        assert view._selected is input_section and view.scroll_y == 20
        assert view._detail_is_visible(input_section)
        view.toggle_selected()
        assert input_section.collapsed and view.scroll_y == 0
        await pilot.click("#chat", offset=(4, 0))
        assert not input_section.collapsed
        view.scroll_to(y=20, animate=False, immediate=True)
        await pilot.click("#chat", offset=(4, 0))
        assert input_section.collapsed
        assert view.scroll_y == 0
        assert evidence in view.plain() and "Shared input 20" not in view.visible_plain()
        reasoning = next(section for bubble in view.bubbles for section in bubble.sections if section.kind == "reasoning")
        view.scroll_to(y=view._detail_rows[reasoning], animate=False, immediate=True)
        assert "Bull Researcher" in "".join(segment.text for segment in view.render_line(0))
        assert not view._detail_is_visible(reasoning)
        await pilot.click("#chat", offset=(4, 0))
        assert reasoning.collapsed
        bull = next(bubble for bubble in view.bubbles if bubble.agent == "Bull Researcher")
        assert bull.collapsed and view._selected is bull
        assert "Bull text" not in view.visible_plain()
        await pilot.click("#chat", offset=(4, 0))
        assert not bull.collapsed and reasoning.collapsed
        view.select_detail(1)
        assert view._selected is reasoning
        assert view._detail_is_visible(reasoning)
        row = view._detail_rows[reasoning] - int(view.scroll_y)
        assert row > 0
        view.toggle_selected()
        assert not reasoning.collapsed


async def test_turn_disclosure_retains_streamed_content_and_nested_detail_states() -> None:
    """Collapse a running turn, retain new output, and restore independently expanded tool details."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 30)) as pilot:
        for event in (
            AgentEvent("turn_start", "Aggressive Analyst", "risk", data={"round": 3, "turn": 1, "total_rounds": 3}),
            AgentEvent("reasoning", "Aggressive Analyst", "risk", "Check evidence.", {"message_id": "thinking"}),
            AgentEvent("tool_start", "Aggressive Analyst", "risk", '{"expression":"2+2"}', {"tool": "calculate", "tool_call_id": "one"}),
            AgentEvent("tool_result", "Aggressive Analyst", "risk", '{"result":4}', {"tool_call_id": "one"}),
            AgentEvent("text", "Aggressive Analyst", "risk", "Initial position", {"message_id": "output"}),
        ):
            view.consume(event)
        view.sync()
        await pilot.pause()
        view.navigate("home")
        bubble = view.bubbles[0]
        view.select_detail(1)
        assert view._selected is bubble
        view.toggle_selected()
        assert bubble.collapsed
        assert view.visible_plain() == "Aggressive Analyst"
        assert view.virtual_size.height == 2
        assert "▸ Aggressive Analyst" in "".join(segment.text for segment in view.render_line(0))
        view.consume(AgentEvent("text", "Aggressive Analyst", "risk", " after verification", {"message_id": "output"}))
        view.consume(AgentEvent("agent_end", "Aggressive Analyst", "risk", data={"output": "Initial position after verification"}))
        view.sync()
        assert bubble.collapsed and view.virtual_size.height == 2
        assert "Initial position after verification" in view.plain()
        assert "Initial position" not in view.visible_plain()
        assert "✓" in "".join(segment.text for segment in view.render_line(0))
        view.toggle_selected()
        assert not bubble.collapsed and "Initial position after verification" in view.visible_plain()
        assert "Check evidence." not in view.visible_plain() and '{"result":4}' not in view.visible_plain()
        view.select_detail(1)
        view.select_detail(1)
        tool = view._selected
        view.toggle_selected()
        assert '{"result":4}' in view.visible_plain()
        view.navigate("home")
        await pilot.click("#chat", offset=(4, 0))
        assert bubble.collapsed and tool is not None and not tool.collapsed
        await pilot.click("#chat", offset=(4, 0))
        assert not bubble.collapsed and '{"result":4}' in view.visible_plain()


async def test_activity_animation_repaints_headers_without_reindexing_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Animate both a pinned active turn and its collapsed header while retaining cached output."""
    view = ChatView(id="chat", debate=True)
    async with ChatTestApp(view).run_test(size=(100, 24)) as pilot:
        view.consume(AgentEvent("text", "Bull Researcher", "research", "Evidence\n" * 100, {"message_id": "output"}))
        view.sync()
        await pilot.pause()
        view.navigate("home")
        view.scroll_to(y=20, animate=False, immediate=True)
        parts, offsets = view._parts, tuple(view._offsets)
        revisions = [section.document.revision for section in view.bubbles[0].sections]

        def forbid_sync(index: TranscriptIndex, width: int) -> bool:
            """Fail if an animation tick tries to wrap historical output again."""
            raise AssertionError("Animation reindexed transcript content")

        with monkeypatch.context() as patch:
            patch.setattr(TranscriptIndex, "sync", forbid_sync)
            now = time.monotonic()
            view.refresh_activity(now)
            first = "".join(segment.text for segment in view.render_line(0))
            view.refresh_activity(now + 0.1)
            second = "".join(segment.text for segment in view.render_line(0))
            assert first != second and "Bull Researcher" in second and "streaming" not in second
            assert view._parts is parts and tuple(view._offsets) == offsets
            assert [section.document.revision for section in view.bubbles[0].sections] == revisions
        await pilot.click("#chat", offset=(4, 0))
        assert view.bubbles[0].collapsed
        first = "".join(segment.text for segment in view.render_line(0))
        view.refresh_activity(now + 0.2)
        assert first != "".join(segment.text for segment in view.render_line(0))


async def test_separate_system_panes_do_not_duplicate_inline_content_or_disclosures() -> None:
    """Retain complete system source while dedicating the debate chat to shared inputs and turns."""
    view = ChatView(id="chat", debate=True, show_systems=False)
    async with ChatTestApp(view).run_test(size=(100, 30)):
        view.consume(AgentEvent("system", "Bull Researcher", "research", "Separate system guidance"))
        view.sync()
        assert not view.has_details
        assert "Separate system guidance" in view.plain()
        assert not view.visible_plain() and not view._parts
        view.consume(AgentEvent("text", "Bull Researcher", "research", "Public argument"))
        view.sync()
        assert view.has_details
        assert view.visible_plain() == "Bull Researcher\nPublic argument"
        assert all(section not in view._detail_rows for section in view.systems.values())
        assert view._parts[-1].kind == "bottom"
