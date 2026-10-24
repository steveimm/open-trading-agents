from __future__ import annotations

import time

from textual import events
from textual.app import App, ComposeResult
from textual.scrollbar import ScrollTo

from open_trading_agents.chat import ChatView
from open_trading_agents.domain import AgentEvent
from open_trading_agents.transcript import Transcript, TranscriptView


class ChatTestApp(App[None]):
    def __init__(self, view: ChatView | TranscriptView) -> None:
        """Mount one viewport for component interaction tests."""
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        """Provide the isolated conversation viewport."""
        yield self.view


async def test_debate_bubbles_keep_shared_input_once_and_collapsed_exact_inputs() -> None:
    """Present each public contribution once while retaining native input and hidden details."""
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
        assert "Private reasoning 1" in view.plain()
        assert len(view.systems) == 2
        assert len(view.bubbles) == 3
        assert all(bubble.status == "complete" for bubble in view.bubbles)
        view.toggle_system()
        assert "Bull Researcher instructions" in view.visible_plain()
        view.toggle_system()
        assert "Bull Researcher instructions" not in view.visible_plain()


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
        view.consume(AgentEvent("system", "Bear Researcher", "research", "New participant's system prompt"))
        view.consume(AgentEvent("conversation_input", "Input", "research", "Shared evidence\n" * 20))
        view.sync()
        await pilot.pause()
        after = "".join(segment.text for segment in view.render_line(0))
        assert before == after
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
