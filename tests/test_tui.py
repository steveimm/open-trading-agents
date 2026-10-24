from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest
from rich.color import Color
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.strip import Strip
from textual.widgets import Collapsible, ContentSwitcher, Static, Tab

from open_trading_agents.analysis.models import ANALYSTS, STAGES, AgentEvent, EventSink, Usage
from open_trading_agents.persistence.events import EventRecorder
from open_trading_agents.persistence.storage import RunStore
from open_trading_agents.ui.app import TradingApp, usage_label
from open_trading_agents.ui.chat import ChatView
from open_trading_agents.ui.transcript import Transcript, TranscriptView


async def test_parallel_streams_keep_complete_source_and_usage_without_raw_panel() -> None:
    """Retain concurrent source text and counts while removing the payload UI and filler."""
    app = TradingApp(ticker="NVDA")
    async with app.run_test(size=(150, 45)) as pilot:
        for agent in ANALYSTS.values():
            await app.receive(AgentEvent("agent_start", agent, "analysis"))
            await app.receive(AgentEvent("system", agent, "analysis", "Research using supplied tools."))
            await app.receive(AgentEvent("prompt", agent, "analysis", "Analyze NVDA."))
        for index in range(300):
            for agent in ANALYSTS.values():
                await app.receive(AgentEvent("text", agent, "analysis", f"{agent} {index} 日本語 분석\n", {"message_id": "reply"}))
        for agent in ANALYSTS.values():
            usage = {"request_id": "reply", "input_tokens": 150, "output_tokens": 40, "cache_read_tokens": 100}
            await app.receive(AgentEvent("usage", agent, "analysis", data=usage))
            await app.receive(AgentEvent("usage", agent, "analysis", data=usage))
            text = "".join(f"{agent} {index} 日本語 분석\n" for index in range(300))
            await app.receive(AgentEvent("agent_end", agent, "analysis", data={"output": text}))
        app.flush_events()
        app.flush_events()
        await pilot.pause()
        assert len(app.usage) == 4
        assert sum(value.input_tokens for value in app.usage.values()) == 600
        for agent in ANALYSTS.values():
            chat = app.agent_views[agent]
            assert f"{agent} 0 日本語 분석" in chat.plain()
            assert chat.plain().count(f"{agent} 299 日本語 분석") == 1
            assert "Research using supplied tools." in chat.plain()
            assert "Research using supplied tools." not in chat.visible_plain()
        assert not app.query("#details, #raw-events, #transcript-hint, #payload-hint")
        assert "write" not in str(app.query_one("#usage", Static).render())
        assert all(binding.key not in {"d", "s"} for binding in app.BINDINGS)
        await pilot.press("d")
        assert app.active_stage == "analysis"


async def test_stage_tabs_distinguish_running_complete_and_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep running-stage progress visible even when the user views another tab."""
    app = TradingApp()
    async with app.run_test(size=(140, 40)) as pilot:
        for event in (
            AgentEvent("stage_start", "Workflow", "analysis"),
            AgentEvent("stage_end", "Workflow", "analysis"),
            AgentEvent("stage_start", "Workflow", "research"),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        assert app.active_stage == "analysis"
        assert "✓" in str(app.query_one("#tab-analysis", Tab).label)
        running_tab = app.query_one("#tab-research", Tab)
        start = app._running_spinner.start_time
        assert start is not None
        app._refresh_tabs(start)
        first_frame = str(running_tab.render())
        relabelled = []
        post = running_tab.post_message

        def record_message(message: object) -> bool:
            """Record tab layout notifications while preserving normal event delivery."""
            if isinstance(message, Tab.Relabelled):
                relabelled.append(message)
            return post(message)

        monkeypatch.setattr(running_tab, "post_message", record_message)
        app._refresh_tabs(start + 0.15)
        assert str(running_tab.render()) != first_frame
        assert str(running_tab.render()).startswith("Research Debate ")
        assert "[run]" not in str(running_tab.render())
        assert not relabelled
        assert "✓" in str(app.query_one("#tab-analysis", Tab).label)
        assert app.query_one("#tab-analysis", Tab).has_class("complete")
        assert app.query_one("#tab-research", Tab).has_class("running")
        app.refresh_clocks()
        header = str(app.query_one("#run-status", Static).render())
        assert "Total" in header and "Running" not in header and "Research Debate" not in header
        await app.receive(AgentEvent("error", "Workflow", "reports", "Provider failed"))
        app.flush_events()
        assert "[failed]" in str(app.query_one("#tab-research", Tab).label)
        app._refresh_tabs(start + 1)
        assert "[failed]" in str(running_tab.label)
        assert "✓" in str(app.query_one("#tab-analysis", Tab).label)
        assert "Provider failed" in str(app.query_one("#run-error", Static).render())


async def test_tabs_fullscreen_scroll_anchor_and_bottom_follow() -> None:
    """Preserve scrollback and resume following by scrolling down, without requiring End."""
    app = TradingApp()
    async with app.run_test(size=(140, 42)) as pilot:
        await app.receive(AgentEvent("text", "Sentiment Analyst", "analysis", "\n".join(f"{i:04} 한국어 日本語 " * 4 for i in range(600))))
        app.flush_events()
        await pilot.pause()
        pane = app.query_one("#pane-1", Vertical)
        await pilot.press("right", "f")
        view = app.agent_views["Sentiment Analyst"]
        assert app.fullscreen and view.size.width > 130 and view.size.height > 20
        await pilot.press("home", "pagedown")
        assert not view.following
        top = view.scroll_y
        await app.receive(AgentEvent("text", "Sentiment Analyst", "analysis", "\nNew streaming text"))
        app.flush_events()
        await pilot.pause()
        assert view.scroll_y == top
        await pilot.resize_terminal(100, 36)
        await pilot.pause()
        assert not view.following and "0000 한국어 日本語" in view.plain()
        view.scroll_to(y=max(0, view.max_scroll_y - 1), animate=False, immediate=True)
        await pilot.press("down")
        assert view.following
        await app.receive(AgentEvent("text", "Sentiment Analyst", "analysis", "\nNewest output"))
        app.flush_events()
        await pilot.pause()
        assert view.scroll_y == view.max_scroll_y
        await pilot.press("f", "tab")
        assert app.active_stage == "research"
        assert app.query_one("#stages", ContentSwitcher).current == "stage-research"
        await pilot.press("shift+tab")
        assert app.query_one("#pane-1", Vertical) is pane and not app.fullscreen
        await pilot.press("shift+tab")
        assert app.active_stage == STAGES[-1][0]


async def test_debate_chat_has_shared_input_once_and_completed_role_bubbles() -> None:
    """Display shared evidence and each public contribution once, hiding repeated native context."""
    app = TradingApp()
    evidence = "Analyst evidence: verified data."
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.press("tab")
        await app.receive(AgentEvent("conversation_input", "Workflow", "research", evidence))
        for turn, agent, text in ((1, "Bull Researcher", "Bull case"), (2, "Bear Researcher", "Bear response")):
            identity = f"research:1:{turn}"
            data = {"round": 1, "turn": turn, "total_rounds": 3, "conversation_turn_id": identity}
            await app.receive(AgentEvent("turn_start", agent, "research", data=data))
            await app.receive(AgentEvent("agent_start", agent, "research", data=data))
            await app.receive(AgentEvent("system", agent, "research", f"Instructions for {agent}"))
            await app.receive(AgentEvent("prompt", agent, "research", evidence + "\nExact native turn input", data))
            await app.receive(AgentEvent("reasoning", agent, "research", "Weighing evidence.", {**data, "message_id": agent}))
            await app.receive(AgentEvent("text", agent, "research", text, {**data, "message_id": agent}))
            await app.receive(AgentEvent("text", agent, "research", text, {**data, "message_id": agent, "final": True}))
            await app.receive(AgentEvent("agent_end", agent, "research", data={**data, "output": text}))
            await app.receive(AgentEvent("conversation_output", agent, "research", text, data))
        app.flush_events()
        await pilot.pause()
        view = app.stage_views["research"]
        visible = view.visible_plain()
        assert visible.count(evidence) == 1
        assert visible.count("Bull case") == visible.count("Bear response") == 1
        assert visible.index("Bull case") < visible.index("Bear response")
        assert "Exact native turn input" not in visible and "Weighing evidence." not in visible
        assert "Exact native turn input" not in view.plain()
        assert not any(section.kind == "prompt" for bubble in view.bubbles if bubble.kind == "agent" for section in bubble.sections)
        assert [bubble.agent for bubble in view.bubbles] == ["Input", "Bull Researcher", "Bear Researcher"]
        assert all(b.status == "complete" for b in view.bubbles)
        assert not app.query("#current-research, .debate-speaker")
        await pilot.press("s")
        assert "Instructions for Bull Researcher" not in view.visible_plain()
        assert app.prompt_views["research"]["Bull Researcher"].document.plain() == "Instructions for Bull Researcher"
        assert app.prompt_views["research"]["Bear Researcher"].document.plain() == "Instructions for Bear Researcher"
        assert not view.show_systems


async def test_shortcuts_and_actions_follow_stage_focus_and_available_sections() -> None:
    """Do not advertise or execute controls belonging to another stage or background pane."""
    app = TradingApp()
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        footer = app.query_one("#shortcuts", Static)
        assert "fullscreen" in str(footer.render())
        assert "Space" not in str(footer.render())
        assert not app.check_action("detail", (1,))
        await app.receive(AgentEvent("reasoning", "Market Analyst", "analysis", "Check evidence.", {"message_id": "r"}))
        app.flush_events()
        assert "[ prev · ] next" in str(footer.render())
        assert "Space expand/collapse" in str(footer.render())
        await pilot.press("f")
        assert app.fullscreen
        await pilot.press("tab")
        assert app.active_stage == "research"
        assert "f toggle fullscreen" in str(footer.render()) and "grid" not in str(footer.render())
        assert app.check_action("fullscreen", ())
        await pilot.press("f", "escape")
        assert app.fullscreen and app._debate_fullscreen["research"] is app.stage_views["research"]
        await pilot.press("shift+tab", "escape")
        assert app.fullscreen
        await pilot.press("f")
        assert not app.fullscreen
        for event in (
            AgentEvent("agent_start", "Reflection 1", "analysis"),
            AgentEvent("reasoning", "Reflection 1", "analysis", "Supporting work.", {"message_id": "m"}),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.press("f", "m")
        await pilot.pause()
        assert app.memory_view.has_focus and app.fullscreen
        assert not app.check_action("fullscreen", ())
        assert "fullscreen" not in str(footer.render()) and "grid" not in str(footer.render())
        await pilot.press("f", "escape")
        assert app.fullscreen
        await pilot.press("m", "shift+tab")
        assert app.active_stage == "reports"
        assert not app.check_action("detail", (1,)) and not app.check_action("toggle_detail", ())
        assert "Space" not in str(footer.render()) and "section" not in str(footer.render())


async def test_single_analyst_arrows_scroll_without_a_redundant_fullscreen_mode() -> None:
    """A single pane already fills its stage and should use arrows for scrolling."""
    app = TradingApp(selected_analysts=("market",))
    async with app.run_test(size=(100, 35)) as pilot:
        await app.receive(AgentEvent("text", "Market Analyst", "analysis", "Line\n" * 100))
        app.flush_events()
        await pilot.pause()
        assert not app.check_action("fullscreen", ())
        await pilot.press("home", "down")
        assert app.agent_views["Market Analyst"].scroll_y > 0
        await pilot.press("f")
        assert not app.fullscreen


async def test_following_survives_contextual_control_height_changes() -> None:
    """Keep the latest line visible when new supporting controls reduce viewport height."""
    app = TradingApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.receive(AgentEvent("text", "Market Analyst", "analysis", "line\n" * 100, {"message_id": "x"}))
        await app.receive(AgentEvent("reasoning", "Market Analyst", "analysis", "Check data.", {"message_id": "y"}))
        app.flush_events()
        await pilot.pause()
        heights = {view: view.size.height for view in app.agent_views.values()}
        await app.receive(AgentEvent("prompt", "Reflection 1", "analysis", "Review prior outcome."))
        app.flush_events()
        await pilot.pause()
        resized = [view for view, height in heights.items() if view.size.height < height]
        assert resized
        view = app.agent_views["Market Analyst"]
        assert view.following and view.scroll_y == view.max_scroll_y
        view.navigate("home")
        await pilot.resize_terminal(100, 35)
        await pilot.pause()
        assert not view.following and view.scroll_y == 0


async def test_reasoning_and_tools_can_be_revealed_from_keyboard() -> None:
    """Use the app's keyboard controls to reveal collapsed model work."""
    app = TradingApp(selected_analysts=("market",))
    async with app.run_test(size=(110, 36)) as pilot:
        for event in (
            AgentEvent("agent_start", "Market Analyst", "analysis"),
            AgentEvent("reasoning", "Market Analyst", "analysis", "Check units.", {"message_id": "a"}),
            AgentEvent("tool_start", "Market Analyst", "analysis", '{"command":"python check.py"}', {"tool": "bash", "tool_call_id": "t"}),
            AgentEvent("tool_result", "Market Analyst", "analysis", "25", {"tool_call_id": "t"}),
            AgentEvent("text", "Market Analyst", "analysis", "The verified result is 25.", {"message_id": "a"}),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        view = app.agent_views["Market Analyst"]
        assert "Check units." not in view.visible_plain()
        assert "python check.py" not in view.visible_plain()
        await pilot.press("]", "]", "space")
        assert "Check units." in view.visible_plain()
        await pilot.press("]", "space")
        assert "python check.py" in view.visible_plain() and "Result\n25" in view.visible_plain()


@pytest.mark.parametrize("stage", ["research", "risk"])
async def test_debate_prompt_panes_keep_independent_scroll_and_toggle_fullscreen(stage: str) -> None:
    """Read each participant's actual system prompt beside the shared chat using keyboard or mouse focus."""
    app = TradingApp()
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range([key for key, _ in STAGES].index(stage)):
            await pilot.press("tab")
        await pilot.pause()
        assert app.active_stage == stage
        for agent in app.prompt_views[stage]:
            await app.receive(AgentEvent("system", agent, stage, f"{agent} instructions\n" + "Verify cited evidence.\n" * 80))
        await app.receive(AgentEvent("text", next(iter(app.prompt_views[stage])), stage, "Public debate output.\n" * 80))
        app.flush_events()
        await pilot.pause()
        chat = app.stage_views[stage]
        panes = list(app.prompt_views[stage].values())
        assert chat.size.width > panes[0].size.width * 2.5
        assert panes[0].region.x < chat.region.x
        assert panes[0].region.y == chat.region.y
        assert panes[-1].region.bottom == chat.region.bottom
        assert all("instructions" in pane.document.plain() for pane in panes)
        assert "instructions" not in chat.visible_plain()
        await pilot.press("left")
        selected = panes[-1]
        assert app._active_view() is selected and selected.has_focus
        assert "f toggle fullscreen" in str(app.query_one("#shortcuts", Static).render())
        await pilot.press("home", "down")
        position = selected.scroll_y
        assert position > 0
        name = app._prompt_targets[selected][1]
        await app.receive(AgentEvent("system", name, stage, selected.document.plain()))
        app.flush_events()
        await pilot.pause()
        assert selected.scroll_y == position
        await pilot.press("f")
        assert app._debate_fullscreen[stage] is selected
        assert selected.size.width > 150 and not chat.display
        assert all(not pane.display for pane in panes if pane is not selected)
        await pilot.press("enter", "escape")
        assert app._debate_fullscreen[stage] is selected
        await pilot.press("f")
        assert app._debate_fullscreen[stage] is None and chat.display
        await pilot.click(f"#{panes[0].id}", offset=(3, 2))
        assert app._active_view() is panes[0] and panes[0].has_focus
        await pilot.press("left")
        assert app._active_view() is chat and chat.has_focus


@pytest.mark.parametrize("size", [(190, 43), (100, 32), (80, 24)])
async def test_debate_geometry_fills_available_area_without_outer_scrollbars(size: tuple[int, int]) -> None:
    """Reserve scrolling inside panes while keeping stage borders and footer within the terminal."""
    app = TradingApp()
    async with app.run_test(size=size) as pilot:
        for _ in range(4):
            await pilot.press("tab")
        for agent in app.prompt_views["risk"]:
            await app.receive(AgentEvent("system", agent, "risk", "System instructions\n" * 150))
        await app.receive(AgentEvent("text", "Aggressive Analyst", "risk", "Evidence and calculations\n" * 300))
        app.flush_events()
        await pilot.pause()
        chat = app.stage_views["risk"]
        stage = app.query_one("#stage-risk")
        footer = app.query_one("#shortcuts", Static)
        assert app.screen.max_scroll_x == app.screen.max_scroll_y == 0
        assert not app.screen.show_horizontal_scrollbar and not app.screen.show_vertical_scrollbar
        assert stage.region.right == size[0] and chat.region.right == stage.region.right
        assert chat.region.bottom == stage.region.bottom
        assert footer.region.bottom <= size[1]
        assert chat.max_scroll_x == 0 and chat.max_scroll_y > 0
        assert all(pane.size.height >= 2 for pane in app.prompt_views["risk"].values())


async def test_analyst_completion_and_fullscreen_use_requested_controls() -> None:
    """Keep the analyst header to its name and time, with fullscreen on f alone."""
    app = TradingApp()
    async with app.run_test(size=(140, 40)) as pilot:
        await app.receive(AgentEvent("agent_start", "Market Analyst", "analysis"))
        await app.receive(AgentEvent("agent_end", "Market Analyst", "analysis", data={"output": "Verified report"}))
        app.flush_events()
        app.refresh_clocks()
        status = str(app.query_one("#agent-status-0", Static).render())
        assert re.fullmatch(r"Market Analyst · \d{2}:\d{2}", status.splitlines()[0])
        await pilot.press("enter", "escape")
        assert not app.fullscreen
        await pilot.press("f")
        assert app.fullscreen
        footer = str(app.query_one("#shortcuts", Static).render())
        assert "f toggle fullscreen" in footer and "Esc" not in footer and "Enter" not in footer
        await pilot.press("enter", "escape")
        assert app.fullscreen
        await pilot.press("f")
        assert not app.fullscreen


async def test_clocks_tick_without_reindexing_chats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep elapsed time responsive while a provider is silent and content is unchanged."""
    release = asyncio.Event()
    rendered = 0
    original = ChatView.render_line

    def count_render(view: ChatView, y: int) -> Strip:
        """Count visible row rendering while only clocks change."""
        nonlocal rendered
        rendered += 1
        return original(view, y)

    monkeypatch.setattr(ChatView, "render_line", count_render)

    async def run(sink: EventSink) -> None:
        """Produce a long transcript and then wait for an external response."""
        await sink(AgentEvent("stage_start", "Workflow", "analysis"))
        await sink(AgentEvent("agent_start", "Market Analyst", "analysis"))
        await sink(AgentEvent("text", "Market Analyst", "analysis", "Long transcript\n" * 3000))
        await release.wait()

    app = TradingApp(run)
    async with app.run_test(size=(130, 40)) as pilot:
        await pilot.pause(0.15)
        offsets = app.agent_views["Market Analyst"]._offsets
        before = rendered
        await pilot.pause(1.05)
        assert "00:01" in str(app.query_one("#run-status", Static).render())
        assert app.agent_views["Market Analyst"]._offsets is offsets
        assert rendered - before < 100
        await pilot.press("tab", "shift+tab")
        release.set()


async def test_quit_cancels_active_workflow_and_waits_for_cleanup() -> None:
    """Do not leave running native models behind when the UI closes."""
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def run(sink: EventSink) -> None:
        """Wait for cancellation and record completion of cleanup."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async with TradingApp(run).run_test() as pilot:
        await started.wait()
        await pilot.press("q")
    assert cleaned.is_set()


async def test_ui_callbacks_ignore_events_after_screen_teardown() -> None:
    """Ignore late timer and refresh callbacks after the screen has been removed."""
    app = TradingApp()
    async with app.run_test() as pilot:
        await pilot.pause()
    event = AgentEvent("prompt", "Reflection 1", "analysis", "Final cleanup event.")
    await app.receive(event)
    app.flush_events()
    app.refresh_clocks()
    app._refresh_shortcuts()
    app._refresh_tabs()
    app._adapt_layout()
    assert list(app._pending) == [event]
    assert not app.check_action("fullscreen", ())


async def test_compact_grid_and_unicode_row_renderer() -> None:
    """Retain useful compact pane sizes and exact Unicode rendering after extraction."""
    app = TradingApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.agent_views["Market Analyst"].size.height >= 3
        assert app.screen.has_class("compact")
    source = "日本語한국어" * 60
    document = Transcript()
    document.append(source, "green")
    view = TranscriptView(document)

    class Renderer(App):
        def compose(self) -> ComposeResult:
            """Mount the common viewport independently of chat borders."""
            yield view

    async with Renderer().run_test(size=(79, 24)) as pilot:
        await pilot.pause()
        rendered = "".join("".join(segment.text for segment in view.render_line(row)).rstrip() for row in range(view.virtual_size.height))
        assert rendered == source
        document.append("\nFinal colored line", "green")
        view.sync()
        row = view.render_line(view.virtual_size.height - 1)
        assert all(segment.style.color == Color.parse("green") for segment in row if segment.text.strip() and segment.style)


async def test_supporting_tasks_and_errors_preserve_recorded_clocks() -> None:
    """Keep optional failures local and retain original times during replay."""
    app = TradingApp()
    base = time.time() - 30
    async with app.run_test() as pilot:
        for event in (
            AgentEvent("stage_start", "Workflow", "analysis", timestamp=base),
            AgentEvent("agent_start", "Market Analyst", "analysis", timestamp=base),
            AgentEvent("agent_start", "News Analyst", "analysis", timestamp=base),
            AgentEvent("agent_end", "News Analyst", "analysis", timestamp=base + 2),
            AgentEvent("agent_start", "Reflection 42", "analysis", timestamp=base),
            AgentEvent("prompt", "Reflection 42", "analysis", "Review a resolved decision.", timestamp=base),
            AgentEvent("error", "Reflection 42", "analysis", "Optional provider unavailable", timestamp=base + 4),
        ):
            await app.receive(event)
        app.flush_events()
        assert app._failure is None
        assert app.agents["Reflection 42"].elapsed(time.monotonic()) == pytest.approx(4, abs=0.1)
        assert "Review a resolved decision." in app.memory_view.plain()
        await pilot.press("m")
        assert not app.query_one("#memory-details", Collapsible).collapsed
        await app.receive(AgentEvent("error", "Workflow", "reports", "Provider failed", timestamp=base + 5))
        app.flush_events()
        header = str(app.query_one("#run-status", Static).render())
        assert "Failed" in header and "00:05" in header

        await pilot.pause()
        assert app.agents["News Analyst"].status == "complete"
        assert app.agents["Market Analyst"].status == "stopped"
        assert app.agents["Market Analyst"].elapsed(time.monotonic()) == pytest.approx(5, abs=0.1)
        assert "00:05" in str(app.query_one("#run-status", Static).render())


async def test_prefetch_time_is_not_reset_by_model_start() -> None:
    """Include source collection time in the analyst's running clock."""
    app = TradingApp()
    async with app.run_test():
        started = time.time() - 10
        await app.receive(AgentEvent("tool_start", "Sentiment Analyst", "analysis", timestamp=started,
                                     data={"tool": "get_social_sentiment", "tool_call_id": "prefetch"}))
        await app.receive(AgentEvent("agent_start", "Sentiment Analyst", "analysis", timestamp=started + 9))
        app.flush_events()
        assert app.agents["Sentiment Analyst"].elapsed(time.monotonic()) >= 9


async def test_completed_report_is_loaded_without_export_controls(tmp_path: Path) -> None:
    """The Reports stage reads the automatically saved full report, without another export action."""
    (tmp_path / "reports").mkdir()
    full = "# Complete report\n\nMarket evidence\n\nBull and bear\n\nFinal recommendation"
    (tmp_path / "reports" / "complete_report.md").write_text(full)
    app = TradingApp(run_dir=tmp_path)
    async with app.run_test() as pilot:
        await app.receive(AgentEvent("report", "Portfolio Manager", "reports", "Final recommendation"))
        await app.receive(AgentEvent("run_end", "Workflow", "reports"))
        app.flush_events()
        await pilot.pause()
        assert app.report_document.plain() == full
        assert not app.query("Button")
        assert "e" not in {binding.key for binding in app.BINDINGS}


async def test_large_results_remain_complete_on_disk_without_payload_ui(tmp_path: Path) -> None:
    """Keep full tool results and ordinary output while offering concise expandable previews."""
    final = "HEAD-分析-" + "data123 " * 131_072 + "TAIL-日本語"
    prompt, output = "Input:" + "P" * 25_000, "Output:" + "O" * 25_000
    events = [
        AgentEvent("prompt", "Market Analyst", "analysis", prompt),
        AgentEvent("tool_start", "Market Analyst", "analysis", "python check.py", {"tool": "bash", "tool_call_id": "large"}),
        AgentEvent("tool_result", "Market Analyst", "analysis", final, {"tool_call_id": "large", "final": True}),
        AgentEvent("text", "Market Analyst", "analysis", output, {"message_id": "answer", "final": True}),
    ]
    recorder = EventRecorder(tmp_path)
    app = TradingApp(selected_analysts=("market",), run_dir=tmp_path)
    try:
        async with app.run_test(size=(140, 42)) as pilot:
            for event in events:
                recorder.append(event)
                await app.receive(event)
            app.flush_events()
            await pilot.pause()
            view = app.agent_views["Market Analyst"]
            assert final in view.plain() and prompt in view.plain() and output in view.plain()
            assert final not in view.visible_plain()
            tool = next(s for b in view.bubbles for s in b.sections if s.kind == "tool_result")
            tool.collapsed = False
            assert f"[omitted {len(final) - 8_000} chars]" in view.visible_plain()
            assert final[:6000] in view.visible_plain() and final[-2000:] in view.visible_plain()
            assert not app.query("#transcript-hint, #details, #raw-events")
    finally:
        await asyncio.to_thread(recorder.close)
    store = RunStore(tmp_path)
    assert store.events() == events
    store.close()
    exported = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert exported[2]["text"] == final


def test_conditional_cache_write_formatting() -> None:
    """Only show cache creation counts when a provider actually supplied them."""
    assert "write" not in usage_label([Usage(input_tokens=100)])
    assert "write 0" in usage_label([Usage(cache_write_tokens=0)])
    assert "write 40+?" in usage_label([Usage(cache_write_tokens=40), Usage()])
