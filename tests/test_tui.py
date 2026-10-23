from __future__ import annotations

import asyncio

import pytest
from textual.containers import Vertical
from textual.strip import Strip
from textual.widgets import Collapsible, ContentSwitcher, Static

from open_trading_agents.domain import ANALYSTS, STAGES, AgentEvent, EventSink, Usage
from open_trading_agents.tui import TradingApp, Transcript, TranscriptView, usage_label


@pytest.mark.asyncio
async def test_parallel_streams_keep_complete_transcripts_and_usage() -> None:
    """Keep every concurrent agent fragment and count final usage once per request."""
    app = TradingApp(ticker="NVDA")
    async with app.run_test(size=(150, 45)) as pilot:
        for agent in ANALYSTS.values():
            await app.receive(AgentEvent("agent_start", agent, "analysis"))
            await app.receive(AgentEvent("system", agent, "analysis", "Research using the supplied tools."))
            await app.receive(AgentEvent("prompt", agent, "analysis", "Analyze NVDA."))
        for index in range(300):
            for agent in ANALYSTS.values():
                await app.receive(AgentEvent("text", agent, "analysis", f"{agent} {index} 日本語 분석\n", {"message_id": "reply"}))
        for agent in ANALYSTS.values():
            await app.receive(
                AgentEvent(
                    "usage",
                    agent,
                    "analysis",
                    data={
                        "request_id": "reply",
                        "input_tokens": 150,
                        "output_tokens": 40,
                        "cache_read_tokens": 100,
                        "cache_write_tokens": 0,
                    },
                )
            )
            await app.receive(
                AgentEvent(
                    "usage",
                    agent,
                    "analysis",
                    data={
                        "request_id": "reply",
                        "input_tokens": 150,
                        "output_tokens": 40,
                        "cache_read_tokens": 100,
                        "cache_write_tokens": 0,
                    },
                )
            )
            await app.receive(AgentEvent("agent_end", agent, "analysis", data={"output": "Duplicate final text"}))
        app.flush_events()
        app.flush_events()
        await pilot.pause()

        assert len(app.usage) == 4
        assert sum(value.input_tokens for value in app.usage.values()) == 600
        assert sum(value.cache_read_tokens or 0 for value in app.usage.values()) == 400
        for agent in ANALYSTS.values():
            transcript = app.agent_documents[agent].plain()
            assert f"{agent} 0 日本語 분석" in transcript
            assert f"{agent} 299 日本語 분석" in transcript
            assert "Duplicate final text" not in transcript
            assert "[Output]" not in transcript
            assert "Research using the supplied tools." in transcript
        assert "cache_read_tokens" in app.raw_document.plain()


@pytest.mark.asyncio
async def test_tabs_selection_fullscreen_scrolling_and_resize() -> None:
    """Preserve complete Unicode scrollback while stage and pane controls stay independent."""
    app = TradingApp()
    async with app.run_test(size=(140, 42)) as pilot:
        await app.receive(
            AgentEvent("text", "Sentiment Analyst", "analysis", "\n".join(f"{index:04} 한국어 日本語 " * 4 for index in range(600)))
        )
        app.flush_events()
        await pilot.pause()
        original_pane = app.query_one("#pane-1", Vertical)
        await pilot.press("right", "enter")
        assert app.selected_agent == 1
        assert app.fullscreen
        assert app.query_one("#pane-0").has_class("hidden")
        view = app.query_one("#analyst-1", TranscriptView)
        await pilot.press("home", "pagedown")
        assert not view.following
        top = view.scroll_y
        await app.receive(AgentEvent("text", "Sentiment Analyst", "analysis", "\nNew streaming text"))
        app.flush_events()
        await pilot.pause()
        assert view.scroll_y == top
        await pilot.resize_terminal(100, 36)
        await pilot.pause()
        assert not view.following
        assert "0000 한국어 日本語" in view.document.plain()
        assert len(view._cache) <= 256
        await pilot.press("end")
        assert view.following
        await pilot.press("escape", "tab")
        assert app.active_stage == "research"
        assert app.query_one("#stages", ContentSwitcher).current == "stage-research"
        await pilot.press("shift+tab")
        assert app.active_stage == "analysis"
        assert not app.fullscreen
        assert app.query_one("#pane-1", Vertical) is original_pane
        await pilot.press("shift+tab")
        assert app.active_stage == STAGES[-1][0]


@pytest.mark.asyncio
async def test_debate_timeline_side_pane_and_channel_boundaries() -> None:
    """Show the entire conversation chronologically and the current speaker separately."""
    app = TradingApp()
    async with app.run_test(size=(150, 44)) as pilot:
        await pilot.press("tab")
        for turn, agent, text in ((1, "Bull Researcher", "Bull case"), (2, "Bear Researcher", "Bear response")):
            await app.receive(AgentEvent("agent_start", agent, "research"))
            await app.receive(AgentEvent("turn_start", agent, "research", data={"round": 1, "turn": turn, "total_rounds": 3}))
            await app.receive(AgentEvent("prompt", agent, "research", "Read the earlier argument."))
            await app.receive(AgentEvent("reasoning", agent, "research", "Weighing the evidence.", {"message_id": agent}))
            await app.receive(AgentEvent("text", agent, "research", text, {"message_id": agent}))
            await app.receive(AgentEvent("text", agent, "research", text, {"message_id": agent, "final": True}))
            await app.receive(AgentEvent("agent_end", agent, "research", data={"output": text}))
        app.flush_events()
        await pilot.pause()
        main = app.stage_documents["research"].plain()
        assert main.index("Bull case") < main.index("Bear response")
        assert main.count("Bull case") == 1
        assert main.count("Bear response") == 1
        assert "Round 1/3 · Turn 2" in main
        assert "[Reasoning]" not in main
        assert "Read the earlier argument." in main
        side = app.query_one("#current-research", TranscriptView)
        assert side.document is app.agent_documents["Bear Researcher"]
        await pilot.press("d")
        assert not app.query_one("#details", Collapsible).collapsed


@pytest.mark.asyncio
async def test_clocks_tick_during_silent_work_and_do_not_reindex_transcripts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep time and navigation responsive while the workflow awaits a silent provider."""
    release = asyncio.Event()
    rendered_lines: dict[str, int] = {}
    original_render = TranscriptView.render_line

    def count_render(view: TranscriptView, y: int) -> Strip:
        """Count viewport work performed while only the clock is changing.

        Args:
            view: Transcript being rendered.
            y: Viewport row.

        Returns:
            Original rendered line.
        """
        key = view.id or ""
        rendered_lines[key] = rendered_lines.get(key, 0) + 1
        return original_render(view, y)

    monkeypatch.setattr(TranscriptView, "render_line", count_render)

    async def run(sink: EventSink) -> None:
        """Simulate an agent waiting for an external response.

        Args:
            sink: UI event receiver.
        """
        await sink(AgentEvent("stage_start", "", "analysis"))
        await sink(AgentEvent("agent_start", "Market Analyst", "analysis"))
        await sink(AgentEvent("text", "Market Analyst", "analysis", "A complete long transcript\n" * 3000))
        await release.wait()

    app = TradingApp(run)
    async with app.run_test(size=(130, 40)) as pilot:
        await pilot.pause(0.15)
        view = app.query_one("#analyst-0", TranscriptView)
        offsets = view._offsets
        rendered = rendered_lines.get("analyst-0", 0)
        await pilot.pause(1.05)
        assert "00:01" in str(app.query_one("#run-status", Static).render())
        assert view._offsets is offsets
        assert rendered_lines.get("analyst-0", 0) - rendered < 100
        await pilot.press("tab", "shift+tab")
        assert app.active_stage == "analysis"
        release.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_quit_cancels_workflow_and_waits_for_cleanup() -> None:
    """Terminate active tasks cleanly rather than leaving model processes behind."""
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def run(sink: EventSink) -> None:
        """Wait until cancelled and record cleanup completion.

        Args:
            sink: UI event receiver.
        """
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    app = TradingApp(run)
    async with app.run_test() as pilot:
        await started.wait()
        await pilot.press("q")
    assert cleaned.is_set()


def test_usage_distinguishes_missing_cache_information_from_zero() -> None:
    """Keep provider omissions explicit and show cache values as input subsets."""
    assert "Cache read —" in usage_label([Usage(input_tokens=100)])
    assert "Cache read 0 · write 0" in usage_label([Usage(input_tokens=100, cache_read_tokens=0, cache_write_tokens=0)])
    mixed = usage_label([Usage(input_tokens=100, cache_read_tokens=70), Usage(input_tokens=50)])
    assert "In 150" in mixed
    assert "Cache read 70+?" in mixed


@pytest.mark.asyncio
async def test_compact_panes_preserve_wide_characters_and_readable_viewports() -> None:
    """Keep wide characters intact at odd cell widths and reserve useful pane space."""
    app = TradingApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one("#analyst-0", TranscriptView)
        source = "日本語한국어" * 60
        document = Transcript()
        document.append(source, "green")
        view.set_document(document)
        view.navigate("home")
        await pilot.pause()
        assert view.size.height >= 3
        assert app.screen.has_class("compact")
        rendered = "".join("".join(segment.text for segment in view.render_line(row)).rstrip() for row in range(view.virtual_size.height))
        assert rendered == source
