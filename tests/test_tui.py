from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from rich.color import Color
from textual.containers import Vertical
from textual.strip import Strip
from textual.widgets import Collapsible, ContentSwitcher, Static

from open_trading_agents.domain import ANALYSTS, STAGES, AgentEvent, EventSink, Usage
from open_trading_agents.events import EventRecorder
from open_trading_agents.reporting import write_transcript
from open_trading_agents.storage import RunStore
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
        assert view.size.width > 130
        assert view.size.height > 20
        await pilot.press("home", "pagedown")
        assert not view.following
        top = view.scroll_y
        await app.receive(AgentEvent("text", "Sentiment Analyst", "analysis", "\nNew streaming text"))
        app.flush_events()
        await pilot.pause()
        assert view.scroll_y == top
        await pilot.resize_terminal(100, 36)
        await pilot.pause()
        assert app.screen.has_class("compact")
        assert not view.following
        assert "0000 한국어 日本語" in view.document.plain()
        assert len(view._cache) <= 256
        await pilot.resize_terminal(140, 42)
        await pilot.pause()
        assert not app.screen.has_class("compact")
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
        document.append("\nFinal colored line", "green")
        view.sync()
        last = view.render_line(view.virtual_size.height - 1)
        colors = [segment.style.color for segment in last if segment.text.strip() and segment.style]
        assert colors and all(color == Color.parse("green") for color in colors)


@pytest.mark.asyncio
async def test_native_payloads_tool_deltas_memory_and_replayed_timings() -> None:
    """Preserve all inputs without duplicating native payloads or tool headings."""
    app = TradingApp()
    base = time.time() - 100
    async with app.run_test(size=(140, 42)) as pilot:
        agent = "Reflection 42"
        events = [
            AgentEvent("agent_start", agent, "analysis", timestamp=base),
            AgentEvent("system", agent, "analysis", "Application system instructions"),
            AgentEvent("system", agent, "analysis", '{"messages": ["raw native input"]}', {"source": "native_request"}),
            AgentEvent("prompt", agent, "analysis", "Review the resolved decision."),
            AgentEvent("reasoning", agent, "analysis", "Consider the measured outcome.", {"message_id": "one"}),
            AgentEvent("text", agent, "analysis", "First lesson.", {"message_id": "one", "final": True}),
            AgentEvent("agent_end", agent, "analysis", data={"output": "First lesson."}, timestamp=base + 5),
            AgentEvent("agent_start", agent, "analysis", timestamp=base + 60),
            AgentEvent("tool_start", agent, "analysis", "{}", {"tool": "get_stock_data", "tool_call_id": "tool-1"}),
            AgentEvent("tool_start", agent, "analysis", '{"ticker":', {"tool_call_id": "tool-1", "delta": True}),
            AgentEvent("tool_start", agent, "analysis", '"NVDA"}', {"tool_call_id": "tool-1", "delta": True}),
            AgentEvent("tool_result", agent, "analysis", "Verified rows", {"tool_call_id": "tool-1"}),
            AgentEvent("agent_end", agent, "analysis", data={"output": "Second lesson."}, timestamp=base + 68),
        ]
        for event in events:
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        assert app.agents[agent].elapsed(time.monotonic()) == pytest.approx(13, abs=0.1)
        transcript = app.memory_document.plain()
        assert "Application system instructions" in transcript
        assert "raw native input" not in transcript
        assert "raw native input" in app.raw_document.plain()
        assert transcript.count("First lesson.") == 1
        assert '{"ticker":"NVDA"}' in transcript
        assert transcript.count("get_stock_data") == 2
        assert "Second lesson." in transcript
        panel = app.query_one("#memory-details", Collapsible)
        assert not panel.has_class("hidden")
        assert panel.collapsed
        await pilot.press("m")
        assert not panel.collapsed
        await pilot.press("tab")
        assert panel.has_class("hidden")


@pytest.mark.asyncio
async def test_final_snapshots_complete_partial_streams_and_report_duplicates() -> None:
    """Keep authoritative final text when a provider's last snapshot adds content."""
    app = TradingApp(selected_analysts=("market",))
    async with app.run_test(size=(100, 35)) as pilot:
        agent = "Market Analyst"
        for event in (
            AgentEvent("text", agent, "analysis", "Partial ", {"message_id": "one"}),
            AgentEvent("text", agent, "analysis", "Partial completed response", {"message_id": "one", "final": True}),
            AgentEvent("report", agent, "analysis", "Rendered final report"),
            AgentEvent("report", agent, "reports", "Rendered final report"),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        transcript = app.agent_documents[agent].plain()
        assert transcript.count("Partial completed response") == 1
        assert app.stage_documents["reports"].plain().count("Rendered final report") == 1
        assert app.query_one("#analyst-0", TranscriptView).size.width > 90


@pytest.mark.asyncio
async def test_script_arguments_output_deltas_and_replaced_snapshots() -> None:
    """Keep native code readable and reconcile streamed output without duplicate results."""
    app = TradingApp()
    command = "python - <<'PY'\nimport math\nprint(math.sqrt(2))\nPY"
    async with app.run_test(size=(140, 42)) as pilot:
        agent = "Market Analyst"
        events = [
            AgentEvent("tool_start", agent, "analysis", json.dumps({"command": command}), {"tool": "Bash", "tool_call_id": "code"}),
            AgentEvent("tool_result", agent, "analysis", "1.414", {"tool_call_id": "code", "delta": True}),
            AgentEvent("tool_result", agent, "analysis", "21356\n", {"tool_call_id": "code", "delta": True}),
            AgentEvent("tool_result", agent, "analysis", "1.41421356\n", {"tool_call_id": "code", "final": True}),
            AgentEvent(
                "tool_result",
                agent,
                "analysis",
                '{"content":[{"text":"preliminary output\\n"}]}',
                {"tool": "Python", "tool_call_id": "snapshot", "snapshot": True},
            ),
            AgentEvent(
                "tool_result",
                "News Analyst",
                "analysis",
                "Independent search result",
                {"tool": "web_search", "tool_call_id": "search", "final": True},
            ),
            AgentEvent(
                "tool_result",
                agent,
                "analysis",
                '{"content":[{"text":"corrected output\\nnext line\\n"}]}',
                {"tool_call_id": "snapshot", "snapshot": True},
            ),
            AgentEvent(
                "tool_result",
                agent,
                "analysis",
                '{"content":[{"text":"corrected output\\nnext line\\n"}]}',
                {"tool_call_id": "snapshot", "final": True},
            ),
            AgentEvent(
                "tool_result",
                agent,
                "analysis",
                "stdout\n\x1b]0;changed title\x07\rstderr\x00",
                {"tool": "Bash", "tool_call_id": "controls", "final": True},
            ),
        ]
        for event in events:
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        transcript = app.agent_documents[agent].plain()
        shared = app.stage_documents["analysis"].plain()
        assert command in transcript
        assert transcript.count("1.41421356") == 1
        assert "preliminary output" not in transcript
        assert transcript.count("corrected output") == 1
        assert "corrected output\nnext line\n" in transcript
        assert "Independent search result" in shared
        assert "preliminary output" not in shared
        assert "\x1b" not in transcript and "\x00" not in transcript
        assert "\\x1b]0;changed title\\x07\\rstderr\\x00" in transcript
        view = app.query_one("#analyst-0", TranscriptView)
        await pilot.press("f", "home", "pagedown", "end")
        assert view.following


@pytest.mark.asyncio
async def test_terminal_workflow_error_stops_unfinished_agent_clocks() -> None:
    """Make failed replays visibly terminal while preserving completed analyst status."""
    app = TradingApp()
    timestamp = time.time() - 30
    async with app.run_test() as pilot:
        for event in (
            AgentEvent("stage_start", "Workflow", "analysis", timestamp=timestamp),
            AgentEvent("agent_start", "Market Analyst", "analysis", timestamp=timestamp),
            AgentEvent("agent_start", "News Analyst", "analysis", timestamp=timestamp),
            AgentEvent("agent_end", "News Analyst", "analysis", timestamp=timestamp + 2),
            AgentEvent("error", "Workflow", "reports", "Provider output limit reached", timestamp=timestamp + 5),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.pause(0.15)
        assert "Failed" in str(app.query_one("#run-status", Static).render())
        assert "00:05" in str(app.query_one("#run-status", Static).render())
        assert app.agents["Market Analyst"].status == "stopped"
        assert app.agents["News Analyst"].status == "complete"
        assert app.stages["analysis"].status == "failed"
        assert app.agents["Market Analyst"].elapsed(time.monotonic()) == pytest.approx(5, abs=0.1)
        await pilot.pause(0.15)
        assert app.agents["Market Analyst"].elapsed(time.monotonic()) == pytest.approx(5, abs=0.1)


@pytest.mark.asyncio
async def test_optional_screening_failure_remains_visible_without_stopping_analysis() -> None:
    """Expose supporting rollouts and stop their clocks when optional work fails."""
    app = TradingApp()
    base = time.time() - 20
    async with app.run_test() as pilot:
        name = "Sentiment Screening abc123"
        for event in (
            AgentEvent("agent_start", "Market Analyst", "analysis", timestamp=base),
            AgentEvent("agent_start", name, "analysis", timestamp=base),
            AgentEvent("prompt", name, "analysis", "Screen the fetched source posts."),
            AgentEvent("error", name, "analysis", "Provider unavailable", timestamp=base + 4),
        ):
            await app.receive(event)
        app.flush_events()
        await pilot.pause()
        assert "Screen the fetched source posts." in app.memory_document.plain()
        assert "Provider unavailable" in app.memory_document.plain()
        assert app.agents[name].status == "error"
        assert app.agents[name].elapsed(time.monotonic()) == pytest.approx(4, abs=0.1)
        assert app.agents["Market Analyst"].finished is None
        assert app._failure is None
        assert not app.query_one("#memory-details", Collapsible).has_class("hidden")


async def test_prefetch_time_remains_part_of_agent_wall_time() -> None:
    """Source collection starts the role clock and model startup does not reset it."""
    import time

    app = TradingApp()
    async with app.run_test(size=(100, 35)):
        started = time.time() - 10
        await app.receive(
            AgentEvent(
                "tool_start",
                "Sentiment Analyst",
                "analysis",
                timestamp=started,
                data={"tool": "get_social_sentiment", "tool_call_id": "prefetch"},
            )
        )
        await app.receive(AgentEvent("agent_start", "Sentiment Analyst", "analysis", timestamp=started + 9))
        app.flush_events()
        assert app.agents["Sentiment Analyst"].elapsed(time.monotonic()) >= 9


@pytest.mark.asyncio
async def test_megabyte_tool_previews_preserve_exact_events_and_ordinary_transcripts(tmp_path: Path) -> None:
    """Bound large displayed payloads while real persistence retains every original character."""
    agent = "Market Analyst"
    complete = "HEAD-分析-" + "data123 " * 131_072 + "TAIL-日本語"
    final = complete + " final stderr tail"
    native = json.dumps({"messages": [{"role": "system", "content": "Financial evidence\\n" * 60_000}]})
    prompt = "Ordinary input: " + "P" * 25_000
    output = "Ordinary output: " + "O" * 25_000
    events = [
        AgentEvent("system", agent, "analysis", native, {"source": "native_request"}),
        AgentEvent("prompt", agent, "analysis", prompt),
        AgentEvent("tool_start", agent, "analysis", "python calculation.py", {"tool": "Bash", "tool_call_id": "large"}),
        AgentEvent("tool_result", agent, "analysis", complete[:600_000], {"tool_call_id": "large", "snapshot": True}),
        AgentEvent("tool_result", agent, "analysis", complete[600_000:], {"tool_call_id": "large", "delta": True}),
        AgentEvent("tool_result", agent, "analysis", final, {"tool_call_id": "large", "final": True}),
        AgentEvent("text", agent, "analysis", output, {"message_id": "answer"}),
        AgentEvent("text", agent, "analysis", output, {"message_id": "answer", "final": True}),
    ]
    assert len(final.encode()) > 1_000_000 and len(native.encode()) > 1_000_000
    recorder = EventRecorder(tmp_path)
    app = TradingApp(selected_analysts=("market",), run_dir=tmp_path)
    try:
        async with app.run_test(size=(150, 44)) as pilot:
            for event in events:
                recorder.append(event)
                await app.receive(event)
            app.flush_events()
            await pilot.pause()
            transcript = app.agent_documents[agent].plain()
            marker = f"...[omitted {len(final) - 8_000} chars]..."
            assert final[:6_000] in transcript and final[-2_000:] in transcript
            assert transcript.count(marker) == 1
            assert final not in transcript
            assert app._tool_outputs[agent, "large"] == final
            assert prompt in transcript and output in transcript
            assert len(app.raw_document.plain()) < 40_000
            assert str(tmp_path / "transcript.jsonl") in str(app.query_one("#transcript-hint", Static).render())
            await pilot.press("d", "home", "pagedown", "end", "d")
            view = app.query_one("#analyst-0", TranscriptView)
            assert len(view._cache) <= 256
            assert all(strip.cell_length <= view.size.width for strip in view._cache.values())
    finally:
        await asyncio.to_thread(recorder.close)
    store = RunStore(tmp_path)
    try:
        saved = store.events()
    finally:
        store.close()
    assert saved == events
    write_transcript(tmp_path / "transcript.jsonl", saved)
    exported = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert exported[0]["text"] == native
    assert exported[5]["text"] == final
    assert events[5].text == final
