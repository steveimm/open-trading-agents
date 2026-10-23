from __future__ import annotations

import asyncio
import bisect
import json
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.cells import cell_len, chop_cells
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Collapsible, ContentSwitcher, Static, Tab, Tabs

from open_trading_agents.domain import ANALYSTS, STAGES, AgentEvent, EventSink, Usage

CHANNEL_COLORS = {
    "system": "#939caa",
    "prompt": "#e8b86d",
    "reasoning": "#d8ca81",
    "text": "#91d5ac",
    "tool_start": "#78cbd4",
    "tool_result": "#a6bfee",
    "error": "#f2858a",
    "report": "#b5e3c8",
}
DEBATE_STAGES = {"research", "risk"}


class Transcript:
    def __init__(self) -> None:
        """Keep complete logical lines independently of terminal rendering."""
        self.lines = [Text()]
        self.revision = 0
        self.last_channel = ""
        self.last_agent = ""

    def append(self, text: str, style: str = "") -> None:
        """Append styled text without reprocessing earlier lines.

        Args:
            text: Literal content from an event.
            style: Rich text style for the added content.
        """
        if not text:
            return
        pieces = text.replace("\r\n", "\n").replace("\r", "").expandtabs(4).split("\n")
        self.lines[-1].append(pieces[0], style)
        self.lines.extend(Text(piece, style=style) for piece in pieces[1:])
        self.revision += 1

    def heading(self, title: str) -> None:
        """Separate a new agent turn or input block with one visible heading.

        Args:
            title: Short contextual heading.
        """
        prefix = "\n\n" if any(line.plain for line in self.lines) else ""
        self.append(f"{prefix}{title}\n", "bold #b7c2d4")

    def plain(self) -> str:
        """Return the full transcript for export and verification.

        Returns:
            Every logical line, including content outside the visible viewport.
        """
        return "\n".join(line.plain for line in self.lines)


class TranscriptView(ScrollView):
    DEFAULT_CSS = """
    TranscriptView { height: 1fr; width: 1fr; scrollbar-size: 1 1; background: #111821; }
    TranscriptView:focus { background: #141e2a; }
    """
    can_focus = True

    def __init__(self, document: Transcript, *, id: str | None = None) -> None:
        """Display a complete transcript while rendering only visible lines.

        Args:
            document: Complete transcript shared with the run state.
            id: Textual widget identifier.
        """
        super().__init__(id=id)
        self.document = document
        self.following = True
        self._width = 0
        self._revision = -1
        self._offsets = [0]
        self._wrap_starts: list[list[int]] = []
        self._cache: OrderedDict[int, Strip] = OrderedDict()

    def set_document(self, document: Transcript) -> None:
        """Switch the current speaker without replacing the widget.

        Args:
            document: Transcript for the newly selected speaker.
        """
        if self.document is document:
            return
        self.document = document
        self._revision = -1
        self._width = 0
        self._offsets = [0]
        self._wrap_starts.clear()
        self._cache.clear()
        self.following = True
        self.sync()

    def sync(self) -> None:
        """Update line indexes after content or width changes, preserving scrollback."""
        width = max(1, self.scrollable_content_region.width)
        if width == self._width and self._revision == self.document.revision:
            return
        old_source = bisect.bisect_right(self._offsets, int(self.scroll_y)) - 1
        resized = width != self._width
        start = 0 if resized or self._revision < 0 else max(0, len(self._wrap_starts) - 1)
        if resized:
            self._cache.clear()
        else:
            for index in list(self._cache):
                if index >= start:
                    del self._cache[index]
        self._wrap_starts[start:] = [self._line_starts(line.plain, width) for line in self.document.lines[start:]]
        self._offsets = self._offsets[: start + 1]
        for starts in self._wrap_starts[start:]:
            self._offsets.append(self._offsets[-1] + len(starts))
        self._width = width
        self._revision = self.document.revision
        self.virtual_size = Size(width, self._offsets[-1])
        if self.following:
            self.scroll_end(animate=False, immediate=True)
        elif resized:
            anchor = self._offsets[min(max(0, old_source), len(self._offsets) - 1)]
            self.scroll_to(y=anchor, animate=False, immediate=True)
        self.refresh()

    @staticmethod
    def _line_starts(text: str, width: int) -> list[int]:
        """Find soft-wrap offsets without splitting wide Unicode graphemes.

        Args:
            text: One logical line of plain text.
            width: Viewport width in terminal cells.

        Returns:
            Cell offsets of the line's wrapped rows.
        """
        starts, offset = [], 0
        for chunk in chop_cells(text, width):
            starts.append(offset)
            offset += cell_len(chunk)
        return starts or [0]

    def render_line(self, y: int) -> Strip:
        width = max(1, self.scrollable_content_region.width)
        row = int(self.scroll_y) + y
        index = bisect.bisect_right(self._offsets, row) - 1
        if index < 0 or index >= len(self._wrap_starts):
            return Strip.blank(width, self.rich_style)
        if index not in self._cache:
            self._cache[index] = Strip(self.document.lines[index].render(self.app.console))
            if len(self._cache) > 256:
                self._cache.popitem(last=False)
        strip = self._cache[index]
        self._cache.move_to_end(index)
        local_row = row - self._offsets[index]
        offset = self._wrap_starts[index][local_row] if local_row < len(self._wrap_starts[index]) else 0
        return strip.crop_extend(offset, offset + width, self.rich_style).apply_style(self.rich_style)

    def on_resize(self, event: events.Resize) -> None:
        """Reindex line widths when the terminal or pane changes size.

        Args:
            event: Textual resize notification.
        """
        self.sync()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Keep incoming output from moving a reader away from older content.

        Args:
            event: Mouse wheel notification.
        """
        self.following = False

    def navigate(self, direction: str) -> None:
        """Scroll a transcript or explicitly return to following new output.

        Args:
            direction: up, down, pageup, pagedown, home, or end.
        """
        self.following = direction == "end"
        if direction == "end":
            self.scroll_end(animate=False, immediate=True)
        elif direction == "home":
            self.scroll_home(animate=False, immediate=True)
        else:
            amount = max(1, self.size.height - 1) if direction.startswith("page") else 1
            sign = -1 if direction in {"up", "pageup"} else 1
            self.scroll_relative(y=sign * amount, animate=False, immediate=True)


@dataclass
class Activity:
    started: float | None = None
    finished: float | None = None
    status: str = "waiting"
    last_event: float | None = None
    round: int | None = None
    turn: int | None = None
    total_rounds: int | None = None

    def elapsed(self, now: float) -> float:
        """Calculate elapsed duration independently of incoming model events.

        Args:
            now: Current monotonic timestamp.

        Returns:
            Seconds spent between start and completion or now.
        """
        return max(0, (self.finished or now) - self.started) if self.started is not None else 0


def duration(seconds: float) -> str:
    """Format elapsed duration for compact terminal status labels.

    Args:
        seconds: Elapsed seconds.

    Returns:
        Minutes and seconds, or hours, minutes, and seconds for long runs.
    """
    value = int(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{second:02}" if hours else f"{minutes:02}:{second:02}"


def usage_label(usages: list[Usage], compact: bool = False) -> str:
    """Format input totals and cache subsets without counting cached input twice.

    Args:
        usages: Authoritative per-request usage records.
        compact: Use shorter counters in narrow panes.

    Returns:
        Compact usage description with unknown cache counts shown explicitly.
    """
    if not usages:
        return "In 0 Out 0 Cache R— W—" if compact else "In 0 · Out 0 · Cache read — · write —"
    read_text = _cache_count([value.cache_read_tokens for value in usages], compact)
    write_text = _cache_count([value.cache_write_tokens for value in usages], compact)
    inputs = sum(value.input_tokens for value in usages)
    outputs = sum(value.output_tokens for value in usages)
    if compact:
        return f"In {_short(inputs)} Out {_short(outputs)} Cache R{read_text} W{write_text}"
    return f"In {inputs:,} · Out {outputs:,} · Cache read {read_text} · write {write_text}"


def _cache_count(values: list[int | None], compact: bool) -> str:
    """Distinguish incomplete provider counts from reported cache misses.

    Args:
        values: Cache counts for each completed request.
        compact: Abbreviate known totals for narrow panes.

    Returns:
        Known count, partial count, or an unavailable marker.
    """
    total = sum(value or 0 for value in values)
    count = _short(total) if compact else f"{total:,}"
    if all(value is not None for value in values):
        return count
    return f"{count}+?" if total else "—"


def _short(value: int) -> str:
    """Abbreviate a token count for narrow status panes.

    Args:
        value: Exact token count retained in the run state.

    Returns:
        Compact display value.
    """
    return f"{value / 1_000_000:.1f}m" if value >= 1_000_000 else f"{value / 1000:.1f}k" if value >= 1000 else str(value)


class TradingApp(App[None]):
    TITLE = "Open Trading Agents"
    CSS = """
    Screen { background: #0d131c; color: #d9e2ef; }
    #run-status { height: 2; padding: 0 1; background: #1a2635; }
    #stage-tabs { height: 3; }
    #stage-status { height: 2; padding: 0 1; color: #a9bcd3; }
    #stages { height: 1fr; }
    .stage { height: 1fr; }
    #stage-analysis { grid-size: 2 2; grid-gutter: 1; padding: 0 1; }
    #stage-analysis.one-agent { grid-size: 1 1; }
    #stage-analysis.two-agents { grid-size: 2 1; }
    .agent-pane { border: round #34475e; height: 1fr; }
    .agent-pane.selected { border: round #bd9bfa; }
    .agent-status { height: 3; padding: 0 1; color: #9fb2ca; }
    .debate-main { width: 2fr; border: round #34475e; }
    .debate-speaker { width: 1fr; border: round #75559d; }
    .speaker-status { height: 3; padding: 0 1; }
    .single-stage { border: round #34475e; margin: 0 1; }
    .full-grid { grid-size: 1 1 !important; }
    .hidden { display: none; }
    #details { height: auto; max-height: 12; padding: 0; border: none; }
    #raw-events { height: 8; }
    #usage { height: 2; padding: 0 1; background: #1a2635; }
    #shortcuts { height: 1; padding: 0 1; color: #8799af; }
    Screen.compact #run-status, Screen.compact #stage-status, Screen.compact #usage { height: 1; }
    Screen.compact #stage-tabs { height: 2; }
    Screen.compact .agent-status { height: 2; }
    Screen.compact #stage-analysis { grid-gutter: 0; }
    """
    BINDINGS = [
        Binding("tab", "next_stage", "Next stage", priority=True),
        Binding("shift+tab", "previous_stage", "Previous stage", priority=True),
        Binding("up", "navigate('up')", show=False, priority=True),
        Binding("down", "navigate('down')", show=False, priority=True),
        Binding("left", "navigate('left')", show=False, priority=True),
        Binding("right", "navigate('right')", show=False, priority=True),
        Binding("pageup", "navigate('pageup')", show=False, priority=True),
        Binding("pagedown", "navigate('pagedown')", show=False, priority=True),
        Binding("home", "navigate('home')", show=False, priority=True),
        Binding("end", "navigate('end')", show=False, priority=True),
        Binding("enter,f", "fullscreen", "Expand", priority=True),
        Binding("escape", "restore", "Restore", priority=True),
        Binding("d", "details", "Payloads", priority=True),
        Binding("ctrl+c,q", "stop", "Quit", priority=True),
    ]

    def __init__(
        self,
        run: Callable[[EventSink], Awaitable[object]] | None = None,
        *,
        selected_analysts: tuple[str, ...] = tuple(ANALYSTS),
        ticker: str = "",
        run_dir: Path | None = None,
    ) -> None:
        """Create the live research interface without starting model work.

        Args:
            run: Async workflow receiving the app event sink.
            selected_analysts: Enabled analyst keys in stable display order.
            ticker: Instrument shown in the run header.
            run_dir: Local result directory displayed after completion.
        """
        super().__init__()
        self.run_callback = run
        self.selected_analysts = selected_analysts
        self.ticker = ticker
        self.run_dir = run_dir
        self.error: Exception | None = None
        self.result: object | None = None
        self.stage_index = 0
        self.selected_agent = 0
        self.fullscreen = False
        self.agent_documents: dict[str, Transcript] = {}
        self.stage_documents = {stage: Transcript() for stage, _ in STAGES}
        self.raw_document = Transcript()
        self.agents: dict[str, Activity] = {}
        self.stages = {stage: Activity() for stage, _ in STAGES}
        self.usage: dict[tuple[str, str], Usage] = {}
        self.current_speakers: dict[str, str] = {}
        self._pending: deque[AgentEvent] = deque()
        self._event_ids: set[str] = set()
        self._render_views: list[TranscriptView] = []
        self._started = time.monotonic()
        self._finished: float | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._failure: str | None = None
        self._streamed: set[tuple[str, str, str]] = set()
        self._turn_streamed: set[str] = set()
        self._status_widgets: dict[str, Static] = {}
        self._agent_stages: dict[str, str] = {}
        for key in selected_analysts:
            name = ANALYSTS.get(key, key)
            self.agent_documents[name] = Transcript()
            self.agents[name] = Activity()

    def _view(self, document: Transcript, widget_id: str) -> TranscriptView:
        """Register a persistent transcript viewport.

        Args:
            document: Complete transcript backing the viewport.
            widget_id: Stable Textual identifier.

        Returns:
            Registered transcript widget.
        """
        view = TranscriptView(document, id=widget_id)
        self._render_views.append(view)
        return view

    def compose(self) -> ComposeResult:
        """Build stable stage tabs and transcript panes once.

        Yields:
            Widgets composing the research interface.
        """
        yield Static(id="run-status")
        yield Tabs(*(Tab(label, id=f"tab-{stage}") for stage, label in STAGES), id="stage-tabs")
        yield Static(id="stage-status")
        with ContentSwitcher(initial="stage-analysis", id="stages"):
            with Grid(
                id="stage-analysis",
                classes=f"stage {'one-agent' if len(self.selected_analysts) == 1 else 'two-agents' if len(self.selected_analysts) == 2 else ''}",
            ):
                for index, key in enumerate(self.selected_analysts):
                    name = ANALYSTS.get(key, key)
                    with Vertical(id=f"pane-{index}", classes="agent-pane selected" if index == 0 else "agent-pane"):
                        yield Static(name, id=f"agent-status-{index}", classes="agent-status")
                        yield self._view(self.agent_documents[name], f"analyst-{index}")
            for stage, _ in STAGES[1:]:
                if stage in DEBATE_STAGES:
                    with Horizontal(id=f"stage-{stage}", classes="stage"):
                        with Vertical(classes="debate-main"):
                            yield self._view(self.stage_documents[stage], f"transcript-{stage}")
                        with Vertical(classes="debate-speaker"):
                            yield Static("Waiting for speaker", id=f"speaker-{stage}", classes="speaker-status")
                            yield self._view(Transcript(), f"current-{stage}")
                else:
                    with Vertical(id=f"stage-{stage}", classes="stage single-stage"):
                        yield self._view(self.stage_documents[stage], f"transcript-{stage}")
        with Collapsible(title="Raw event payloads", collapsed=True, id="details"):
            yield self._view(self.raw_document, "raw-events")
        yield Static(id="usage")
        yield Static(
            "Tab/Shift+Tab stage · Arrows select/scroll · Enter/f expand · Esc restore · End live · d payloads · q quit", id="shortcuts"
        )

    def on_mount(self) -> None:
        """Start independent display timers and launch the supplied workflow."""
        self._status_widgets = {widget.id: widget for widget in self.query(Static) if widget.id}
        self._adapt_layout()
        self.set_interval(0.04, self.flush_events)
        self.set_interval(0.1, self.refresh_clocks)
        self.refresh_clocks()
        if self.run_callback is not None:
            self._run_task = asyncio.create_task(self._execute())

    async def _execute(self) -> None:
        """Run the workflow without blocking terminal event processing."""
        try:
            assert self.run_callback is not None
            self.result = await self.run_callback(self.receive)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc
            self._failure = str(exc)
            await self.receive(AgentEvent("error", "Workflow", self.active_stage, str(exc)))
        finally:
            self._finished = time.monotonic()
            self.flush_events()

    async def receive(self, event: AgentEvent) -> None:
        """Queue an event quickly so model streams never wait on rendering.

        Args:
            event: Workflow or harness event to display.
        """
        self._pending.append(event)

    @property
    def active_stage(self) -> str:
        """Return the stage currently selected by the user.

        Returns:
            Stable stage key.
        """
        return STAGES[self.stage_index][0]

    def flush_events(self) -> None:
        """Apply a batch of queued events and refresh only changed visible transcripts."""
        for _ in range(min(len(self._pending), 1000)):
            event = self._pending.popleft()
            if event.event_id not in self._event_ids:
                self._event_ids.add(event.event_id)
                self._apply(event)
        for view in self._render_views:
            if view.is_on_screen:
                view.sync()

    def _apply(self, event: AgentEvent) -> None:
        """Update local presentation state from one event.

        Args:
            event: Deduplicated run event.
        """
        now = time.monotonic()
        kind = {"system_prompt": "system", "text_delta": "text", "thinking_delta": "reasoning", "output": "text"}.get(
            event.kind, event.kind
        )
        stage = event.stage if event.stage in self.stage_documents else self.active_stage
        agent = event.agent or "Workflow"
        document = self.agent_documents.setdefault(agent, Transcript())
        activity = self.agents.setdefault(agent, Activity())
        self._agent_stages[agent] = stage
        activity.last_event = now
        self.raw_document.append(json.dumps(asdict(event), ensure_ascii=False, default=str) + "\n", "#899bb3")
        if kind == "stage_start":
            self.stages[stage] = Activity(started=now, status="running")
        elif kind == "stage_end":
            self.stages[stage].finished = now
            self.stages[stage].status = "complete"
        elif kind == "agent_start":
            activity.started = activity.started or now
            activity.status = "running"
            activity.finished = None
            self._turn_streamed.discard(agent)
        elif kind == "agent_end":
            activity.finished = now
            activity.status = "complete"
            output = event.data.get("output", "")
            if output and agent not in self._turn_streamed:
                self._append_event(document, self.stage_documents[stage], agent, "text", str(output))
        elif kind == "turn_start":
            activity.round = event.data.get("round")
            activity.turn = event.data.get("turn")
            activity.total_rounds = event.data.get("total_rounds")
            activity.status = "running"
            self._turn_streamed.discard(agent)
            self._speaker(stage, agent)
            self.stage_documents[stage].heading(
                f"{agent} · Round {activity.round or 1}/{activity.total_rounds or '?'} · Turn {activity.turn or 1}"
            )
            self.stage_documents[stage].last_agent = agent
        elif kind == "usage":
            raw = event.data.get("usage", event.data)
            if isinstance(raw, Usage):
                value = raw
            else:
                value = Usage(**{key: raw[key] for key in Usage.__dataclass_fields__ if key in raw})
            request_id = str(event.data.get("request_id") or event.data.get("message_id") or event.event_id)
            self.usage[agent, request_id] = value
        elif kind == "status":
            activity.status = event.text or str(event.data.get("status", "running"))
        elif kind == "run_end":
            self._finished = now
        elif kind in CHANNEL_COLORS:
            text = event.text
            if kind in {"text", "reasoning"}:
                message_id = str(event.data.get("message_id", ""))
                key = agent, message_id, kind
                if event.data.get("final") and key in self._streamed:
                    return
                self._streamed.add(key)
                if kind == "text":
                    self._turn_streamed.add(agent)
            if kind in {"tool_start", "tool_result"}:
                name = event.data.get("name") or event.data.get("tool_name") or "tool"
                payload = event.data.get("arguments") if kind == "tool_start" else event.data.get("result")
                if not text and payload is not None:
                    text = json.dumps(payload, ensure_ascii=False, default=str)
                text = f"\n{name}{' result' if kind == 'tool_result' else ''}\n{text}\n"
                activity.status = "using tool" if kind == "tool_start" else "running"
            if kind == "error":
                activity.status = "error"
                activity.finished = now
            if kind == "report":
                self.stage_documents["reports"].heading(agent)
                self.stage_documents["reports"].append(text + "\n", CHANNEL_COLORS[kind])
                return
            if stage in DEBATE_STAGES and kind in {"text", "reasoning", "prompt"}:
                self._speaker(stage, agent)
            self._append_event(document, self.stage_documents[stage], agent, kind, text)

    def _speaker(self, stage: str, agent: str) -> None:
        """Point a debate side pane at the current speaker's complete transcript.

        Args:
            stage: Debate stage key.
            agent: Current speaker name.
        """
        if stage not in DEBATE_STAGES or self.current_speakers.get(stage) == agent:
            return
        self.current_speakers[stage] = agent
        self.query_one(f"#current-{stage}", TranscriptView).set_document(self.agent_documents[agent])

    def _append_event(self, document: Transcript, stage: Transcript, agent: str, kind: str, text: str) -> None:
        """Append styled content with a heading only at meaningful boundaries.

        Args:
            document: Per-agent transcript.
            stage: Chronological stage transcript.
            agent: Source agent name.
            kind: Input/output channel.
            text: Literal event content.
        """
        for target in (document, stage):
            if target.last_agent != agent or kind in {"prompt", "system"}:
                title = f"{agent} · {'System instructions' if kind == 'system' else 'Input' if kind == 'prompt' else 'Response'}"
                target.heading(title)
            elif target.last_channel != kind and target.last_channel:
                target.append("\n")
            target.append(text, CHANNEL_COLORS[kind])
            target.last_agent = agent
            target.last_channel = kind

    def refresh_clocks(self) -> None:
        """Refresh clocks and compact statuses without touching transcript content."""
        if not self._status_widgets or not self._status_widgets["run-status"].is_attached:
            return
        now = time.monotonic()
        elapsed = duration((self._finished or now) - self._started)
        status = "Failed" if self._failure else "Complete" if self._finished else "Running"
        compact = self.screen.has_class("compact")
        separator = " · " if compact else "\n"
        self._status_widgets["run-status"].update(f"Open Trading Agents  {self.ticker}{separator}{status} · Total {elapsed}")
        stage = self.stages[self.active_stage]
        stage_text = f"{STAGES[self.stage_index][1]} · {stage.status} · {duration(stage.elapsed(now))}"
        participants = [name for name, key in self._agent_stages.items() if key == self.active_stage]
        if self.active_stage not in {"analysis", "reports", *DEBATE_STAGES} and participants:
            name = participants[-1]
            activity = self.agents[name]
            values = [value for (agent, _), value in self.usage.items() if agent == name]
            stage_text += f" · {name} {duration(activity.elapsed(now))}"
            if not compact:
                stage_text += "\n" + usage_label(values)
        self._status_widgets["stage-status"].update(stage_text)
        for index, key in enumerate(self.selected_analysts):
            name = ANALYSTS.get(key, key)
            activity = self.agents[name]
            values = [value for (agent, _), value in self.usage.items() if agent == name]
            label = name.removesuffix(" Analyst") if compact else name
            self._status_widgets[f"agent-status-{index}"].update(
                f"{label} · {activity.status} · {duration(activity.elapsed(now))}\n{usage_label(values, compact=compact)}"
            )
        for debate, name in self.current_speakers.items():
            activity = self.agents[name]
            values = [value for (agent, _), value in self.usage.items() if agent == name]
            self._status_widgets[f"speaker-{debate}"].update(
                f"{name} · {activity.status} · {duration(activity.elapsed(now))}\nRound {activity.round or 1}/{activity.total_rounds or '?'} · Turn {activity.turn or 1}\n{usage_label(values)}"
            )
        explanation = "" if compact else "\nInput includes cached tokens · — unavailable · +? incomplete cache reporting"
        self._status_widgets["usage"].update(usage_label(list(self.usage.values()), compact=compact) + explanation)

    def on_resize(self, event: events.Resize) -> None:
        """Keep usable transcript space in shorter terminals.

        Args:
            event: New terminal dimensions.
        """
        self._adapt_layout()

    def _adapt_layout(self) -> None:
        """Adjust status density without replacing transcript widgets."""
        compact = self.size.height < 32 or self.size.width < 110
        self.screen.set_class(compact, "compact")
        if "shortcuts" in self._status_widgets:
            shortcuts = "Tab stage · Arrows pane/scroll · f expand · Esc grid · End live · d payloads · q quit"
            self._status_widgets["shortcuts"].update(shortcuts)
        self.refresh_clocks()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Change stage content when a tab is selected.

        Args:
            event: Activated stage tab.
        """
        stage = (event.tab.id or "").removeprefix("tab-")
        if stage not in self.stage_documents:
            return
        self.stage_index = [key for key, _ in STAGES].index(stage)
        self.query_one("#stages", ContentSwitcher).current = f"stage-{stage}"
        self.call_after_refresh(self.flush_events)
        self.refresh_clocks()

    def action_next_stage(self) -> None:
        """Select the next workflow stage without changing keyboard focus order."""
        self._change_stage(1)

    def action_previous_stage(self) -> None:
        """Select the previous workflow stage."""
        self._change_stage(-1)

    def _change_stage(self, offset: int) -> None:
        """Select a stage relative to the current one.

        Args:
            offset: Signed stage offset.
        """
        self.stage_index = (self.stage_index + offset) % len(STAGES)
        self.query_one("#stage-tabs", Tabs).active = f"tab-{self.active_stage}"

    def action_navigate(self, direction: str) -> None:
        """Select an analyst in grid view or scroll the active transcript.

        Args:
            direction: Keyboard navigation direction.
        """
        focused = self.focused
        if isinstance(focused, TranscriptView) and focused.id in {"raw-events", f"current-{self.active_stage}"}:
            if direction not in {"left", "right"}:
                focused.navigate(direction)
            return
        if self.active_stage == "analysis" and not self.fullscreen and direction in {"left", "right", "up", "down"}:
            if not self.selected_analysts:
                return
            step = {"left": -1, "right": 1, "up": -2, "down": 2}[direction]
            self.selected_agent = (self.selected_agent + step) % len(self.selected_analysts)
            self._update_selection()
            return
        if direction in {"left", "right"}:
            return
        view_id = f"analyst-{self.selected_agent}" if self.active_stage == "analysis" else f"transcript-{self.active_stage}"
        self.query_one(f"#{view_id}", TranscriptView).navigate(direction)

    def _update_selection(self) -> None:
        """Apply selection and fullscreen visibility to existing analyst panes."""
        for index in range(len(self.selected_analysts)):
            pane = self.query_one(f"#pane-{index}", Vertical)
            pane.set_class(index == self.selected_agent, "selected")
            pane.set_class(self.fullscreen and index != self.selected_agent, "hidden")
        self.query_one("#stage-analysis", Grid).set_class(self.fullscreen, "full-grid")
        if self.selected_analysts:
            self.query_one(f"#analyst-{self.selected_agent}", TranscriptView).focus()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Select an analyst pane when its transcript receives mouse focus.

        Args:
            event: Focused descendant widget.
        """
        identifier = event.widget.id or ""
        if identifier.startswith("analyst-"):
            index = int(identifier.removeprefix("analyst-"))
            if index != self.selected_agent:
                self.selected_agent = index
                self._update_selection()

    def action_fullscreen(self) -> None:
        """Expand the selected analyst pane while other agents keep running."""
        if self.active_stage == "analysis":
            self.fullscreen = not self.fullscreen
            self._update_selection()

    def action_restore(self) -> None:
        """Return from fullscreen to the analyst grid."""
        if self.fullscreen:
            self.fullscreen = False
            self._update_selection()

    def action_details(self) -> None:
        """Toggle the complete local raw-event transcript."""
        details = self.query_one("#details", Collapsible)
        details.collapsed = not details.collapsed
        if not details.collapsed:
            self.query_one("#raw-events", TranscriptView).focus()
        self.call_after_refresh(self.flush_events)

    async def action_stop(self) -> None:
        """Cancel active workflow work and restore the terminal."""
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            await asyncio.gather(self._run_task, return_exceptions=True)
        self.exit()

    async def on_unmount(self) -> None:
        """Release workflow tasks when the application is closed externally."""
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            await asyncio.gather(self._run_task, return_exceptions=True)
