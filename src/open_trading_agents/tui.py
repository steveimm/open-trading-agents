from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.content import Content
from textual.geometry import Size
from textual.widgets import Collapsible, ContentSwitcher, Static, Tab, Tabs

from open_trading_agents.chat import ChatView
from open_trading_agents.domain import ANALYSTS, STAGES, AgentEvent, EventSink, Usage
from open_trading_agents.transcript import Transcript, TranscriptView
from open_trading_agents.usage import format_usage as usage_label
from open_trading_agents.usage import request_usage

DEBATE_STAGES = {"research", "risk"}


@dataclass
class Activity:
    started: float | None = None
    finished: float | None = None
    status: str = "waiting"
    last_event: float | None = None
    round: int | None = None
    turn: int | None = None
    total_rounds: int | None = None
    previous_seconds: float = 0

    def elapsed(self, now: float) -> float:
        """Calculate accumulated role or stage duration independently of model events.

        Args:
            now: Current monotonic timestamp.

        Returns:
            Elapsed active seconds.
        """
        current = max(0, (self.finished if self.finished is not None else now) - self.started) if self.started is not None else 0
        return self.previous_seconds + current


def duration(seconds: float) -> str:
    """Format elapsed time for a compact terminal status line.

    Args:
        seconds: Elapsed seconds.

    Returns:
        Minutes/seconds or hours/minutes/seconds.
    """
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, second = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{second:02}" if hours else f"{minutes:02}:{second:02}"


class TradingApp(App[None]):
    TITLE = "Open Trading Agents"
    CSS = """
    Screen { background: #0d131c; color: #d9e2ef; }
    #run-status { height: 2; padding: 0 1; background: #1a2635; }
    #stage-tabs { height: 3; }
    #stage-tabs Tab.running { color: #e8b86d; text-style: bold; }
    #stage-tabs Tab.complete { color: #91d5ac; }
    #stage-tabs Tab.failed { color: #f2858a; }
    #stage-status { height: 2; padding: 0 1; color: #a9bcd3; }
    #stages, .stage { height: 1fr; }
    #stage-analysis { grid-size: 2 2; grid-gutter: 1; padding: 0 1; }
    #stage-analysis.one-agent { grid-size: 1 1; }
    #stage-analysis.two-agents { grid-size: 2 1; }
    #stage-analysis.full-grid { grid-size: 1 1; }
    .agent-pane { border: round #34475e; height: 1fr; }
    .agent-pane.selected { border: round #bd9bfa; }
    .agent-status { height: 3; padding: 0 1; color: #9fb2ca; }
    .single-stage { border: round #34475e; margin: 0 1; }
    .hidden { display: none; }
    #memory-details { height: auto; max-height: 14; padding: 0; border: none; }
    #memory-transcript { height: 9; }
    #memory-status { height: 2; }
    #usage { height: 1; padding: 0 1; background: #1a2635; }
    #shortcuts { height: 1; padding: 0 1; color: #8799af; }
    Screen.compact #run-status, Screen.compact #stage-status { height: 1; }
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
        Binding("end", "navigate('end')", "Follow stream", priority=True),
        Binding("enter,f", "fullscreen", "Expand", priority=True),
        Binding("escape", "restore", "Restore", priority=True),
        Binding("left_square_bracket", "detail(-1)", "Previous detail", show=False, priority=True),
        Binding("right_square_bracket", "detail(1)", "Next detail", show=False, priority=True),
        Binding("space", "toggle_detail", "Toggle detail", priority=True),
        Binding("s", "system", "Instructions", priority=True),
        Binding("m", "memory", "Supporting analysis", show=False, priority=True),
        Binding("ctrl+c,q", "stop", "Quit", priority=True),
    ]

    def __init__(self, run: Callable[[EventSink], Awaitable[object]] | None = None, *,
                 selected_analysts: tuple[str, ...] = tuple(ANALYSTS), ticker: str = "", run_dir: Path | None = None) -> None:
        """Create stage views without starting native model work.

        Args:
            run: Workflow callback receiving the UI event sink.
            selected_analysts: Analyst keys shown in the initial grid.
            ticker: Instrument shown in the run header.
            run_dir: Saved run location used to load the completed report.
        """
        super().__init__()
        self.run_callback, self.selected_analysts, self.ticker, self.run_dir = run, selected_analysts, ticker, run_dir
        self.error: Exception | None = None
        self.result: object | None = None
        self.stage_index, self.selected_agent, self.fullscreen = 0, 0, False
        self.agents = {ANALYSTS.get(key, key): Activity() for key in selected_analysts}
        self.stages = {stage: Activity() for stage, _ in STAGES}
        self.usage: dict[tuple[str, str], Usage] = {}
        self.current_speakers: dict[str, str] = {}
        self.agent_views = {ANALYSTS.get(key, key): ChatView(id=f"analyst-{index}") for index, key in enumerate(selected_analysts)}
        self.stage_views = {stage: ChatView(debate=stage in DEBATE_STAGES, id=f"transcript-{stage}")
                            for stage, _ in STAGES[1:] if stage != "reports"}
        self.memory_view = ChatView(id="memory-transcript")
        self.report_document = Transcript()
        self.report_view = TranscriptView(self.report_document, id="transcript-reports")
        self._render_views = [*self.agent_views.values(), *self.stage_views.values(), self.memory_view, self.report_view]
        self._pending: deque[AgentEvent] = deque()
        self._event_ids: set[str] = set()
        self._started = time.monotonic()
        self._finished: float | None = None
        self._failure: str | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._report_task: asyncio.Task[None] | None = None
        self._stage_agents: dict[str, list[str]] = {stage: [] for stage, _ in STAGES}
        self._memory_agents: list[str] = []
        self._reports: dict[str, str] = {}
        self._tab_statuses: dict[str, str] = {}
        self._status_widgets: dict[str, Static] = {}

    def compose(self) -> ComposeResult:
        """Create stable viewports so updates do not rebuild the stage layout.

        Yields:
            Tabs, analyst panes, chronological chats, and counters.
        """
        yield Static(id="run-status", markup=False)
        yield Tabs(*(Tab(label, id=f"tab-{stage}") for stage, label in STAGES), id="stage-tabs")
        yield Static(id="stage-status", markup=False)
        with ContentSwitcher(initial="stage-analysis", id="stages"):
            layout = "one-agent" if len(self.selected_analysts) == 1 else "two-agents" if len(self.selected_analysts) == 2 else ""
            with Grid(id="stage-analysis", classes=f"stage {layout}"):
                for index, key in enumerate(self.selected_analysts):
                    name = ANALYSTS.get(key, key)
                    with Vertical(id=f"pane-{index}", classes="agent-pane selected" if index == 0 else "agent-pane"):
                        yield Static(name, id=f"agent-status-{index}", classes="agent-status", markup=False)
                        yield self.agent_views[name]
            for stage, _ in STAGES[1:]:
                with Vertical(id=f"stage-{stage}", classes="stage single-stage"):
                    yield self.report_view if stage == "reports" else self.stage_views[stage]
        with Collapsible(title="Supporting analysis", collapsed=True, id="memory-details", classes="hidden"):
            yield Static(id="memory-status", markup=False)
            yield self.memory_view
        yield Static(id="usage", markup=False)
        yield Static(id="shortcuts", markup=False)

    def on_mount(self) -> None:
        """Start independent rendering and clock timers."""
        self._status_widgets = {widget.id: widget for widget in self.query(Static) if widget.id}
        self._adapt_layout()
        self.set_interval(0.04, self.flush_events)
        self.set_interval(0.1, self.refresh_clocks)
        self._refresh_tabs()
        if self.agent_views:
            next(iter(self.agent_views.values())).focus()
        if self.run_callback:
            self._run_task = asyncio.create_task(self._execute())

    async def _execute(self) -> None:
        """Run or replay events without blocking terminal interaction."""
        try:
            assert self.run_callback is not None
            self.result = await self.run_callback(self.receive)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc
            if not any(event.kind == "error" and event.agent == "Workflow" for event in self._pending):
                await self.receive(AgentEvent("error", "Workflow", self.active_stage, str(exc)))
        finally:
            self.flush_events()
            if self._finished is None and not self._pending:
                self._finished = time.monotonic()

    async def receive(self, event: AgentEvent) -> None:
        """Queue one event without making generation wait for rendering.

        Args:
            event: A normalized workflow or harness event.
        """
        self._pending.append(event)

    @property
    def active_stage(self) -> str:
        """Return the stage selected for viewing."""
        return STAGES[self.stage_index][0]

    def flush_events(self) -> None:
        """Apply a bounded event batch and update only visible viewports."""
        for _ in range(min(len(self._pending), 1000)):
            event = self._pending.popleft()
            if event.event_id not in self._event_ids:
                self._event_ids.add(event.event_id)
                self._apply(event)
        for view in self._render_views:
            if view.is_on_screen:
                view.sync()

    def _apply(self, event: AgentEvent) -> None:
        """Update lifecycle counters and route content to its chronological chat.

        Args:
            event: A deduplicated event with immutable source content.
        """
        if event.kind == "system" and event.data.get("source") == "native_request":
            return
        now = time.monotonic() - max(0, time.time() - event.timestamp)
        self._started = min(self._started, now)
        kind = {"system_prompt": "system", "text_delta": "text", "thinking_delta": "reasoning", "output": "text"}.get(event.kind, event.kind)
        stage = event.stage if event.stage in self.stages else self.active_stage
        agent = event.agent or "Workflow"
        activity = self.agents.setdefault(agent, Activity())
        supporting = agent == "Memory" or agent.startswith(("Reflection ", "Sentiment Screening "))
        if agent != "Workflow" and not supporting and agent not in self._stage_agents[stage]:
            self._stage_agents[stage].append(agent)
        if supporting and agent not in self._memory_agents:
            self._memory_agents.append(agent)
            self.query_one("#memory-details", Collapsible).set_class(self.active_stage != "analysis", "hidden")
        activity.last_event = now
        if activity.started is None and agent != "Workflow" and kind in {"system", "prompt", "tool_start", "text", "reasoning"}:
            activity.started = now
        if kind == "stage_start":
            self._finished, self._failure = None, None
            self.stages[stage] = Activity(started=now, status="running")
        elif kind == "stage_end":
            self.stages[stage].finished, self.stages[stage].status = now, "complete"
        elif kind == "agent_start":
            if activity.finished is not None:
                activity.previous_seconds = activity.elapsed(now)
                activity.started = now
            elif activity.started is None:
                activity.started = now
            activity.status, activity.finished = "running", None
        elif kind == "agent_end":
            activity.finished, activity.status = now, "complete"
        elif kind == "conversation_output":
            if (activity.started is None or now >= activity.started) and (activity.finished is None or now >= activity.finished):
                activity.finished, activity.status = now, "complete"
        elif kind == "turn_start":
            activity.round, activity.turn, activity.total_rounds = (event.data.get(key) for key in ("round", "turn", "total_rounds"))
            activity.status = "running"
            self.current_speakers[stage] = agent
        elif kind == "usage":
            key, value = request_usage(event)
            self.usage[key] = value
        elif kind == "status":
            activity.status = event.text or str(event.data.get("status", "running"))
        elif kind == "tool_start":
            activity.status = "using tool"
        elif kind == "tool_result":
            activity.status = "using tool" if event.data.get("delta") or event.data.get("snapshot") else "running"
        elif kind == "error":
            activity.status, activity.finished = "error", now
            if agent == "Workflow":
                self._failure, self._finished = event.text, now
                for entry in self.agents.values():
                    if entry.started is not None and entry.finished is None:
                        entry.status, entry.finished = "stopped", now
                for entry in self.stages.values():
                    if entry.status == "running":
                        entry.status, entry.finished = "failed", now
        elif kind == "run_end":
            self._finished = now
            if self.run_dir and self._report_task is None:
                self._report_task = asyncio.create_task(self._load_completed_report())
        if kind in {"stage_start", "stage_end", "error", "run_end"}:
            self._refresh_tabs()
        if kind == "report" and self._reports.get(agent) != event.text:
            self._reports[agent] = event.text
            document = Transcript()
            for name, report in self._reports.items():
                document.heading(name)
                document.append(report + "\n", "#b5e3c8")
            self.report_document = document
            self.report_view.set_document(document)
        if supporting:
            self.memory_view.consume(event)
        elif stage == "analysis" and agent in self.agent_views:
            self.agent_views[agent].consume(event)
        elif stage in self.stage_views:
            self.stage_views[stage].consume(event)

    async def _load_completed_report(self) -> None:
        """Load the already-exported complete report into its stage without rewriting files."""
        assert self.run_dir is not None
        path = self.run_dir / "reports" / "complete_report.md"
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except OSError:
            return
        document = Transcript()
        document.append(text, "#b5e3c8")
        self.report_document = document
        self.report_view.set_document(document)

    def _refresh_tabs(self) -> None:
        """Change stage indicators only when lifecycle status changes."""
        if not self.is_mounted:
            return
        suffixes = {"running": " [run]", "complete": " [done]", "failed": " [failed]"}
        for stage, label in STAGES:
            status = self.stages[stage].status
            if self._tab_statuses.get(stage) == status:
                continue
            self._tab_statuses[stage] = status
            tab = self.query_one(f"#tab-{stage}", Tab)
            tab.label = Content(label + suffixes.get(status, ""))
            for category in ("running", "complete", "failed"):
                tab.set_class(status == category, category)

    def refresh_clocks(self) -> None:
        """Refresh clocks and counters without re-rendering historical chat content."""
        if not self._status_widgets or not self._status_widgets["run-status"].is_attached:
            return
        now = time.monotonic()
        elapsed = duration((self._finished if self._finished is not None else now) - self._started)
        status = "Failed" if self._failure else "Complete" if self._finished is not None else "Running"
        compact = self.screen.has_class("compact")
        separator = " · " if compact else "\n"
        self._status_widgets["run-status"].update(f"Open Trading Agents  {self.ticker}{separator}{status} · Total {elapsed}")
        stage = self.stages[self.active_stage]
        stage_text = f"{STAGES[self.stage_index][1]} · {stage.status} · {duration(stage.elapsed(now))}"
        participants = self._stage_agents[self.active_stage]
        if self.active_stage not in {"analysis", "reports"} and participants:
            name = self.current_speakers.get(self.active_stage, participants[-1])
            activity = self.agents[name]
            stage_text += f" · {name}"
            if activity.round:
                stage_text += f" · Round {activity.round}/{activity.total_rounds or '?'}"
            if not compact:
                values = [value for (agent, _), value in self.usage.items() if agent == name]
                stage_text += "\n" + usage_label(values)
        self._status_widgets["stage-status"].update(stage_text)
        for index, key in enumerate(self.selected_analysts):
            name = ANALYSTS.get(key, key)
            activity = self.agents[name]
            values = [value for (agent, _), value in self.usage.items() if agent == name]
            label = name.removesuffix(" Analyst") if compact else name
            self._status_widgets[f"agent-status-{index}"].update(
                f"{label} · {activity.status} · {duration(activity.elapsed(now))}\n{usage_label(values, compact=compact)}")
        if self._memory_agents:
            name = self._memory_agents[-1]
            activity = self.agents[name]
            values = [value for (agent, _), value in self.usage.items() if agent in self._memory_agents]
            self._status_widgets["memory-status"].update(
                f"{name} · {activity.status} · {duration(activity.elapsed(now))}\n{usage_label(values, compact=compact)}")
        self._status_widgets["usage"].update(usage_label(list(self.usage.values()), compact=compact))

    def on_resize(self, event: events.Resize) -> None:
        """Adapt status density to the terminal dimensions.

        Args:
            event: New terminal size.
        """
        self._adapt_layout(event.size)

    def _adapt_layout(self, size: Size | None = None) -> None:
        """Keep useful content space in smaller terminals.

        Args:
            size: Optional newly reported terminal dimensions.
        """
        size = size or self.size
        compact = size.height < 32 or size.width < 110
        self.screen.set_class(compact, "compact")
        if "shortcuts" in self._status_widgets:
            text = "Tab stage · f expand · [ ] details · Space toggle · s instructions · q quit"
            if not compact:
                text = "Tab/Shift+Tab stage · Arrows select/scroll · f expand · [ ] details · Space toggle · s instructions · End follow · q quit"
            self._status_widgets["shortcuts"].update(text)
        self.refresh_clocks()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Switch stages without disturbing another stage's scroll position.

        Args:
            event: Activated stage tab.
        """
        stage = (event.tab.id or "").removeprefix("tab-")
        if stage not in self.stages:
            return
        self.stage_index = [key for key, _ in STAGES].index(stage)
        self.query_one("#stages", ContentSwitcher).current = f"stage-{stage}"
        self.query_one("#memory-details", Collapsible).set_class(stage != "analysis" or not self._memory_agents, "hidden")
        self._active_view().focus()
        self.call_after_refresh(self.flush_events)
        self.refresh_clocks()

    def _active_view(self) -> ChatView | TranscriptView:
        """Find the selected or focused content viewport.

        Returns:
            The viewport receiving scrolling and disclosure controls.
        """
        if self.memory_view.has_focus and self.memory_view.is_on_screen:
            return self.memory_view
        if self.active_stage == "analysis" and self.selected_analysts:
            return self.agent_views[ANALYSTS.get(self.selected_analysts[self.selected_agent], self.selected_analysts[self.selected_agent])]
        if self.active_stage == "analysis":
            return self.report_view
        return self.report_view if self.active_stage == "reports" else self.stage_views[self.active_stage]

    def action_next_stage(self) -> None:
        """Select the next stage."""
        self._change_stage(1)

    def action_previous_stage(self) -> None:
        """Select the previous stage."""
        self._change_stage(-1)

    def _change_stage(self, offset: int) -> None:
        """Select a stage relative to the current view.

        Args:
            offset: Signed stage offset.
        """
        self.stage_index = (self.stage_index + offset) % len(STAGES)
        self.query_one("#stage-tabs", Tabs).active = f"tab-{self.active_stage}"

    def action_navigate(self, direction: str) -> None:
        """Select analyst panes or scroll the selected content.

        Args:
            direction: Keyboard movement direction.
        """
        if self.memory_view.has_focus and self.memory_view.is_on_screen:
            if direction not in {"left", "right"}:
                self.memory_view.navigate(direction)
            return
        if self.active_stage == "analysis" and not self.fullscreen and direction in {"left", "right", "up", "down"}:
            if self.selected_analysts:
                step = {"left": -1, "right": 1, "up": -2, "down": 2}[direction]
                self.selected_agent = (self.selected_agent + step) % len(self.selected_analysts)
                self._update_selection()
            return
        if direction not in {"left", "right"}:
            self._active_view().navigate(direction)

    def _update_selection(self) -> None:
        """Update analyst pane borders and fullscreen dimensions."""
        for index in range(len(self.selected_analysts)):
            pane = self.query_one(f"#pane-{index}", Vertical)
            pane.set_class(index == self.selected_agent, "selected")
            pane.set_class(self.fullscreen and index != self.selected_agent, "hidden")
        self.query_one("#stage-analysis", Grid).set_class(self.fullscreen, "full-grid")
        if self.selected_analysts:
            self._active_view().focus()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Select a pane when its chat receives mouse focus.

        Args:
            event: Focused descendant.
        """
        identifier = event.widget.id or ""
        if identifier.startswith("analyst-"):
            index = int(identifier.removeprefix("analyst-"))
            if index != self.selected_agent:
                self.selected_agent = index
                self._update_selection()

    def action_fullscreen(self) -> None:
        """Expand or restore the selected analyst pane."""
        if self.active_stage == "analysis":
            self.fullscreen = not self.fullscreen
            self._update_selection()

    def action_restore(self) -> None:
        """Restore the analyst grid."""
        if self.fullscreen:
            self.fullscreen = False
            self._update_selection()

    def action_detail(self, offset: int) -> None:
        """Select a collapsible section without changing stages.

        Args:
            offset: Previous or next section.
        """
        view = self._active_view()
        if isinstance(view, ChatView):
            view.select_detail(offset)

    def action_toggle_detail(self) -> None:
        """Expand or collapse the selected reasoning, tool, or input section."""
        view = self._active_view()
        if isinstance(view, ChatView):
            view.toggle_selected()

    def action_system(self) -> None:
        """Toggle system instructions in the active conversation."""
        view = self._active_view()
        if isinstance(view, ChatView):
            view.toggle_system()

    def action_memory(self) -> None:
        """Show optional memory and screening conversations."""
        if self.active_stage == "analysis" and self._memory_agents:
            panel = self.query_one("#memory-details", Collapsible)
            panel.collapsed = not panel.collapsed
            if not panel.collapsed:
                self.memory_view.focus()
            self.call_after_refresh(self.flush_events)

    async def action_stop(self) -> None:
        """Cancel active work and close the terminal interface."""
        await self._cancel_tasks()
        self.exit()

    async def _cancel_tasks(self) -> None:
        """Wait for active workflow cleanup without disturbing completed report files."""
        tasks = [task for task in (self._run_task, self._report_task) if task is not None and not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def on_unmount(self) -> None:
        """Clean up background tasks on external terminal shutdown."""
        await self._cancel_tasks()
