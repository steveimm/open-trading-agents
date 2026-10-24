from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from rich.control import Control
from rich.spinner import Spinner
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.content import Content
from textual.geometry import Size
from textual.widgets import Collapsible, ContentSwitcher, Static, Tab, Tabs

from open_trading_agents.analysis.models import ANALYSTS, STAGES, AgentEvent, EventSink, HarnessConfig, Usage
from open_trading_agents.analysis.prompts import ROLES, system_prompt
from open_trading_agents.reporting.usage import format_usage as usage_label
from open_trading_agents.reporting.usage import request_usage
from open_trading_agents.ui.chat import ChatView
from open_trading_agents.ui.transcript import Transcript, TranscriptView, literal_text

DEBATE_ROLES = {
    "research": ("bull", "bear"),
    "risk": ("aggressive", "conservative", "neutral"),
}
DEBATE_STAGES = set(DEBATE_ROLES)


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


def _model_selection(config: HarnessConfig) -> str:
    """Describe the harness, model, and effort selected for a role.

    Args:
        config: Effective runtime configuration for the role.

    Returns:
        One sanitized line of model settings.
    """
    values = (config.kind, config.model or "Harness default", config.effort or "Harness default")
    return " · ".join(" ".join(literal_text(value).splitlines()) for value in values)


def model_configuration(working: HarnessConfig | None, decision: HarnessConfig | None) -> str:
    """Describe the selected native runtimes without displaying commands or environment values.

    Args:
        working: Runtime used by analysts, debaters, and the trader.
        decision: Runtime used by decision managers.

    Returns:
        One line for shared choices, or separate working and manager lines.
    """
    choices = {}
    for label, config in (("Working", working), ("Manager", decision)):
        if config is None:
            continue
        choices[label] = _model_selection(config)
    if len(choices) == 2 and choices["Working"] == choices["Manager"]:
        return f"Agents · {choices['Working']}"
    return "\n".join(f"{label} · {value}" for label, value in choices.items())


class TradingApp(App[None]):
    TITLE = "Open Trading Agents"
    CSS = """
    Screen { background: #0d131c; color: #d9e2ef; overflow: hidden hidden; }
    #run-status { height: 1; padding: 0 1; background: #1a2635; }
    #model-config { height: auto; padding: 0 1; background: #1a2635; color: #a9bcd3; }
    #run-error { height: auto; max-height: 5; padding: 0 1; color: #f2858a; }
    #stage-tabs { height: 3; }
    #stage-tabs Tab.running { color: #e8b86d; text-style: bold; }
    #stage-tabs Tab.complete { color: #91d5ac; }
    #stage-tabs Tab.failed { color: #f2858a; }
    #stage-status { height: 2; padding: 0 1; color: #a9bcd3; }
    #stages, .stage { height: 1fr; min-height: 0; }
    #stage-analysis { grid-size: 2 2; grid-gutter: 1; padding: 0 1; }
    #stage-analysis.one-agent { grid-size: 1 1; }
    #stage-analysis.two-agents { grid-size: 2 1; }
    #stage-analysis.full-grid { grid-size: 1 1; }
    .agent-pane { border: round #34475e; height: 1fr; }
    .agent-pane.selected { border: round #bd9bfa; }
    .agent-status { height: 3; padding: 0 1; color: #9fb2ca; }
    .single-stage { border: round #34475e; margin: 0 1; }
    .prompt-column { width: 1fr; min-width: 18; height: 1fr; }
    .prompt-pane {
        border: round #34475e; border-title-color: #a9bcd3; border-title-style: bold;
        padding: 0 1; height: 1fr; min-height: 3;
    }
    .prompt-pane:focus { border: round #bd9bfa; }
    .debate-transcript { width: 3fr; height: 1fr; border: round #34475e; }
    .debate-stage.prompt-fullscreen .prompt-column { width: 1fr; }
    .hidden { display: none; }
    #memory-details { height: auto; max-height: 14; padding: 0; border: none; }
    #memory-transcript { height: 9; }
    #memory-status { height: 2; }
    #usage { height: 1; padding: 0 1; background: #1a2635; }
    #shortcuts { height: auto; max-height: 2; padding: 0 1; color: #8799af; }
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
        Binding("f", "fullscreen", "Toggle fullscreen", priority=True),
        Binding("left_square_bracket", "detail(-1)", "Prev", show=False, priority=True),
        Binding("right_square_bracket", "detail(1)", "Next", show=False, priority=True),
        Binding("space", "toggle_detail", "Expand/collapse section", priority=True),
        Binding("m", "memory", "Supporting analysis", show=False, priority=True),
        Binding("ctrl+c,q", "stop", "Quit", priority=True),
    ]

    def __init__(
        self,
        run: Callable[[EventSink], Awaitable[object]] | None = None,
        *,
        selected_analysts: tuple[str, ...] = tuple(ANALYSTS),
        ticker: str = "",
        run_dir: Path | None = None,
        working_config: HarnessConfig | None = None,
        decision_config: HarnessConfig | None = None,
        agent_configs: dict[str, HarnessConfig] | None = None,
        output_language: str = "English",
    ) -> None:
        """Create stage views without starting native model work.

        Args:
            run: Workflow callback receiving the UI event sink.
            selected_analysts: Analyst keys shown in the initial grid.
            ticker: Instrument shown in the run header.
            run_dir: Saved run location used to load the completed report.
            working_config: Selected working-agent harness, provider, model, and effort.
            decision_config: Selected decision-manager harness, provider, model, and effort.
            agent_configs: Effective per-role settings keyed by display name when individual assignments are used.
            output_language: Run language used to build the initial debate system prompts.
        """
        super().__init__()
        self.run_callback, self.selected_analysts, self.ticker, self.run_dir = run, selected_analysts, ticker, run_dir
        self._model_config_text = model_configuration(working_config, decision_config)
        self._agent_config_text = {name: _model_selection(config) for name, config in (agent_configs or {}).items()}
        self._displayed_model_config_text = ""
        self.error: Exception | None = None
        self.result: object | None = None
        self.stage_index, self.selected_agent, self.fullscreen = 0, 0, False
        self.agents = {ANALYSTS.get(key, key): Activity() for key in selected_analysts}
        self.stages = {stage: Activity() for stage, _ in STAGES}
        self.usage: dict[tuple[str, str], Usage] = {}
        self.current_speakers: dict[str, str] = {}
        self.agent_views = {ANALYSTS.get(key, key): ChatView(id=f"analyst-{index}") for index, key in enumerate(selected_analysts)}
        self.stage_views = {stage: ChatView(debate=stage in DEBATE_STAGES, show_systems=stage not in DEBATE_STAGES, id=f"transcript-{stage}")
                            for stage, _ in STAGES[1:] if stage != "reports"}
        self.prompt_views: dict[str, dict[str, TranscriptView]] = {}
        for stage, roles in DEBATE_ROLES.items():
            self.prompt_views[stage] = {}
            for key in roles:
                name = ROLES[key].name
                document = Transcript()
                document.append(system_prompt(ROLES[key], output_language), "#a9bcd3")
                view = TranscriptView(document, id=f"system-{stage}-{key}")
                view.add_class("prompt-pane")
                view.border_title = name
                view.following = False
                self.prompt_views[stage][name] = view
            self.stage_views[stage].add_class("debate-transcript")
        self._prompt_targets = {view: (stage, name) for stage, views in self.prompt_views.items() for name, view in views.items()}
        self._debate_focus = {stage: len(views) for stage, views in self.prompt_views.items()}
        self._debate_fullscreen: dict[str, ChatView | TranscriptView | None] = {stage: None for stage in DEBATE_STAGES}
        self.memory_view = ChatView(id="memory-transcript")
        self.report_document = Transcript()
        self.report_view = TranscriptView(self.report_document, id="transcript-reports")
        self._render_views = [*self.agent_views.values(), *self.stage_views.values(), *self._prompt_targets, self.memory_view, self.report_view]
        self._pending: deque[AgentEvent] = deque()
        self._event_ids: set[str] = set()
        self._started = time.monotonic()
        self._wall_offset = self._started - time.time()
        self._finished: float | None = None
        self._failure: str | None = None
        self._run_cancelled = False
        self._run_task: asyncio.Task[None] | None = None
        self._report_task: asyncio.Task[None] | None = None
        self._stage_agents: dict[str, list[str]] = {stage: [] for stage, _ in STAGES}
        self._memory_agents: list[str] = []
        self._reports: dict[str, str] = {}
        self._tab_statuses: dict[str, tuple[str, str]] = {}
        self._running_spinner = Spinner("dots")
        self._terminal_title_started = False
        self._last_terminal_title: str | None = None
        self._status_widgets: dict[str, Static] = {}
        self._shortcuts_text = ""

    def compose(self) -> ComposeResult:
        """Create stable viewports so updates do not rebuild the stage layout.

        Yields:
            Tabs, analyst panes, chronological chats, and counters.
        """
        yield Static(id="run-status", markup=False)
        self._displayed_model_config_text = self._current_model_configuration()
        configuration = Static(self._displayed_model_config_text, id="model-config",
                               classes="" if self._displayed_model_config_text else "hidden", markup=False)
        assignments = "\n".join(f"{name} · {value}" for name, value in self._agent_config_text.items())
        configuration.tooltip = Content(assignments or self._model_config_text)
        yield configuration
        yield Static(id="run-error", classes="hidden", markup=False)
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
                if stage in DEBATE_STAGES:
                    with Horizontal(id=f"stage-{stage}", classes="stage debate-stage"):
                        with Vertical(id=f"prompts-{stage}", classes="prompt-column"):
                            yield from self.prompt_views[stage].values()
                        yield self.stage_views[stage]
                else:
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
            self._run_cancelled = True
            raise
        except Exception as exc:
            self.error = exc
            if not any(event.kind == "error" and event.agent == "Workflow" for event in self._pending):
                await self.receive(AgentEvent("error", "Workflow", self.active_stage, str(exc)))
        finally:
            self.flush_events()
            if self._finished is None and not self._pending:
                self._finished = time.monotonic()
            self._refresh_terminal_title(self._running_spinner.render(time.monotonic()).plain)

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
        if not self.is_running:
            return

        lifecycle_changed = False
        for _ in range(min(len(self._pending), 1000)):
            event = self._pending.popleft()
            if event.event_id not in self._event_ids:
                self._event_ids.add(event.event_id)
                self._apply(event)
                lifecycle_changed = lifecycle_changed or event.kind in {
                    "stage_start", "stage_end", "agent_start", "agent_end", "turn_start", "error", "run_end",
                }

        for view in self._render_views:
            if view.is_on_screen:
                view.sync()

        if lifecycle_changed:
            self.refresh_clocks()
        else:
            self._refresh_shortcuts()

    def _apply(self, event: AgentEvent) -> None:
        """Update lifecycle counters and route content to its chronological chat.

        Args:
            event: A deduplicated event with immutable source content.
        """
        if event.kind in {"system", "system_prompt"} and event.data.get("source") == "native_request":
            return
        now = min(time.monotonic(), event.timestamp + self._wall_offset)
        self._started = min(self._started, now)
        kind = {"system_prompt": "system", "text_delta": "text", "thinking_delta": "reasoning", "output": "text"}.get(event.kind, event.kind)
        stage = event.stage if event.stage in self.stages else self.active_stage
        agent = event.agent or "Workflow"
        if kind == "system" and agent in self.prompt_views.get(stage, {}):
            view = self.prompt_views[stage][agent]
            if view.document.plain() != event.text:
                document = Transcript()
                document.append(event.text, "#a9bcd3")
                view.set_document(document)
                view.navigate("home")
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
            self.query_one("#run-error", Static).add_class("hidden")
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
                banner = self.query_one("#run-error", Static)
                banner.update(literal_text(event.text))
                banner.remove_class("hidden")
                for entry in self.agents.values():
                    if entry.started is not None and entry.finished is None:
                        entry.status, entry.finished = "stopped", now
                for entry in self.stages.values():
                    if entry.status == "running":
                        entry.status, entry.finished = "failed", now
                for view in self._render_views:
                    if isinstance(view, ChatView):
                        view.finish("stopped")
        elif kind == "run_end":
            self._finished = now
            for view in self._render_views:
                if isinstance(view, ChatView):
                    view.finish("complete")
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

    def _refresh_tabs(self, now: float | None = None) -> None:
        """Animate running indicators while keeping inactive tab labels unchanged.

        Args:
            now: Optional monotonic time supplied by the display timer.
        """
        if not self.is_running:
            return
        frame = self._running_spinner.render(time.monotonic() if now is None else now).plain
        suffixes = {"running": f" {frame}", "failed": " [failed]"}
        for stage, label in STAGES:
            status = self.stages[stage].status
            indicator = suffixes.get(status, "")
            previous = self._tab_statuses.get(stage)
            if previous == (status, indicator):
                continue
            self._tab_statuses[stage] = (status, indicator)
            tab = self.query_one(f"#tab-{stage}", Tab)
            content = Content(label + indicator)
            if previous is None or previous[0] != status:
                tab.label = content
                for category in ("running", "complete", "failed"):
                    tab.set_class(status == category, category)
            else:
                # Spinner frames have equal width, so they need no tab relayout.
                Static.update(tab, content, layout=False)

    def _refresh_terminal_title(self, frame: str) -> None:
        """Update the terminal tab title using the current run state and spinner frame.

        Args:
            frame: Frame already rendered by the shared activity spinner.
        """
        driver = self._driver
        if not self.is_running or driver is None or driver.is_headless or driver.is_web:
            return

        name = f"{self.TITLE} · {self.ticker}" if self.ticker else self.TITLE
        if self._run_cancelled:
            title = f"Stopped · {name}"
        elif self._failure:
            title = f"Failed · {name}"
        elif self._finished is not None:
            title = name
        elif self.run_callback is not None or any(stage.status == "running" for stage in self.stages.values()):
            title = f"{frame} {name}"
        else:
            title = name

        title = "".join(character for character in title if character.isprintable())
        if title == self._last_terminal_title:
            return

        if not self._terminal_title_started:
            driver.write("\x1b[22;0t")
            self._terminal_title_started = True
        driver.write(str(Control.title(title)))
        driver.flush()
        self._last_terminal_title = title

    def _restore_terminal_title(self) -> None:
        """Clear activity and restore the previous title on terminals with title-stack support."""
        if not self._terminal_title_started:
            return

        self._terminal_title_started = False
        self._last_terminal_title = None
        if self._driver is not None:
            self._driver.write(str(Control.title("")) + "\x1b[23;0t")
            self._driver.flush()

    def _current_model_configuration(self) -> str:
        """Describe the selected role while retaining grouped settings for ordinary runs.

        Returns:
            The current role's configuration, or the shared working/manager description.
        """
        if not self._agent_config_text:
            return self._model_config_text

        name: str | None = None
        if self.memory_view.has_focus and self.memory_view.is_on_screen and self._memory_agents:
            latest = self._memory_agents[-1]
            name = next((ROLES[key].name for key in ("reflection", "screening")
                         if latest == ROLES[key].name or latest.startswith(ROLES[key].name + " ")), None)
        if name is None and self.active_stage == "analysis" and self.selected_analysts:
            key = self.selected_analysts[self.selected_agent]
            name = ANALYSTS.get(key, key)
        elif name is None and self.active_stage == "reports":
            name = ROLES["portfolio_manager"].name
        elif name is None:
            view = self._active_view()
            if view in self._prompt_targets:
                name = self._prompt_targets[view][1]
            else:
                participants = self._stage_agents[self.active_stage]
                name = self.current_speakers.get(self.active_stage) or (participants[-1] if participants else None)
                if name is None:
                    name = next((role.name for role in ROLES.values() if role.stage == self.active_stage), None)

        description = self._agent_config_text.get(name) if name is not None else None
        return f"{name} · {description}" if description is not None else self._model_config_text

    def _refresh_model_configuration(self) -> None:
        """Update the role header only when selection or active speaker changes."""
        if not self._agent_config_text or "model-config" not in self._status_widgets:
            return

        text = self._current_model_configuration()
        if text == self._displayed_model_config_text:
            return

        self._displayed_model_config_text = text
        widget = self._status_widgets["model-config"]
        widget.set_class(not text, "hidden")
        widget.update(text)

    def refresh_clocks(self) -> None:
        """Refresh clocks and counters without re-rendering historical chat content."""
        if not self.is_running or not self._status_widgets or not self._status_widgets["run-status"].is_attached:
            return
        now = time.monotonic()
        self._refresh_tabs(now)
        frame = self._running_spinner.render(now).plain
        self._refresh_terminal_title(frame)
        self._refresh_model_configuration()
        status_suffixes = {"running": f" {frame}", "complete": "", "using tool": f" {frame} tool"}
        for view in self._render_views:
            if isinstance(view, ChatView) and view.is_on_screen:
                view.refresh_activity(now)
        elapsed = duration((self._finished if self._finished is not None else now) - self._started)
        header = ["Open Trading Agents"]
        if self.ticker:
            header.append(self.ticker)
        if self._failure:
            header.append("Failed")
        elif self._finished is not None:
            header.append("Complete")
        header.append(f"Total {elapsed}")
        compact = self.screen.has_class("compact")
        self._status_widgets["run-status"].update(" · ".join(header))
        stage = self.stages[self.active_stage]
        stage_suffix = status_suffixes.get(stage.status, f" · {stage.status}")
        stage_text = f"{STAGES[self.stage_index][1]}{stage_suffix} · {duration(stage.elapsed(now))}"
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
                f"{label} · {duration(activity.elapsed(now))}\n{usage_label(values, compact=compact)}")
        if self._memory_agents:
            name = self._memory_agents[-1]
            activity = self.agents[name]
            values = [value for (agent, _), value in self.usage.items() if agent in self._memory_agents]
            suffix = status_suffixes.get(activity.status, f" · {activity.status}")
            self._status_widgets["memory-status"].update(
                f"{name}{suffix} · {duration(activity.elapsed(now))}\n{usage_label(values, compact=compact)}")
        self._status_widgets["usage"].update(usage_label(list(self.usage.values()), compact=compact))
        self._refresh_shortcuts()

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
        if not self.is_running:
            return
        size = size or self.size
        compact = size.height < 32 or size.width < 110
        self.screen.set_class(compact, "compact")
        self.refresh_clocks()

    def _grid_navigation(self) -> bool:
        """Return whether arrow keys select between analyst panes."""
        return (self.active_stage == "analysis" and len(self.selected_analysts) > 1 and not self.fullscreen
                and self._active_view() is not self.memory_view)

    def _action_available(self, action: str, parameters: tuple[object, ...] = ()) -> bool:
        """Apply the same context rules to key handling and visible shortcut hints.

        Args:
            action: Application action name.
            parameters: Optional action arguments.

        Returns:
            Whether the action applies to the current stage and focused content.
        """
        if not self.is_running:
            return False
        view = self._active_view()
        analyst = self.active_stage == "analysis" and len(self.selected_analysts) > 1 and view is not self.memory_view
        if action == "fullscreen":
            return analyst or self.active_stage in DEBATE_STAGES and view in self._debate_views(self.active_stage)
        if action in {"detail", "toggle_detail"}:
            return isinstance(view, ChatView) and view.has_details
        if action == "memory":
            return self.active_stage == "analysis" and bool(self._memory_agents)
        if action == "navigate":
            direction = parameters[0] if parameters else ""
            if direction in {"left", "right"}:
                return self._grid_navigation() or (self.active_stage in DEBATE_STAGES and self._debate_fullscreen[self.active_stage] is None)
            return view.is_on_screen
        return True

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Disable controls that do not apply to the selected view.

        Args:
            action: Requested action.
            parameters: Action arguments parsed from the key binding.

        Returns:
            The current availability, or the framework's behavior for other actions.
        """
        if action in {"fullscreen", "detail", "toggle_detail", "memory", "navigate"}:
            return self._action_available(action, parameters)
        return super().check_action(action, parameters)

    def _refresh_shortcuts(self) -> None:
        """Show only useful controls, with section navigation separate from toggling."""
        if not self.is_running or "shortcuts" not in self._status_widgets:
            return
        view = self._active_view()
        compact = self.screen.has_class("compact")
        parts = ["Tab stages" if compact else "Tab/Shift+Tab stages"]
        if self._grid_navigation():
            parts.append("Arrows pane" if compact else "Arrows select pane")
        else:
            if self._action_available("navigate", ("left",)):
                parts.append("Left/Right pane")
            if view.max_scroll_y > 0:
                parts.append("Up/Down scroll")
        if self._action_available("fullscreen"):
            parts.append("f toggle fullscreen")
        if self._action_available("detail"):
            parts.extend(["[ prev · ] next",
                          "Space toggle" if compact else "Space expand/collapse"])
        if not view.following and view not in self._prompt_targets:
            parts.append("End follow")
        if self._action_available("memory"):
            parts.append("m support")
        parts.append("q quit")
        text = " · ".join(parts)
        if text != self._shortcuts_text:
            self._shortcuts_text = text
            self._status_widgets["shortcuts"].update(text)
            self.refresh_bindings()

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
        if self.active_stage in DEBATE_STAGES:
            return self._debate_fullscreen[self.active_stage] or self._debate_views(self.active_stage)[self._debate_focus[self.active_stage]]
        return self.report_view if self.active_stage == "reports" else self.stage_views[self.active_stage]

    def _debate_views(self, stage: str) -> tuple[ChatView | TranscriptView, ...]:
        """List prompt panes followed by the shared conversation for keyboard focus.

        Args:
            stage: Research or risk debate stage.

        Returns:
            Stable panes in left-to-right focus order.
        """
        return (*self.prompt_views[stage].values(), self.stage_views[stage])

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
        if not self._action_available("navigate", (direction,)):
            return
        if self.memory_view.has_focus and self.memory_view.is_on_screen:
            if direction not in {"left", "right"}:
                self.memory_view.navigate(direction)
            self._refresh_shortcuts()
            return
        if self._grid_navigation() and direction in {"left", "right", "up", "down"}:
            if self.selected_analysts:
                step = {"left": -1, "right": 1, "up": -2, "down": 2}[direction]
                self.selected_agent = (self.selected_agent + step) % len(self.selected_analysts)
                self._update_selection()
            return
        if self.active_stage in DEBATE_STAGES and direction in {"left", "right"}:
            views = self._debate_views(self.active_stage)
            offset = -1 if direction == "left" else 1
            self._debate_focus[self.active_stage] = (self._debate_focus[self.active_stage] + offset) % len(views)
            self._active_view().focus()
            self._refresh_shortcuts()
            return
        if direction not in {"left", "right"}:
            self._active_view().navigate(direction)
        self._refresh_shortcuts()

    def _update_selection(self) -> None:
        """Update analyst pane borders and fullscreen dimensions."""
        for index in range(len(self.selected_analysts)):
            pane = self.query_one(f"#pane-{index}", Vertical)
            pane.set_class(index == self.selected_agent, "selected")
            pane.set_class(self.fullscreen and index != self.selected_agent, "hidden")
        self.query_one("#stage-analysis", Grid).set_class(self.fullscreen, "full-grid")
        if self.selected_analysts:
            self._active_view().focus()
        self._refresh_model_configuration()
        self.call_after_refresh(self._refresh_shortcuts)

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
        for stage in DEBATE_STAGES:
            views = self._debate_views(stage)
            if event.widget in views:
                self._debate_focus[stage] = views.index(event.widget)
                break
        self.call_after_refresh(self._refresh_model_configuration)
        self.call_after_refresh(self._refresh_shortcuts)

    def action_fullscreen(self) -> None:
        """Toggle fullscreen for the selected analyst or debate pane."""
        if not self._action_available("fullscreen"):
            return
        view = self._active_view()
        if self.active_stage in DEBATE_STAGES:
            stage = self.active_stage
            selected = None if self._debate_fullscreen[stage] is view else view
            self._debate_fullscreen[stage] = selected
            chat = self.stage_views[stage]
            self.query_one(f"#stage-{stage}", Horizontal).set_class(selected in self._prompt_targets, "prompt-fullscreen")
            self.query_one(f"#prompts-{stage}", Vertical).set_class(selected is chat, "hidden")
            for prompt in self.prompt_views[stage].values():
                prompt.set_class(selected is not None and prompt is not selected, "hidden")
            chat.set_class(selected is not None and selected is not chat, "hidden")
            view.focus()
            self.call_after_refresh(self.flush_events)
            return
        self.fullscreen = not self.fullscreen
        self._update_selection()

    def action_detail(self, offset: int) -> None:
        """Select a collapsible section without changing stages.

        Args:
            offset: Previous or next section.
        """
        view = self._active_view()
        if self._action_available("detail") and isinstance(view, ChatView):
            view.select_detail(offset)
            self._refresh_shortcuts()

    def action_toggle_detail(self) -> None:
        """Expand or collapse the selected reasoning, tool, or input section."""
        view = self._active_view()
        if self._action_available("toggle_detail") and isinstance(view, ChatView):
            view.toggle_selected()
            self._refresh_shortcuts()

    def action_memory(self) -> None:
        """Show optional memory and screening conversations."""
        if self._action_available("memory"):
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
            if not task.cancelling():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def on_unmount(self) -> None:
        """Clean up background tasks on external terminal shutdown."""
        try:
            await self._cancel_tasks()
        finally:
            self._restore_terminal_title()
