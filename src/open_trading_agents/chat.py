from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, field

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.geometry import Size
from textual.strip import Strip

from open_trading_agents.domain import AgentEvent
from open_trading_agents.transcript import (
    CHANNEL_COLORS,
    FollowScrollView,
    Transcript,
    TranscriptIndex,
    literal_text,
    tool_preview,
    tool_text,
)


@dataclass(eq=False)
class ChatSection:
    key: str
    kind: str
    title: str = ""
    collapsed: bool = False
    collapsible: bool = False
    source: str = ""
    document: Transcript = field(default_factory=Transcript)
    index: TranscriptIndex = field(init=False)

    def __post_init__(self) -> None:
        self.index = TranscriptIndex(self.document)

    def replace(self, text: str, display: str | None = None) -> None:
        """Update a section without rebuilding earlier conversation content.

        Args:
            text: Complete original section text.
            display: Optional bounded representation used only for tool results.
        """
        shown = text if display is None else display
        previous = self.document.plain()
        normalized = literal_text(shown)
        self.source = text
        if normalized == previous:
            return
        style = CHANNEL_COLORS.get(self.kind, "#b7c2d4")
        if normalized.startswith(previous):
            self.document.append(shown[len(previous):] if shown == normalized else normalized[len(previous):], style)
        else:
            self.document = Transcript()
            self.document.append(shown, style)
            self.index = TranscriptIndex(self.document)

    def append(self, text: str) -> None:
        """Append a streamed fragment to this section alone.

        Args:
            text: New model text or exposed reasoning.
        """
        self.source += text
        self.document.append(text, CHANNEL_COLORS.get(self.kind, "#b7c2d4"))


@dataclass(eq=False)
class ChatBubble:
    key: str
    agent: str
    kind: str = "agent"
    title: str = ""
    status: str = "streaming"
    sections: list[ChatSection] = field(default_factory=list)
    round_number: int | None = None
    turn_number: int | None = None


@dataclass(frozen=True)
class ChatPart:
    key: str
    kind: str
    height: int
    bubble: ChatBubble | None = None
    section: ChatSection | None = None


class ChatView(FollowScrollView):
    DEFAULT_CSS = """
    ChatView { height: 1fr; width: 1fr; scrollbar-size: 1 1; background: #101720; }
    ChatView:focus { background: #121c28; }
    """

    def __init__(self, *, id: str | None = None, debate: bool = False) -> None:
        """Create an incremental conversation viewport with collapsible details.

        Args:
            id: Stable Textual widget identifier.
            debate: Whether repeated participant inputs should begin collapsed.
        """
        super().__init__(id=id)
        self.debate = debate
        self.document = self
        self.bubbles: list[ChatBubble] = []
        self.systems: OrderedDict[str, ChatSection] = OrderedDict()
        self._turns: dict[str, ChatBubble] = {}
        self._current: dict[str, ChatBubble] = {}
        self._messages: dict[tuple[str, str, str], ChatSection] = {}
        self._tools: dict[tuple[str, str], ChatSection] = {}
        self._arguments: dict[tuple[str, str], str] = {}
        self._outputs: dict[tuple[str, str], str] = {}
        self._argument_deltas: set[tuple[str, str]] = set()
        self._seen: set[str] = set()
        self._parts: list[ChatPart] = []
        self._offsets = [0]
        self._detail_rows: dict[ChatSection, int] = {}
        self._cache: OrderedDict[tuple[TranscriptIndex, tuple[int, int]], Strip] = OrderedDict()
        self._selected: ChatSection | None = None
        self._shared_input: ChatBubble | None = None
        self._dirty = True
        self._width = 0

    @staticmethod
    def _turn_key(event: AgentEvent) -> str | None:
        """Read the stable workflow turn identity without confusing native turn IDs.

        Args:
            event: Normalized workflow or harness event.

        Returns:
            The workflow identity when supplied.
        """
        value = event.data.get("conversation_turn_id")
        return str(value) if value is not None else None

    def _bubble(self, event: AgentEvent, new: bool = False) -> ChatBubble:
        """Find or create the current contribution for a participant.

        Args:
            event: Incoming event carrying role and optional turn identity.
            new: Whether an unkeyed completed contribution should start another bubble.

        Returns:
            The contribution receiving this event.
        """
        turn = self._turn_key(event)
        if turn and turn in self._turns:
            bubble = self._turns[turn]
        else:
            legacy = next((bubble for bubble in self.bubbles if bubble.agent == event.agent and bubble.kind == "agent"
                           and event.data.get("round") is not None and bubble.round_number == event.data["round"]
                           and bubble.turn_number == event.data.get("turn")), None)
            bubble = legacy or self._current.get(event.agent)
            if bubble is None or (new and bubble.status != "streaming") or (turn and bubble.key in self._turns and bubble.key != turn):
                bubble = ChatBubble(turn or event.event_id, event.agent)
                self.bubbles.append(bubble)
            if turn:
                self._turns[turn] = bubble
                bubble.key = turn
        self._current[event.agent] = bubble
        return bubble

    def consume(self, event: AgentEvent) -> None:
        """Apply one event without rebuilding or rendering the conversation history.

        Args:
            event: Original event retained independently by the workflow recorder.
        """
        if event.event_id in self._seen:
            return
        self._seen.add(event.event_id)
        kind = {"system_prompt": "system", "text_delta": "text", "thinking_delta": "reasoning", "output": "text"}.get(event.kind, event.kind)
        if kind == "system" and event.data.get("source") == "native_request":
            return
        if kind == "system":
            section = self.systems.setdefault(event.agent, ChatSection(f"system:{event.agent}", "system", f"System · {event.agent}", True, True))
            section.replace(event.text)
        elif kind == "conversation_input":
            self._shared(event.text)
        elif kind in {"turn_start", "agent_start"}:
            bubble = self._bubble(event, new=True)
            bubble.status = "streaming"
            if kind == "turn_start":
                number = event.data.get("round", 1)
                total = event.data.get("total_rounds", "?")
                bubble.round_number, bubble.turn_number = number, event.data.get("turn", 1)
                bubble.title = f"Round {number}/{total} · Turn {event.data.get('turn', 1)}"
        elif kind == "prompt":
            if self.debate and self._shared_input is None:
                self._shared(event.text)
            bubble = self._bubble(event, new=True)
            section = ChatSection(f"prompt:{event.event_id}", "prompt", "Turn input" if self.debate else "Input", self.debate, True)
            section.replace(event.text)
            if self.debate or bubble.sections:
                bubble.sections.append(section)
            else:
                input_bubble = ChatBubble(f"input:{event.event_id}", "Input", "input", status="complete", sections=[section])
                self.bubbles.insert(self.bubbles.index(bubble), input_bubble)
        elif kind in {"text", "reasoning"}:
            if not event.text:
                return
            bubble = self._bubble(event)
            message = str(event.data.get("message_id") or bubble.key)
            key = event.agent, message, kind
            section = self._messages.get(key)
            if section is None:
                section = ChatSection(f"{kind}:{event.agent}:{message}", kind, "Reasoning" if kind == "reasoning" else "",
                                      kind == "reasoning", kind == "reasoning")
                self._messages[key] = section
                bubble.sections.append(section)
            if event.data.get("final"):
                section.replace(event.text)
            else:
                section.append(event.text)
        elif kind in {"tool_start", "tool_result"}:
            self._tool(event, kind)
        elif kind == "agent_end":
            bubble = self._bubble(event)
            bubble.status = "complete"
            output = str(event.data.get("output", ""))
            self._complete(bubble, output)
        elif kind == "conversation_output":
            bubble = self._bubble(event)
            bubble.status = "complete"
            bubble.round_number, bubble.turn_number = event.data.get("round", 1), event.data.get("turn", 1)
            bubble.title = f"Round {event.data.get('round', 1)}/{event.data.get('total_rounds', '?')} · Turn {event.data.get('turn', 1)}"
            self._complete(bubble, event.text)
        elif kind == "report":
            if any(b.kind == "report" and b.agent == event.agent and b.sections[0].source == event.text for b in self.bubbles):
                return
            section = ChatSection(f"report:{event.event_id}", "report")
            section.replace(event.text)
            self.bubbles.append(ChatBubble(event.event_id, event.agent, "report", "Report", "complete", [section]))
        elif kind == "error":
            bubble = self._bubble(event)
            bubble.status = "error"
            section = ChatSection(f"error:{event.event_id}", "error")
            section.replace(event.text)
            bubble.sections.append(section)
        else:
            return
        self._dirty = True

    def _shared(self, text: str) -> None:
        """Insert or replace shared evidence once, including legacy saved runs.

        Args:
            text: Authoritative evidence or the first legacy participant input.
        """
        if self._shared_input is None:
            self._shared_input = ChatBubble("shared-input", "Input", "input", "Analysis results", "complete")
            self._shared_input.sections.append(ChatSection("shared-evidence", "prompt", "Analysis results", False, True))
            self.bubbles.insert(0, self._shared_input)
        self._shared_input.sections[0].replace(text)

    def _complete(self, bubble: ChatBubble, output: str) -> None:
        """Reconcile a durable final argument with the existing streamed contribution.

        Args:
            bubble: Existing or recovered participant contribution.
            output: Authoritative completed output.
        """
        if not output:
            return
        if bubble.sections and bubble.sections[-1].kind == "text":
            bubble.sections[-1].replace(output)
        else:
            section = ChatSection(f"final:{bubble.key}", "text")
            section.replace(output)
            bubble.sections.append(section)

    def _tool(self, event: AgentEvent, kind: str) -> None:
        """Group arguments and results for one tool call into one expandable section.

        Args:
            event: Native tool event.
            kind: Normalized tool_start or tool_result category.
        """
        key = event.agent, str(event.data.get("tool_call_id") or event.event_id)
        section = self._tools.get(key)
        name = str(event.data.get("tool") or event.data.get("name") or event.data.get("tool_name") or "Tool")
        if section is None:
            section = ChatSection(f"tool:{key[0]}:{key[1]}", "tool_result", name, True, True)
            self._tools[key] = section
            self._bubble(event).sections.append(section)
        elif name != "Tool":
            section.title = name
        value = tool_text(event)
        if kind == "tool_start":
            if event.data.get("delta"):
                previous = self._arguments.get(key, "") if key in self._argument_deltas else ""
                self._arguments[key] = previous + value
                self._argument_deltas.add(key)
            else:
                self._arguments[key] = value
        else:
            self._outputs[key] = self._outputs.get(key, "") + value if event.data.get("delta") else value
            if event.data.get("is_error"):
                section.kind = "error"
        arguments = self._arguments.get(key, "")
        result = self._outputs.get(key, "")
        source = f"Arguments\n{arguments}" if arguments else ""
        shown = source
        if result:
            prefix = "\n\n" if source else ""
            source += f"{prefix}Result\n{result}"
            shown += f"{prefix}Result\n{tool_preview(result)}"
        section.replace(source, shown)

    def _anchor(self) -> tuple[str, int, int] | None:
        """Capture a logical location before inserting or expanding earlier content.

        Returns:
            Part identity and logical source location, when content exists.
        """
        index = bisect.bisect_right(self._offsets, int(self.scroll_y)) - 1
        if index < 0 or index >= len(self._parts):
            return None
        part = self._parts[index]
        if part.kind == "body" and part.section is not None:
            return (part.key, *part.section.index.source_at(int(self.scroll_y) - self._offsets[index]))
        return part.key, 0, 0

    def sync(self) -> None:
        """Index changed sections and repaint visible rows while retaining scroll anchors."""
        if not self.is_on_screen:
            return
        width = max(1, self.scrollable_content_region.width)
        if not self._dirty and width == self._width:
            return
        anchor = self._anchor()
        parts: list[ChatPart] = []
        self._detail_rows = {}

        def add_section(section: ChatSection, bubble: ChatBubble | None, header: bool = True) -> None:
            """Index one visible section without touching its collapsed body.

            Args:
                section: Conversation detail or output section.
                bubble: Optional containing role bubble.
                header: Whether the section needs its own header below the bubble title.
            """
            if section.collapsible and header:
                parts.append(ChatPart(f"header:{section.key}", "detail", 1, bubble, section))
            if not section.collapsed and section.source:
                section.index.sync(max(1, width - 4))
                parts.append(ChatPart(f"body:{section.key}", "body", section.index.height, bubble, section))

        for section in self.systems.values():
            add_section(section, None)
        if self.systems:
            parts.append(ChatPart("system-gap", "gap", 1))
        for bubble in self.bubbles:
            input_section = bubble.sections[0] if bubble.kind == "input" and bubble.sections else None
            parts.append(ChatPart(f"top:{bubble.key}", "top", 1, bubble, input_section))
            for section in bubble.sections:
                add_section(section, bubble, header=bubble.kind != "input")
            parts.append(ChatPart(f"bottom:{bubble.key}", "bottom", 1, bubble))
            parts.append(ChatPart(f"gap:{bubble.key}", "gap", 1, bubble))
        self._parts = parts
        visible_indexes = {part.section.index for part in parts if part.kind == "body" and part.section is not None}
        for key in list(self._cache):
            if key[0] not in visible_indexes:
                del self._cache[key]
                key[0].cache.pop(key[1], None)
        self._offsets = [0]
        lookup = {}
        for part in parts:
            lookup[part.key] = self._offsets[-1]
            if part.kind in {"detail", "top"} and part.section is not None:
                self._detail_rows[part.section] = self._offsets[-1]
            self._offsets.append(self._offsets[-1] + part.height)
        self._width, self._dirty = width, False
        self.virtual_size = Size(width, self._offsets[-1])
        if self.following:
            self.scroll_end(animate=False, immediate=True)
        elif anchor and anchor[0] in lookup:
            part = next(p for p in parts if p.key == anchor[0])
            offset = part.section.index.row_at(anchor[1], anchor[2]) if part.kind == "body" and part.section else 0
            self.scroll_to(y=lookup[anchor[0]] + offset, animate=False, immediate=True)
        self.refresh()

    def _line(self, text: str, style: Style | str, width: int) -> Strip:
        """Render a bounded header line without interpreting rich markup.

        Args:
            text: Literal header.
            style: Header color and emphasis.
            width: Available terminal cells.

        Returns:
            A terminal-width header strip.
        """
        value = Text(literal_text(text), style=style)
        value.truncate(width)
        return Strip(value.render(self.app.console)).apply_style(self.app.console.get_style(value.style)).adjust_cell_length(width, self.rich_style).apply_style(self.rich_style)

    def render_line(self, y: int) -> Strip:
        """Render one visible bubble row using bounded section caches.

        Args:
            y: Row within this viewport.

        Returns:
            A styled row with input/agent borders or a detail header.
        """
        width = max(1, self.scrollable_content_region.width)
        row = int(self.scroll_y) + y
        index = bisect.bisect_right(self._offsets, row) - 1
        if index < 0 or index >= len(self._parts):
            return Strip.blank(width, self.rich_style)
        part = self._parts[index]
        bubble, section = part.bubble, part.section
        color = "#e8b86d" if bubble and bubble.kind == "input" else "#668ab5"
        if part.kind == "top" and bubble is not None:
            suffix = f" · {bubble.title}" if bubble.title else ""
            active = " · streaming" if bubble.status == "streaming" else " · error" if bubble.status == "error" else ""
            fold = f"{'▸' if section.collapsed else '▾'} " if section is not None else ""
            style = "bold #f1e5b9 on #303547" if section is not None and section is self._selected else f"bold {color}"
            title = Text(f"╭ {fold}{bubble.agent}{suffix}{active} ", style=style)
            title.truncate(max(1, width - 1))
            title.append("─" * max(0, width - title.cell_len - 1) + "╮")
            return Strip(title.render(self.app.console)).apply_style(self.app.console.get_style(title.style)).adjust_cell_length(width, self.rich_style).apply_style(self.rich_style)
        if part.kind == "bottom":
            return self._line("╰" + "─" * max(0, width - 1), color, width)
        if part.kind == "detail" and section is not None:
            marker = "▸" if section.collapsed else "▾"
            style = "bold #f1e5b9 on #303547" if section is self._selected else CHANNEL_COLORS.get(section.kind, "#b7c2d4")
            return self._line(f"{'│ ' if bubble else '  '}{marker} {section.title}", style, width)
        if part.kind == "body" and section is not None:
            content_row = row - self._offsets[index]
            body = section.index.render(content_row, self.app.console, self.rich_style)
            cache_key = section.index, section.index.cache_key(content_row)
            self._cache[cache_key] = body
            self._cache.move_to_end(cache_key)
            while len(self._cache) > 256:
                (old_index, old_key), _ = self._cache.popitem(last=False)
                old_index.cache.pop(old_key, None)
            side = Strip([Segment("│ " if bubble else "  ", Style(color=color))])
            end = Strip([Segment(" │" if bubble else "  ", Style(color=color))])
            return Strip.join([side, body, end]).adjust_cell_length(width, self.rich_style)
        return Strip.blank(width, self.rich_style)

    def _toggle(self, section: ChatSection) -> None:
        """Toggle a detail while keeping the reader anchored.

        Args:
            section: Selected input, system, reasoning, or tool section.
        """
        self.following = False
        self._selected = section
        section.collapsed = not section.collapsed
        self._dirty = True
        self.sync()

    def select_detail(self, offset: int) -> None:
        """Move keyboard selection among expandable details.

        Args:
            offset: Positive for the next detail and negative for the previous one.
        """
        self.sync()
        sections = list(self._detail_rows)
        if not sections:
            return
        if self._selected in sections:
            index = (sections.index(self._selected) + offset) % len(sections)
        else:
            top = int(self.scroll_y)
            visible = [i for i, section in enumerate(sections) if self._detail_rows[section] >= top]
            index = (visible[0] if visible else len(sections) - 1) if offset > 0 else (visible[0] - 1 if visible else len(sections) - 1)
        self._selected = sections[index]
        self.following = False
        target = self._detail_rows[self._selected]
        if target < self.scroll_y or target >= self.scroll_y + self.size.height:
            self.scroll_to(y=max(0, target - self.size.height // 2), animate=False, immediate=True)
        self.refresh()

    def toggle_selected(self) -> None:
        """Toggle the selected detail, choosing a visible detail when none is selected."""
        if self._selected is None:
            self.select_detail(1)
        if self._selected is not None:
            self._toggle(self._selected)

    def toggle_system(self, agent: str | None = None) -> None:
        """Show or hide stable per-role system instructions.

        Args:
            agent: One role, or None to toggle all instruction sections.
        """
        sections = [self.systems[agent]] if agent in self.systems else list(self.systems.values()) if agent is None else []
        if not sections:
            return
        collapse = any(not section.collapsed for section in sections)
        for section in sections:
            section.collapsed = collapse
        self._selected = sections[0]
        self.following = False
        self._dirty = True
        self.sync()
        self.scroll_home(animate=False, immediate=True)

    def on_click(self, event: events.Click) -> None:
        """Toggle a clicked detail header without changing the selected stage.

        Args:
            event: Pointer click within the conversation viewport.
        """
        self.focus()
        row = int(self.scroll_y) + int(event.y)
        index = bisect.bisect_right(self._offsets, row) - 1
        if 0 <= index < len(self._parts):
            part = self._parts[index]
            if part.kind in {"detail", "top"} and part.section is not None:
                self._toggle(part.section)
                event.stop()

    def on_resize(self, event: events.Resize) -> None:
        """Reindex visible sections for the new viewport width.

        Args:
            event: Textual resize event.
        """
        self.sync()

    def plain(self) -> str:
        """Return all retained input, output and detail text for verification.

        Returns:
            Full original section text, including collapsed content and complete tool values.
        """
        parts = [section.source for section in self.systems.values()]
        for bubble in self.bubbles:
            parts.append(bubble.agent)
            parts.extend(section.source for section in bubble.sections)
        return "\n\n".join(parts)

    def visible_plain(self) -> str:
        """Return only currently expanded text for display assertions.

        Returns:
            Role labels, section headers and expanded display content.
        """
        parts = []
        for section in self.systems.values():
            parts.append(section.title)
            if not section.collapsed:
                parts.append(section.document.plain())
        for bubble in self.bubbles:
            parts.append(bubble.agent)
            for section in bubble.sections:
                if section.title:
                    parts.append(section.title)
                if not section.collapsed:
                    parts.append(section.document.plain())
        return "\n".join(parts)
