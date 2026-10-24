from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from dataclasses import dataclass

from rich._wrap import divide_line
from rich.console import Console
from rich.style import Style
from rich.text import Span, Text
from textual import events
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.scrollbar import ScrollDown, ScrollTo, ScrollUp
from textual.strip import Strip

from open_trading_agents.domain import AgentEvent

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
CONTROL_ESCAPES = {code: f"\\x{code:02x}" for code in (*range(32), *range(127, 160)) if code not in (9, 10)}
CONTROL_ESCAPES[13] = "\\r"
TOOL_PREVIEW_LIMIT = 10_000
TOOL_PREVIEW_PREFIX = 6_000
TOOL_PREVIEW_SUFFIX = 2_000


@dataclass
class TranscriptBlock:
    start: int
    end: int
    text: str


def literal_text(text: str) -> str:
    """Render terminal control sequences as visible characters.

    Args:
        text: Untrusted model or tool output.

    Returns:
        Display text with newline and tab layout preserved, and controls escaped.
    """
    return text.replace("\r\n", "\n").translate(CONTROL_ESCAPES).expandtabs(4)


def tool_text(event: AgentEvent) -> str:
    """Extract readable script arguments and tool text without losing raw event data.

    Args:
        event: Native tool input, output delta, or result snapshot.

    Returns:
        Plain text suitable for the transcript, retaining multiline source code.
    """
    text = event.text
    payload = event.data.get("arguments") if event.kind == "tool_start" else event.data.get("result")
    if event.data.get("delta"):
        return text
    if payload is None and text.lstrip().startswith(("{", "[")):
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return text
    if payload is None:
        return text
    if isinstance(payload, dict) and isinstance(payload.get("content"), list):
        blocks = payload["content"]
        return "\n".join(str(block.get("text", f"<{block.get('type', 'content')} content>")) for block in blocks if isinstance(block, dict))
    if event.kind == "tool_result" and not (isinstance(payload, dict) and {"stdout", "stderr"}.intersection(payload)):
        return text or json.dumps(payload, ensure_ascii=False, indent=2)
    if isinstance(payload, dict):
        parts = []
        for key, value in payload.items():
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
            separator = "\n" if "\n" in rendered else " "
            parts.append(f"{key}:{separator}{rendered}")
        return "\n".join(parts)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def tool_preview(text: str) -> str:
    """Shorten large tool results for display while keeping exact head and tail text.

    Args:
        text: Complete decoded tool output retained independently of this preview.

    Returns:
        Original text or a preview naming the exact omitted character count.
    """
    if len(text) <= TOOL_PREVIEW_LIMIT:
        return text
    omitted = len(text) - TOOL_PREVIEW_PREFIX - TOOL_PREVIEW_SUFFIX
    return f"{text[:TOOL_PREVIEW_PREFIX]}\n...[omitted {omitted} chars]...\n{text[-TOOL_PREVIEW_SUFFIX:]}"


class Transcript:
    def __init__(self) -> None:
        """Keep complete logical lines independently of terminal rendering."""
        self.lines = [Text()]
        self.revision = 0
        self.last_channel = ""
        self.last_agent = ""
        self.length = 0
        self.blocks: list[TranscriptBlock] = []
        self.rewrite_revision = -1
        self.rewrite_line = 0

    def append(self, text: str, style: str = "") -> None:
        """Append styled text without reprocessing earlier lines.

        Args:
            text: Literal content from an event.
            style: Rich text style for the added content.
        """
        if not text:
            return
        text = literal_text(text)
        pieces = text.split("\n")
        self.lines[-1].append(pieces[0], style)
        self.lines.extend(Text(piece, style=style) for piece in pieces[1:])
        self.length += len(text)
        self.revision += 1

    def append_block(self, text: str, style: str) -> TranscriptBlock:
        """Append one tool result whose later snapshots may replace its content.

        Args:
            text: Initial tool output.
            style: Output color.

        Returns:
            A tracked region that remains valid when other tools stream.
        """
        start = self.length
        self.append(text, style)
        block = TranscriptBlock(start, self.length, literal_text(text))
        self.blocks.append(block)
        return block

    def replace_block(self, block: TranscriptBlock, text: str, style: str) -> None:
        """Reconcile cumulative tool output without repeating earlier content.

        Args:
            block: Region created for this tool invocation.
            text: Complete current snapshot.
            style: Output color.
        """
        text = literal_text(text)
        if text == block.text:
            return
        if block.end == self.length and text.startswith(block.text):
            self.append(text[len(block.text) :], style)
            block.end, block.text = self.length, text
            return
        complete = Text("\n").join(self.lines)
        replacement = complete[: block.start]
        replacement.append(text, style)
        replacement.append_text(complete[block.end :])
        start_line = complete.plain[: block.start].count("\n")
        difference = len(text) - (block.end - block.start)
        for other in self.blocks:
            if other is not block and other.start >= block.end:
                other.start += difference
                other.end += difference
        self.lines = list(replacement.split("\n", allow_blank=True)) or [Text()]
        self.length += difference
        block.end, block.text = block.start + len(text), text
        self.revision += 1
        self.rewrite_line = min(self.rewrite_line, start_line) if self.rewrite_revision >= 0 else start_line
        self.rewrite_revision = self.revision

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


class TranscriptIndex:
    def __init__(self, document: Transcript) -> None:
        """Index one document independently of a widget or its scroll position.

        Args:
            document: Styled logical lines owned by a transcript section.
        """
        self.document = document
        self.width = 0
        self.revision = -1
        self.offsets = [0]
        self.starts: list[list[int]] = []
        self.cache: OrderedDict[tuple[int, int], Strip] = OrderedDict()

    def sync(self, width: int) -> bool:
        """Reindex only modified logical lines, without rendering hidden rows.

        Args:
            width: Available terminal cells.

        Returns:
            Whether the index changed.
        """
        width = max(1, width)
        if width == self.width and self.revision == self.document.revision:
            return False
        resized = width != self.width
        start = 0 if resized or self.revision < 0 else max(0, len(self.starts) - 1)
        if self.revision < self.document.rewrite_revision:
            start = min(start, self.document.rewrite_line)
        if resized:
            self.cache.clear()
        else:
            for key in list(self.cache):
                if key[0] >= start:
                    del self.cache[key]
        self.starts[start:] = [[0, *divide_line(line.plain, width, fold=True)] for line in self.document.lines[start:]]
        self.offsets = self.offsets[:start + 1]
        for starts in self.starts[start:]:
            self.offsets.append(self.offsets[-1] + len(starts))
        self.width, self.revision = width, self.document.revision
        return True

    @property
    def height(self) -> int:
        """Return the number of wrapped rows."""
        return self.offsets[-1]

    def source_at(self, row: int) -> tuple[int, int]:
        """Locate the logical source at a wrapped row.

        Args:
            row: Wrapped row offset.

        Returns:
            Logical line and character offset for scroll anchoring.
        """
        index = min(max(0, bisect.bisect_right(self.offsets, row) - 1), len(self.starts) - 1)
        if index < 0:
            return 0, 0
        local = min(max(0, row - self.offsets[index]), len(self.starts[index]) - 1)
        return index, self.starts[index][local]

    def row_at(self, line: int, character: int) -> int:
        """Find a wrapped row after a resize or earlier insertion.

        Args:
            line: Logical line index.
            character: Character offset previously visible at the top.

        Returns:
            The corresponding wrapped row.
        """
        if not self.starts:
            return 0
        line = min(line, len(self.starts) - 1)
        return self.offsets[line] + max(0, bisect.bisect_right(self.starts[line], character) - 1)

    def render(self, row: int, console: Console, base_style: Style) -> Strip:
        """Render one bounded row rather than cropping an entire large logical line.

        Args:
            row: Wrapped row offset.
            console: Current terminal renderer.
            base_style: Widget background and default foreground.

        Returns:
            One terminal-width row with a bounded render cache.
        """
        index, local = self.cache_key(row)
        if index < 0 or index >= len(self.starts):
            return Strip.blank(self.width, base_style)
        key = index, local
        if key not in self.cache:
            line = self.document.lines[index]
            starts = self.starts[index]
            start = starts[local]
            stop = starts[local + 1] if local + 1 < len(starts) else len(line)
            visible = Text(line.plain[start:stop], style=line.style)
            visible.spans = [Span(max(0, span.start - start), min(stop, span.end) - start, span.style)
                             for span in line.spans if span.end > start and span.start < stop]
            self.cache[key] = Strip(visible.render(console)).apply_style(console.get_style(visible.style))
            if len(self.cache) > 256:
                self.cache.popitem(last=False)
        strip = self.cache[key]
        self.cache.move_to_end(key)
        return strip.adjust_cell_length(self.width, base_style).apply_style(base_style)

    def cache_key(self, row: int) -> tuple[int, int]:
        """Identify a rendered row for a shared viewport cache budget.

        Args:
            row: Wrapped row offset.

        Returns:
            Logical line and its wrapped row index.
        """
        index = bisect.bisect_right(self.offsets, row) - 1
        return index, row - self.offsets[index] if index >= 0 else 0


class FollowScrollView(ScrollView):
    can_focus = True

    def __init__(self, *, id: str | None = None) -> None:
        """Create a viewport whose follow mode changes only through explicit navigation.

        Args:
            id: Textual widget identifier.
        """
        super().__init__(id=id)
        self.following = True

    def _at_bottom(self) -> bool:
        """Determine whether manual scrolling targets the last available row.

        Returns:
            Whether the scroll target has reached the bottom.
        """
        return self.scroll_target_y >= self.max_scroll_y - 0.5

    def _follow_after_resize(self) -> None:
        """Keep following at the newest row after surrounding controls change height."""
        if self.following and self.is_on_screen:
            self.scroll_end(animate=False, immediate=True)

    def navigate(self, direction: str) -> None:
        """Scroll manually and resume following when downward navigation reaches the end.

        Args:
            direction: up, down, pageup, pagedown, home, or end.
        """
        self.following = False
        if direction == "end":
            self.scroll_end(animate=False, immediate=True)
            self.following = True
        elif direction == "home":
            self.scroll_home(animate=False, immediate=True)
        else:
            amount = max(1, self.size.height - 1) if direction.startswith("page") else 1
            sign = -1 if direction in {"up", "pageup"} else 1
            self.scroll_relative(y=sign * amount, animate=False, immediate=True)
            self.following = sign > 0 and self._at_bottom()

    def _on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Pause following on an upward wheel movement.

        Args:
            event: Native pointer scroll event.
        """
        self.following = False
        super()._on_mouse_scroll_up(event)

    def _on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Return to following when a downward wheel movement reaches the bottom.

        Args:
            event: Native pointer scroll event.
        """
        super()._on_mouse_scroll_down(event)
        if not (event.ctrl or event.shift):
            self.following = self._at_bottom()

    def _on_scroll_to(self, message: ScrollTo) -> None:
        """Use scrollbar dragging as an explicit follow-mode decision.

        Args:
            message: Scrollbar target selected by the user.
        """
        super()._on_scroll_to(message)
        if message.y is not None:
            self.following = self._at_bottom()

    def _on_scroll_up(self, event: ScrollUp) -> None:
        """Pause following when the user clicks above the scrollbar thumb.

        Args:
            event: Scrollbar page-up request.
        """
        self.following = False
        super()._on_scroll_up(event)

    def _on_scroll_down(self, event: ScrollDown) -> None:
        """Resume following when a scrollbar page-down request reaches the end.

        Args:
            event: Scrollbar page-down request.
        """
        super()._on_scroll_down(event)
        self.following = self._at_bottom()


class TranscriptView(FollowScrollView):
    DEFAULT_CSS = """
    TranscriptView { height: 1fr; width: 1fr; scrollbar-size: 1 1; background: #111821; }
    TranscriptView:focus { background: #141e2a; }
    """

    def __init__(self, document: Transcript, *, id: str | None = None) -> None:
        """Display an existing transcript with independently cached visible rows.

        Args:
            document: The complete styled transcript.
            id: Textual widget identifier.
        """
        super().__init__(id=id)
        self.document = document
        self.index = TranscriptIndex(document)

    @property
    def _cache(self) -> OrderedDict[tuple[int, int], Strip]:
        """Expose the bounded render cache for diagnostics."""
        return self.index.cache

    @property
    def _offsets(self) -> list[int]:
        """Expose logical row offsets for render diagnostics."""
        return self.index.offsets

    def set_document(self, document: Transcript) -> None:
        """Select a different transcript without replacing the viewport widget.

        Args:
            document: Newly selected transcript.
        """
        if self.document is not document:
            self.document = document
            self.index = TranscriptIndex(document)
            self.following = True
            self.sync()

    def sync(self) -> None:
        """Update changed rows while preserving an anchored reader during resize."""
        if not self.is_on_screen:
            return
        width = max(1, self.scrollable_content_region.width)
        resized = width != self.index.width
        anchor = self.index.source_at(int(self.scroll_y))
        if not self.index.sync(width):
            return
        self.virtual_size = Size(width, self.index.height)
        if self.following:
            self.scroll_end(animate=False, immediate=True)
        elif resized:
            self.scroll_to(y=self.index.row_at(*anchor), animate=False, immediate=True)
        self.refresh()
        self.call_after_refresh(self.sync)

    def render_line(self, y: int) -> Strip:
        """Render one visible wrapped row.

        Args:
            y: Row offset within the viewport.

        Returns:
            A cached terminal-width strip.
        """
        return self.index.render(int(self.scroll_y) + y, self.app.console, self.rich_style)

    def on_resize(self, event: events.Resize) -> None:
        """Reindex the transcript after a viewport size change.

        Args:
            event: Textual resize event.
        """
        self.sync()
        self.call_after_refresh(self._follow_after_resize)


