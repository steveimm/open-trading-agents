from __future__ import annotations

from collections.abc import Iterator

from markdown_it.token import Token
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from rich.theme import Theme


class MarkdownLayout:
    def __init__(self) -> None:
        """Cache formatted blocks independently of the original transcript text."""
        self.console = Console(theme=Theme({
            "markdown.paragraph": "#d9e2ef",
            "markdown.h1": "bold #bd9bfa",
            "markdown.h2": "bold #78cbd4",
            "markdown.h3": "bold #a6bfee",
            "markdown.h4": "bold #a6bfee",
            "markdown.h5": "bold #a6bfee",
            "markdown.h6": "bold #a6bfee",
            "markdown.em": "italic",
            "markdown.strong": "bold",
            "markdown.code": "#e8b86d on #1a2635",
            "markdown.link": "underline #78cbd4",
            "markdown.link_url": "#8799af",
            "markdown.item.bullet": "#78cbd4",
            "markdown.item.number": "#78cbd4",
            "markdown.block_quote": "#a9bcd3",
            "markdown.table.header": "bold #78cbd4",
            "markdown.table.border": "#34475e",
        }))
        self.cache: dict[tuple[int, str], tuple[Text, ...]] = {}
        self.anchors: list[tuple[int, int]] = []

    @staticmethod
    def _blocks(tokens: list[Token]) -> Iterator[list[Token]]:
        """Group top-level tokens while retaining nested lists, tables, and quotes.

        Args:
            tokens: Parsed Markdown tokens.

        Yields:
            One complete top-level block at a time.
        """
        start = 0
        for index, token in enumerate(tokens):
            if token.level == 0 and token.nesting != 1:
                yield tokens[start:index + 1]
                start = index + 1

    def _render_block(self, tokens: list[Token], width: int) -> tuple[Text, ...]:
        """Format one block, preserving prose line breaks during streaming.

        Args:
            tokens: A top-level block with resolved inline markup.
            width: Available terminal columns.

        Returns:
            Styled lines without surrounding layout padding.
        """
        if tokens[0].type == "paragraph_open":
            inline = tokens[1]
            if all(child.type in {"text", "softbreak"} for child in inline.children or []):
                text = "".join("\n" if child.type == "softbreak" else child.content for child in inline.children or [])
                return tuple(Text(line, style="#d9e2ef") for line in text.split("\n"))

        for token in tokens:
            for child in token.children or []:
                if child.type == "softbreak":
                    child.type = "hardbreak"
        markdown = Markdown("", hyperlinks=False, code_theme="github-dark", style="#d9e2ef")
        markdown.parsed = tokens
        rows = self.console.render_lines(markdown, self.console.options.update(width=width), pad=False)
        lines = [Text.assemble(*[(segment.text, segment.style) for segment in row]) for row in rows]
        if tokens[0].type not in {"fence", "code_block"}:
            for line in lines:
                line.rstrip()
        while lines and not lines[0].plain.strip():
            lines.pop(0)
        while lines and not lines[-1].plain.strip():
            lines.pop()
        return tuple(lines)

    def render(self, source: str, width: int) -> list[Text]:
        """Reuse unchanged blocks while rebuilding the current Markdown layout.

        Args:
            source: Complete sanitized source, retained separately for export.
            width: Available terminal columns.

        Returns:
            Styled display lines, with source anchors stored alongside them.
        """
        parsed = Markdown(source, hyperlinks=False)
        lines: list[Text] = []
        anchors: list[tuple[int, int]] = []
        cache = {}
        previous_end = 0
        previous_kind = ""
        spaced = {"heading_open", "blockquote_open", "fence", "code_block", "table_open", "hr"}
        for tokens in self._blocks(parsed.parsed):
            start, end = tokens[0].map or (previous_end, previous_end)
            kind = tokens[0].type
            plain = kind == "paragraph_open" and all(child.type in {"text", "softbreak"} for child in tokens[1].children or [])
            key = width, repr(tokens)
            rendered = self.cache.get(key)
            if rendered is None:
                rendered = self._render_block(tokens, width)
            cache[key] = rendered
            if lines and (start > previous_end or kind in spaced or previous_kind in spaced):
                lines.append(Text())
                anchors.append((previous_end, 0))
            for offset, line in enumerate(rendered):
                lines.append(line)
                anchors.append((start + offset, 0) if plain else (start, offset * width))
            previous_end = end
            previous_kind = kind
        self.cache, self.anchors = cache, anchors or [(0, 0)]
        return lines or [Text()]
