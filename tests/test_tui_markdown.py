import pytest
from markdown_it.token import Token
from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual.geometry import Offset
from textual.selection import Selection

from open_trading_agents.analysis.models import AgentEvent
from open_trading_agents.ui.app import TradingApp
from open_trading_agents.ui.chat import ChatSection
from open_trading_agents.ui.markdown import MarkdownLayout
from open_trading_agents.ui.transcript import Transcript, TranscriptIndex


@pytest.mark.parametrize("width", [28, 60, 110])
def test_markdown_formats_prose_tables_and_code_without_changing_source(width: int) -> None:
    """Render complete financial prose at grid and fullscreen widths.

    Args:
        width: Available viewport columns.
    """
    source = '''# Outlook

**Hold** with *caution* and `cash`.

| Metric | Value |
|---|---:|
| Revenue | 119.8 |
| 한국어 | 24.2% |

- First point
- Second point

> Source dates matter.

```python
value = 123
print(value)
```

[Source](https://example.org)
'''
    document = Transcript(markdown=True)
    document.append(source)
    index = TranscriptIndex(document)
    index.sync(width)
    rows = [index.render(row, Console(), Style()) for row in range(index.height)]
    visible = '\n'.join(row.text for row in rows)
    assert document.plain() == source
    assert all(row.cell_length == width for row in rows)
    for expected in ("Outlook", "Hold", "caution", "cash", "Revenue", "119.8", "한국어", "24.2%", "First point", "value = 123"):
        assert expected in visible
    for marker in ("# Outlook", "**Hold**", "*caution*", "`cash`", "|---", "```python"):
        assert marker not in visible
    assert any(segment.style and segment.style.bold and "Hold" in segment.text for row in rows for segment in row)
    assert any(segment.style and segment.style.italic and "caution" in segment.text for row in rows for segment in row)
    assert any(segment.style and segment.style.bgcolor and "value" in segment.text for row in rows for segment in row)
    assert "https://example.org" in visible


def test_streamed_markdown_reuses_finished_blocks_and_reconciles_partial_syntax(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complete inline markers, table rows, and fences without duplicate or stale blocks.

    Args:
        monkeypatch: Count block rendering without making model requests.
    """
    calls = []
    original = MarkdownLayout._render_block

    def render_block(layout: MarkdownLayout, tokens: list[Token], width: int) -> tuple[Text, ...]:
        """Record which source blocks are actually formatted."""
        calls.append(tokens[0].type)
        return original(layout, tokens, width)

    monkeypatch.setattr(MarkdownLayout, "_render_block", render_block)
    document = Transcript(markdown=True)
    index = TranscriptIndex(document)
    for chunk in ("## Summary\n\n**Bu", "y**\n\n| Metric | Value |\n|---|---|\n| Cash |", " 10 |\n\n```python\nx =", " 2\n```"):
        document.append(chunk)
        index.sync(50)
    assert calls.count("heading_open") == 1
    visible = '\n'.join(index.render(row, Console(), Style()).text for row in range(index.height))
    assert visible.count("Summary") == 1 and visible.count("Buy") == 1
    assert "**" not in visible and "Cash" in visible and "10" in visible and "x = 2" in visible
    rendered = len(calls)
    assert not index.sync(50) and len(calls) == rendered
    index.sync(30)
    assert calls.count("heading_open") == 2


def test_tool_payloads_stay_literal() -> None:
    """Formatting must not reinterpret source data, shell scripts, or JSON from tools."""
    source = 'Arguments\ncommand: echo "**literal**"\n\nResult\n{"heading": "# source"}'
    section = ChatSection("tool", "tool_result")
    section.replace(source)
    section.index.sync(80)
    assert not section.document.markdown
    assert '\n'.join(line.plain for line in section.index.lines) == source


async def test_formatted_text_remains_selectable_and_source_stays_complete() -> None:
    """Select displayed emphasis without copying Markdown markers or losing the source."""
    app = TradingApp(selected_analysts=("market",))
    async with app.run_test(size=(100, 30)) as pilot:
        source = "## Recommendation\n\n**Buy** with *caution*."
        await app.receive(AgentEvent("text", "Market Analyst", "analysis", source, {"message_id": "markdown"}))
        app.flush_events()
        view = app.agent_views["Market Analyst"]
        view.navigate("home")
        await pilot.pause()
        row = next(row for row in range(view.virtual_size.height) if "Buy with caution." in view.render_line(row).text)
        region = view.scrollable_content_region
        start = (region.x - view.region.x + 2, region.y - view.region.y + row)
        await pilot.mouse_down(view, offset=start)
        await pilot.hover(view, offset=(start[0] + 2, start[1]))
        await pilot.mouse_up(view, offset=(start[0] + 2, start[1]))
        await pilot.pause()
        assert app.clipboard == "Buy"
        assert view.bubbles[0].sections[0].source == source
        assert view.get_selection(Selection(Offset(2, row), Offset(5, row))) == ("Buy", "\n")
