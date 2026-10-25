from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from html import escape
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt
from markdown_it.renderer import RendererHTML
from markdown_it.token import Token

if TYPE_CHECKING:
    from open_trading_agents.reporting.reports import ReportSection

_TONE = {"Buy": "up", "Overweight": "up", "Hold": "level", "Underweight": "down", "Sell": "down"}

_CSS = """
:root {
  --paper: #fbfbfa; --ink: #1c2126; --muted: #5d6670; --rule: #e2e5e8; --wash: #f2f4f5;
  --accent: #087a59; --up: #087a59; --down: #b4372f; --level: #5d6670; --edge: #d3d8dc;
  --serif: Charter, "Bitstream Charter", "Sitka Text", Cambria, Georgia, serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #131a1f; --ink: #e4e8eb; --muted: #9aa6af; --rule: #263239; --wash: #1a242a;
    --accent: #14c290; --up: #14c290; --down: #ef6f63; --level: #9aa6af; --edge: #3d4c55;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 17px/1.62 var(--serif); font-kerning: normal; }
.page { max-width: 46rem; margin: 0 auto; padding: 3.5rem 1.5rem 6rem; }
main h2, main h3 { scroll-margin-top: 1rem; }
a { color: var(--accent); text-underline-offset: 0.15em; overflow-wrap: anywhere; }
h1, h2, h3, h4 { text-wrap: balance; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.masthead { padding-bottom: 1.75rem; border-bottom: 1px solid var(--rule); }
.brand { margin: 0 0 2.25rem; color: var(--accent); font: 600 0.9rem/1.4 var(--sans); letter-spacing: 0.02em; }
.masthead h1 {
  margin: 0; font: 600 2.6rem/1.1 var(--sans); letter-spacing: -0.02em;
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.2em 0.5em; overflow-wrap: anywhere;
}
.masthead .rating { font-weight: 500; }
.rating.up { color: var(--up); } .rating.down { color: var(--down); } .rating.level { color: var(--level); }
.rating.review { color: var(--muted); font-style: italic; }
.masthead .date { margin: 0.55rem 0 1.5rem; font: 1rem/1.4 var(--sans); color: var(--muted); }
.run { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: 0.45rem 1.25rem; font: 0.85rem/1.5 var(--sans); }
.run > div { display: contents; }
.run dt { color: var(--muted); }
.run dd { margin: 0; overflow-wrap: anywhere; }
.contents { margin: 2.25rem 0 0; font: 0.9rem/1.45 var(--sans); }
.contents ul { margin: 0; padding: 0; list-style: none; }
.contents > ul > li { margin: 0 0 0.75rem; }
.contents ul ul { margin-top: 0.2rem; padding-left: 1.6rem; font-size: 0.85rem; }
.contents ul ul li { margin: 0.1rem 0; }
.contents a { display: flex; align-items: baseline; gap: 0.6rem; color: var(--muted); text-decoration: none;
  padding: 0.2rem 0.4rem; border-left: 2px solid transparent; border-radius: 3px; }
.contents .label { min-width: 0; }
.contents a[aria-current="location"] { color: var(--accent); background: var(--wash); border-left-color: var(--accent); }
.contents li.active-section > a, .contents li.active-section > a .num { color: var(--accent); font-weight: 600; }
.contents-toggle { display: none; padding: 0.5rem 0.8rem; border: 1px solid var(--edge); border-radius: 4px;
  background: var(--paper); color: var(--accent); font: 600 0.9rem/1.4 var(--sans); cursor: pointer; }
.contents-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.js .contents-toggle { display: block; }
.js .contents > ul { display: none; margin-top: 0.75rem; }
.js .contents.is-open > ul { display: block; }
.contents > ul > li > a { color: var(--ink); }
.contents .num { flex: none; min-width: 1.6rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.contents a:hover .label, .contents a:focus-visible .label { color: var(--accent); text-decoration: underline; }
.to-top { position: fixed; right: 1rem; bottom: calc(1rem + env(safe-area-inset-bottom));
  padding: 0.25rem 0.6rem; border: 1px solid var(--edge); border-radius: 4px; background: var(--paper);
  color: var(--muted); font: 0.8rem/1.4 var(--sans); text-decoration: none; }
@media (min-width: 72rem) {
  .page { margin-left: max(21rem, calc((100% - 46rem) / 2)); margin-right: auto; }
  .contents { position: fixed; left: 1rem; top: 50%; transform: translateY(-50%); z-index: 1;
    width: 18rem; max-height: 86vh; overflow-y: auto; overscroll-behavior: contain; margin: 0;
    padding: 0.7rem; border: 1px solid var(--rule); border-radius: 6px; background: var(--paper); }
  .contents > ul > li { margin: 0 0 0.6rem; }
  .contents ul ul { padding-left: 1rem; }
  .contents a { min-height: 1.5rem; }
  .contents .num { min-width: 1.4rem; }
  .js .contents > ul { display: block; margin-top: 0; }
  .js .contents-toggle, .to-top { display: none; }
}
.credit { margin-top: 4.5rem; padding-top: 1.25rem; border-top: 1px solid var(--rule);
  color: var(--muted); font: 0.8rem/1.4 var(--sans); }
main { overflow-wrap: anywhere; }
main h2 { margin: 4.5rem 0 0.25rem; padding-top: 1.5rem; border-top: 2px solid var(--ink);
  font: 600 1.45rem/1.3 var(--sans); letter-spacing: -0.01em; }
main section:first-child h2 { margin-top: 3rem; }
main h3 { margin: 2.75rem 0 0.75rem; padding-top: 1.1rem; border-top: 1px solid var(--rule);
  font: 600 1.15rem/1.35 var(--sans); color: var(--accent); }
main h2 + h3 { margin-top: 1.25rem; padding-top: 0; border-top: 0; }
main h4, main h5, main h6 { margin: 1.75rem 0 0.5rem; font: 600 1rem/1.4 var(--sans); }
main h5, main h6 { color: var(--muted); }
main p, main ul, main ol { margin: 0 0 1rem; }
main li { margin: 0.25rem 0; }
main strong { font-weight: 650; }
main blockquote { margin: 1.25rem 0; padding: 0.1rem 0 0.1rem 1rem; border-left: 3px solid var(--rule); color: var(--muted); }
main code { font: 0.88em/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--wash); padding: 0.1em 0.3em; border-radius: 3px; }
main pre { background: var(--wash); padding: 0.9rem 1rem; border-radius: 4px; overflow-x: auto; }
main pre code { background: none; padding: 0; }
main hr { width: 3rem; margin: 2rem auto; border: 0; border-top: 1px solid var(--rule); }
.table { overflow-x: auto; overflow-wrap: normal; margin: 0 0 1.25rem; }
table { border-collapse: collapse; width: 100%; font: 0.86rem/1.45 var(--sans); font-variant-numeric: tabular-nums; }
th, td { padding: 0.45rem 0.7rem; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; border-bottom-color: var(--muted); }
tbody tr:last-child td { border-bottom: 0; }
@media (max-width: 34rem) {
  body { font-size: 16px; }
  .page { padding: 2rem 1rem 3rem; }
  .masthead h1 { font-size: 2rem; }
  .run { grid-template-columns: 1fr; gap: 0; }
  .run dd { margin-bottom: 0.6rem; }
}
@media print {
  :root { --paper: #fff; --ink: #000; --muted: #444; --rule: #bbb; --accent: #000; --wash: #f2f2f2;
    --edge: #bbb; --up: #087a59; --down: #b4372f; --level: #444; }
  body { font-size: 11pt; }
  .table { overflow: visible; }
  .page { max-width: none; margin: 0; padding: 0; }
  .contents, .to-top { display: none; }
  main h2 { break-before: page; border-top: 0; margin-top: 0; }
  main section:first-child h2 { break-before: auto; }
  main h3, main h4 { break-after: avoid; }
  a { color: inherit; text-decoration: none; }
  table { break-inside: avoid; }
}
"""


_SCRIPT = """
(() => {
  const contents = document.querySelector('.contents');
  const toggle = contents.querySelector('.contents-toggle');
  const desktop = window.matchMedia('(min-width: 72rem)');
  const entries = Array.from(contents.querySelectorAll('a[href^="#"]'), link => {
    const target = document.getElementById(link.hash.slice(1));
    return {link, heading: target.matches('section') ? target.querySelector('h2') : target};
  });
  let current = null;
  let scheduled = false;

  function update() {
    scheduled = false;
    toggle.setAttribute('aria-expanded', String(desktop.matches || contents.classList.contains('is-open')));
    let active = entries[0];
    for (const entry of entries) {
      if (entry.heading.getBoundingClientRect().top > 32) break;
      active = entry;
    }
    if (window.scrollY > 0 && window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
      active = entries[entries.length - 1];
    }
    if (!active) return;
    if (active !== current) {
      if (current) {
        current.link.removeAttribute('aria-current');
        current.link.closest('.contents > ul > li').classList.remove('active-section');
      }
      current = active;
      active.link.setAttribute('aria-current', 'location');
      active.link.closest('.contents > ul > li').classList.add('active-section');
    }
    if (desktop.matches) {
      const item = active.link.getBoundingClientRect();
      const viewport = contents.getBoundingClientRect();
      if (item.top < viewport.top) contents.scrollTop += item.top - viewport.top - 8;
      else if (item.bottom > viewport.bottom) contents.scrollTop += item.bottom - viewport.bottom + 8;
    }
  }

  function scheduleUpdate() {
    if (!scheduled) {
      scheduled = true;
      window.requestAnimationFrame(update);
    }
  }

  toggle.addEventListener('click', () => {
    contents.classList.toggle('is-open');
    scheduleUpdate();
  });
  contents.addEventListener('click', event => {
    if (!desktop.matches && event.target.closest('a')) {
      contents.classList.remove('is-open');
      scheduleUpdate();
    }
  });
  window.addEventListener('scroll', scheduleUpdate, {passive: true});
  window.addEventListener('resize', scheduleUpdate);
  window.addEventListener('hashchange', scheduleUpdate);
  window.addEventListener('load', scheduleUpdate);
  document.documentElement.classList.add('js');
  update();
})();
"""
_SCRIPT_HASH = b64encode(sha256(_SCRIPT.encode()).digest()).decode()


def _table_wrapper(renderer: RendererHTML, tokens: list[Token], index: int, options: dict[str, Any], env: dict[str, Any]) -> str:
    """Wrap each table so wide evidence columns can scroll within the reading column.

    Args:
        renderer: Markdown renderer supplied by the render-rule API.
        tokens: Parsed report tokens.
        index: Table opening or closing token index.
        options: Markdown rendering options.
        env: Rendering context.

    Returns:
        Opening or closing table markup with its overflow container.
    """
    return '<div class="table"><table>\n' if tokens[index].type == "table_open" else "</table></div>\n"


def _render_agent_text(markdown: MarkdownIt, text: str) -> str:
    """Place headings authored by an agent below its report heading.

    Args:
        markdown: Renderer with raw HTML and images disabled.
        text: Complete agent output.

    Returns:
        Safe rendered HTML with report hierarchy preserved.
    """
    tokens = markdown.parse(text)
    for token in tokens:
        if token.type in ("heading_open", "heading_close"):
            token.tag = f"h{min(int(token.tag[1]) + 3, 6)}"
    return markdown.renderer.render(tokens, markdown.options, {})


def render_report(ticker: str, trade_date: str, rating: str, details: list[tuple[str, str]], parts: list[ReportSection]) -> str:
    """Render a self-contained report with provenance and linked team sections.

    Args:
        ticker: Instrument symbol.
        trade_date: Analysis cutoff date.
        rating: Validated final rating or REVIEW.
        details: Plain-text run metadata shared with the Markdown report.
        parts: Ordered sections and their complete agent outputs.

    Returns:
        A standalone HTML page with local navigation and no remotely loaded assets.
    """
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
    markdown.add_render_rule("table_open", _table_wrapper)
    markdown.add_render_rule("table_close", _table_wrapper)
    metadata = "".join(f"<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>" for label, value in details)
    contents, body = [], []
    for section_index, (heading, _folder, entries) in enumerate(parts, 1):
        agent_links = "".join(
            f'<li><a href="#s{section_index}-{index}"><span class="label">{escape(name)}</span></a></li>'
            for index, (name, _filename, _text) in enumerate(entries, 1)
        )
        numeral, separator, name = heading.partition(". ")
        if not separator:
            numeral, name = "+", heading
        contents.append(
            f'<li><a href="#s{section_index}"><span class="num">{escape(numeral)}</span><span class="label">{escape(name)}</span></a><ul>{agent_links}</ul></li>'
        )
        body.append(f'<section id="s{section_index}"><h2>{escape(heading)}</h2>')
        for index, (name, _filename, text) in enumerate(entries, 1):
            body.append(f'<h3 id="s{section_index}-{index}">{escape(name)}</h3>\n{_render_agent_text(markdown, text)}')
        body.append("</section>")
    tone = _TONE.get(rating, "review")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'sha256-{_SCRIPT_HASH}'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; form-action 'none'">
<title>{escape(ticker)} {escape(trade_date)} {escape(rating)}: Open Trading Agents report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page" id="top">
<header class="masthead">
<p class="brand">Open Trading Agents</p>
<h1><span class="ticker">{escape(ticker)}</span> <span class="rating {tone}">{escape(rating)}</span></h1>
<p class="date">Analysis of {escape(trade_date)}</p>
<dl class="run">{metadata}</dl>
</header>
<nav class="contents" aria-label="Sections">
<button class="contents-toggle" type="button" aria-expanded="false" aria-controls="report-contents">Contents</button>
<ul id="report-contents">{"".join(contents)}</ul>
</nav>
<main>{"".join(body)}</main>
<footer class="credit">Open Trading Agents</footer>
</div>
<a class="to-top" href="#top">Top</a>
<script id="report-navigation">{_SCRIPT}</script>
</body>
</html>
"""
