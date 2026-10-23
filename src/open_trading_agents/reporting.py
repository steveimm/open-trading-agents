from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from open_trading_agents.domain import AgentEvent


def write_reports(directory: Path, ticker: str, trade_date: str, outputs: dict[str, str], rating: str,
                  settings: dict[str, Any], events: list[AgentEvent], include_html: bool = True) -> Path:
    """Write individual reports, combined reports, and a local event transcript.

    Args:
        directory: The run directory.
        ticker: Instrument symbol.
        trade_date: Analysis cutoff date.
        outputs: Ordered role reports and debate transcripts.
        rating: Validated final rating or REVIEW.
        settings: Non-secret run settings for reproducibility.
        events: Persisted events to export as JSON lines.
        include_html: Whether to generate the standalone HTML report.

    Returns:
        The combined Markdown report path.
    """
    report_dir = directory / "reports"
    report_dir.mkdir(exist_ok=True)
    parts = [f"# Trading analysis: {ticker}\n\nAnalysis date: {trade_date}\n\nFinal rating: **{rating}**"]
    for index, (name, text) in enumerate(outputs.items(), 1):
        slug = "_".join(word.lower() for word in name.split() if word.isalnum())
        (report_dir / f"{index:02}_{slug}.md").write_text(text, encoding="utf-8")
        parts.append(f"## {name}\n\n{text}")
    combined = "\n\n".join(parts) + "\n"
    target = report_dir / "complete_report.md"
    target.write_text(combined, encoding="utf-8")
    (report_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    from dataclasses import asdict

    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    if include_html:
        renderer = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
        body = renderer.render(combined)
        page = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; form-action 'none'">
<title>{html.escape(ticker)} analysis</title>
<style>body{{max-width:960px;margin:2rem auto;padding:0 1rem;font:16px/1.6 system-ui;color:#17202a;background:#fafafa}}pre{{overflow:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.4rem;text-align:left}}h1,h2{{line-height:1.2}}code{{overflow-wrap:anywhere}}</style>
<body>{body}</body></html>"""
        (report_dir / "complete_report.html").write_text(page, encoding="utf-8")
    return target
