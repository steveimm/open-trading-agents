from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from open_trading_agents.domain import AgentEvent
from open_trading_agents.report_html import render_report
from open_trading_agents.usage import summarize_usage

type ReportEntry = tuple[str, str, str]
type ReportSection = tuple[str, str, list[ReportEntry]]

_SECTIONS = (
    ("I. Analyst Team Reports", "1_analysts", (
        ("Market Analyst", "market.md"), ("Sentiment Analyst", "sentiment.md"),
        ("News Analyst", "news.md"), ("Fundamentals Analyst", "fundamentals.md"),
    )),
    ("II. Research Team Decision", "2_research", (("Research Debate", "debate.md"), ("Research Manager", "manager.md"))),
    ("III. Trading Team Plan", "3_trading", (("Trader", "trader.md"),)),
    ("IV. Risk Management Team Decision", "4_risk", (("Risk Debate", "debate.md"),)),
    ("V. Portfolio Manager Decision", "5_portfolio", (("Portfolio Manager", "decision.md"),)),
)


def _debate_entries(name: str, text: str, events: list[AgentEvent]) -> list[ReportEntry]:
    """Use recorded speaker turns when they account for the entire public debate.

    Args:
        name: Research or risk debate output name.
        text: Complete chronological discussion from the workflow.
        events: Recorded workflow events with authoritative speaker attribution.

    Returns:
        Chronological turns, or the original complete discussion when records differ.
    """
    stage = "research" if name == "Research Debate" else "risk"
    turns: dict[tuple[str, int, int], AgentEvent] = {}
    for event in events:
        if event.kind != "conversation_output" or event.stage != stage:
            continue
        round_number, turn_number = event.data.get("round"), event.data.get("turn")
        if isinstance(round_number, int) and isinstance(turn_number, int):
            turns[(event.agent, round_number, turn_number)] = event
    ordered = sorted(turns.values(), key=lambda event: (event.data["round"], event.data["turn"]))
    reconstructed = "\n\n".join(f"### Round {event.data['round']}: {event.agent}\n\n{event.text}" for event in ordered)
    if not ordered or reconstructed != text:
        return [(name, "debate.md", text)]
    return [(f"{event.agent} · Round {event.data['round']}", f"turn_{index:02}.md", event.text)
            for index, event in enumerate(ordered, 1)]


def _report_parts(outputs: dict[str, str], events: list[AgentEvent]) -> list[ReportSection]:
    """Group all available outputs in workflow order, retaining partial and extra reports.

    Args:
        outputs: Role reports and complete debate transcripts.
        events: Recorded events for debate speaker attribution.

    Returns:
        Ordered section headings, directories, and report entries.
    """
    parts: list[ReportSection] = []
    consumed: set[str] = set()
    for heading, folder, roles in _SECTIONS:
        entries: list[ReportEntry] = []
        for name, filename in roles:
            consumed.add(name)
            if text := outputs.get(name):
                entries.extend(_debate_entries(name, text, events) if name.endswith("Debate") else [(name, filename, text)])
        if entries:
            parts.append((heading, folder, entries))
    extra = [(name, f"{index:02}.md", text) for index, (name, text) in enumerate(outputs.items(), 1)
             if name not in consumed and text]
    if extra:
        parts.append(("Additional Reports", "additional", extra))
    return parts


def _runtime_description(harness: Any, provider: Any, model: Any) -> str:
    """Describe runtime fields without inferring native harness model defaults.

    Args:
        harness: Observed or configured CLI harness.
        provider: Observed or configured provider.
        model: Observed or configured model identifier.

    Returns:
        A readable runtime description with explicit unknown fields.
    """
    return f"Harness: {harness or 'not reported'} · Provider: {provider or 'not reported'} · Model: {model or 'not reported'}"


def report_details(settings: dict[str, Any], generated: str) -> list[tuple[str, str]]:
    """Describe run provenance for both human report formats.

    Args:
        settings: Non-secret configured and observed runtime metadata.
        generated: Timestamp shared by the Markdown and HTML exports.

    Returns:
        Plain-text labels and values, without usage statistics.
    """
    details = [("Generated", generated)]
    runtimes: dict[tuple[Any, Any, Any], list[str]] = {}
    for runtime in settings.get("runtime_agents") or []:
        identity = (runtime.get("harness"), runtime.get("provider"), runtime.get("model"))
        agents = runtimes.setdefault(identity, [])
        if runtime.get("agent") and runtime["agent"] not in agents:
            agents.append(runtime["agent"])
    for identity, agents in runtimes.items():
        description = _runtime_description(*identity)
        if len(runtimes) > 1 and agents:
            description += f" ({', '.join(agents)})"
        details.append(("Runtime", description))

    if not runtimes or any(not identity[2] for identity in runtimes):
        configured: dict[tuple[Any, Any, Any], list[str]] = {}
        for tier in ("quick", "deep"):
            identity = (settings.get(f"{tier}_harness") or settings.get("harness"),
                        settings.get(f"{tier}_provider"), settings.get(f"{tier}_model"))
            if not runtimes or identity[2]:
                configured.setdefault(identity, []).append(tier)
        for identity, tiers in configured.items():
            label = "Configured runtime" if len(tiers) == 2 else f"Configured {tiers[0]} runtime"
            details.append((label, _runtime_description(*identity)))
    if analysts := settings.get("analysts"):
        details.append(("Analysts", ", ".join("sentiment" if name == "social" else name for name in analysts)))
    rounds = [f"{name}: {settings[f'{name}_rounds']}" for name in ("research", "risk") if f"{name}_rounds" in settings]
    if rounds:
        details.append(("Debate rounds", " · ".join(rounds)))
    vendors = {**(settings.get("data_vendors") or {}), **(settings.get("tool_vendors") or {})}
    if vendors:
        details.append(("Data vendors", " · ".join(f"{name.replace('_', ' ')}: {value}" for name, value in vendors.items())))
    if version := settings.get("version"):
        details.append(("Version", f"Open Trading Agents {version}"))
    return details


def _markdown_value(value: str) -> str:
    """Keep report metadata on one line and escape Markdown formatting characters.

    Args:
        value: Plain-text metadata value.

    Returns:
        An inline Markdown value that cannot add headings or embedded markup.
    """
    return re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", " ".join(value.splitlines()))


def write_transcript(path: Path, events: list[AgentEvent], append: bool = False) -> None:
    """Export ordered events as JSON lines.

    Args:
        path: Local transcript path.
        events: Ordered event batch.
        append: Whether to append final lifecycle events after report export.
    """
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")


def write_reports(
    directory: Path,
    ticker: str,
    trade_date: str,
    outputs: dict[str, str],
    rating: str,
    settings: dict[str, Any],
    events: list[AgentEvent],
    include_html: bool = True,
) -> Path:
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
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    details = report_details(settings, generated)
    header = [f"# Trading Analysis Report: {_markdown_value(ticker)}", "",
              f"- Analysis date: {_markdown_value(trade_date)}", f"- Rating: **{_markdown_value(rating)}**"]
    header.extend(f"- {label}: {_markdown_value(value)}" for label, value in details)
    parts = _report_parts(outputs, events)
    sections = []
    usage = summarize_usage(events)
    (report_dir / "usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    for heading, folder, entries in parts:
        (report_dir / folder).mkdir(exist_ok=True)
        content = []
        for name, filename, text in entries:
            (report_dir / folder / filename).write_text(text, encoding="utf-8")
            content.append(f"### {_markdown_value(name)}\n\n{text}")
        sections.append(f"## {heading}\n\n" + "\n\n".join(content))
    combined = "\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n"
    target = report_dir / "complete_report.md"
    target.write_text(combined, encoding="utf-8")
    (report_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    write_transcript(directory / "transcript.jsonl", events)
    if include_html:
        page = render_report(ticker, trade_date, rating, details, parts)
        (report_dir / "complete_report.html").write_text(page, encoding="utf-8")
    return target
