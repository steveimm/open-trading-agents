import json
from dataclasses import asdict
from pathlib import Path

from open_trading_agents.domain import AgentEvent
from open_trading_agents.reporting import write_reports


def test_automatic_report_writer_saves_both_formats_under_reports(tmp_path: Path) -> None:
    """Keep the complete research in the documented reports subdirectory."""
    outputs = {
        "Market Analyst": "Dated market evidence and a verified calculation: **25%**.",
        "Research Debate": "Bull: growth.\n\nBear: valuation.",
        "Research Manager": "**Recommendation**: Hold",
        "Trader": "**Action**: Hold",
        "Risk Debate": "Aggressive: upside.\nConservative: downside.\nNeutral: balance.",
        "Portfolio Manager": "**Rating**: Hold\n\n**Executive Summary**: Wait for stronger evidence.",
    }
    settings = {"harness": "pi", "quick_model": "configured-model", "research_rounds": 1, "risk_rounds": 1}
    target = write_reports(tmp_path, "NVDA", "2024-03-15", outputs, "Hold", settings, [])
    assert target == tmp_path / "reports" / "complete_report.md"
    assert not (tmp_path / "complete_report.md").exists()
    markdown = target.read_text()
    page = target.with_suffix(".html").read_text()
    for name, text in outputs.items():
        assert f"## {name}" in markdown
        assert text in markdown
        assert name in page
    assert "2024-03-15" in markdown
    assert "Final rating: **Hold**" in markdown
    assert json.loads((target.parent / "settings.json").read_text()) == settings


def test_report_export_keeps_full_events_and_reconciled_usage(tmp_path: Path) -> None:
    """Display previews cannot shorten the saved source or duplicate token snapshots."""
    output = "prefix" + "X" * 100_000 + "suffix"
    events = [
        AgentEvent("tool_result", "Market Analyst", "analysis", output),
        AgentEvent("usage", "Market Analyst", "analysis", data={"request_id": "one", "input_tokens": 100,
                                                                 "output_tokens": 20, "cache_read_tokens": 40}),
        AgentEvent("usage", "Market Analyst", "analysis", data={"request_id": "one", "input_tokens": 120,
                                                                 "output_tokens": 25, "cache_read_tokens": 50}),
    ]
    target = write_reports(tmp_path, "NVDA", "2024-03-15", {"Portfolio Manager": "Hold"}, "Hold", {}, events)
    restored = [json.loads(line) for line in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert restored == [asdict(event) for event in events]
    usage = json.loads((target.parent / "usage.json").read_text())["totals"]
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 25
    assert usage["cache_read_tokens"] == 50
    assert usage["cache_write_tokens"] is None


def test_html_export_keeps_model_markup_inert(tmp_path: Path) -> None:
    """Prevent model text from introducing scripts, remote images, or JavaScript links."""
    target = write_reports(tmp_path, 'TEST<script>bad()</script>', "2024-03-15",
                           {"Report": '<script>alert(1)</script>\n\n![pixel](https://example.com/pixel.png)\n\n[bad](javascript:alert(1))'},
                           "REVIEW", {}, [])
    page = target.with_suffix(".html").read_text()
    assert "<script>" not in page
    assert "<img" not in page
    assert 'href="javascript:' not in page
    assert "&lt;script&gt;" in page
    assert "default-src 'none'" in page
