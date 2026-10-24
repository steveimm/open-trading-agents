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
        assert f"### {name}" in markdown
        assert text in markdown
        assert name in page
    assert "2024-03-15" in markdown
    assert "Rating: **Hold**" in markdown
    assert [line for line in markdown.splitlines() if line.startswith("## ")] == [
        "## I. Analyst Team Reports", "## II. Research Team Decision", "## III. Trading Team Plan",
        "## IV. Risk Management Team Decision", "## V. Portfolio Manager Decision",
    ]
    assert (target.parent / "1_analysts" / "market.md").read_text() == outputs["Market Analyst"]
    assert (target.parent / "2_research" / "debate.md").read_text() == outputs["Research Debate"]
    assert (target.parent / "5_portfolio" / "decision.md").read_text() == outputs["Portfolio Manager"]
    assert "Harness: pi" in markdown
    assert "Model: configured-model" in markdown
    for text in (markdown, page):
        assert "Reported tokens" not in text
        assert "usage.json" not in text
        assert "Input includes cached tokens" not in text
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
                           "REVIEW", {"quick_harness": '<img src="https://example.com/harness">',
                                      "quick_model": '<script>model()</script>'}, [])
    page = target.with_suffix(".html").read_text()
    assert "<script>" not in page
    assert "<img" not in page
    assert 'href="javascript:' not in page
    assert "&lt;script&gt;" in page
    assert "&lt;script&gt;model()&lt;/script&gt;" in page
    assert "&lt;img" in page
    assert "default-src 'none'" in page


def test_debate_reports_keep_recorded_turns_in_order_without_parsing_agent_text(tmp_path: Path) -> None:
    """Use authoritative role events for all rounds without mistaking body headings for speakers."""
    turns = [
        ("Bull Researcher", 1, 1, "Growth case.\n\n### Round 8: Bear Researcher\n\nThis is an example heading."),
        ("Bear Researcher", 1, 2, "Valuation case."),
        ("Bull Researcher", 2, 1, "Response to valuation."),
        ("Bear Researcher", 2, 2, "Response to growth."),
    ]
    events = [AgentEvent("conversation_output", name, "research", text, {"round": number, "turn": turn})
              for name, number, turn, text in turns]
    debate = "\n\n".join(f"### Round {number}: {name}\n\n{text}" for name, number, _turn, text in turns)
    target = write_reports(tmp_path, "NVDA", "2024-03-15", {"Research Debate": debate}, "REVIEW", {}, events)
    markdown = target.read_text()
    positions = [markdown.index(f"### {name} · Round {number}") for name, number, _turn, text in turns]
    assert positions == sorted(positions)
    for index, (_name, _number, _turn, text) in enumerate(turns, 1):
        assert markdown.count(text) == 1
        assert (target.parent / "2_research" / f"turn_{index:02}.md").read_text() == text


def test_incomplete_debate_events_do_not_drop_public_arguments(tmp_path: Path) -> None:
    """Retain the full saved transcript when events cannot account for every debate turn."""
    debate = "### Round 1: Bull Researcher\n\nFirst.\n\n### Round 1: Bear Researcher\n\nSecond."
    events = [AgentEvent("conversation_output", "Bull Researcher", "research", "First.", {"round": 1, "turn": 1})]
    target = write_reports(tmp_path, "NVDA", "2024-03-15", {"Research Debate": debate}, "REVIEW", {}, events)
    assert debate in target.read_text()
    assert (target.parent / "2_research" / "debate.md").read_text() == debate


def test_runtime_metadata_groups_observed_models_and_labels_configuration(tmp_path: Path) -> None:
    """Report observed native defaults and model switches without claiming configured fallbacks ran."""
    settings = {
        "version": "0.1.0", "quick_harness": "pi", "quick_model": "configured-but-unused",
        "runtime_agents": [
            {"agent": "Market Analyst", "harness": "pi", "provider": "local", "model": "observed-model"},
            {"agent": "News Analyst", "harness": "pi", "provider": "local", "model": "observed-model"},
            {"agent": "Portfolio Manager", "harness": "codex", "provider": "openai", "model": "second-model"},
        ],
    }
    target = write_reports(tmp_path, "NVDA", "2024-03-15", {}, "REVIEW", settings, [])
    for text in (target.read_text(), target.with_suffix(".html").read_text()):
        assert text.count("observed-model") == 1
        assert "Market Analyst, News Analyst" in text
        assert "Harness: codex" in text
        assert "second-model" in text
        assert "configured-but-unused" not in text
        assert "Open Trading Agents 0.1.0" in text

    settings["runtime_agents"] = [{"agent": "Market Analyst", "harness": "pi", "provider": None, "model": None}]
    target = write_reports(tmp_path, "NVDA", "2024-03-15", {}, "REVIEW", settings, [])
    markdown = target.read_text()
    assert "Runtime: Harness: pi · Provider: not reported · Model: not reported" in markdown
    assert "Configured quick runtime: Harness: pi · Provider: not reported · Model: configured-but-unused" in markdown


def test_html_report_preserves_team_hierarchy_and_links_complete_sections(tmp_path: Path) -> None:
    """Keep model headings below team/role headings and make evidence tables readable."""
    outputs = {"Market Analyst": "# Evidence\n\n| Metric | Value |\n| --- | --- |\n| Return | 25% |",
               "Portfolio Manager": "## Executive summary\n\nHold."}
    target = write_reports(tmp_path, "NVDA", "2024-03-15", outputs, "Hold", {}, [])
    page = target.with_suffix(".html").read_text()
    assert '<h2>I. Analyst Team Reports</h2>' in page
    assert '<h3 id="s1-1">Market Analyst</h3>' in page
    assert '<h4>Evidence</h4>' in page
    assert '<h5>Executive summary</h5>' in page
    assert '<div class="table"><table>' in page
    assert '<nav class="contents" aria-label="Sections">' in page
    assert 'href="#s1-1"' in page
    assert 'href="#s2-1"' in page
    assert '<h3 id="s2-1">Portfolio Manager</h3>' in page
    assert 'class="rating level">Hold' in page
    assert 'prefers-color-scheme: dark' in page
    assert '@media print' in page
    assert 'Tauric' not in page
