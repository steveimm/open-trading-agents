import json
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from open_trading_agents import cli
from open_trading_agents.analysis.models import AgentEvent, EventSink
from open_trading_agents.analysis.workflow import RunResult


@pytest.mark.parametrize(
    "cache,expected",
    [
        (
            {"cache_read_tokens": 6_900_000, "cache_read_tokens_complete": True,
             "cache_write_tokens": 0, "cache_write_tokens_complete": True},
            "Cache read: 6,900,000 / write: 0.",
        ),
        (
            {"cache_read_tokens": 300, "cache_read_tokens_complete": False,
             "cache_write_tokens": 12, "cache_write_tokens_complete": False},
            "Cache read: 300+? / write: 12+?.",
        ),
        ({"cache_read_tokens": 0, "cache_read_tokens_complete": True}, "Cache read: 0."),
        ({"cache_read_tokens": None, "cache_write_tokens": None}, "Cache read: —."),
        ({}, "Cache read: —."),
    ],
)
def test_completed_run_reports_cache_and_saved_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cache: dict[str, Any],
    expected: str,
) -> None:
    """Display persisted counts and the report directory without redundant run or usage-file lines.

    Args:
        tmp_path: Isolated output directory.
        capsys: Captured completion output.
        cache: Reported cache totals and their completeness flags.
        expected: Expected cache summary, including explicit zero or partial data.
    """
    directory = tmp_path / "custom-output" / "run-id"
    reports = directory / "reports"
    reports.mkdir(parents=True)
    usage = reports / "usage.json"
    usage.write_text(json.dumps({"totals": {"input_tokens": 7_444_519, "output_tokens": 463_423, **cache}}))
    result = RunResult("run-id", directory, "Hold", {})

    async def run(emit: EventSink) -> RunResult:
        """Return a completed fixture without executing a model.

        Args:
            emit: Unused event observer for this completed fixture.

        Returns:
            The saved run result.
        """
        return result

    workflow = SimpleNamespace(run=run, ticker="NVDA", directory=directory, run_id="run-id")
    assert cli.run_workflow(workflow, headless=True, events=False) is result

    output = capsys.readouterr().out
    summary = next(line for line in output.splitlines() if line.startswith("Reported tokens:"))
    assert summary == f"Reported tokens: 7,444,519 input / 463,423 output. {expected}"
    assert f"Run saved to {reports}" in output
    assert not any(line.startswith(("Reports:", "Run:", "Usage details:")) for line in output.splitlines())


def test_json_event_mode_does_not_append_human_usage_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Keep event output parseable without requiring or printing the human usage report.

    Args:
        tmp_path: Isolated run directory with no usage file.
        capsys: Captured event output.
    """
    result = RunResult("events-only", tmp_path, "Hold", {})

    async def run(emit: EventSink) -> RunResult:
        """Emit one completion event without creating any report files.

        Args:
            emit: Observer receiving the completion event.

        Returns:
            The synthetic completed run.
        """
        await emit(AgentEvent("run_end", "Workflow", "reports"))
        return result

    workflow = SimpleNamespace(run=run, ticker="NVDA", directory=tmp_path, run_id="events-only")
    cli.run_workflow(workflow, headless=True, events=True)

    output = capsys.readouterr().out
    assert json.loads(output)["kind"] == "run_end"
    assert "Reported tokens:" not in output and "Usage details:" not in output


@pytest.fixture
def completed_interactive_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[SimpleNamespace, RunResult]:
    """Provide a completed TUI run with saved reports and no live model or browser activity.

    Args:
        tmp_path: Isolated output directory, including spaces to exercise file URIs.
        monkeypatch: Terminal and TUI adapters.

    Returns:
        Workflow stand-in and its completed result.
    """
    from open_trading_agents.config import Settings
    from open_trading_agents.ui import app as tui
    from open_trading_agents.ui import configuration as configuration_ui

    directory = tmp_path / "run with spaces"
    reports = directory / "reports"
    reports.mkdir(parents=True)
    (reports / "usage.json").write_text(json.dumps({"totals": {"input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 80}}))
    (reports / "complete_report.html").write_text("<!doctype html><title>Saved report</title>")
    (reports / "complete_report.md").write_text("# Complete report\n\n**Evidence**\n\n" + "Report evidence.\n\n" * 100 + "\x1b[31mEND COMPLETE REPORT\x1b[0m\n")
    result = RunResult("fixture-run", directory, "Underweight", {})

    async def run(emit: EventSink) -> RunResult:
        """Return saved work for unattended-mode assertions without opening sessions.

        Args:
            emit: Event observer for the run.

        Returns:
            The already-completed result.
        """
        await emit(AgentEvent("run_end", "Workflow", "reports"))
        return result

    def unexpected(*args: Any, **kwargs: Any) -> None:
        """Reject unmocked prompts or browser launches."""
        raise AssertionError("Unexpected interactive action")

    workflow = SimpleNamespace(run=run, settings=Settings(analysts=("market",)), ticker="NFLX", asset_type="stock", directory=directory)
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True), stdout=SimpleNamespace(isatty=lambda: True)))
    monkeypatch.setattr(tui, "TradingApp", lambda *args, **kwargs: SimpleNamespace(run=lambda: None, result=result, error=None))
    monkeypatch.setattr(configuration_ui, "_confirm", unexpected)
    monkeypatch.setattr(webbrowser, "open_new_tab", unexpected)
    return workflow, result


def test_interactive_completion_has_no_viewing_prompts(
    completed_interactive_run: tuple[SimpleNamespace, RunResult], capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit directly with the saved result and summary, without optional report prompts."""
    workflow, result = completed_interactive_run
    assert cli.run_workflow(workflow, headless=False, events=False) is result
    output = capsys.readouterr().out
    assert "Run saved to" in output
    assert "END COMPLETE REPORT" not in output


@pytest.mark.parametrize("mode", ["headless", "events", "redirected"])
def test_noninteractive_completion_never_asks_to_view_reports(
    completed_interactive_run: tuple[SimpleNamespace, RunResult], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], mode: str,
) -> None:
    """Keep unattended output usable without launching a browser or waiting for input.

    Args:
        completed_interactive_run: Saved fixture with prompts and browser launches forbidden.
        monkeypatch: Redirected output adapter.
        capsys: Captured completion or event output.
        mode: Invocation mode that must remain non-interactive.
    """
    workflow, result = completed_interactive_run
    if mode == "redirected":
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    assert cli.run_workflow(workflow, headless=mode == "headless", events=mode == "events") is result
    output = capsys.readouterr().out
    if mode == "events":
        assert json.loads(output)["kind"] == "run_end"
    else:
        assert "Run saved to" in output
