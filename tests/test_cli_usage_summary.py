import json
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
    monkeypatch.setattr(cli.webbrowser, "open_new_tab", unexpected)
    return workflow, result


@pytest.mark.parametrize("open_browser,print_terminal", [(True, True), (True, False), (False, True), (False, False)])
def test_completion_offers_browser_then_full_terminal_report(
    completed_interactive_run: tuple[SimpleNamespace, RunResult], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], open_browser: bool, print_terminal: bool,
) -> None:
    """Independently honor both choices, render the complete Markdown, and keep terminal controls inert.

    Args:
        completed_interactive_run: Saved fixture and TUI result.
        monkeypatch: Prompt and browser adapters.
        capsys: Captured completion and rendered report output.
        open_browser: Whether to open the exported HTML.
        print_terminal: Whether to render the exported Markdown.
    """
    from open_trading_agents.ui import configuration as configuration_ui

    workflow, result = completed_interactive_run
    actions = []
    answers = iter((open_browser, print_terminal))

    def confirm(label: str, default: bool = False) -> bool:
        """Record prompts in the same sequence as their actions."""
        actions.append(label)
        return next(answers)

    def open_page(uri: str) -> bool:
        """Capture the correctly escaped saved report URI."""
        actions.append(uri)
        return True

    monkeypatch.setattr(configuration_ui, "_confirm", confirm)
    monkeypatch.setattr(cli.webbrowser, "open_new_tab", open_page)
    assert cli.run_workflow(workflow, headless=False, events=False) is result

    expected = ["Open report in browser?"]
    if open_browser:
        expected.append((result.directory / "reports" / "complete_report.html").as_uri())
    expected.append("Print full report in terminal?")
    assert actions == expected
    output = capsys.readouterr().out
    assert f"Run saved to {result.directory / 'reports'}" in output
    assert "Usage details:" not in output and "Run: fixture-run" not in output
    assert ("END COMPLETE REPORT" in output) is print_terminal
    assert "\x1b" not in output


@pytest.mark.parametrize("failure", [False, OSError("fixture browser error"), cli.webbrowser.Error("fixture browser error")])
def test_browser_failure_does_not_prevent_terminal_report_or_fail_completed_run(
    completed_interactive_run: tuple[SimpleNamespace, RunResult], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], failure: bool | Exception,
) -> None:
    """Keep viewing failures separate from successful analysis and still offer the second view.

    Args:
        completed_interactive_run: Saved fixture and TUI result.
        monkeypatch: Prompt and browser adapters.
        capsys: Captured failure notice and terminal report.
        failure: Failed launch return value or launch exception.
    """
    from open_trading_agents.ui import configuration as configuration_ui

    workflow, result = completed_interactive_run
    monkeypatch.setattr(configuration_ui, "_confirm", lambda *args, **kwargs: True)

    def open_page(uri: str) -> bool:
        """Simulate an unavailable browser without starting a process."""
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(cli.webbrowser, "open_new_tab", open_page)
    assert cli.run_workflow(workflow, headless=False, events=False) is result
    output = capsys.readouterr().out
    assert "Could not open a browser." in output and "complete_report.html" in output
    assert "END COMPLETE REPORT" in output


@pytest.mark.parametrize("cancel_at", [1, 2])
def test_cancelling_report_prompts_keeps_successful_result(
    completed_interactive_run: tuple[SimpleNamespace, RunResult], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], cancel_at: int,
) -> None:
    """Cancelling optional viewing must not label the already-completed research as cancelled.

    Args:
        completed_interactive_run: Saved fixture and TUI result.
        monkeypatch: Prompt interruption adapter.
        capsys: Captured completion summary.
        cancel_at: Viewing question interrupted by the user.
    """
    import typer

    from open_trading_agents.ui import configuration as configuration_ui

    workflow, result = completed_interactive_run
    count = 0

    def confirm(label: str, default: bool = False) -> bool:
        """Interrupt the selected question and decline any earlier question."""
        nonlocal count
        count += 1
        if count == cancel_at:
            raise typer.Abort()
        return False

    monkeypatch.setattr(configuration_ui, "_confirm", confirm)
    assert cli.run_workflow(workflow, headless=False, events=False) is result
    output = capsys.readouterr().out
    assert "Run saved to" in output and "cancelled" not in output.lower()
    assert (result.directory / "reports" / "complete_report.md").exists()


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
