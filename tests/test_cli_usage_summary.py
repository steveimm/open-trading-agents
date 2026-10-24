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
def test_completed_run_reports_cache_and_actual_usage_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cache: dict[str, Any],
    expected: str,
) -> None:
    """Display persisted counts faithfully and link details to the actual run directory.

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
    assert f"Usage details: {usage}" in output


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
