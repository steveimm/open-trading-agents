from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from open_trading_agents import cli
from open_trading_agents.config import HarnessSettings, Portfolio, Settings, save_preset
from open_trading_agents.ui import configuration as configuration_ui

runner = CliRunner()


@pytest.fixture
def run_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create isolated preferences and expose an interactive terminal to the CLI."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True), stdout=SimpleNamespace(isatty=lambda: True)))
    path = tmp_path / ".open-trading-agents" / "config.toml"
    path.parent.mkdir()
    path.write_text('research_rounds = 1\nrisk_rounds = 1\n[harness]\nkind = "pi"\nmodel = "base-model"\n')
    save_preset("available", HarnessSettings(kind="pi", model="base-model"))
    return path


def capture_runs(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> list[tuple[str, str, Settings]]:
    """Capture chosen settings before workflow construction can create files or model sessions.

    Args:
        monkeypatch: Scoped replacements.
        calls: Ordered wizard and execution observations.

    Returns:
        Captured ticker, analysis date, and settings for each invocation.
    """
    received: list[tuple[str, str, Settings]] = []

    def workflow(ticker: str, trade_date: str, settings: Settings, portfolio: Portfolio | None) -> Settings:
        """Record the finalized run choices without starting any external work."""
        calls.append("workflow")
        received.append((ticker, trade_date, settings))
        return settings

    monkeypatch.setattr(cli, "Workflow", workflow)
    monkeypatch.setattr(cli, "run_workflow", lambda workflow, headless, events: None)
    return received


def test_interactive_run_asks_ticker_date_depth_and_agents_before_starting(run_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect startup choices in the requested order without modifying global configuration."""
    calls: list[str] = []
    original = run_config.read_text()
    received = capture_runs(monkeypatch, calls)

    def prompt(label: str, default: str | None = None) -> str:
        """Record ticker and date before either selection menu opens."""
        calls.append(label)
        if label == "Ticker":
            return "NVDA"
        assert default == date.today().isoformat()
        return "2024-03-15"

    def depth(settings: Settings) -> Settings:
        """Select medium research before selecting any runtime profiles."""
        calls.append("depth")
        return settings.model_copy(update={"research_rounds": 3, "risk_rounds": 3})

    def agents(settings: Settings) -> Settings:
        """Select different runtime profiles for working and manager roles."""
        calls.append("agents")
        assert settings.research_rounds == 3 and settings.risk_rounds == 3
        return settings.model_copy(update={
            "quick": HarnessSettings(kind="pi", provider="local", model="worker", effort="low"),
            "deep": HarnessSettings(kind="claude", model="manager", effort="high"),
        })

    monkeypatch.setattr(cli.typer, "prompt", prompt)
    monkeypatch.setattr(configuration_ui, "select_research_depth", depth)
    monkeypatch.setattr(configuration_ui, "select_agent_settings", agents)
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert calls == ["Ticker", "Analysis date (YYYY-MM-DD)", "depth", "agents", "workflow"]
    ticker, trade_date, settings = received[0]
    assert ticker == "NVDA" and trade_date == "2024-03-15"
    assert settings.model_settings().model == "worker" and settings.model_settings().effort == "low"
    assert settings.model_settings(True).model == "manager" and settings.model_settings(True).effort == "high"
    assert run_config.read_text() == original
    assert {path.name for path in run_config.parent.iterdir()} == {"config.toml", "presets"}


@pytest.mark.parametrize("answer,expected,errors", [("\n", None, 0), ("not-a-date\n2999-01-01\n2024-03-15\n", "2024-03-15", 2)])
def test_date_prompt_defaults_to_today_and_retries_invalid_values_before_menus(
    run_config: Path, monkeypatch: pytest.MonkeyPatch, answer: str, expected: str | None, errors: int,
) -> None:
    """Accept Enter as today or a valid historical cutoff without allowing malformed or future dates."""
    calls: list[str] = []
    received = capture_runs(monkeypatch, calls)
    monkeypatch.setattr(configuration_ui, "select_research_depth", lambda settings: settings)
    monkeypatch.setattr(configuration_ui, "select_agent_settings", lambda settings: settings)
    result = runner.invoke(cli.app, ["--ticker", "NVDA"], input=answer)
    assert result.exit_code == 0, result.output
    assert received[0][1] == (expected or date.today().isoformat())
    assert result.output.count("Enter a valid date") == errors
    assert date.today().isoformat() in result.output
    assert calls == ["workflow"]


@pytest.mark.parametrize("value", ["invalid", "2024-02-30", "2999-01-01", ""])
def test_invalid_date_flag_is_rejected_before_menus_or_workflow(run_config: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Validate explicit cutoffs before discovering models or creating any run state."""
    calls: list[str] = []
    received = capture_runs(monkeypatch, calls)

    def unexpected(settings: Settings) -> Settings:
        """Reject any downstream interaction before the cutoff is valid."""
        raise AssertionError("Date was not validated before the menus")

    monkeypatch.setattr(configuration_ui, "select_research_depth", unexpected)
    monkeypatch.setattr(configuration_ui, "select_agent_settings", unexpected)
    result = runner.invoke(cli.app, ["--ticker", "NVDA", "--date", value])
    assert result.exit_code == 1
    assert received == [] and calls == []
    assert {path.name for path in run_config.parent.iterdir()} == {"config.toml", "presets"}


def test_cancelling_date_prompt_does_not_create_run_state(run_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An abandoned date question must not advance to runtime selection or create a run."""
    calls: list[str] = []
    received = capture_runs(monkeypatch, calls)

    def unexpected(settings: Settings) -> Settings:
        """Reject later selections after date input has been cancelled."""
        raise AssertionError("Date cancellation reached a later menu")

    monkeypatch.setattr(configuration_ui, "select_research_depth", unexpected)
    result = runner.invoke(cli.app, ["--ticker", "NVDA"], input="")
    assert result.exit_code == 1
    assert received == [] and calls == []
    assert {path.name for path in run_config.parent.iterdir()} == {"config.toml", "presets"}


@pytest.mark.parametrize("round_flags,expected", [(["--research-rounds", "2"], (2, 5)), (["--research-rounds", "2", "--risk-rounds", "4"], (2, 4))])
def test_supplied_ticker_still_selects_depth_and_agents_and_explicit_flags_win(
    run_config: Path, monkeypatch: pytest.MonkeyPatch, round_flags: list[str], expected: tuple[int, int],
) -> None:
    """Skip only the supplied ticker prompt and apply explicit runtime flags after wizard choices."""
    calls: list[str] = []
    received = capture_runs(monkeypatch, calls)

    def prompt(label: str) -> str:
        """Reject any redundant ticker or date question."""
        raise AssertionError(f"Unexpected prompt: {label}")

    def depth(settings: Settings) -> Settings:
        """Choose deep research while retaining explicitly provided runtime defaults."""
        calls.append("depth")
        assert settings.harness.model == "flag-model" and settings.harness.effort == "medium"
        return settings.model_copy(update={"research_rounds": 5, "risk_rounds": 5})

    def agents(settings: Settings) -> Settings:
        """Return distinct choices to verify explicit native flags remain authoritative."""
        calls.append("agents")
        return settings.model_copy(update={
            "quick": HarnessSettings(kind="claude", model="picked-worker", effort="high", command=("custom-claude",)),
            "deep": HarnessSettings(kind="codex", model="picked-manager", effort="high", extra_args=("--profile", "old")),
        })

    monkeypatch.setattr(cli.typer, "prompt", prompt)
    monkeypatch.setattr(configuration_ui, "select_research_depth", depth)
    monkeypatch.setattr(configuration_ui, "select_agent_settings", agents)
    result = runner.invoke(cli.app, [
        "--ticker", "NVDA", "--date", "2026-01-02", "--harness", "pi", "--provider", "local", "--model", "flag-model",
        "--effort", "medium", *round_flags,
    ])
    assert result.exit_code == 0, result.output
    assert calls == ["depth", "agents", "workflow"]
    assert received[0][1] == "2026-01-02"
    settings = received[0][2]
    assert (settings.research_rounds, settings.risk_rounds) == expected
    for deep in (False, True):
        assert settings.model_settings(deep) == HarnessSettings(kind="pi", provider="local", model="flag-model", effort="medium")


def test_cancelling_run_selection_creates_no_run_or_settings_changes(run_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel after depth selection before any workflow, directory, or native session exists."""
    original = run_config.read_text()
    calls: list[str] = []
    received = capture_runs(monkeypatch, calls)
    monkeypatch.setattr(configuration_ui, "select_research_depth", lambda settings: settings.model_copy(update={"research_rounds": 5}))

    def cancel(settings: Settings) -> Settings:
        """Simulate cancellation at either agent selection menu."""
        raise typer.Abort()

    monkeypatch.setattr(configuration_ui, "select_agent_settings", cancel)
    result = runner.invoke(cli.app, ["--ticker", "NVDA", "--date", "2024-03-15"])
    assert result.exit_code == 1
    assert received == [] and calls == []
    assert run_config.read_text() == original
    assert {path.name for path in run_config.parent.iterdir()} == {"config.toml", "presets"}


@pytest.mark.parametrize("mode", ["headless", "events", "stdin-pipe", "stdout-pipe"])
def test_unattended_analysis_never_opens_startup_menus(run_config: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Avoid terminal selection prompts for explicit unattended modes or redirected streams."""
    monkeypatch.setattr(cli, "sys", SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: mode != "stdin-pipe"), stdout=SimpleNamespace(isatty=lambda: mode != "stdout-pipe"),
    ))
    calls: list[str] = []
    received = capture_runs(monkeypatch, calls)

    def unexpected(settings: Settings) -> Settings:
        """Fail if an unattended command asks for interactive selections."""
        raise AssertionError("Unexpected startup menu")

    monkeypatch.setattr(configuration_ui, "select_research_depth", unexpected)
    monkeypatch.setattr(configuration_ui, "select_agent_settings", unexpected)
    flags = [f"--{mode}"] if mode in ("headless", "events") else []
    result = runner.invoke(cli.app, ["--ticker", "NVDA", *flags])
    assert result.exit_code == 0, result.output
    assert calls == ["workflow"]
    assert received[0][1] == date.today().isoformat()
    assert received[0][2].harness.model == "base-model"
    assert received[0][2].research_rounds == 1 and received[0][2].risk_rounds == 1
