import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from open_trading_agents import cli, config
from open_trading_agents.config import (
    HarnessSettings,
    Portfolio,
    Position,
    default_portfolio_path,
    load_portfolio,
    save_portfolio,
    save_preset,
)
from open_trading_agents.diagnostics import doctor_checks
from open_trading_agents.ui import configuration as ui

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate portfolio/config storage and expose an interactive terminal.

    Args:
        tmp_path: Temporary home directory.
        monkeypatch: Scoped home and terminal adapters.

    Returns:
        The isolated application directory.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True), stdout=SimpleNamespace(isatty=lambda: True)))
    return tmp_path / ".open-trading-agents"


def menu_answers(monkeypatch: pytest.MonkeyPatch, path: Path, answers: list[str]) -> None:
    """Drive wizard choices and ensure pending edits never reach disk before Save.

    Args:
        monkeypatch: Menu replacement.
        path: Portfolio destination whose initial contents must remain intact.
        answers: Choice labels, with exhaustion cancelling the wizard.
    """
    original = path.read_bytes() if path.exists() else None
    remaining = iter(answers)

    def choose(label: str, choices: list[str], default: int = 0, **kwargs: Any) -> int:
        """Select a displayed row while preserving atomic-save behavior."""
        assert (path.read_bytes() if path.exists() else None) == original
        try:
            answer = next(remaining)
        except StopIteration as exc:
            raise typer.Abort() from exc
        matches = [index for index, choice in enumerate(choices) if choice.split(" · ", 1)[0] == answer]
        assert len(matches) == 1, (answer, label, choices)
        return matches[0]

    monkeypatch.setattr(ui, "_choose", choose)


def test_wizard_creates_edits_and_removes_positions_with_numeric_validation(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save a validated portfolio through the public command without requiring agent presets.

    Args:
        home: Isolated application storage.
        monkeypatch: Keyboard-menu replacement.
    """
    path = default_portfolio_path()
    menu_answers(monkeypatch, path, ["Currency", "KRW", "Available cash", "Add position", "Add position", "NVDA", "Edit", "AAPL", "Remove", "Save"])
    result = runner.invoke(cli.app, ["configure", "portfolio"], input="nan\n10000.123456789\nnvda\nnan\n10\n-1\n120\naapl\n5.5\nunknown\n\n\nunknown\n")

    assert result.exit_code == 0, result.output
    assert json.loads(path.read_text()) == {"cash": 10000.123456789, "currency": "KRW", "positions": [{"ticker": "NVDA", "quantity": 10.0}]}
    assert path.stat().st_mode & 0o777 == 0o600
    assert "Saved portfolio:" in result.output and str(path) in result.output
    assert result.output.count("finite number") == 3
    assert {file.name for file in home.iterdir()} == {"portfolio.json"}


def test_edit_keeps_precise_values_on_enter_and_supports_custom_destination(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Editing an existing fractional holding must not round its prefilled values.

    Args:
        home: Isolated application storage.
        monkeypatch: Keyboard-menu replacement.
    """
    path = home.parent / "custom" / "holdings.json"
    original = Portfolio(currency="USD", cash=12345.6789123, positions=[Position(ticker="BTCUSD", quantity=0.123456789, average_price=98765.4321987)])
    save_portfolio(path, original)
    menu_answers(monkeypatch, path, ["Available cash", "BTC-USD", "Edit", "Save"])
    result = runner.invoke(cli.app, ["configure", "portfolio", "--file", str(path)], input="\n\n\n\n")

    assert result.exit_code == 0, result.output
    assert load_portfolio(path) == original
    assert not default_portfolio_path().exists()
    checked = runner.invoke(cli.app, ["doctor", "--portfolio", str(path)])
    assert checked.exit_code == 0, checked.output
    assert f"Portfolio: Valid: {path}" in checked.output
    assert "98765.4321987" not in checked.output


@pytest.mark.parametrize("existing", [False, True])
def test_cancellation_after_portfolio_editing_preserves_files(home: Path, monkeypatch: pytest.MonkeyPatch, existing: bool) -> None:
    """Discard edits without creating a portfolio or replacing an existing one.

    Args:
        home: Isolated application storage.
        monkeypatch: Menu replacement that cancels after an edit.
        existing: Whether an existing file must be preserved.
    """
    path = default_portfolio_path()
    if existing:
        save_portfolio(path, Portfolio(cash=100))
    original = path.read_bytes() if existing else None
    menu_answers(monkeypatch, path, ["Available cash"])
    result = runner.invoke(cli.app, ["configure", "portfolio"], input="999\n")

    assert result.exit_code == 130
    assert "Cancelled." in result.output and "failed" not in result.output
    assert (path.read_bytes() if path.exists() else None) == original
    assert not list(home.glob(".*.tmp"))


def test_invalid_saved_portfolio_can_be_replaced_only_on_save(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover from invalid JSON without deleting it before the replacement is ready.

    Args:
        home: Isolated application directory.
        monkeypatch: Menu replacement that accepts a replacement draft.
    """
    home.mkdir()
    path = default_portfolio_path()
    path.write_text('{"cash": invalid}')
    menu_answers(monkeypatch, path, ["Yes", "Save"])
    result = runner.invoke(cli.app, ["configure", "portfolio"])

    assert result.exit_code == 0, result.output
    assert load_portfolio(path) == Portfolio(currency="USD", positions=[])


def test_failed_atomic_save_preserves_portfolio_and_cleans_temporary_file(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not lose the previous holdings when the final file replacement fails.

    Args:
        home: Isolated application directory.
        monkeypatch: Filesystem replacement failure.
    """
    path = save_portfolio(default_portfolio_path(), Portfolio(cash=100))
    original = path.read_bytes()

    def fail(*args: object) -> None:
        """Simulate a failed atomic replacement."""
        raise OSError("fixture replacement failure")

    monkeypatch.setattr(config.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failure"):
        save_portfolio(path, Portfolio(cash=200))
    assert path.read_bytes() == original
    assert list(home.iterdir()) == [path]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("mode", ["yes", "no", "headless", "events", "redirected", "explicit"])
def test_run_portfolio_is_opt_in_and_explicit_files_skip_the_question(home: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Pass the selected holdings to the workflow without silently using saved holdings in unattended runs.

    Args:
        home: Isolated application storage.
        monkeypatch: Runtime and menu adapters that avoid all model requests.
        mode: Interactive answer or unattended/explicit-file invocation mode.
    """
    save_preset("daily", HarnessSettings())
    saved = Portfolio(cash=1000, currency="USD", positions=[])
    save_portfolio(default_portfolio_path(), saved)
    explicit = Portfolio(positions=[Position(ticker="NVDA", quantity=4)])
    alternate = save_portfolio(home / "alternate.json", explicit)
    received = []
    questions = []

    def choose(label: str, choices: list[str], default: int = 0) -> int:
        """Record the portfolio question, rejecting unexpected interactive prompts."""
        questions.append(label)
        assert mode in {"yes", "no"}
        assert choices == ["Yes", "No"]
        return 0 if mode == "yes" else 1

    def workflow(ticker: str, trade_date: str, settings: object, portfolio: Portfolio | None, **kwargs: Any) -> object:
        """Capture the holdings passed to the workflow without creating a run."""
        received.append(portfolio)
        return object()

    monkeypatch.setattr(ui, "_choose", choose)
    monkeypatch.setattr(ui, "select_research_depth", lambda settings: settings)
    monkeypatch.setattr(ui, "select_agent_settings", lambda settings: settings)
    monkeypatch.setattr(ui, "select_output_directory", lambda path: None)
    monkeypatch.setattr(cli, "Workflow", workflow)
    monkeypatch.setattr(cli, "run_workflow", lambda *args: None)
    if mode == "redirected":
        monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)
    flags = ["--ticker", "NVDA", "--date", "2024-03-15"]
    flags += ["--portfolio", str(alternate)] if mode == "explicit" else ["--" + mode] if mode in {"headless", "events"} else []
    result = runner.invoke(cli.app, flags)

    assert result.exit_code == 0, result.output
    assert received == [saved if mode == "yes" else explicit if mode == "explicit" else None]
    assert len(questions) == int(mode in {"yes", "no"})
    assert load_portfolio(default_portfolio_path()) == saved
    assert not (home / "runs").exists()


@pytest.mark.parametrize("contents", [None, '{"cash": "private-invalid-value"}', '{"positions": [{"ticker": "NVDA", "quantity": NaN}]}'])
def test_missing_or_invalid_opted_in_portfolio_blocks_run_with_setup_guidance(
    home: Path, monkeypatch: pytest.MonkeyPatch, contents: str | None,
) -> None:
    """Fail before creating a run when the user requests unavailable or invalid holdings.

    Args:
        home: Isolated application storage.
        monkeypatch: Menu and workflow adapters.
        contents: Invalid JSON/schema, or None for an absent portfolio.
    """
    save_preset("daily", HarnessSettings())
    if contents is not None:
        default_portfolio_path().write_text(contents)
    monkeypatch.setattr(ui, "_choose", lambda *args, **kwargs: 0)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        """Reject any workflow construction with an invalid selected portfolio."""
        raise AssertionError("Created a workflow with invalid holdings")

    monkeypatch.setattr(cli, "Workflow", unexpected)
    result = runner.invoke(cli.app, ["--ticker", "NVDA", "--date", "2024-03-15"])
    assert result.exit_code == 1
    assert "configure portfolio" in result.output
    assert "private-invalid-value" not in result.output
    assert not (home / "runs").exists()


@pytest.mark.parametrize("contents,status", [
    (None, "Not configured (optional)"),
    ('{"cash": invalid}', "Invalid portfolio JSON or holdings"),
    ('{"cash": 7654321, "positions": [{"ticker": "NVDA", "quantity": 12345}]}', "Valid:"),
    ('{"positions": [{"ticker": "NVDA", "quantity": Infinity}]}', "Invalid portfolio JSON or holdings"),
])
def test_doctor_checks_portfolio_even_without_config_and_does_not_echo_holdings(home: Path, contents: str | None, status: str) -> None:
    """Validate optional holdings independently of model/data setup without exposing balances.

    Args:
        home: Isolated application storage.
        contents: Portfolio file contents, or None for missing.
        status: Expected diagnostic status.
    """
    if contents is not None:
        home.mkdir()
        default_portfolio_path().write_text(contents)
    checks = dict(doctor_checks(home / "config.toml"))
    assert status in checks["Portfolio"]
    assert not any(value in checks["Portfolio"] for value in ("7654321", "12345", "NVDA"))
    assert not (home / "config.toml").exists()
