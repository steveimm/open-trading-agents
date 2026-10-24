import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import typer
from typer.testing import CliRunner

from open_trading_agents import cli
from open_trading_agents.analysis import backtest
from open_trading_agents.cli import HarnessName, app, configured
from open_trading_agents.config import HarnessSettings, Portfolio, Position, Settings, default_config_path, save_preset, set_default_preset
from open_trading_agents.ui import app as tui
from open_trading_agents.ui import configuration as configuration_ui

runner = CliRunner()


def test_help_omits_completion_replay_and_analyze() -> None:
    """Expose only the supported user-facing commands and options."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "install-completion" not in result.output
    assert "show-completion" not in result.output
    assert "replay" not in result.output
    assert "analyze" not in result.output
    assert runner.invoke(app, ["replay"]).exit_code == 2
    assert runner.invoke(app, ["analyze"]).exit_code == 2


@pytest.mark.parametrize("prefix", [[], ["backtest"]])
def test_run_commands_reject_removed_preset_argument(prefix: list[str]) -> None:
    """Presets are selected per agent role in startup rather than through a run-level flag."""
    help_result = runner.invoke(app, [*prefix, "--help"])
    assert help_result.exit_code == 0
    assert "--preset" not in help_result.output
    rejected = runner.invoke(app, [*prefix, "--preset", "daily"])
    assert rejected.exit_code == 2
    assert "No such option: --preset" in rejected.output


def test_run_rejects_shared_storage_override() -> None:
    """Keep shared application storage fixed while exposing the per-run output option."""
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "--state-dir" not in help_result.output
    assert "--output-dir" in help_result.output

    rejected = runner.invoke(app, ["--state-dir", "./storage"])
    assert rejected.exit_code == 2
    assert "No such option: --state-dir" in rejected.output


def test_configure_no_longer_selects_a_run_tier() -> None:
    """Configure edits one reusable profile while role selection belongs to analysis startup."""
    help_result = runner.invoke(app, ["configure", "agents", "--help"])
    assert help_result.exit_code == 0
    assert "--tier" not in help_result.output
    assert runner.invoke(app, ["configure", "agents", "--tier", "deep"]).exit_code == 2


@pytest.mark.parametrize("flags,existing_config", [([], False), (["--ticker", "NVDA", "--headless"], True),
                                                  (["--ticker", "NVDA", "--events"], False)])
def test_missing_preset_exits_before_input_settings_or_native_work(
    flags: list[str], existing_config: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh launch explains setup without asking questions, reading config, or creating run files."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True), stdout=SimpleNamespace(isatty=lambda: True)))
    if existing_config:
        path = default_config_path()
        path.parent.mkdir()
        path.write_text("deliberately invalid TOML")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    def unexpected(*args: object, **kwargs: object) -> None:
        """Reject any setup, filesystem-producing workflow, or model lookup before the preset check."""
        raise AssertionError("Work started before requiring a saved preset")

    monkeypatch.setattr(cli.typer, "prompt", unexpected)
    monkeypatch.setattr(cli, "configured", unexpected)
    monkeypatch.setattr(cli, "Workflow", unexpected)
    monkeypatch.setattr(configuration_ui, "discover_models", unexpected)
    result = runner.invoke(app, flags)
    assert result.exit_code == 1, result.output
    assert result.output.strip() == "Run uv run open-trading-agents configure first to create a preset."
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    if existing_config:
        assert default_config_path().read_text() == "deliberately invalid TOML"


def test_maintenance_commands_do_not_require_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep setup, inspection, help, and resuming saved work reachable before a preset exists."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    def unexpected() -> tuple[str, ...]:
        """Reject a preset lookup for commands that do not start a new analysis."""
        raise AssertionError("Maintenance command checked presets")

    monkeypatch.setattr(cli, "list_presets", unexpected)
    for options in (["--help"], ["configure", "--help"], ["configure", "agents", "--help"],
                    ["configure", "data", "--help"], ["backtest", "--help"], ["resume", "--help"], ["doctor"]):
        result = runner.invoke(app, options)
        assert result.exit_code == 0, result.output
    configured_result = runner.invoke(app, ["configure", "agents", "--harness", "pi", "--model", "setup-model"])
    assert configured_result.exit_code == 0, configured_result.output
    directory = tmp_path / "saved-run"
    directory.mkdir()
    resumed = object()
    monkeypatch.setattr(cli, "Workflow", SimpleNamespace(resume=lambda path: resumed))
    monkeypatch.setattr(cli, "run_workflow", lambda workflow, headless, events: resumed)
    assert runner.invoke(app, ["resume", str(directory), "--headless"]).exit_code == 0


@pytest.mark.parametrize("section", ["agents", "data"])
def test_configure_menu_exposes_agent_and_data_editors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str) -> None:
    """Route the top-level setup menu without requiring a preset or creating run files.

    Args:
        tmp_path: Isolated application and configuration location.
        monkeypatch: Scoped setup-menu replacements.
        section: Selected configuration category.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))
    monkeypatch.setattr(configuration_ui, "select_configuration_section", lambda: section)
    received: list[tuple[str, Path]] = []
    monkeypatch.setattr(configuration_ui, "configure_interactively", lambda path: received.append(("agents", path)))
    monkeypatch.setattr(configuration_ui, "configure_data_interactively", lambda path: received.append(("data", path)))
    path = tmp_path / "selected.toml"

    result = runner.invoke(app, ["configure", "--config", str(path)])

    assert result.exit_code == 0, result.output
    assert received == [(section, path)]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("placement", ["parent", "command"])
def test_configure_data_accepts_the_selected_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placement: str) -> None:
    """Expose data setup directly with either supported configuration-option placement.

    Args:
        tmp_path: Isolated configuration location.
        monkeypatch: Scoped data-editor replacement.
        placement: Whether --config appears before or after the data subcommand.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))
    received: list[Path] = []
    monkeypatch.setattr(configuration_ui, "configure_data_interactively", received.append)
    path = tmp_path / "selected.toml"
    args = ["configure", "--config", str(path), "data"] if placement == "parent" else ["configure", "data", "--config", str(path)]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert received == [path]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("placement", ["parent", "command"])
def test_doctor_reports_selected_data_config_without_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placement: str) -> None:
    """Show data-source readiness for the selected file while keeping credential values out of output.

    Args:
        tmp_path: Isolated application and custom configuration location.
        monkeypatch: Scoped home-directory replacement.
        placement: Whether --config appears before or after doctor.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    source = tmp_path / "custom.toml"
    source.write_text('[data]\nfred_api_key = "fake-doctor-private-key"\n')
    before = source.read_bytes()
    args = ["--config", str(source), "doctor"] if placement == "parent" else ["doctor", "--config", str(source)]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert str(source) in result.output
    assert "FRED: Configured" in result.output
    assert "Profiles: Missing" in result.output
    assert "fake-doctor-private-key" not in result.output
    assert source.read_bytes() == before
    assert not (tmp_path / ".open-trading-agents").exists()


@pytest.mark.parametrize("command", ["run", "run-events", "resume", "resume-events", "backtest", "configure"])
@pytest.mark.parametrize("exception", [typer.Abort, KeyboardInterrupt, RuntimeError])
def test_commands_distinguish_cancellation_from_runtime_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    exception: type[BaseException],
) -> None:
    """Keep explicit cancellation at exit 130 while retaining actual failure messages and exit 1.

    Args:
        tmp_path: Isolated storage and an existing directory for resume validation.
        monkeypatch: Scoped workflow and configuration replacements.
        command: Public command whose protected work is interrupted.
        exception: User cancellation or genuine runtime failure.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "list_presets", lambda: ("fixture",))

    def fail(*args: object, **kwargs: object) -> NoReturn:
        """Stop the command before filesystem or native harness work.

        Args:
            args: Command-specific inputs.
            kwargs: Optional native runtime selections.
        """
        raise exception("fixture interruption")

    monkeypatch.setattr(cli, "configured", fail)
    monkeypatch.setattr(cli, "Workflow", SimpleNamespace(resume=fail))
    monkeypatch.setattr(cli, "save_harness_selection", fail)
    arguments = {
        "run": ["--ticker", "NVDA", "--headless"],
        "run-events": ["--ticker", "NVDA", "--events"],
        "resume": ["resume", str(tmp_path), "--headless"],
        "resume-events": ["resume", str(tmp_path), "--events"],
        "backtest": ["backtest", "NVDA", "--start", "2024-03-15", "--end", "2024-03-15"],
        "configure": ["configure", "agents", "--model", "fixture"],
    }
    result = runner.invoke(app, arguments[command])

    if exception is RuntimeError:
        label = {"run": "Analysis", "run-events": "Analysis", "resume-events": "Resume", "configure": "Configuration"}.get(
            command, command.capitalize(),
        )
        assert result.exit_code == 1
        assert f"{label} failed: fixture interruption" in result.output
        assert "Cancelled." not in result.output
    else:
        assert result.exit_code == 130
        assert "Cancelled." in result.stderr and "failed" not in result.output
        assert result.stdout == ""

    assert list(tmp_path.iterdir()) == []


def test_tui_cancellation_preserves_resume_command_and_exit_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave a resumable run notice intact when the terminal UI exits without a result.

    Args:
        tmp_path: Isolated run directory containing spaces.
        monkeypatch: Scoped UI and workflow replacements without native execution.
    """
    settings = Settings(state_dir=tmp_path)
    workflow = SimpleNamespace(settings=settings, ticker="NVDA", asset_type="stock", directory=tmp_path / "saved run", run=object())
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True), stdout=SimpleNamespace(isatty=lambda: True)))
    monkeypatch.setattr(cli, "list_presets", lambda: ("fixture",))
    monkeypatch.setattr(cli, "configured", lambda *args: settings)
    monkeypatch.setattr(cli, "Workflow", lambda *args, **kwargs: workflow)
    monkeypatch.setattr(configuration_ui, "select_research_depth", lambda settings: settings)
    monkeypatch.setattr(configuration_ui, "select_agent_settings", lambda settings: settings)
    monkeypatch.setattr(tui, "TradingApp", lambda *args, **kwargs: SimpleNamespace(run=lambda: None, error=None, result=None))

    result = runner.invoke(app, ["--ticker", "NVDA", "--date", "2024-03-15", "--output-dir", str(tmp_path)])

    assert result.exit_code == 130
    assert "Run cancelled. Resume with:" in result.output
    assert "failed" not in result.output and "interrupted" not in result.output
    command = result.output.split("Resume with: ", 1)[1].strip()
    assert shlex.split(command) == ["open-trading-agents", "resume", str(workflow.directory)]
    assert list(tmp_path.iterdir()) == []


def test_headless_requires_explicit_ticker_before_any_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unattended invocations fail with actionable input instructions."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("available", HarnessSettings())
    result = runner.invoke(app, ["--headless"])
    assert result.exit_code == 2
    assert "provide --ticker" in result.output


def test_explicit_config_keeps_global_application_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load selected model preferences without moving shared storage beside the configuration file."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config" / "settings.toml"
    config.parent.mkdir()
    config.write_text('[harness]\nkind = "pi"\nmodel = "configured-model"\n')

    settings = configured(config, None, None, None)

    assert settings.state_dir == tmp_path / ".open-trading-agents"
    assert settings.harness.model == "configured-model"
    assert settings.harness.provider is None


def test_unknown_harness_is_rejected_by_cli() -> None:
    """Reject a misspelled backend rather than falling back silently."""
    result = runner.invoke(app, ["--ticker", "NVDA", "--harness", "wrong", "--headless"])
    assert result.exit_code == 2
    assert "wrong" in result.output


def test_explicit_model_selection_overrides_configured_tiers(tmp_path: Path) -> None:
    """Explicit CLI choices apply to group defaults and individual agent profiles."""
    config = tmp_path / "settings.toml"
    config.write_text('''[quick]
kind = "pi"
model = "old-quick"
provider = "old-provider"
[deep]
kind = "pi"
model = "old-deep"
[agents.bull]
kind = "claude"
model = "old-bull"
command = ["custom-claude"]
[agents.portfolio_manager]
model = "old-manager"
''')
    settings = configured(config, HarnessName.codex, "new-model", None)
    for deep in (False, True):
        assert settings.model_settings(deep).kind == "codex"
        assert settings.model_settings(deep).model == "new-model"
        assert settings.model_settings(deep).provider is None
    for role in ("bull", "portfolio_manager"):
        assert settings.model_settings(role=role) == HarnessSettings(kind="codex", model="new-model")


def test_portfolio_alias_is_not_mistaken_for_an_absent_position() -> None:
    """Recognize existing holdings after instrument normalization."""
    portfolio = Portfolio(positions=[Position(ticker="BTCUSD", quantity=2)])
    rendered = portfolio.render("BTC-USD")
    assert "No current position" not in rendered
    assert "BTC-USD: 2 units" in rendered


@pytest.mark.parametrize("harness,effort", [(None, "medium"), (HarnessName.pi, None)])
def test_flags_preserve_role_inheritance_after_group_selection(harness: HarnessName | None, effort: str | None) -> None:
    """Apply flags before and after the wizard without fixing inherited fields to the old group.

    Args:
        harness: Optional CLI harness change that must preserve omitted role fields.
        effort: Optional CLI effort override applied to every role.
    """
    settings = Settings(
        harness=HarnessSettings(kind="claude", model="old-worker"),
        agents={"bull": HarnessSettings(effort="high")},
    )
    initial = cli.runtime_overrides(settings, harness, None, None, effort)
    selected = initial.model_copy(update={"quick": HarnessSettings(kind="pi", provider="local", model="new-worker")})
    final = cli.runtime_overrides(selected, harness, None, None, effort)

    expected = HarnessSettings(kind="pi", provider="local", model="new-worker", effort=effort)
    assert final.model_settings(role="bull") == expected
    restored = Settings.model_validate_json(final.model_dump_json())
    assert restored.model_settings(role="bull") == expected


def test_switching_harness_discards_old_backend_specific_configuration(tmp_path: Path) -> None:
    """Changing CLIs cannot retain another harness's executable or provider/model selectors."""
    config = tmp_path / "settings.toml"
    config.write_text('[harness]\nkind="pi"\nmodel="local-model"\nprovider="local-provider"\neffort="high"\ncommand=["special-pi"]\nextra_args=["--thinking","high"]\n')
    settings = configured(config, HarnessName.claude, None, None)
    assert settings.harness.kind == "claude"
    assert settings.harness.model is None
    assert settings.harness.provider is None
    assert settings.harness.effort is None
    assert settings.harness.command == ()
    assert settings.harness.extra_args == ()


def test_partial_tier_inherits_harness_and_survives_saved_run_roundtrip(tmp_path: Path) -> None:
    """Selecting only a deep model does not silently switch its CLI back to pi."""
    config = tmp_path / "settings.toml"
    config.write_text('[harness]\nkind="claude"\nmodel="working-model"\n[deep]\nmodel="manager-model"\n')
    settings = configured(config, None, None, None)
    assert settings.model_settings(True).kind == "claude"
    assert settings.model_settings(True).model == "manager-model"
    restored = Settings.model_validate(settings.model_dump(mode="json"))
    assert restored.model_settings(True) == settings.model_settings(True)


def test_default_profile_overrides_global_models_before_explicit_cli_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the default runtime profile without rewriting the shared configuration."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = tmp_path / ".open-trading-agents" / "config.toml"
    path.parent.mkdir()
    path.write_text('research_rounds=3\n[harness]\nkind="pi"\nmodel="global-model"\n[deep]\nmodel="stale-manager"\n')
    save_preset("claude-work", HarnessSettings(kind="claude", model="preset-model", effort="high"))
    set_default_preset(path, "claude-work")
    original = path.read_text()
    selected = configured(None, None, None, None)
    assert selected.research_rounds == 3
    assert selected.harness == HarnessSettings(kind="claude", model="preset-model", effort="high")
    assert selected.quick is None and selected.deep is None
    overridden = configured(None, HarnessName.pi, "one-run-model", "local", "low")
    assert overridden.harness == HarnessSettings(kind="pi", model="one-run-model", provider="local", effort="low")
    assert path.read_text() == original


def test_cli_flags_override_default_profile_for_both_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve a shared default profile and explicit runtime flags without rewriting the default selection."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("daily", HarnessSettings(model="daily-model", effort="high"))
    path = default_config_path()
    set_default_preset(path, "daily")
    original = path.read_text()
    default = configured(None, None, None, None)
    assert default.model_settings().model == "daily-model"
    assert default.model_settings(True).model == "daily-model"
    assert default.model_settings(True).effort == "high"
    assert set(default.agent_presets.values()) == {"daily"}
    retained = configured(None, None, "daily-model", None)
    assert retained.agent_presets == default.agent_presets
    overridden = configured(None, HarnessName.codex, "one-run-model", "local", "medium")
    for deep in (False, True):
        assert overridden.model_settings(deep) == HarnessSettings(kind="codex", model="one-run-model", provider="local", effort="medium")
    assert overridden.agent_presets == {}
    assert path.read_text() == original


def test_flags_only_remove_preset_labels_for_changed_role_profiles() -> None:
    """Keep a preset label only when explicit flags leave that role's four selectors intact."""
    settings = Settings(
        harness=HarnessSettings(model="worker", effort="low"),
        agents={"bull": HarnessSettings(effort="high")},
        agent_presets={"market": "worker-low", "bull": "worker-high"},
    )

    updated = cli.runtime_overrides(settings, None, None, None, "high")

    assert updated.model_settings(role="market").effort == "high"
    assert updated.model_settings(role="bull").effort == "high"
    assert updated.agent_presets == {"bull": "worker-high"}


@pytest.mark.parametrize("command", ["run", "backtest"])
def test_unattended_runs_use_the_default_profile(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the configured default profile into unattended runs without changing native credentials."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("local", HarnessSettings(kind="pi", provider="local", model="selected-model", effort="high"))
    set_default_preset(default_config_path(), "local")
    received: list[Settings] = []

    def workflow(
        ticker: str,
        trade_date: str,
        settings: Settings,
        portfolio: Portfolio | None,
        *,
        output_dir: Path | None = None,
        config_path: Path | None = None,
    ) -> Settings:
        """Capture a new analysis before any native sessions or data requests start."""
        received.append(settings)
        return settings

    async def sweep(tickers: list[str], grid: object, settings: Settings,
                    portfolio: Portfolio | None, run_id: str | None, *, config_path: Path | None = None) -> SimpleNamespace:
        """Capture a new backtest before any native sessions or data requests start."""
        received.append(settings)
        return SimpleNamespace(summary="Saved fixture", directory=tmp_path, failures=[])

    monkeypatch.setattr(cli, "Workflow", workflow)
    monkeypatch.setattr(cli, "run_workflow", lambda workflow, headless, events: None)
    monkeypatch.setattr(backtest, "run_backtest", sweep)
    options = ["--ticker", "NVDA", "--headless"] if command == "run" else ["backtest", "NVDA", "--start", "2026-01-02", "--end", "2026-01-02"]
    result = runner.invoke(app, options)
    assert result.exit_code == 0, result.output
    assert len(received) == 1
    assert received[0].harness == HarnessSettings(kind="pi", provider="local", model="selected-model", effort="high")
