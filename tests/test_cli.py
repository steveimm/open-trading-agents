from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from open_trading_agents import cli
from open_trading_agents.analysis import backtest
from open_trading_agents.cli import HarnessName, app, configured
from open_trading_agents.config import HarnessSettings, Portfolio, Position, Settings, default_config_path, save_preset, set_default_preset
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


def test_configure_no_longer_selects_a_run_tier() -> None:
    """Configure edits one reusable profile while role selection belongs to analysis startup."""
    help_result = runner.invoke(app, ["configure", "--help"])
    assert help_result.exit_code == 0
    assert "--tier" not in help_result.output
    assert runner.invoke(app, ["configure", "--tier", "deep"]).exit_code == 2


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
    for options in (["--help"], ["configure", "--help"], ["backtest", "--help"], ["resume", "--help"], ["doctor"]):
        result = runner.invoke(app, options)
        assert result.exit_code == 0, result.output
    configured_result = runner.invoke(app, ["configure", "--harness", "pi", "--model", "setup-model"])
    assert configured_result.exit_code == 0, configured_result.output
    directory = tmp_path / "saved-run"
    directory.mkdir()
    resumed = object()
    monkeypatch.setattr(cli, "Workflow", SimpleNamespace(resume=lambda path: resumed))
    monkeypatch.setattr(cli, "run_workflow", lambda workflow, headless, events: resumed)
    assert runner.invoke(app, ["resume", str(directory), "--headless"]).exit_code == 0


def test_headless_requires_explicit_ticker_before_any_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unattended invocations fail with actionable input instructions."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("available", HarnessSettings())
    result = runner.invoke(app, ["--headless"])
    assert result.exit_code == 2
    assert "provide --ticker" in result.output


def test_config_resolves_storage_relative_to_its_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load fresh app settings without inheriting reference-project preferences."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config" / "settings.toml"
    config.parent.mkdir()
    config.write_text('state_dir = "./results"\n[harness]\nkind = "pi"\nmodel = "configured-model"\n')
    settings = configured(config, None, None, None)
    assert settings.state_dir.resolve() == config.parent / "results"
    assert settings.harness.model == "configured-model"
    assert settings.harness.provider is None


def test_unknown_harness_is_rejected_by_cli() -> None:
    """Reject a misspelled backend rather than falling back silently."""
    result = runner.invoke(app, ["--ticker", "NVDA", "--harness", "wrong", "--headless"])
    assert result.exit_code == 2
    assert "wrong" in result.output


def test_explicit_model_selection_overrides_configured_tiers(tmp_path: Path) -> None:
    """An explicit CLI backend choice applies to manager and working-agent tiers."""
    config = tmp_path / "settings.toml"
    config.write_text('[quick]\nkind="pi"\nmodel="old-quick"\nprovider="old-provider"\n[deep]\nkind="pi"\nmodel="old-deep"\n')
    settings = configured(config, HarnessName.codex, "new-model", None)
    for deep in (False, True):
        assert settings.model_settings(deep).kind == "codex"
        assert settings.model_settings(deep).model == "new-model"
        assert settings.model_settings(deep).provider is None


def test_portfolio_alias_is_not_mistaken_for_an_absent_position() -> None:
    """Recognize existing holdings after instrument normalization."""
    portfolio = Portfolio(positions=[Position(ticker="BTCUSD", quantity=2)])
    rendered = portfolio.render("BTC-USD")
    assert "No current position" not in rendered
    assert "BTC-USD: 2 units" in rendered


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
    overridden = configured(None, HarnessName.codex, "one-run-model", "local", "medium")
    for deep in (False, True):
        assert overridden.model_settings(deep) == HarnessSettings(kind="codex", model="one-run-model", provider="local", effort="medium")
    assert path.read_text() == original


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
    ) -> Settings:
        """Capture a new analysis before any native sessions or data requests start."""
        received.append(settings)
        return settings

    async def sweep(tickers: list[str], grid: object, settings: Settings,
                    portfolio: Portfolio | None, run_id: str | None) -> SimpleNamespace:
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
