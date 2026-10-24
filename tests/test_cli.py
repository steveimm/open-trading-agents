from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from open_trading_agents import backtest, cli
from open_trading_agents.cli import HarnessName, app, configured
from open_trading_agents.config import HarnessSettings, Portfolio, Position, Settings, default_config_path, save_preset, set_default_preset

runner = CliRunner()


def test_help_omits_completion_and_replay() -> None:
    """Expose only the supported user-facing commands and options."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "install-completion" not in result.output
    assert "show-completion" not in result.output
    assert "replay" not in result.output
    assert runner.invoke(app, ["replay"]).exit_code == 2


def test_headless_requires_explicit_ticker_before_any_model_call() -> None:
    """Unattended invocations fail with actionable input instructions."""
    result = runner.invoke(app, ["analyze", "--headless"])
    assert result.exit_code == 2
    assert "provide a ticker" in result.output


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
    result = runner.invoke(app, ["analyze", "NVDA", "--harness", "wrong", "--headless"])
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
    config.write_text('[harness]\nkind="pi"\nmodel="local-model"\nprovider="local-provider"\ncommand=["special-pi"]\nextra_args=["--thinking","high"]\n')
    settings = configured(config, HarnessName.claude, None, None)
    assert settings.harness.kind == "claude"
    assert settings.harness.model is None
    assert settings.harness.provider is None
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


def test_preset_overrides_global_models_before_explicit_cli_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply a named runtime to one invocation without rewriting the shared configuration."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = tmp_path / ".open-trading-agents" / "config.toml"
    path.parent.mkdir()
    original = 'research_rounds=3\n[harness]\nkind="pi"\nmodel="global-model"\n[deep]\nmodel="stale-manager"\n'
    path.write_text(original)
    save_preset("claude-work", Settings(harness=HarnessSettings(kind="claude", model="preset-model")))
    selected = configured(None, None, None, None, "claude-work")
    assert selected.research_rounds == 3
    assert selected.harness == HarnessSettings(kind="claude", model="preset-model")
    assert selected.quick is None and selected.deep is None
    overridden = configured(None, HarnessName.pi, "one-run-model", "local", "claude-work")
    assert overridden.harness == HarnessSettings(kind="pi", model="one-run-model", provider="local")
    assert path.read_text() == original


def test_explicit_preset_and_cli_flags_override_default_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve default, named override, and command-line selectors without rewriting the default selection."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("daily", Settings(harness=HarnessSettings(model="daily-model"), deep=HarnessSettings(model="daily-manager")))
    save_preset("other", Settings(harness=HarnessSettings(kind="claude", model="other-model")))
    path = default_config_path()
    set_default_preset(path, "daily")
    original = path.read_text()
    default = configured(None, None, None, None)
    assert default.model_settings().model == "daily-model"
    assert default.model_settings(True).model == "daily-manager"
    explicit = configured(None, None, None, None, preset="other")
    assert explicit.model_settings().kind == "claude" and explicit.model_settings().model == "other-model"
    overridden = configured(None, HarnessName.codex, "one-run-model", "local")
    for deep in (False, True):
        assert overridden.model_settings(deep) == HarnessSettings(kind="codex", model="one-run-model", provider="local")
    assert path.read_text() == original


@pytest.mark.parametrize("command", ["analyze", "backtest"])
@pytest.mark.parametrize("explicit", [False, True])
def test_new_runs_use_the_requested_preset(command: str, explicit: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route default and explicitly named selections into both execution paths without changing native credentials."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    save_preset("local", Settings(harness=HarnessSettings(kind="pi", provider="local", model="selected-model")))
    set_default_preset(default_config_path(), "local")
    received: list[Settings] = []

    def workflow(ticker: str, trade_date: str, settings: Settings, portfolio: Portfolio | None) -> Settings:
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
    options = ["--headless"] if command == "analyze" else ["--start", "2026-01-02", "--end", "2026-01-02"]
    preset = ["--preset", "local"] if explicit else []
    result = runner.invoke(app, [command, "NVDA", *preset, *options])
    assert result.exit_code == 0, result.output
    assert len(received) == 1
    assert received[0].harness == HarnessSettings(kind="pi", provider="local", model="selected-model")
