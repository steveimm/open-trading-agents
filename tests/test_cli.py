from pathlib import Path

import pytest
from typer.testing import CliRunner

from open_trading_agents.cli import HarnessName, app, configured
from open_trading_agents.config import Portfolio, Position

runner = CliRunner()


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
