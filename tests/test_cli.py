from pathlib import Path

import pytest
from typer.testing import CliRunner

from open_trading_agents.cli import app, configured

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
