import os
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from open_trading_agents import cli
from open_trading_agents.config import load_settings, save_harness_selection

runner = CliRunner()


def test_configure_creates_local_settings_without_model_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist a reusable selection through the same command advertised to users."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["configure", "--harness", "pi", "--provider", "local", "--model", "research-model"])
    assert result.exit_code == 0, result.output
    settings = load_settings(tmp_path / "tradingagents.toml")
    assert settings.harness.kind == "pi"
    assert settings.harness.provider == "local"
    assert settings.harness.model == "research-model"
    assert "Saved default: pi / local / research-model" in result.output


def test_configuration_preserves_unrelated_values_comments_and_permissions(tmp_path: Path) -> None:
    """Changing a model must retain data vendor choices and custom native CLI arguments."""
    path = tmp_path / "tradingagents.toml"
    path.write_text('# Research preferences\nresearch_rounds = 3\n\n[harness]\nkind = "pi"\nmodel = "old" # selected model\n'
                    'command = ["custom-pi"]\nextra_args = ["--thinking", "high"]\n\n[data]\ncache_dir = "./cache" # local cache\n')
    path.chmod(0o640)
    settings = save_harness_selection(path, model="new-model")
    assert settings.research_rounds == 3
    assert settings.harness.command == ("custom-pi",)
    assert settings.harness.extra_args == ("--thinking", "high")
    assert settings.data == {"cache_dir": "./cache"}
    assert '# Research preferences' in path.read_text()
    assert 'model = "new-model" # selected model' in path.read_text()
    assert 'cache_dir = "./cache" # local cache' in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o640


def test_switching_default_harness_clears_inherited_backend_choices(tmp_path: Path) -> None:
    """Switch inherited tiers together while retaining explicitly separate native harnesses."""
    path = tmp_path / "tradingagents.toml"
    path.write_text('[harness]\nkind="pi"\nmodel="old-model"\nprovider="old-provider"\n'
                    'command=["custom-pi"]\nextra_args=["--thinking","high"]\n'
                    '[quick]\nmodel="quick-model"\nextra_args=["--thinking","low"]\n'
                    '[deep]\nkind="pi"\nmodel="manager-model"\n')
    settings = save_harness_selection(path, harness="codex", model="new-model")
    for tier in (settings.harness, settings.quick):
        assert tier is not None
        assert tier.kind == "codex"
        assert tier.model == "new-model"
        assert tier.provider is None
        assert tier.command == ()
        assert tier.extra_args == ()
    assert settings.deep is not None
    assert settings.deep.kind == "pi"
    assert settings.deep.model == "manager-model"


def test_tier_can_select_native_defaults_without_inheriting_shared_model(tmp_path: Path) -> None:
    """Empty model/provider selections must reset a tier even when shared values exist."""
    path = tmp_path / "tradingagents.toml"
    path.write_text('[harness]\nkind="pi"\nmodel="working-model"\nprovider="local"\n')
    result = runner.invoke(cli.app, ["configure", "--config", str(path), "--tier", "deep", "--model", "", "--provider", ""])
    assert result.exit_code == 0, result.output
    settings = load_settings(path)
    assert settings.harness.model == "working-model"
    assert settings.deep is not None
    assert settings.deep.kind == "pi"
    assert settings.deep.model is None
    assert settings.deep.provider is None
    assert tomllib.loads(path.read_text())["deep"] == {"model": "", "provider": ""}


def test_tier_harness_switch_to_shared_kind_uses_native_defaults(tmp_path: Path) -> None:
    """An explicit backend switch discards stale arguments and selectors."""
    path = tmp_path / "tradingagents.toml"
    path.write_text('[harness]\nkind="codex"\nmodel="shared-model"\nprovider="shared-provider"\n'
                    'command=["custom-codex"]\nextra_args=["--profile","work"]\n'
                    '[deep]\nkind="pi"\nmodel="old-model"\ncommand=["custom-pi"]\n')
    settings = save_harness_selection(path, tier="deep", harness="codex")
    assert settings.deep is not None
    assert settings.deep.kind == "codex"
    assert settings.deep.model is None
    assert settings.deep.provider is None
    assert settings.deep.command == ()
    assert settings.deep.extra_args == ()
    assert settings.harness.model == "shared-model"


def test_interactive_configuration_uses_native_defaults_for_blank_answers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt accepts a harness choice and clearly resets blank model/provider choices."""
    path = tmp_path / "tradingagents.toml"
    path.write_text('[harness]\nkind="pi"\nmodel="old-model"\nprovider="old-provider"\n')
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))
    result = runner.invoke(cli.app, ["configure", "--config", str(path)], input="codex\n\n\n")
    assert result.exit_code == 0, result.output
    settings = load_settings(path)
    assert settings.harness.kind == "codex"
    assert settings.harness.model is None
    assert settings.harness.provider is None
    assert "blank uses native default" in result.output


def test_noninteractive_configuration_requires_selection_flags(tmp_path: Path) -> None:
    """Unattended invocations provide actionable guidance instead of waiting for input."""
    path = tmp_path / "tradingagents.toml"
    result = runner.invoke(cli.app, ["configure", "--config", str(path)])
    assert result.exit_code == 2
    assert "provide --harness, --model or --provider" in result.output
    assert not path.exists()


def test_invalid_configuration_is_not_overwritten(tmp_path: Path) -> None:
    """Validate every existing application setting before replacing its configuration."""
    path = tmp_path / "tradingagents.toml"
    original = 'risk_rounds = 0\n[harness]\nkind="pi"\nmodel="old"\n'
    path.write_text(original)
    result = runner.invoke(cli.app, ["configure", "--config", str(path), "--model", "new"])
    assert result.exit_code == 1
    assert "risk_rounds" in result.output
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


def test_failed_atomic_replace_retains_original_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write failure leaves the original file intact and removes the temporary file."""
    path = tmp_path / "tradingagents.toml"
    original = '[harness]\nkind="pi"\nmodel="old"\n'
    path.write_text(original)
    monkeypatch.setattr(os, "replace", Mock(side_effect=OSError("write unavailable")))
    with pytest.raises(OSError, match="write unavailable"):
        save_harness_selection(path, model="new")
    assert path.read_text() == original
    assert list(tmp_path.iterdir()) == [path]


def test_claude_provider_configuration_is_left_to_native_harness(tmp_path: Path) -> None:
    """Reject an ineffective provider flag instead of saving it as an applied selection."""
    path = tmp_path / "tradingagents.toml"
    result = runner.invoke(cli.app, ["configure", "--config", str(path), "--harness", "claude", "--provider", "local"])
    assert result.exit_code == 1
    assert "Claude Code selects its provider" in result.output
    assert not path.exists()
