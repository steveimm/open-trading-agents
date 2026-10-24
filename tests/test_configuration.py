import os
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from open_trading_agents import cli
from open_trading_agents.config import HarnessSettings, Settings, load_preset, load_settings, preset_path, save_harness_selection

runner = CliRunner()


def test_configure_creates_user_settings_without_model_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist a reusable selection through the same command advertised to users."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    result = runner.invoke(cli.app, ["configure", "--harness", "pi", "--provider", "local", "--model", "research-model"])
    assert result.exit_code == 0, result.output
    settings = load_settings(tmp_path / ".open-trading-agents" / "config.toml")
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
    settings = save_harness_selection(path, tier="deep", model="", provider="")
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


def test_interactive_configuration_keeps_highlighted_current_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirming each highlighted choice keeps all configured native selectors intact."""
    from open_trading_agents.harnesses.catalog import ModelCatalog
    from open_trading_agents.ui import configuration as configuration_ui

    path = tmp_path / "tradingagents.toml"
    path.write_text('[harness]\nkind="pi"\nmodel="old-model"\nprovider="old-provider"\n')
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))
    monkeypatch.setattr(configuration_ui, "discover_models", lambda settings: ModelCatalog(()))
    monkeypatch.setattr(configuration_ui, "effort_choices", lambda settings, catalog: ())
    monkeypatch.setattr(configuration_ui, "_choose", lambda label, choices, default=0: default)
    result = runner.invoke(cli.app, ["configure", "--config", str(path)], input="retained-profile\n")
    assert result.exit_code == 0, result.output
    settings = load_settings(path)
    assert settings.harness.kind == "pi"
    assert settings.harness.model == "old-model"
    assert settings.harness.provider == "old-provider"
    assert settings.quick is None and settings.deep is None
    assert "Working agents" not in result.output
    assert "Decision manager" not in result.output


def test_noninteractive_configuration_requires_selection_flags(tmp_path: Path) -> None:
    """Unattended invocations provide actionable guidance instead of waiting for input."""
    path = tmp_path / "tradingagents.toml"
    result = runner.invoke(cli.app, ["configure", "--config", str(path)])
    assert result.exit_code == 2
    assert "provide --harness" in result.output and "--effort" in result.output
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


def test_claude_provider_override_is_rejected_by_all_configuration_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the native-provider restriction to files, profiles, and run flags.

    Args:
        tmp_path: Isolated configuration storage.
        monkeypatch: Redirect the profile directory away from user settings.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    invalid = {"kind": "claude", "provider": "ignored-provider"}
    path = tmp_path / "config.toml"
    path.write_text('[harness]\nkind = "claude"\nprovider = "ignored-provider"\n')
    profile = preset_path("invalid")
    profile.parent.mkdir(parents=True)
    profile.write_text('harness = "claude"\nprovider = "ignored-provider"\n')

    with pytest.raises(ValueError, match="Claude Code selects its provider"):
        HarnessSettings.model_validate(invalid)
    with pytest.raises(ValueError, match="Claude Code selects its provider"):
        load_settings(path)
    with pytest.raises(ValueError, match="Claude Code selects its provider"):
        load_preset("invalid")
    with pytest.raises(ValueError, match="Claude Code selects its provider"):
        cli.runtime_overrides(Settings(), cli.HarnessName.claude, "selected-model", "ignored-provider")

    assert HarnessSettings(kind="claude", provider="  ").provider is None
