import tomllib
from pathlib import Path

import pytest

from open_trading_agents.config import (
    HarnessSettings,
    Settings,
    application_dir,
    default_config_path,
    list_presets,
    load_settings,
    preset_path,
    save_configuration,
    save_preset,
    set_default_preset,
)


def test_global_settings_are_independent_of_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the global config and resolve its relative storage path consistently."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert Settings().state_dir == tmp_path / ".open-trading-agents"
    assert load_settings().state_dir == application_dir()
    path = default_config_path()
    path.parent.mkdir()
    path.write_text('state_dir = "./results"\n[harness]\nkind = "pi"\nmodel = "global-model"\n')
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    settings = load_settings()
    assert settings.harness.model == "global-model"
    assert settings.state_dir == path.parent / "results"


def test_guided_save_preserves_preferences_and_explicit_tier_defaults(tmp_path: Path) -> None:
    """Save one complete setup while keeping data settings, comments, and native arguments."""
    path = tmp_path / "config.toml"
    path.write_text('# research preferences\nrisk_rounds = 2\n[harness]\nkind = "pi"\nmodel = "old" # chosen model\n'
                    '[quick]\nkind = "codex"\nmodel = "old-quick"\n[data]\nweb_search_backend = "yahoo"\n')
    path.chmod(0o640)
    settings = save_configuration(
        path, HarnessSettings(kind="pi", model="working", provider="local", extra_args=("--thinking", "high")),
        None, HarnessSettings(kind="pi"),
    )
    assert settings.quick is None
    assert settings.deep is not None and settings.deep.model is None and settings.deep.provider is None
    assert settings.deep.extra_args == ()
    assert settings.harness.extra_args == ("--thinking", "high")
    assert settings.risk_rounds == 2 and settings.data["web_search_backend"] == "yahoo"
    assert '# research preferences' in path.read_text() and '# chosen model' in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o640


def test_presets_replace_all_harness_tiers_without_changing_other_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore a complete model selection without inheriting stale base models or copying state paths."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = default_config_path()
    config.parent.mkdir()
    config.write_text('state_dir = "./research"\nrisk_rounds = 3\n[harness]\nkind = "codex"\nmodel = "base-model"\n'
                      '[quick]\nmodel = "base-quick"\n[deep]\nmodel = "base-deep"\n[data]\nweb_search_backend = "yahoo"\n')
    selected = Settings(harness=HarnessSettings(kind="pi", provider="local", model="working"),
                        deep=HarnessSettings(kind="claude", model="manager"), state_dir=tmp_path / "unrelated", risk_rounds=1)
    path = save_preset("local-research", selected)
    assert path == application_dir() / "presets" / "local-research.toml"
    assert set(tomllib.loads(path.read_text())) == {"harness", "deep"}
    loaded = load_settings(preset="local-research")
    assert loaded.harness == selected.harness
    assert loaded.quick is None and loaded.model_settings() == selected.harness
    assert loaded.deep == selected.deep
    assert loaded.risk_rounds == 3
    assert loaded.state_dir == config.parent / "research"
    assert loaded.data == {"web_search_backend": "yahoo"}
    assert load_settings().harness.model == "base-model"
    assert list_presets() == ("local-research",)


def test_presets_preserve_native_resets_and_require_explicit_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native defaults stay distinct from inherited choices and saved presets are not silently overwritten."""
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = Settings(harness=HarnessSettings(model="working", provider="local"), deep=HarnessSettings(model=None, provider=None))
    path = save_preset("native-managers", settings)
    original = path.read_text()
    with pytest.raises(FileExistsError):
        save_preset("native-managers", Settings())
    assert path.read_text() == original
    assert tuple(path.parent.iterdir()) == (path,)
    loaded = load_settings(preset="native-managers")
    assert loaded.deep is not None and loaded.deep.model is None and loaded.deep.provider is None
    save_preset("native-managers", Settings(harness=HarnessSettings(kind="claude")), overwrite=True)
    assert load_settings(preset="native-managers").harness.kind == "claude"


def test_invalid_or_missing_presets_do_not_fall_back_silently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject path-like names and missing selections before a workflow can use unintended models."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("", "../outside", "/tmp/outside", "nested/name", "bad name", ".hidden", "a" * 65):
        with pytest.raises(ValueError, match="Preset names"):
            preset_path(name)
    save_preset("existing", Settings())
    with pytest.raises(ValueError, match="Unknown preset 'missing'. Available presets: existing"):
        load_settings(preset="missing")
    preset_path("invalid").write_text('state_dir = "../other"\n[harness]\nkind = "pi"\n')
    with pytest.raises(ValueError, match="harness selections only"):
        load_settings(preset="invalid")


def test_default_preset_changes_runtime_choices_without_rewriting_base_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the selected default only to new runs and retain editable configuration choices."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = default_config_path()
    path.parent.mkdir()
    path.write_text('# application preferences\nresearch_rounds = 3\n[harness]\nkind = "pi"\nmodel = "base" # saved choice\n'
                    '[deep]\nmodel = "base-manager"\n[data]\nweb_search_backend = "yahoo"\n')
    path.chmod(0o640)
    selected = Settings(harness=HarnessSettings(kind="claude", model="default-model"))
    save_preset("daily", selected)
    raw = set_default_preset(path, "daily")
    assert raw.default_preset == "daily"
    assert raw.harness.model == "base" and raw.deep is not None and raw.deep.model == "base-manager"
    stored = path.read_text()
    assert '# application preferences' in stored and '# saved choice' in stored
    assert path.stat().st_mode & 0o777 == 0o640
    runtime = load_settings(use_default_preset=True)
    assert runtime.harness == selected.harness and runtime.quick is None and runtime.deep is None
    assert runtime.research_rounds == 3 and runtime.data == {"web_search_backend": "yahoo"}
    assert runtime.default_preset == "daily"
    assert load_settings().harness.model == "base"
    assert path.read_text() == stored
    preset_path("daily").unlink()
    restored = Settings.model_validate(runtime.model_dump(mode="json"))
    assert restored.harness == selected.harness


def test_default_preset_can_be_cleared_and_invalid_selection_preserves_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject missing or malformed preset files before persisting a default and allow reverting to base choices."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = default_config_path()
    save_configuration(path, HarnessSettings(model="base-model"), None, None)
    save_preset("daily", Settings(harness=HarnessSettings(model="selected-model")))
    set_default_preset(path, "daily")
    original = path.read_text()
    with pytest.raises(ValueError, match="Unknown preset 'missing'"):
        set_default_preset(path, "missing")
    preset_path("invalid").write_text('[harness]\nkind = "unknown"\n')
    with pytest.raises(ValueError):
        set_default_preset(path, "invalid")
    assert path.read_text() == original
    cleared = set_default_preset(path, None)
    assert cleared.default_preset is None
    assert "default_preset" not in tomllib.loads(path.read_text())
    assert load_settings(use_default_preset=True).harness.model == "base-model"


def test_missing_default_has_an_actionable_error_but_explicit_preset_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never silently change the runtime when a default is deleted, while allowing an explicit replacement."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = default_config_path()
    path.parent.mkdir()
    path.write_text('default_preset = "removed"\n[harness]\nkind = "pi"\nmodel = "base-model"\n')
    with pytest.raises(ValueError, match="Default preset 'removed'.*configure.*--preset"):
        load_settings(use_default_preset=True)
    save_preset("replacement", Settings(harness=HarnessSettings(kind="codex", model="replacement-model")))
    explicit = load_settings(preset="replacement", use_default_preset=True)
    assert explicit.harness.kind == "codex" and explicit.harness.model == "replacement-model"
    assert explicit.default_preset == "removed"
    assert load_settings().harness.model == "base-model"
