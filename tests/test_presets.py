import tomllib
from pathlib import Path

import pytest

from open_trading_agents.config import (
    HarnessSettings,
    Settings,
    application_dir,
    default_config_path,
    list_presets,
    load_preset,
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


def test_presets_store_only_one_agents_four_selectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep presets reusable across roles without copying native commands or application settings."""
    monkeypatch.setenv("HOME", str(tmp_path))
    config = default_config_path()
    config.parent.mkdir()
    config.write_text('state_dir = "./research"\nrisk_rounds = 3\n[harness]\nkind = "codex"\nmodel = "base-model"\n'
                      '[quick]\nmodel = "base-quick"\n[deep]\nmodel = "base-deep"\n[data]\nweb_search_backend = "yahoo"\n')
    selected = HarnessSettings(kind="pi", provider="local", model="working", effort="high",
                               command=("custom-pi",), extra_args=("--offline",))
    path = save_preset("local-research", selected)
    assert path == application_dir() / "presets" / "local-research.toml"
    assert tomllib.loads(path.read_text()) == {"harness": "pi", "provider": "local", "model": "working", "effort": "high"}
    loaded = load_preset("local-research")
    assert loaded == HarnessSettings(kind="pi", provider="local", model="working", effort="high")
    assert loaded.command == () and loaded.extra_args == ()
    current = load_settings()
    assert current.research_rounds == 1 and current.risk_rounds == 3
    assert current.state_dir == config.parent / "research"
    assert current.data == {"web_search_backend": "yahoo"}
    assert load_settings().harness.model == "base-model"
    assert list_presets() == ("local-research",)


def test_presets_preserve_native_resets_and_require_explicit_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native defaults stay distinct from inherited choices and saved presets are not silently overwritten."""
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = HarnessSettings(model=None, provider=None, effort=None)
    path = save_preset("native-managers", settings)
    original = path.read_text()
    with pytest.raises(FileExistsError):
        save_preset("native-managers", HarnessSettings(kind="codex", effort="high"))
    assert path.read_text() == original
    assert tuple(path.parent.iterdir()) == (path,)
    loaded = load_preset("native-managers")
    assert loaded.model is None and loaded.provider is None and loaded.effort is None
    save_preset("native-managers", HarnessSettings(kind="claude"), overwrite=True)
    assert load_preset("native-managers").kind == "claude"


def test_invalid_or_missing_presets_do_not_fall_back_silently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject path-like names and missing selections before a workflow can use unintended models."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("", "../outside", "/tmp/outside", "nested/name", "bad name", ".hidden", "a" * 65):
        with pytest.raises(ValueError, match="Preset names"):
            preset_path(name)
    save_preset("existing", HarnessSettings())
    with pytest.raises(ValueError, match="Unknown preset 'missing'. Available presets: existing"):
        load_preset("missing")
    preset_path("invalid").write_text('state_dir = "../other"\n[harness]\nkind = "pi"\n')
    with pytest.raises(ValueError, match="only harness, provider, model, and effort"):
        load_preset("invalid")
    preset_path("invalid").write_text('harness = "pi"\nmodel = "test"\ncommand = ["custom-pi"]\n')
    with pytest.raises(ValueError, match="only harness, provider, model, and effort"):
        load_preset("invalid")


def test_default_preset_changes_runtime_choices_without_rewriting_base_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the selected default only to new runs and retain editable configuration choices."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = default_config_path()
    path.parent.mkdir()
    path.write_text('# application preferences\nresearch_rounds = 3\n[harness]\nkind = "pi"\nmodel = "base" # saved choice\n'
                    '[deep]\nmodel = "base-manager"\n[data]\nweb_search_backend = "yahoo"\n')
    path.chmod(0o640)
    selected = HarnessSettings(kind="claude", model="default-model", effort="high")
    save_preset("daily", selected)
    raw = set_default_preset(path, "daily")
    assert raw.default_preset == "daily"
    assert raw.harness.model == "base" and raw.deep is not None and raw.deep.model == "base-manager"
    stored = path.read_text()
    assert '# application preferences' in stored and '# saved choice' in stored
    assert path.stat().st_mode & 0o777 == 0o640
    runtime = load_settings(use_default_preset=True)
    assert runtime.harness == selected and runtime.quick is None and runtime.deep is None
    assert runtime.research_rounds == 3 and runtime.data == {"web_search_backend": "yahoo"}
    assert runtime.default_preset == "daily"
    assert load_settings().harness.model == "base"
    assert path.read_text() == stored
    preset_path("daily").unlink()
    restored = Settings.model_validate(runtime.model_dump(mode="json"))
    assert restored.harness == selected


def test_default_preset_can_be_cleared_and_invalid_selection_preserves_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject missing or malformed preset files before persisting a default and allow reverting to base choices."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = default_config_path()
    save_configuration(path, HarnessSettings(model="base-model"), None, None)
    save_preset("daily", HarnessSettings(model="selected-model"))
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


def test_missing_default_can_be_repaired_without_loading_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a missing default while allowing configure to read and replace the saved selection."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = default_config_path()
    path.parent.mkdir()
    path.write_text('default_preset = "removed"\n[harness]\nkind = "pi"\nmodel = "base-model"\n')
    with pytest.raises(ValueError, match="Default preset 'removed'.*configure"):
        load_settings(use_default_preset=True)
    save_preset("replacement", HarnessSettings(kind="codex", model="replacement-model"))
    set_default_preset(path, "replacement")
    runtime = load_settings(use_default_preset=True)
    assert runtime.harness.kind == "codex" and runtime.harness.model == "replacement-model"
    assert runtime.default_preset == "replacement"
    assert load_settings().harness.model == "base-model"
