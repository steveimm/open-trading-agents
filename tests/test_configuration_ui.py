import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from typer.testing import CliRunner

from open_trading_agents import cli, configuration_ui
from open_trading_agents.config import HarnessSettings, Settings, load_settings, preset_path, save_preset
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice

runner = CliRunner()


def select_choices(monkeypatch: pytest.MonkeyPatch, answers: list[int | None]) -> list[tuple[str, list[str]]]:
    """Supply selections while retaining the native choices presented by each wizard step.

    Args:
        monkeypatch: Scoped selection adapter replacement.
        answers: Zero-based selections, with None accepting the highlighted default.

    Returns:
        The labels and available choices shown by the wizard.
    """
    remaining = iter(answers)
    shown: list[tuple[str, list[str]]] = []

    def choose(label: str, choices: list[str], default: int = 0) -> int:
        """Select one queued answer or abort when the user stops setup."""
        shown.append((label, choices))
        try:
            answer = next(remaining)
        except StopIteration as exc:
            raise typer.Abort() from exc
        return default if answer is None else answer

    monkeypatch.setattr(configuration_ui, "_choose", choose)
    return shown


@pytest.fixture
def wizard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate user storage and expose deterministic native choices without model calls.

    Args:
        tmp_path: Isolated test directory.
        monkeypatch: Scoped attribute replacements.

    Returns:
        The default user configuration path.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))
    catalog = ModelCatalog((ModelChoice("small", "Small model", "local"), ModelChoice("large", "Large model", "local")))
    monkeypatch.setattr(configuration_ui, "discover_models", lambda settings: catalog)
    return tmp_path / ".open-trading-agents" / "config.toml"


def test_one_session_selects_both_tiers_and_saves_a_reusable_preset(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Choose listed models, inherit the working tier, and save a distinct manager tier."""
    shown = select_choices(monkeypatch, [None, 1, 1, None, 1, 1, 2, 0, None])
    result = runner.invoke(cli.app, ["configure"], input="local-research\n")
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness == HarnessSettings(kind="pi", provider="local", model="small")
    assert settings.quick is None
    assert settings.deep == HarnessSettings(kind="pi", provider="local", model="large")
    assert shown[0] == ("Harness", ["pi", "codex", "claude"])
    assert "Small model (small)" in shown[2][1] and "Large model (large)" in shown[2][1]
    assert "Working agents (quick)" in result.output and "Decision managers (deep)" in result.output
    assert preset_path("local-research").exists()
    assert tomllib.loads(wizard.read_text())["default_preset"] == "local-research"
    restored = load_settings(preset="local-research")
    assert restored.harness == settings.harness and restored.deep == settings.deep
    assert wizard.exists()


def test_saving_another_preset_keeps_the_existing_default(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Declining the new default selection leaves the previous named preset selected."""
    previous = save_preset("daily", Settings(harness=HarnessSettings(kind="claude", model="stable-model")))
    previous_contents = previous.read_text()
    wizard.write_text('default_preset = "daily"\n[harness]\nkind = "pi"\n')
    select_choices(monkeypatch, [None, 1, 1, None, None, 0, None])
    result = runner.invoke(cli.app, ["configure"], input="experiment\n")
    assert result.exit_code == 0, result.output
    assert tomllib.loads(wizard.read_text())["default_preset"] == "daily"
    assert previous.read_text() == previous_contents
    assert load_settings(preset="experiment").harness == HarnessSettings(kind="pi", provider="local", model="small")
    assert "Default preset: daily" in result.output


def test_replacing_the_selected_preset_keeps_it_as_default(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Updating the selected named preset preserves its default status and new model choices."""
    save_preset("daily", Settings(harness=HarnessSettings(kind="pi", provider="local", model="small")))
    wizard.write_text('default_preset = "daily"\n[harness]\nkind = "pi"\nprovider = "local"\nmodel = "small"\n')
    select_choices(monkeypatch, [None, None, 2, None, None, 0, 0, None])
    result = runner.invoke(cli.app, ["configure"], input="daily\n")
    assert result.exit_code == 0, result.output
    assert tomllib.loads(wizard.read_text())["default_preset"] == "daily"
    assert load_settings(preset="daily").harness.model == "large"
    assert "Default preset: daily" in result.output


def test_cancel_at_default_preset_selection_writes_nothing(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gather the final preset preference before changing the config or creating a preset."""
    wizard.parent.mkdir(parents=True)
    original = '[harness]\nkind = "pi"\nmodel = "saved-model"\n'
    wizard.write_text(original)
    select_choices(monkeypatch, [None, None, None, None, None, 0])
    result = runner.invoke(cli.app, ["configure"], input="uncommitted\n")
    assert result.exit_code == 1, result.output
    assert wizard.read_text() == original
    assert not preset_path("uncommitted").exists()


def test_native_defaults_are_explicit_choices_and_cancel_preserves_config(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset native selectors explicitly and leave the file intact if setup is abandoned."""
    wizard.parent.mkdir(parents=True)
    original = '# keep this comment\n[harness]\nkind="pi"\nprovider="local"\nmodel="small"\n'
    wizard.write_text(original)
    select_choices(monkeypatch, [None, 0, 0])
    cancelled = runner.invoke(cli.app, ["configure"])
    assert cancelled.exit_code == 1
    assert wizard.read_text() == original
    select_choices(monkeypatch, [None, 0, 0, None, None, 1])
    result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness.provider is None and settings.harness.model is None
    assert "# keep this comment" in wizard.read_text()
    assert not preset_path("unused").exists()


def test_existing_preset_is_not_overwritten_without_confirmation(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep named choices intact when a user declines replacement at the final prompt."""
    original = Settings(harness=HarnessSettings(kind="claude", model="existing-model"))
    path = save_preset("saved", original)
    contents = path.read_text()
    shown = select_choices(monkeypatch, [None, None, None, None, None, 0, 1, 1])
    result = runner.invoke(cli.app, ["configure"], input="saved\n")
    assert result.exit_code == 0, result.output
    assert path.read_text() == contents
    assert wizard.exists()
    assert ("Replace preset 'saved'?", ["Yes", "No"]) in shown


def test_unavailable_catalog_still_allows_custom_models(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native discovery failure does not require users to abandon their configured backend."""
    monkeypatch.setattr(configuration_ui, "discover_models", lambda settings: ModelCatalog(notice="Catalog unavailable."))
    select_choices(monkeypatch, [None, 1, 1, None, None, 1])
    result = runner.invoke(cli.app, ["configure"], input="custom-provider\ncustom-model\n")
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness.provider == "custom-provider"
    assert settings.harness.model == "custom-model"
    assert "Catalog unavailable." in result.output


def test_changing_harness_clears_backend_arguments_in_the_wizard(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching the native CLI cannot carry another harness's command or profile flags."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="pi"\nprovider="local"\nmodel="small"\ncommand=["custom-pi"]\nextra_args=["--thinking","high"]\n')
    shown = select_choices(monkeypatch, [2, 0, None, None, 1])
    result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness == HarnessSettings(kind="claude")
    assert all(label != "Provider" for label, _ in shown)


def test_new_codex_provider_uses_its_own_model_choices(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider change must not present the previous provider's model catalog."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="codex"\nprovider="old-provider"\nmodel="old-model"\n')

    def catalog(settings: HarnessSettings) -> ModelCatalog:
        """Report models scoped to each configured provider."""
        model = "new-model" if settings.provider == "new-provider" else "old-model"
        return ModelCatalog((ModelChoice(model, model, settings.provider),))

    monkeypatch.setattr(configuration_ui, "discover_models", catalog)
    shown = select_choices(monkeypatch, [None, 2, 1, None, None, 1])
    result = runner.invoke(cli.app, ["configure"], input="new-provider\n")
    assert result.exit_code == 0, result.output
    assert load_settings().harness == HarnessSettings(kind="codex", provider="new-provider", model="new-model")
    assert shown[2] == ("Model", ["Native default", "new-model", "Enter a model manually"])


@pytest.mark.parametrize(("keys", "expected"), [("\r", 1), ("\x1b[A\r", 0), ("\x1b[B\r", 2)])
def test_arrow_keys_select_from_the_highlighted_current_value(keys: str, expected: int) -> None:
    """Exercise actual terminal key handling for both directions and the current default."""
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text(keys)
        assert configuration_ui._choose("Harness", ["pi", "codex", "claude"], default=1) == expected


def test_model_choices_can_be_filtered_by_typing() -> None:
    """Find a model within a longer native catalog without entering a numeric index."""
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("cloud\r")
        assert configuration_ui._choose("Model", ["local-small", "cloud-large", "local-large"]) == 1


def test_keyboard_cancellation_keeps_existing_configuration(wizard: Path) -> None:
    """Ctrl-C from the actual selection menu cancels setup without saving anything."""
    wizard.parent.mkdir(parents=True)
    original = '[harness]\nkind="pi"\nmodel="saved-model"\n'
    wizard.write_text(original)
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\x03")
        result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 1, result.output
    assert wizard.read_text() == original
