from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from open_trading_agents import cli, configuration_ui
from open_trading_agents.config import HarnessSettings, Settings, load_settings, preset_path, save_preset
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice

runner = CliRunner()


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


def test_one_session_selects_both_tiers_and_saves_a_reusable_preset(wizard: Path) -> None:
    """Choose listed models, inherit the working tier, and save a distinct manager tier."""
    result = runner.invoke(cli.app, ["configure"], input="\n2\n2\n\n2\n2\n3\ny\nlocal-research\n")
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness == HarnessSettings(kind="pi", provider="local", model="small")
    assert settings.quick is None
    assert settings.deep == HarnessSettings(kind="pi", provider="local", model="large")
    assert "1. pi" in result.output and "2. codex" in result.output and "3. claude" in result.output
    assert "Small model (small)" in result.output and "Large model (large)" in result.output
    assert "Working agents (quick)" in result.output and "Decision managers (deep)" in result.output
    assert preset_path("local-research").exists()
    restored = load_settings(preset="local-research")
    assert restored.harness == settings.harness and restored.deep == settings.deep
    assert wizard.exists()


def test_native_defaults_are_explicit_choices_and_cancel_preserves_config(wizard: Path) -> None:
    """Reset native selectors explicitly and leave the file intact if setup is abandoned."""
    wizard.parent.mkdir(parents=True)
    original = '# keep this comment\n[harness]\nkind="pi"\nprovider="local"\nmodel="small"\n'
    wizard.write_text(original)
    cancelled = runner.invoke(cli.app, ["configure"], input="\n1\n1\n")
    assert cancelled.exit_code == 1
    assert wizard.read_text() == original
    result = runner.invoke(cli.app, ["configure"], input="\n1\n1\n\n\nn\n")
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness.provider is None and settings.harness.model is None
    assert "# keep this comment" in wizard.read_text()
    assert not preset_path("unused").exists()


def test_existing_preset_is_not_overwritten_without_confirmation(wizard: Path) -> None:
    """Keep named choices intact when a user declines replacement at the final prompt."""
    original = Settings(harness=HarnessSettings(kind="claude", model="existing-model"))
    path = save_preset("saved", original)
    contents = path.read_text()
    result = runner.invoke(cli.app, ["configure"], input="\n\n\n\n\ny\nsaved\nn\nn\n")
    assert result.exit_code == 0, result.output
    assert path.read_text() == contents
    assert wizard.exists()
    assert "Replace preset 'saved'?" in result.output


def test_unavailable_catalog_still_allows_custom_models(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native discovery failure does not require users to abandon their configured backend."""
    monkeypatch.setattr(configuration_ui, "discover_models", lambda settings: ModelCatalog(notice="Catalog unavailable."))
    result = runner.invoke(cli.app, ["configure"], input="\n2\ncustom-provider\n2\ncustom-model\n\n\nn\n")
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness.provider == "custom-provider"
    assert settings.harness.model == "custom-model"
    assert "Catalog unavailable." in result.output


def test_changing_harness_clears_backend_arguments_in_the_wizard(wizard: Path) -> None:
    """Switching the native CLI cannot carry another harness's command or profile flags."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="pi"\nprovider="local"\nmodel="small"\ncommand=["custom-pi"]\nextra_args=["--thinking","high"]\n')
    result = runner.invoke(cli.app, ["configure"], input="3\n1\n\n\nn\n")
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness == HarnessSettings(kind="claude")
    assert "Provider:" not in result.output


def test_new_codex_provider_uses_its_own_model_choices(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider change must not present the previous provider's model catalog."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="codex"\nprovider="old-provider"\nmodel="old-model"\n')

    def catalog(settings: HarnessSettings) -> ModelCatalog:
        """Report models scoped to each configured provider."""
        model = "new-model" if settings.provider == "new-provider" else "old-model"
        return ModelCatalog((ModelChoice(model, model, settings.provider),))

    monkeypatch.setattr(configuration_ui, "discover_models", catalog)
    result = runner.invoke(cli.app, ["configure"], input="\n3\nnew-provider\n2\n\n\nn\n")
    assert result.exit_code == 0, result.output
    assert load_settings().harness == HarnessSettings(kind="codex", provider="new-provider", model="new-model")
    assert "new-model" in result.output
    assert "2. old-model" not in result.output
