import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from typer.testing import CliRunner

from open_trading_agents import cli
from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.config import HarnessSettings, Settings, load_preset, load_settings, preset_path, save_preset
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice
from open_trading_agents.ui import configuration as configuration_ui

runner = CliRunner()


def select_choices(monkeypatch: pytest.MonkeyPatch, answers: list[int | None]) -> list[tuple[str, list[str]]]:
    """Supply selections while retaining the choices presented by each wizard step.

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
    """Isolate user storage and provide native model and effort choices without generation.

    Args:
        tmp_path: Isolated test directory.
        monkeypatch: Scoped attribute replacements.

    Returns:
        The default user configuration path.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True), stdout=SimpleNamespace(isatty=lambda: True)))
    catalog = ModelCatalog((ModelChoice("small", "Small model", "local"), ModelChoice("large", "Large model", "local")))
    monkeypatch.setattr(configuration_ui, "discover_models", lambda settings: catalog)
    monkeypatch.setattr(configuration_ui, "effort_choices", lambda settings, catalog: ("low", "high"))
    return tmp_path / ".open-trading-agents" / "config.toml"


def test_configure_saves_one_profile_with_effort_and_no_role_assignments(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create a reusable single-agent preset without configuring the two runtime role groups."""
    shown = select_choices(monkeypatch, [None, 1, 1, 2, 0, None])
    result = runner.invoke(cli.app, ["configure"], input="local-research\n")
    assert result.exit_code == 0, result.output
    profile = HarnessSettings(kind="pi", provider="local", model="small", effort="high")
    assert load_settings().harness == profile
    assert load_settings().quick is None and load_settings().deep is None
    assert load_preset("local-research") == profile
    assert tomllib.loads(preset_path("local-research").read_text()) == {
        "harness": "pi", "provider": "local", "model": "small", "effort": "high",
    }
    assert tomllib.loads(wizard.read_text())["default_preset"] == "local-research"
    assert [label for label, _ in shown[:4]] == ["Harness", "Provider", "Model", "Reasoning effort"]
    assert "Working agents" not in result.output and "Decision manager" not in result.output


@pytest.mark.parametrize("confirmation", [None, "cancel"])
def test_delete_profile_requires_confirmation_and_does_not_discover_models(
    wizard: Path, monkeypatch: pytest.MonkeyPatch, confirmation: str | None,
) -> None:
    """Reject or cancel deletion without changing the saved profile or its default selection."""
    path = save_preset("daily", HarnessSettings(kind="pi", model="saved", effort="high"))
    wizard.write_text('default_preset = "daily"\n[harness]\nkind = "pi"\n')
    before = {file: file.read_bytes() for file in (wizard, path)}

    def unexpected(settings: HarnessSettings) -> ModelCatalog:
        """Deletion must not start native model discovery."""
        raise AssertionError("Unexpected discovery during deletion")

    monkeypatch.setattr(configuration_ui, "discover_models", unexpected)
    answers = [1, 0, confirmation] if confirmation is None else [1, 0]
    shown = select_choices(monkeypatch, answers)
    result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == (0 if confirmation is None else 1), result.output
    assert shown[0] == ("Configuration", ["Create or update profile", "Delete saved profile"])
    assert "saved" in shown[1][1][0] and "high" in shown[1][1][0]
    assert shown[2][0] == "Delete saved profile 'daily'?"
    assert {file: file.read_bytes() for file in before} == before


def test_configure_deletes_an_invalid_default_profile_after_confirmation(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow recovery from an unreadable profile and clear its default before the next run."""
    path = preset_path("broken")
    path.parent.mkdir(parents=True)
    path.write_text("invalid TOML {{")
    wizard.write_text('default_preset = "broken"\noutput_language = "Korean"\n[harness]\nkind = "pi"\n')
    shown = select_choices(monkeypatch, [1, 0, 0])
    result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 0, result.output
    assert "Unreadable profile" in shown[1][1][0]
    assert "Deleted profile: broken" in result.output
    assert not path.exists()
    assert load_settings().default_preset is None
    assert load_settings().output_language == "Korean"
    assert "Harness" not in [label for label, _ in shown]


def test_delete_profile_empty_menu_does_not_create_files(wizard: Path) -> None:
    """An empty profile directory does not need a picker or create configuration state."""
    configuration_ui.delete_profile_interactively(wizard)
    assert not wizard.parent.exists()


def test_configure_preserves_manually_configured_role_tiers(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Updating a profile does not erase explicit unattended-run tiers or unrelated settings."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('risk_rounds = 3\n[harness]\nkind="pi"\n[quick]\nkind="claude"\nmodel="worker"\n[deep]\nkind="codex"\nmodel="manager"\n')
    select_choices(monkeypatch, [None, 1, 1, None, 1])
    result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness.model == "small"
    assert settings.quick == HarnessSettings(kind="claude", model="worker")
    assert settings.deep == HarnessSettings(kind="codex", model="manager")
    assert settings.risk_rounds == 3


def test_saving_another_profile_keeps_the_existing_default(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Declining the new default selection leaves the previous profile selected."""
    previous = save_preset("daily", HarnessSettings(kind="claude", model="stable-model"))
    contents = previous.read_text()
    wizard.write_text('default_preset = "daily"\n[harness]\nkind = "pi"\n')
    select_choices(monkeypatch, [None, None, 1, 1, None, 0, None])
    result = runner.invoke(cli.app, ["configure"], input="experiment\n")
    assert result.exit_code == 0, result.output
    assert load_settings().default_preset == "daily"
    assert previous.read_text() == contents
    assert load_preset("experiment").model == "small"
    assert "Default preset: daily" in result.output


@pytest.mark.parametrize(("answer", "expected"), [(None, "daily"), (1, None)])
def test_replacing_selected_profile_updates_its_default_status(
    wizard: Path, monkeypatch: pytest.MonkeyPatch, answer: int | None, expected: str | None,
) -> None:
    """The final choice can retain or clear the selected profile's default marker."""
    save_preset("daily", HarnessSettings(kind="pi", provider="local", model="small"))
    wizard.write_text('default_preset = "daily"\n[harness]\nkind="pi"\nprovider="local"\nmodel="small"\n')
    select_choices(monkeypatch, [None, None, None, 2, None, 0, 0, answer])
    result = runner.invoke(cli.app, ["configure"], input="daily\n")
    assert result.exit_code == 0, result.output
    assert load_settings().default_preset == expected
    assert load_preset("daily").model == "large"


def test_cancel_at_default_profile_selection_writes_nothing(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gather all profile preferences before modifying config or creating a preset."""
    wizard.parent.mkdir(parents=True)
    original = '[harness]\nkind="pi"\nmodel="saved-model"\n'
    wizard.write_text(original)
    select_choices(monkeypatch, [None, None, None, None, 0])
    result = runner.invoke(cli.app, ["configure"], input="uncommitted\n")
    assert result.exit_code == 1, result.output
    assert wizard.read_text() == original
    assert not preset_path("uncommitted").exists()


def test_pi_native_model_choice_resets_provider_and_effort(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pi cannot start with only a provider, so native defaults must clear both selectors."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="pi"\nprovider="local"\nmodel="small"\neffort="high"\n')
    shown = select_choices(monkeypatch, [None, None, 0, None, 1])
    result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 0, result.output
    assert load_settings().harness == HarnessSettings(kind="pi")
    assert shown[2][1][0] == "Harness default (provider and model)"


def test_model_change_does_not_reuse_previous_effort(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An old model's explicit effort cannot silently carry into a different model."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="pi"\nprovider="local"\nmodel="small"\neffort="high"\n')
    select_choices(monkeypatch, [None, None, 2, None, 1])
    result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 0, result.output
    assert load_settings().harness.model == "large"
    assert load_settings().harness.effort is None


def test_unavailable_catalog_allows_custom_model_without_effort_choices(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery failure leaves manual runtime selection available without inventing effort support."""
    monkeypatch.setattr(configuration_ui, "discover_models", lambda settings: ModelCatalog(notice="Catalog unavailable."))
    monkeypatch.setattr(configuration_ui, "effort_choices", lambda settings, catalog: ())
    shown = select_choices(monkeypatch, [None, 1, 1, 1])
    result = runner.invoke(cli.app, ["configure"], input="custom-provider\ncustom-model\n")
    assert result.exit_code == 0, result.output
    assert load_settings().harness == HarnessSettings(provider="custom-provider", model="custom-model")
    assert "Catalog unavailable." in result.output
    assert all(label != "Reasoning effort" for label, _ in shown)


def test_new_codex_provider_uses_its_own_model_choices(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider-specific discovery must not present the previous provider's model catalog."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="codex"\nprovider="old-provider"\nmodel="old-model"\n')

    def catalog(settings: HarnessSettings) -> ModelCatalog:
        """Report models scoped to each configured provider."""
        model = "new-model" if settings.provider == "new-provider" else "old-model"
        return ModelCatalog((ModelChoice(model, model, settings.provider),))

    monkeypatch.setattr(configuration_ui, "discover_models", catalog)
    shown = select_choices(monkeypatch, [None, 2, 1, None, 1])
    result = runner.invoke(cli.app, ["configure"], input="new-provider\n")
    assert result.exit_code == 0, result.output
    assert load_settings().harness == HarnessSettings(kind="codex", provider="new-provider", model="new-model")
    assert shown[2] == ("Model", ["Harness default", "new-model", "Enter a model manually"])


@pytest.mark.parametrize(("choice", "rounds"), [(0, 1), (1, 3), (2, 5)])
def test_research_depth_controls_both_discussions(choice: int, rounds: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Match the reference shallow, medium, and deep discussion counts."""
    original = Settings()
    select_choices(monkeypatch, [choice])
    selected = configuration_ui.select_research_depth(original)
    assert (selected.research_rounds, selected.risk_rounds) == (rounds, rounds)
    assert (original.research_rounds, original.risk_rounds) == (1, 1)


def test_research_depth_preserves_custom_counts_when_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing unequal discussion limits stay available instead of silently becoming a standard depth."""
    select_choices(monkeypatch, [None])
    selected = configuration_ui.select_research_depth(Settings(research_rounds=2, risk_rounds=4))
    assert (selected.research_rounds, selected.risk_rounds) == (2, 4)


def test_run_selects_independent_profiles_without_changing_saved_files(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one saved profile for working agents and another for decision managers."""
    working = HarnessSettings(provider="local", model="small", effort="low")
    decision = HarnessSettings(kind="codex", model="large", effort="high")
    paths = [save_preset("alpha", working), save_preset("zeta", decision)]
    before = {path: path.read_bytes() for path in paths}
    override = HarnessSettings(kind="claude", model="news-model")
    original = Settings(harness=working, default_preset="alpha", agents={"news": override})
    shown = select_choices(monkeypatch, [None, 2, None])
    selected = configuration_ui.select_agent_settings(original)
    assert selected.quick == working and selected.deep == decision
    assert selected.agents == original.agents == {"news": override}
    assert original.quick is None and original.deep is None
    assert [label for label, _ in shown] == ["Working agents", "Decision manager", "Agent configuration"]
    assert shown[-1][1] == ["Continue", "Advanced"]
    assert shown[0][1][0].startswith("alpha ·")
    assert " / low" in shown[0][1][0] and " / high" in shown[1][1][2]
    assert {path: path.read_bytes() for path in paths} == before
    assert not wizard.exists()


def test_manager_can_follow_newly_selected_working_profile(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirming the manager's same-as-working choice uses this run's working selection."""
    profile = HarnessSettings(kind="claude", model="manager", effort="high")
    save_preset("shared", profile)
    select_choices(monkeypatch, [1, None, None])
    selected = configuration_ui.select_agent_settings(Settings())
    assert selected.quick == profile and selected.deep == profile
    assert selected.quick is not selected.deep
    assert not wizard.exists()


def test_partial_role_override_inherits_new_working_profile(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain an effort-only role override while inheriting the working model selected for this run."""
    profile = HarnessSettings(provider="local", model="new-model", effort="low")
    path = save_preset("new-worker", profile)
    before = path.read_bytes()
    original = Settings(
        harness=HarnessSettings(provider="local", model="old-model", effort="low"),
        agents={"bull": HarnessSettings(effort="high")},
    )
    select_choices(monkeypatch, [1, None, None])

    selected = configuration_ui.select_agent_settings(original)

    assert selected.model_settings(role="bull") == HarnessSettings(provider="local", model="new-model", effort="high")
    assert selected.agents["bull"].model_dump(exclude_unset=True) == {"effort": "high"}
    assert original.model_settings(role="bull").model == "old-model"
    assert path.read_bytes() == before
    assert not wizard.exists()


def test_custom_runtime_choices_are_only_applied_to_the_run(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure an unsaved model and effort during role selection without editing global defaults."""
    wizard.parent.mkdir(parents=True)
    original = '[harness]\nkind="pi"\nprovider="local"\nmodel="small"\neffort="low"\n'
    wizard.write_text(original)
    select_choices(monkeypatch, [2, None, None, 2, 2, None, None])
    selected = configuration_ui.select_agent_settings(load_settings())
    assert selected.quick == HarnessSettings(provider="local", model="large", effort="high")
    assert selected.deep == selected.quick
    assert wizard.read_text() == original
    assert not (wizard.parent / "presets").exists()


def test_run_role_cancellation_leaves_source_settings_unchanged(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete working/manager selection cannot mutate the caller's saved choices."""
    original = Settings(harness=HarnessSettings(model="saved-model"))
    select_choices(monkeypatch, [None])
    with pytest.raises(typer.Abort):
        configuration_ui.select_agent_settings(original)
    assert original.quick is None and original.deep is None
    assert not wizard.exists()


def test_advanced_run_selection_edits_multiple_roles_without_saving(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use saved profiles and an unsaved runtime across working, manager, and supporting roles."""
    working = HarnessSettings(provider="local", model="small", effort="low")
    decision = HarnessSettings(kind="codex", model="large", effort="high")
    news = HarnessSettings(kind="claude", model="news-model")
    paths = [save_preset("alpha", working), save_preset("zeta", decision)]
    before = {path: path.read_bytes() for path in paths}
    original = Settings(harness=working, deep=decision, agents={"news": news})
    original_values = original.model_dump()
    roles = list(ROLES)
    answers = [None, None, 1]
    answers.extend([roles.index("market") + 1, 2])
    answers.extend([roles.index("research_manager") + 1, 1])
    answers.extend([roles.index("reflection") + 1, 4, None, None, 2, 2])
    answers.append(0)
    shown = select_choices(monkeypatch, answers)

    selected = configuration_ui.select_agent_settings(original)

    reflection = HarnessSettings(provider="local", model="large", effort="high")
    assert selected.agents == {"news": news, "market": decision, "research_manager": working, "reflection": reflection}
    assert selected.model_settings(role="market") == decision
    assert selected.model_settings(role="research_manager") == working
    assert selected.model_settings(role="portfolio_manager") == decision
    assert selected.model_settings(role="reflection") == reflection
    role_menus = [choices for label, choices in shown if label == "Advanced agent configuration"]
    assert len(role_menus) == 4
    assert role_menus[0][0] == "Continue"
    assert [label.split(" · ", 1)[0] for label in role_menus[0][1:]] == [role.name for role in ROLES.values()]
    assert role_menus[-1][roles.index("market") + 1] == f"Market Analyst · {configuration_ui._describe(decision)}"
    assert role_menus[-1][roles.index("research_manager") + 1] == f"Research Manager · {configuration_ui._describe(working)}"
    assert role_menus[-1][roles.index("reflection") + 1] == f"Reflection · {configuration_ui._describe(reflection)}"
    assert original.model_dump() == original_values
    assert {path: path.read_bytes() for path in paths} == before
    assert not wizard.exists()


@pytest.mark.parametrize("saved_profile", [False, True])
def test_advanced_harness_defaults_reset_inherited_selectors(
    wizard: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_profile: bool,
) -> None:
    """Treat selected default profiles as complete choices rather than inheriting the group's explicit selectors."""
    original = Settings(harness=HarnessSettings(
        provider="local", model="small", effort="high", command=("custom-pi",), extra_args=("--profile", "custom"),
    ))
    if saved_profile:
        save_preset("harness-default", HarnessSettings())
    select_choices(monkeypatch, [None, None, 1, list(ROLES).index("bull") + 1, 1, 0])

    selected = configuration_ui.select_agent_settings(original)
    restored = Settings.model_validate(selected.model_dump())

    assert selected.model_settings(role="bull") == HarnessSettings()
    assert restored.model_settings(role="bull") == HarnessSettings()
    assert selected.agents["bull"].model_dump(exclude_unset=True) == HarnessSettings().model_dump()
    assert selected.model_settings() == original.harness
    assert not wizard.exists()


@pytest.mark.parametrize("role_key", ["market", "portfolio_manager"])
def test_advanced_role_can_return_to_its_group_default(wizard: Path, monkeypatch: pytest.MonkeyPatch, role_key: str) -> None:
    """Remove a role override through its explicit inheritance choice without changing other assignments."""
    working = HarnessSettings(provider="local", model="small")
    decision = HarnessSettings(kind="codex", model="large")
    override = HarnessSettings(kind="claude", model="role-model")
    original = Settings(harness=working, deep=decision, agents={role_key: override, "bull": override})
    shown = select_choices(monkeypatch, [None, None, 1, list(ROLES).index(role_key) + 1, 0, 0])

    selected = configuration_ui.select_agent_settings(original)

    role = ROLES[role_key]
    inherited = decision if role.deep else working
    group = "decision manager" if role.deep else "working agents"
    role_choices = next(choices for label, choices in shown if label == role.name)
    assert role_choices[0] == f"Same as {group} · {configuration_ui._describe(inherited)}"
    assert selected.agents == {"bull": override}
    assert selected.model_settings(role=role_key) == inherited
    assert original.agents == {role_key: override, "bull": override}
    assert not wizard.exists()


def test_advanced_role_cancellation_discards_all_unsaved_edits(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel after an override is changed without mutating the original settings or any saved file."""
    profile = save_preset("daily", HarnessSettings(provider="local", model="small"))
    wizard.write_text('[harness]\nkind="pi"\nmodel="small"\n[agents.market]\nkind="claude"\nmodel="override"\n')
    before = {path: path.read_bytes() for path in (wizard, profile)}
    original = load_settings()
    original_values = original.model_dump()
    roles = list(ROLES)
    select_choices(monkeypatch, [None, None, 1, roles.index("market") + 1, 0, roles.index("bull") + 1])

    with pytest.raises(typer.Abort):
        configuration_ui.select_agent_settings(original)

    assert original.model_dump() == original_values
    assert {path: path.read_bytes() for path in before} == before
    assert {path.name for path in wizard.parent.iterdir()} == {"config.toml", "presets"}
    assert [path.name for path in profile.parent.iterdir()] == ["daily.toml"]


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
