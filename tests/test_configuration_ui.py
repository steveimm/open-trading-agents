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


def select_choices(monkeypatch: pytest.MonkeyPatch, answers: list[int | str | None]) -> list[tuple[str, list[str]]]:
    """Supply selections while retaining the choices presented by each wizard step.

    Args:
        monkeypatch: Scoped selection adapter replacement.
        answers: Indexes or label prefixes, with None accepting the highlighted default.

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
        if isinstance(answer, str):
            matches = [index for index, choice in enumerate(choices)
                       if choice.split(" · ", 1)[0].removesuffix(" (default)") == answer]
            assert len(matches) == 1, f"Expected one {answer!r} choice in {choices!r}"
            return matches[0]
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
    shown = select_choices(monkeypatch, [None, 1, 1, 2, None])
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
    assert [label for label, _ in shown] == [
        "Harness", "Provider", "Model", "Reasoning effort", "Use preset 'local-research' by default?",
    ]
    assert "Save these choices as a preset?" not in result.output
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
    assert result.exit_code == (0 if confirmation is None else 130), result.output
    if confirmation == "cancel":
        assert "Cancelled." in result.output
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
    result = runner.invoke(cli.app, ["configure"], input="worker\n")
    assert result.exit_code == 0, result.output
    settings = load_settings()
    assert settings.harness.model == "small"
    assert settings.quick == HarnessSettings(kind="claude", model="worker")
    assert settings.deep == HarnessSettings(kind="codex", model="manager")
    assert settings.risk_rounds == 3
    assert load_preset("worker").model == "small"
    assert settings.default_preset is None


def test_saving_another_profile_keeps_the_existing_default(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Declining the new default selection leaves the previous profile selected."""
    previous = save_preset("daily", HarnessSettings(kind="claude", model="stable-model"))
    contents = previous.read_text()
    wizard.write_text('default_preset = "daily"\n[harness]\nkind = "pi"\n')
    select_choices(monkeypatch, [None, None, 1, 1, None, None])
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
    select_choices(monkeypatch, [None, None, None, 2, None, 0, answer])
    result = runner.invoke(cli.app, ["configure"], input="daily\n")
    assert result.exit_code == 0, result.output
    assert load_settings().default_preset == expected
    assert load_preset("daily").model == "large"


def test_cancel_at_default_profile_selection_writes_nothing(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gather all profile preferences before modifying config or creating a preset."""
    wizard.parent.mkdir(parents=True)
    original = '[harness]\nkind="pi"\nmodel="saved-model"\n'
    wizard.write_text(original)
    shown = select_choices(monkeypatch, [None, None, None, None])
    result = runner.invoke(cli.app, ["configure"], input="uncommitted\n")
    assert result.exit_code == 130, result.output
    assert "Cancelled." in result.output
    assert wizard.read_text() == original
    assert not preset_path("uncommitted").exists()
    assert shown[-1][0] == "Use preset 'uncommitted' by default?"


@pytest.mark.parametrize("retry_name", ["another-profile", None])
def test_declining_preset_replacement_retries_name_or_cancels_without_writes(
    wizard: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_name: str | None,
) -> None:
    """Ask for a different name after a collision without changing the existing preset on refusal or cancellation."""
    previous = save_preset("daily", HarnessSettings(provider="local", model="small"))
    wizard.write_text('default_preset="daily"\n[harness]\nkind="pi"\nprovider="local"\nmodel="small"\n')
    before = {path: path.read_bytes() for path in (wizard, previous)}
    answers = [None, None, None, 2, None, None]
    if retry_name is not None:
        answers.append(None)
    shown = select_choices(monkeypatch, answers)
    names = "daily\n" + (f"{retry_name}\n" if retry_name is not None else "")

    result = runner.invoke(cli.app, ["configure"], input=names)

    assert result.exit_code == (0 if retry_name is not None else 130), result.output
    assert result.output.count("Preset name") == 2
    labels = [label for label, _ in shown]
    assert "Replace preset 'daily'?" in labels
    assert "Choose another preset name?" not in labels
    assert "Save these choices as a preset?" not in labels
    assert previous.read_bytes() == before[previous]
    if retry_name is None:
        assert "Cancelled." in result.output
        assert wizard.read_bytes() == before[wizard]
        assert [path.name for path in previous.parent.iterdir()] == ["daily.toml"]
    else:
        assert load_preset(retry_name).model == "large"
        assert load_settings().default_preset == "daily"


def test_pi_native_model_choice_resets_provider_and_effort(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pi cannot start with only a provider, so native defaults must clear both selectors."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="pi"\nprovider="local"\nmodel="small"\neffort="high"\n')
    shown = select_choices(monkeypatch, [None, None, 0, None, 1])
    result = runner.invoke(cli.app, ["configure"], input="harness-default\n")
    assert result.exit_code == 0, result.output
    assert load_settings().harness == HarnessSettings(kind="pi")
    assert shown[2][1][0] == "Harness default (provider and model)"


def test_model_change_does_not_reuse_previous_effort(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An old model's explicit effort cannot silently carry into a different model."""
    wizard.parent.mkdir(parents=True)
    wizard.write_text('[harness]\nkind="pi"\nprovider="local"\nmodel="small"\neffort="high"\n')
    select_choices(monkeypatch, [None, None, 2, None, 1])
    result = runner.invoke(cli.app, ["configure"], input="large-model\n")
    assert result.exit_code == 0, result.output
    assert load_settings().harness.model == "large"
    assert load_settings().harness.effort is None


def test_unavailable_catalog_allows_custom_model_without_effort_choices(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery failure leaves manual runtime selection available without inventing effort support."""
    monkeypatch.setattr(configuration_ui, "discover_models", lambda settings: ModelCatalog(notice="Catalog unavailable."))
    monkeypatch.setattr(configuration_ui, "effort_choices", lambda settings, catalog: ())
    shown = select_choices(monkeypatch, [None, 1, 1, 1])
    result = runner.invoke(cli.app, ["configure"], input="custom-provider\ncustom-model\nmanual-profile\n")
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
    result = runner.invoke(cli.app, ["configure"], input="new-provider\nupdated-provider\n")
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
    """Re-edit either Simple row and display the selected names without changing saved profiles."""
    working = HarnessSettings(provider="local", model="small", effort="low")
    alternate = HarnessSettings(kind="claude", model="alternate", effort="high")
    decision = HarnessSettings(kind="codex", model="large", effort="high")
    paths = [save_preset("alpha", working), save_preset("beta", alternate), save_preset("zeta", decision)]
    wizard.write_text('default_preset="alpha"\n[harness]\nkind="pi"\nprovider="local"\nmodel="small"\neffort="low"\n')
    paths.append(wizard)
    before = {path: path.read_bytes() for path in paths}
    override = HarnessSettings(kind="claude", model="news-model")
    original = Settings(harness=working, default_preset="alpha", agents={"news": override})
    shown = select_choices(monkeypatch, ["Simple", "Working agents", "beta", "Decision manager", "zeta", None, "alpha", "Continue"])

    selected = configuration_ui.select_agent_settings(original)

    assert selected.model_settings() == alternate and selected.model_settings(True) == working
    assert selected.agent_presets["market"] == selected.agent_presets["bull"] == "beta"
    assert selected.agent_presets["research_manager"] == selected.agent_presets["portfolio_manager"] == "alpha"
    assert selected.agents == {}
    assert original.agents == {"news": override}
    assert original.quick is None and original.deep is None
    assert shown[0] == ("Agent configuration", ["Simple", "Advanced"])
    editors = [choices for label, choices in shown if label == "Simple agent configuration"]
    assert editors == [
        ["Working agents · alpha (default)", "Decision manager · alpha (default)", "Continue"],
        ["Working agents · beta", "Decision manager · beta", "Continue"],
        ["Working agents · beta", "Decision manager · zeta", "Continue"],
        ["Working agents · beta", "Decision manager · alpha (default)", "Continue"],
    ]
    assert [label for label, _ in shown].count("Decision manager") == 2
    profile_menus = [choices for label, choices in shown if label in {"Working agents", "Decision manager"}]
    for choices in profile_menus:
        assert [choice for choice in choices if "(default)" in choice] == ["alpha (default) · pi / small / low"]
    assert selected.default_preset == original.default_preset == load_settings().default_preset == "alpha"
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize("independent", [False, True])
def test_manager_can_inherit_or_select_an_equal_independent_profile(
    wizard: Path,
    monkeypatch: pytest.MonkeyPatch,
    independent: bool,
) -> None:
    """Keep an explicitly named manager independent even when it initially equals the working profile."""
    original_profile = HarnessSettings(model="original")
    next_profile = HarnessSettings(kind="claude", model="next")
    save_preset("alpha", original_profile)
    save_preset("alpha-copy", original_profile)
    save_preset("beta", next_profile)
    original = Settings(harness=original_profile, default_preset="alpha")
    manager_choice = "alpha-copy" if independent else "Same as working agents"
    shown = select_choices(monkeypatch, ["Simple", "Decision manager", manager_choice, "Working agents", "beta", "Continue"])

    selected = configuration_ui.select_agent_settings(original)

    assert selected.model_settings() == next_profile
    assert selected.model_settings(True) == (original_profile if independent else next_profile)
    assert selected.agent_presets["research_manager"] == ("alpha-copy" if independent else "beta")
    editors = [choices for label, choices in shown if label == "Simple agent configuration"]
    assert editors[1][1] == f"Decision manager · {'alpha-copy' if independent else 'alpha (default)'}"
    assert editors[-1][1] == f"Decision manager · {'alpha-copy' if independent else 'beta'}"
    assert original.deep is None and original.harness == original_profile
    assert not wizard.exists()


def test_confirming_unnamed_equal_manager_keeps_it_independent(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept an existing independent manager without linking it to later worker edits."""
    initial = HarnessSettings(model="original")
    next_profile = HarnessSettings(kind="claude", model="next")
    save_preset("next-worker", next_profile)
    original = Settings(harness=initial, deep=initial)
    before = original.model_dump()
    select_choices(monkeypatch, ["Simple", "Decision manager", None, "Working agents", "next-worker", "Continue"])

    selected = configuration_ui.select_agent_settings(original)

    assert selected.model_settings() == next_profile
    assert selected.model_settings(True) == initial
    assert original.model_dump() == before
    assert not wizard.exists()


def test_simple_defaults_need_no_profile_questions_and_clear_role_overrides(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept Simple's current group assignments immediately and omit configured individual overrides for this run."""
    original = Settings(
        harness=HarnessSettings(provider="private-provider", model="worker", effort="low"),
        deep=HarnessSettings(kind="claude", model="manager", effort="high"),
        agents={"bull": HarnessSettings(effort="high")},
    )
    before = original.model_dump()
    shown = select_choices(monkeypatch, [None, None])

    selected = configuration_ui.select_agent_settings(original)

    assert selected.model_settings(role="bull") == original.harness
    assert selected.model_settings(True) == original.deep
    assert selected.agents == {}
    assert shown == [
        ("Agent configuration", ["Simple", "Advanced"]),
        ("Simple agent configuration", ["Working agents · pi / worker / low", "Decision manager · claude / manager / high", "Continue"]),
    ]
    assert original.model_dump() == before
    assert not wizard.exists()


def test_custom_runtime_choices_are_only_applied_to_the_run(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure an unsaved model and effort during role selection without editing global defaults."""
    wizard.parent.mkdir(parents=True)
    original = '[harness]\nkind="pi"\nprovider="local"\nmodel="small"\neffort="low"\n'
    wizard.write_text(original)
    shown = select_choices(monkeypatch, ["Simple", "Working agents", "Configure for this run", None, None, 2, 2, "Continue"])
    selected = configuration_ui.select_agent_settings(load_settings())
    assert selected.quick == HarnessSettings(provider="local", model="large", effort="high")
    assert selected.model_settings(True) == selected.quick
    assert shown[-1] == (
        "Simple agent configuration", ["Working agents · pi / large / high", "Decision manager · pi / large / high", "Continue"],
    )
    assert wizard.read_text() == original
    assert not (wizard.parent / "presets").exists()


def test_run_role_cancellation_leaves_source_settings_unchanged(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel after editing a Simple row without mutating original groups or individual overrides."""
    original = Settings(harness=HarnessSettings(model="saved-model"), agents={"bull": HarnessSettings(effort="high")})
    profile = save_preset("temporary", HarnessSettings(model="other-model"))
    saved_profile = profile.read_bytes()
    before = original.model_dump()
    select_choices(monkeypatch, ["Simple", "Working agents", "temporary", "Decision manager"])
    with pytest.raises(typer.Abort):
        configuration_ui.select_agent_settings(original)
    assert original.model_dump() == before
    assert profile.read_bytes() == saved_profile
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
    answers: list[int | str | None] = ["Advanced", "Market Analyst", "zeta", "Research Manager", "alpha"]
    answers.extend(["Reflection", "Configure for this run", None, None, 2, 2, "Continue"])
    shown = select_choices(monkeypatch, answers)

    selected = configuration_ui.select_agent_settings(original)

    reflection = HarnessSettings(provider="local", model="large", effort="high")
    assert selected.agents == {"news": news, "market": decision, "research_manager": working, "reflection": reflection}
    assert selected.model_settings(role="market") == decision
    assert selected.model_settings(role="research_manager") == working
    assert selected.model_settings(role="portfolio_manager") == decision
    assert selected.model_settings(role="reflection") == reflection
    assert selected.agent_presets["market"] == "zeta"
    assert selected.agent_presets["research_manager"] == "alpha"
    assert "reflection" not in selected.agent_presets
    assert selected.model_settings() == working
    assert selected.model_settings(True) == decision
    assert shown[0] == ("Agent configuration", ["Simple", "Advanced"])
    assert "Working agents" not in [label for label, _ in shown]
    assert "Decision manager" not in [label for label, _ in shown]
    role_menus = [choices for label, choices in shown if label == "Advanced agent configuration"]
    assert len(role_menus) == 4
    assert role_menus[0][-1] == "Continue"
    assert [label.split(" · ", 1)[0] for label in role_menus[0][:-1]] == [role.name for role in ROLES.values()]
    assert "Market Analyst · zeta" in role_menus[-1]
    assert "Research Manager · alpha" in role_menus[-1]
    assert "Reflection · pi / large / high" in role_menus[-1]
    assert original.model_dump() == original_values
    assert {path: path.read_bytes() for path in paths} == before
    assert not wizard.exists()


def test_confirming_unnamed_equal_role_retains_explicit_override(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept an existing explicit role without replacing it with its equal group default."""
    profile = HarnessSettings.model_validate(HarnessSettings(model="original").model_dump())
    original = Settings(harness=profile, agents={"bull": profile})
    before = original.model_dump()
    select_choices(monkeypatch, ["Advanced", "Bull Researcher", None, "Continue"])

    selected = configuration_ui.select_agent_settings(original)

    assert selected.agents["bull"].model_dump(exclude_unset=True) == profile.model_dump()
    selected.quick = HarnessSettings(model="new-worker")
    assert selected.model_settings(role="bull") == profile
    assert original.model_dump() == before
    assert not wizard.exists()


def test_saved_harness_defaults_reset_inherited_selectors(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Select saved empty selectors explicitly without an additional synthetic default-profile choice."""
    original = Settings(
        harness=HarnessSettings(provider="local", model="small", effort="high", command=("custom-pi",), extra_args=("--profile", "custom")),
        default_preset="daily",
    )
    save_preset("daily", HarnessSettings())
    shown = select_choices(monkeypatch, ["Advanced", "Bull Researcher", "daily", "Continue"])

    selected = configuration_ui.select_agent_settings(original)
    restored = Settings.model_validate(selected.model_dump())

    assert selected.model_settings(role="bull") == HarnessSettings()
    assert restored.model_settings(role="bull") == HarnessSettings()
    assert selected.agents["bull"].model_dump(exclude_unset=True) == HarnessSettings().model_dump()
    assert selected.model_settings() == original.harness
    role_choices = next(choices for label, choices in shown if label == "Bull Researcher")
    assert "daily (default) · pi / Harness default / Harness default" in role_choices
    assert not any(choice.startswith(("Harness default", "Default profile")) for choice in role_choices)
    assert not wizard.exists()


@pytest.mark.parametrize("role_key", ["market", "portfolio_manager"])
def test_advanced_role_can_select_the_saved_default(wizard: Path, monkeypatch: pytest.MonkeyPatch, role_key: str) -> None:
    """Select the named saved default as an independent role profile without a synthetic reset entry."""
    working = HarnessSettings(provider="local", model="small")
    decision = HarnessSettings(kind="codex", model="large")
    override = HarnessSettings(kind="claude", model="role-model")
    saved_default = HarnessSettings(model="saved-default")
    save_preset("daily", saved_default)
    original = Settings(harness=working, deep=decision, agents={role_key: override, "bull": override}, default_preset="daily")
    shown = select_choices(monkeypatch, ["Advanced", ROLES[role_key].name, "daily", "Continue"])

    selected = configuration_ui.select_agent_settings(original)

    role = ROLES[role_key]
    role_choices = next(choices for label, choices in shown if label == role.name)
    assert "daily (default) · pi / saved-default / Harness default" in role_choices
    assert not any(choice.startswith(("Harness default", "Default profile")) for choice in role_choices)
    assert selected.agents == {role_key: saved_default, "bull": override}
    assert selected.agents[role_key].model_dump(exclude_unset=True) == saved_default.model_dump()
    assert selected.model_settings(role=role_key) == saved_default
    assert original.agents == {role_key: override, "bull": override}
    assert not wizard.exists()


def test_advanced_role_cancellation_discards_all_unsaved_edits(wizard: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancel after an override is changed without mutating the original settings or any saved file."""
    profile = save_preset("daily", HarnessSettings(provider="local", model="small"))
    wizard.write_text('[harness]\nkind="pi"\nmodel="small"\n[agents.market]\nkind="claude"\nmodel="override"\n')
    before = {path: path.read_bytes() for path in (wizard, profile)}
    original = load_settings()
    original_values = original.model_dump()
    select_choices(monkeypatch, ["Advanced", "Market Analyst", "daily", "Bull Researcher"])

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


def test_typing_does_not_filter_or_move_model_selection() -> None:
    """Keep the selected model unchanged when arbitrary letters are typed in the arrow-key menu."""
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("cloud\r")
        assert configuration_ui._choose("Model", ["local-small", "cloud-large", "local-large"]) == 0


def test_keyboard_cancellation_keeps_existing_configuration(wizard: Path) -> None:
    """Ctrl-C from the actual selection menu cancels setup without saving anything."""
    wizard.parent.mkdir(parents=True)
    original = '[harness]\nkind="pi"\nmodel="saved-model"\n'
    wizard.write_text(original)
    with create_pipe_input() as keyboard, create_app_session(input=keyboard, output=DummyOutput()):
        keyboard.send_text("\x03")
        result = runner.invoke(cli.app, ["configure"])
    assert result.exit_code == 130, result.output
    assert "Cancelled." in result.output
    assert wizard.read_text() == original
