from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import questionary
import typer

from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.config import (
    HarnessSettings,
    Settings,
    delete_preset,
    list_presets,
    load_preset,
    load_settings,
    preset_path,
    save_configuration,
    save_preset,
    set_default_preset,
    validate_date,
)
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice, discover_models, effort_choices

_MENU_STYLE = questionary.Style([
    ("selected", "noinherit noreverse"),
    ("highlighted", "noinherit noreverse"),
    ("pointer", "fg:#74b9ff bold noinherit"),
    ("answer", "fg:#74b9ff"),
])


def _choose(label: str, choices: list[str], default: int = 0) -> int:
    """Choose from a keyboard menu with an explicit default label and movable pointer.

    Args:
        label: Name of the selection.
        choices: Human-readable choices in display order.
        default: Initially selected index, marked with a suffix and accepted by Enter.

    Returns:
        The selected zero-based index.

    Raises:
        typer.Abort: The user cancels or terminal input closes.
    """
    options = [questionary.Choice(f"{title} (default)" if index == default and not title.endswith(" (default)") else title, value=index)
               for index, title in enumerate(choices)]
    try:
        selected = questionary.select(
            label, choices=options, default=options[default], pointer="›", style=_MENU_STYLE,
            instruction="(↑/↓ select · Enter confirm)",
            use_arrow_keys=True, use_shortcuts=False, use_jk_keys=False, use_emacs_keys=False, use_search_filter=False,
            erase_when_done=True,
        ).unsafe_ask()
    except (KeyboardInterrupt, EOFError) as exc:
        raise typer.Abort() from exc
    if selected is None:
        raise typer.Abort()
    return selected


def _confirm(label: str, default: bool = False) -> bool:
    """Use the same keyboard selection for a yes-or-no decision.

    Args:
        label: Decision presented to the user.
        default: Initially selected answer.

    Returns:
        Whether the user selected Yes.
    """
    return _choose(label, ["Yes", "No"], default=0 if default else 1) == 0


def _describe(settings: HarnessSettings) -> str:
    """Describe a harness selection without reading native credentials.

    Args:
        settings: Selected harness and optional native selectors.

    Returns:
        A short description suitable for an inheritance choice.
    """
    return f"{settings.kind} / {settings.model or 'Harness default'} / {settings.effort or 'Harness default'}"


def _provider(current: HarnessSettings, catalog: ModelCatalog) -> str | None:
    """Select a known provider or retain a custom provider identifier.

    Args:
        current: Current native selection.
        catalog: Providers and models reported by the harness.

    Returns:
        The selected provider, or None for harness defaults.
    """
    if current.kind == "claude":
        return None
    providers = sorted({model.provider for model in catalog.models if model.provider} | ({current.provider} if current.provider else set()))
    default = providers.index(current.provider) + 1 if current.provider in providers else 0
    choice = _choose("Provider", ["Harness default", *providers, "Enter a provider manually"], default)
    if choice == len(providers) + 1:
        return typer.prompt("Provider identifier", default=current.provider).strip() or None
    return providers[choice - 1] if choice else None


def _model(current: HarnessSettings, catalog: ModelCatalog) -> tuple[str | None, str | None]:
    """Select a native model with its matching provider.

    Args:
        current: Harness selection after the provider step.
        catalog: Model identifiers reported by the native harness.

    Returns:
        The model and provider selectors, with None representing harness defaults.
    """
    models = [model for model in catalog.models if current.provider is None or model.provider in (None, current.provider)]
    if current.model and not any(model.id == current.model and model.provider == current.provider for model in models):
        models.insert(0, ModelChoice(current.model, current.model, current.provider))
    labels = []
    default = 0
    for index, model in enumerate(models, 1):
        label = model.label if model.label == model.id else f"{model.label} ({model.id})"
        labels.append(f"{label} [{model.provider}]" if model.provider and current.provider is None else label)
        if model.id == current.model and model.provider == current.provider:
            default = index
    harness_default = "Harness default (provider and model)" if current.kind == "pi" and current.provider else "Harness default"
    choice = _choose("Model", [harness_default, *labels, "Enter a model manually"], default)
    if choice == len(models) + 1:
        model = typer.prompt("Model identifier", default=current.model).strip() or None
        return model, current.provider if model is not None or current.kind != "pi" else None
    if choice == 0:
        return None, None if current.kind == "pi" else current.provider
    model = models[choice - 1]
    return model.id, model.provider or current.provider


def _catalog(selected: HarnessSettings, catalogs: dict[str, ModelCatalog]) -> ModelCatalog:
    """Reuse native model choices for the same harness and provider in this setup.

    Args:
        selected: Native harness and provider to inspect.
        catalogs: Previously discovered catalogs in this setup session.

    Returns:
        Model choices reported by the selected harness.
    """
    key = selected.model_dump_json(exclude={"model", "effort"})
    if key not in catalogs:
        catalogs[key] = discover_models(selected)
        if catalogs[key].notice:
            typer.echo(catalogs[key].notice)
    return catalogs[key]


def _effort(current: HarnessSettings, catalog: ModelCatalog) -> str | None:
    """Choose a native reasoning level when the harness exposes supported values.

    Args:
        current: Selected harness, provider, and model.
        catalog: Native model metadata available for this selection.

    Returns:
        The native effort selector, or None to retain the harness's default.
    """
    levels = list(dict.fromkeys(effort_choices(current, catalog)))
    labels = levels.copy()
    if current.effort and current.effort not in levels:
        levels.insert(0, current.effort)
        labels.insert(0, f"{current.effort} (current)")
    if not levels:
        return None
    default = levels.index(current.effort) + 1 if current.effort in levels else 0
    choice = _choose("Reasoning effort", ["Harness default", *labels], default)
    return levels[choice - 1] if choice else None


def _runtime(current: HarnessSettings, catalogs: dict[str, ModelCatalog]) -> HarnessSettings:
    """Guide one profile through native harness, provider, model, and effort selection.

    Args:
        current: Existing runtime used to highlight initial choices.
        catalogs: Native discovery results reused within this setup session.

    Returns:
        The selected runtime without changing configuration files.
    """
    kinds = ["pi", "codex", "claude"]
    kind = kinds[_choose("Harness", kinds, kinds.index(current.kind))]
    selected = current if current.kind == kind else HarnessSettings(kind=kind)
    catalog = _catalog(selected, catalogs)
    provider = _provider(selected, catalog)
    if provider != selected.provider:
        selected = selected.model_copy(update={"provider": provider, "model": None, "effort": None})
        if selected.kind == "codex":
            catalog = _catalog(selected, catalogs)
    model, provider = _model(selected, catalog)
    effort = selected.effort if (model, provider) == (selected.model, selected.provider) else None
    selected = selected.model_copy(update={"model": model, "provider": provider, "effort": effort})
    return selected.model_copy(update={"effort": _effort(selected, catalog)})


def select_analysis_date() -> str:
    """Ask for the analysis cutoff, defaulting to the machine's local calendar date.

    Returns:
        A valid YYYY-MM-DD date no later than today.
    """
    while True:
        value = typer.prompt("Analysis date (YYYY-MM-DD)", default=date.today().isoformat()).strip()
        try:
            return validate_date(value)
        except ValueError:
            typer.echo("Enter a valid date as YYYY-MM-DD, today or earlier.")


def select_output_directory(default_directory: Path) -> Path | None:
    """Choose where to create this run's output folder without changing saved preferences.

    Args:
        default_directory: Parent directory used when no override is selected.

    Returns:
        An absolute output directory, or None to retain the current default.
    """
    default_directory = default_directory.expanduser().resolve()

    while True:
        value = typer.prompt("Output directory", default=str(default_directory)).strip()
        if not value:
            return None

        directory = Path(value).expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            typer.echo("Enter a directory path, not an existing file.")
            continue

        return None if directory == default_directory else directory


def select_research_depth(settings: Settings) -> Settings:
    """Select the number of rounds for both research and risk discussions.

    Args:
        settings: Current application settings used to highlight the initial depth.

    Returns:
        New settings with the selected discussion round counts.
    """
    rounds = [(1, 1), (3, 3), (5, 5)]
    labels = ["Shallow · 1 round", "Medium · 3 rounds", "Deep · 5 rounds"]
    current = (settings.research_rounds, settings.risk_rounds)
    if current not in rounds:
        rounds.insert(0, current)
        labels.insert(0, f"Current settings · {current[0]} research rounds / {current[1]} risk rounds")
    research, risk = rounds[_choose("Research depth", labels, rounds.index(current))]
    return settings.model_copy(update={"research_rounds": research, "risk_rounds": risk})


@dataclass(frozen=True)
class ProfileChoice:
    settings: HarnessSettings
    name: str | None = None
    inherited: bool = False


def _profile_label(
    settings: HarnessSettings,
    profiles: dict[str, HarnessSettings],
    name: str | None = None,
    default_preset: str | None = None,
) -> str:
    """Show a selected preset name, falling back to a compact runtime description.

    Args:
        settings: Effective profile shown in the editor.
        profiles: Available saved profiles.
        name: Exact preset chosen during this editing session.
        default_preset: Preferred matching preset when no explicit name was chosen.

    Returns:
        A preset name or harness/model/effort summary.
    """
    for candidate in (name, default_preset):
        if candidate is not None and profiles.get(candidate) == settings:
            return candidate

    return next((key for key, profile in profiles.items() if profile == settings), _describe(settings))


def _agent_profile(
    label: str,
    current: HarnessSettings,
    profiles: dict[str, HarnessSettings],
    default_preset: str | None,
    catalogs: dict[str, ModelCatalog],
    same_as: HarnessSettings | None = None,
    inherit_default: bool = False,
    same_as_label: str = "Same as working agents",
    current_name: str | None = None,
) -> ProfileChoice:
    """Select a saved profile or configure a runtime for a group or individual role.

    Args:
        label: Group or individual role shown as the menu title.
        current: Existing runtime selection for this group or role.
        profiles: Saved named single-agent profiles.
        default_preset: Name to highlight when it matches the current selection.
        catalogs: Shared native discovery results for custom selections.
        same_as: Runtime offered as an explicit inheritance choice.
        inherit_default: Whether inheritance is initially selected.
        same_as_label: Name of the group whose runtime can be inherited.
        current_name: Exact preset name most recently selected for this row.

    Returns:
        The runtime, selected preset name, and whether the inheritance option was chosen.
    """
    labels: list[str] = []
    selections: list[ProfileChoice] = []
    if same_as is not None:
        labels.append(f"{same_as_label} · {_profile_label(same_as, profiles, default_preset=default_preset)}")
        selections.append(ProfileChoice(same_as, inherited=True))
    if current not in profiles.values():
        labels.append(f"Current settings · {_describe(current)}")
        selections.append(ProfileChoice(current))
    preferred = None
    for name, profile in profiles.items():
        labels.append(f"{name} · {_describe(profile)}")
        if name == (current_name or default_preset) and profile == current:
            preferred = len(selections)
        selections.append(ProfileChoice(profile, name))
    native = HarnessSettings(kind=current.kind)
    if not any(choice.settings == native for choice in selections):
        labels.append(f"Harness default · {current.kind}")
        selections.append(ProfileChoice(native))
    default = next((index for index, choice in enumerate(selections) if choice.settings == current and not choice.inherited), 0)
    if preferred is not None:
        default = preferred
    if inherit_default and same_as is not None:
        default = 0
    choice = _choose(label, [*labels, "Configure for this run"], default)
    return ProfileChoice(_runtime(current, catalogs)) if choice == len(selections) else selections[choice]


def select_agent_settings(settings: Settings) -> Settings:
    """Choose a Simple or Advanced editor before assigning profiles for a run.

    Args:
        settings: Current application settings and optional default profile.

    Returns:
        Copied run settings containing the editor's final assignments.
    """
    profiles = {name: load_preset(name) for name in list_presets()}
    catalogs: dict[str, ModelCatalog] = {}
    mode = _choose("Agent configuration", ["Simple", "Advanced"])
    selected = settings.model_copy(deep=True)
    if mode == 0:
        return _simple_agent_settings(selected, profiles, catalogs)

    return _advanced_agent_settings(selected, profiles, catalogs)


def _simple_agent_settings(
    selected: Settings,
    profiles: dict[str, HarnessSettings],
    catalogs: dict[str, ModelCatalog],
) -> Settings:
    """Edit the working and manager profiles in one reusable list.

    Args:
        selected: Private copy of settings for this run.
        profiles: Saved profiles offered for either group.
        catalogs: Cached native choices for custom configuration.

    Returns:
        Group assignments with individual overrides cleared for this run.
    """
    selected.agents = {}
    inherited_manager = selected.deep is None
    if inherited_manager:
        selected.deep = selected.model_settings().model_copy(deep=True)

    groups = (("quick", "Working agents"), ("deep", "Decision manager"))
    names: dict[str, str | None] = {}
    cursor = len(groups)
    while True:
        working = selected.model_settings()
        manager = selected.model_settings(True)
        labels = [f"{label} · {_profile_label(profile, profiles, names.get(key), selected.default_preset)}"
                  for (key, label), profile in zip(groups, (working, manager), strict=True)]
        cursor = _choose("Simple agent configuration", [*labels, "Continue"], cursor)
        if cursor == len(groups):
            return selected

        key, label = groups[cursor]
        choice = _agent_profile(
            label,
            working if cursor == 0 else manager,
            profiles,
            selected.default_preset,
            catalogs,
            same_as=working if cursor == 1 else None,
            inherit_default=inherited_manager if cursor == 1 else False,
            current_name=names.get(key),
        )
        profile = HarnessSettings.model_validate(choice.settings.model_dump())
        if key == "quick":
            selected.quick = profile
            names[key] = choice.name
            if inherited_manager:
                selected.deep = profile.model_copy(deep=True)
                names["deep"] = choice.name
        else:
            inherited_manager = choice.inherited
            selected.deep = profile
            names[key] = names.get("quick") if choice.inherited else choice.name


def _advanced_agent_settings(
    selected: Settings,
    profiles: dict[str, HarnessSettings],
    catalogs: dict[str, ModelCatalog],
) -> Settings:
    """Edit individual roles directly without asking for group assignments first.

    Args:
        selected: Private copy of settings for this run.
        profiles: Saved profiles offered for each role.
        catalogs: Cached native choices for custom configuration.

    Returns:
        Run settings with updated individual role assignments.
    """

    roles = list(ROLES)
    names: dict[str, str | None] = {}
    cursor = len(roles)
    while True:
        labels = [f"{ROLES[key].name} · {_profile_label(selected.model_settings(role=key), profiles, names.get(key), selected.default_preset)}"
                  for key in roles]
        cursor = _choose("Advanced agent configuration", [*labels, "Continue"], cursor)
        if cursor == len(roles):
            return selected

        key = roles[cursor]
        role = ROLES[key]
        inherited = selected.model_settings(deep=role.deep)
        current = selected.model_settings(role=key)
        choice = _agent_profile(
            role.name,
            current,
            profiles,
            selected.default_preset,
            catalogs,
            same_as=inherited,
            inherit_default=key not in selected.agents,
            same_as_label="Default profile",
            current_name=names.get(key),
        )
        if choice.inherited:
            selected.agents.pop(key, None)
            names.pop(key, None)
        else:
            selected.agents[key] = HarnessSettings.model_validate(choice.settings.model_dump())
            names[key] = choice.name


def _preset() -> tuple[str, bool]:
    """Choose a preset name and confirm replacement of an existing profile.

    Returns:
        The preset name and replacement permission.
    """
    if names := list_presets():
        typer.echo(f"Existing presets: {', '.join(names)}")

    while True:
        name = typer.prompt("Preset name").strip()
        try:
            path = preset_path(name)
        except ValueError as exc:
            typer.echo(str(exc))
            continue
        if not path.exists():
            return name, False
        if _confirm(f"Replace preset '{name}'?"):
            return name, True


def delete_profile_interactively(path: Path) -> None:
    """Choose a saved profile and confirm its removal before changing any files.

    Args:
        path: Selected application configuration, whose default is cleared if needed.
    """
    names = list_presets()
    if not names:
        typer.echo("No saved profiles to delete.")
        return
    labels = []
    for name in names:
        try:
            description = _describe(load_preset(name))
        except (ValueError, OSError):
            description = "Unreadable profile"
        labels.append(f"{name} · {description}")
    name = names[_choose("Saved profile to delete", labels)]
    typer.echo("Default selections for this profile will also be cleared.")
    if not _confirm(f"Delete saved profile '{name}'?"):
        typer.echo("Deletion cancelled.")
        return
    delete_preset(name, path)
    typer.echo(f"Deleted profile: {name}")


def configure_interactively(path: Path) -> None:
    """Configure one reusable agent profile without assigning run-specific roles.

    Args:
        path: Application configuration file to create or update.
    """
    if list_presets() and _choose("Configuration", ["Create or update profile", "Delete saved profile"]) == 1:
        delete_profile_interactively(path)
        return
    settings = load_settings(path) if path.exists() else Settings()
    catalogs: dict[str, ModelCatalog] = {}
    typer.echo("Defaults are marked (default). Ctrl+C cancels setup.")
    typer.echo("\nAgent profile")
    harness = _runtime(settings.harness, catalogs)
    name, overwrite = _preset()
    make_default = _confirm(f"Use preset '{name}' by default?", default=settings.default_preset in (None, name))

    saved = save_configuration(path, harness, settings.quick, settings.deep)
    target = save_preset(name, harness, overwrite=overwrite)
    if make_default:
        saved = set_default_preset(path, name)
    elif saved.default_preset == name:
        saved = set_default_preset(path, None)

    typer.echo(f"\nSaved profile: {_describe(saved.harness)}")
    typer.echo(f"Config: {path.expanduser().resolve()}")
    typer.echo(f"Preset: {name} ({target})")
    if saved.default_preset:
        typer.echo(f"Default preset: {saved.default_preset}")
