from __future__ import annotations

from pathlib import Path

import questionary
import typer

from open_trading_agents.config import (
    HarnessSettings,
    Settings,
    list_presets,
    load_preset,
    load_settings,
    preset_path,
    save_configuration,
    save_preset,
    set_default_preset,
)
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice, discover_models, effort_choices

_MENU_STYLE = questionary.Style([
    ("highlighted", "fg:#74b9ff bold noinherit"),
    ("pointer", "fg:#74b9ff bold noinherit"),
    ("answer", "fg:#74b9ff"),
])


def _choose(label: str, choices: list[str], default: int = 0) -> int:
    """Choose from a keyboard menu with the current value initially highlighted.

    Args:
        label: Name of the selection.
        choices: Human-readable choices in display order.
        default: Initially highlighted index, accepted by pressing Enter.

    Returns:
        The selected zero-based index.

    Raises:
        typer.Abort: The user cancels or terminal input closes.
    """
    options = [questionary.Choice(title, value=index) for index, title in enumerate(choices)]
    try:
        selected = questionary.select(
            label, choices=options, default=options[default], pointer="›", style=_MENU_STYLE,
            instruction="(↑/↓ select · Enter confirm · type to filter)",
            use_arrow_keys=True, use_shortcuts=False, use_jk_keys=False, use_search_filter=True,
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
        default: Initially highlighted answer.

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
    return f"{settings.kind} / {settings.provider or 'native provider'} / {settings.model or 'native model'} / {settings.effort or 'native effort'}"


def _provider(current: HarnessSettings, catalog: ModelCatalog) -> str | None:
    """Select a known provider or retain a custom provider identifier.

    Args:
        current: Current native selection.
        catalog: Providers and models reported by the harness.

    Returns:
        The selected provider, or None for native defaults.
    """
    if current.kind == "claude":
        return None
    providers = sorted({model.provider for model in catalog.models if model.provider} | ({current.provider} if current.provider else set()))
    default = providers.index(current.provider) + 1 if current.provider in providers else 0
    choice = _choose("Provider", ["Native default", *providers, "Enter a provider manually"], default)
    if choice == len(providers) + 1:
        return typer.prompt("Provider identifier", default=current.provider).strip() or None
    return providers[choice - 1] if choice else None


def _model(current: HarnessSettings, catalog: ModelCatalog) -> tuple[str | None, str | None]:
    """Select a native model with its matching provider.

    Args:
        current: Harness selection after the provider step.
        catalog: Model identifiers reported by the native harness.

    Returns:
        The model and provider selectors, with None representing native defaults.
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
    native = "Native default (provider and model)" if current.kind == "pi" and current.provider else "Native default"
    choice = _choose("Model", [native, *labels, "Enter a model manually"], default)
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
    choice = _choose("Reasoning effort", ["Native default", *labels], default)
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


def _agent_profile(label: str, current: HarnessSettings, profiles: dict[str, HarnessSettings], default_preset: str | None,
                   catalogs: dict[str, ModelCatalog], same_as: HarnessSettings | None = None,
                   inherit_working: bool = False) -> HarnessSettings:
    """Select a saved profile or configure a runtime for one role group.

    Args:
        label: Working-agent or decision-manager menu title.
        current: Existing runtime selection for this role group.
        profiles: Saved named single-agent profiles.
        default_preset: Name to highlight when it matches the current selection.
        catalogs: Shared native discovery results for custom selections.
        same_as: Working-agent runtime offered to decision managers.
        inherit_working: Whether decision managers initially follow the working agents.

    Returns:
        The selected runtime without persisting the run's role assignments.
    """
    labels: list[str] = []
    selections: list[HarnessSettings] = []
    if same_as is not None:
        labels.append(f"Same as working agents · {_describe(same_as)}")
        selections.append(same_as)
    if current not in profiles.values() and current not in selections:
        labels.append(f"Current settings · {_describe(current)}")
        selections.append(current)
    preferred = None
    for name, profile in profiles.items():
        suffix = " (default)" if name == default_preset else ""
        labels.append(f"{name}{suffix} · {_describe(profile)}")
        if name == default_preset and profile == current:
            preferred = len(selections)
        selections.append(profile)
    native = HarnessSettings(kind=current.kind)
    if native not in selections:
        labels.append(f"Native defaults · {current.kind}")
        selections.append(native)
    default = selections.index(current) if current in selections else 0
    if preferred is not None:
        default = preferred
    if inherit_working and same_as is not None:
        default = 0
    choice = _choose(label, [*labels, "Configure for this run…"], default)
    return _runtime(current, catalogs) if choice == len(selections) else selections[choice]


def select_agent_settings(settings: Settings) -> Settings:
    """Select working and decision runtimes independently for a new analysis.

    Args:
        settings: Current application settings and optional default profile.

    Returns:
        New settings with complete working-agent and decision-manager selections.
    """
    profiles = {name: load_preset(name) for name in list_presets()}
    catalogs: dict[str, ModelCatalog] = {}
    working = _agent_profile("Working agents", settings.model_settings(), profiles, settings.default_preset, catalogs)
    decision = _agent_profile("Decision manager", settings.model_settings(True), profiles, settings.default_preset, catalogs,
                              same_as=working, inherit_working=settings.deep is None)
    return settings.model_copy(update={"quick": working.model_copy(deep=True), "deep": decision.model_copy(deep=True)})


def _preset() -> tuple[str, bool] | None:
    """Offer an optional named preset and confirm replacement of an existing name.

    Returns:
        The preset name and replacement permission, or None when declined.
    """
    if not _confirm("Save these choices as a preset?"):
        return None
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
        if not _confirm("Choose another preset name?", default=True):
            return None


def configure_interactively(path: Path) -> None:
    """Configure one reusable agent profile without assigning run-specific roles.

    Args:
        path: Application configuration file to create or update.
    """
    settings = load_settings(path) if path.exists() else Settings()
    catalogs: dict[str, ModelCatalog] = {}
    typer.echo("Current choices start highlighted. Ctrl+C cancels setup.")
    typer.echo("\nAgent profile")
    harness = _runtime(settings.harness, catalogs)
    preset = _preset()
    make_default = preset is not None and _confirm(
        f"Use preset '{preset[0]}' by default?", default=settings.default_preset in (None, preset[0]),
    )
    saved = save_configuration(path, harness, settings.quick, settings.deep)
    typer.echo(f"\nSaved profile: {_describe(saved.harness)}")
    typer.echo(f"Config: {path.expanduser().resolve()}")
    if preset is not None:
        name, overwrite = preset
        target = save_preset(name, harness, overwrite=overwrite)
        if make_default:
            saved = set_default_preset(path, name)
        elif saved.default_preset == name:
            saved = set_default_preset(path, None)
        typer.echo(f"Preset: {name} ({target})")
    if saved.default_preset:
        typer.echo(f"Default preset: {saved.default_preset}")
