from __future__ import annotations

from pathlib import Path

import questionary
import typer

from open_trading_agents.config import (
    HarnessSettings,
    Settings,
    list_presets,
    load_settings,
    preset_path,
    save_configuration,
    save_preset,
)
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice, discover_models

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
    return f"{settings.kind} / {settings.provider or 'native provider'} / {settings.model or 'native model'}"


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
    choice = _choose("Model", ["Native default", *labels, "Enter a model manually"], default)
    if choice == len(models) + 1:
        return typer.prompt("Model identifier", default=current.model).strip() or None, current.provider
    if choice == 0:
        return None, current.provider
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
    key = selected.model_dump_json(exclude={"model"})
    if key not in catalogs:
        catalogs[key] = discover_models(selected)
        if catalogs[key].notice:
            typer.echo(catalogs[key].notice)
    return catalogs[key]


def _runtime(current: HarnessSettings | None, inherited: HarnessSettings | None,
             catalogs: dict[str, ModelCatalog]) -> HarnessSettings | None:
    """Guide one tier through harness, provider, and model selection.

    Args:
        current: Existing explicit tier, or None when it inherits defaults.
        inherited: Shared defaults offered to a working or decision tier.
        catalogs: Native discovery results reused within this setup session.

    Returns:
        The selected runtime, or None when the tier inherits shared defaults.
    """
    kinds = ["pi", "codex", "claude"]
    labels = kinds.copy()
    offset = int(inherited is not None)
    if inherited is not None:
        labels.insert(0, f"Use default settings ({_describe(inherited)})")
    default = kinds.index(current.kind) + offset if current is not None else 0
    choice = _choose("Harness", labels, default)
    if inherited is not None and choice == 0:
        return None
    kind = kinds[choice - offset]
    previous = current or inherited
    selected = previous if previous is not None and previous.kind == kind else HarnessSettings(kind=kind)
    catalog = _catalog(selected, catalogs)
    provider = _provider(selected, catalog)
    if provider != selected.provider:
        selected = selected.model_copy(update={"provider": provider, "model": None})
        if selected.kind == "codex":
            catalog = _catalog(selected, catalogs)
    model, provider = _model(selected, catalog)
    return selected.model_copy(update={"model": model, "provider": provider})


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
    """Collect every tier and an optional preset before saving configuration.

    Args:
        path: Application configuration file to create or update.
    """
    settings = load_settings(path) if path.exists() else Settings()
    catalogs: dict[str, ModelCatalog] = {}
    typer.echo("Current choices start highlighted. Ctrl+C cancels setup.")
    typer.echo("\nDefault settings")
    harness = _runtime(settings.harness, None, catalogs)
    assert harness is not None
    typer.echo("\nWorking agents (quick)")
    quick = _runtime(settings.quick, harness, catalogs)
    typer.echo("\nDecision managers (deep)")
    deep = _runtime(settings.deep, harness, catalogs)
    preset = _preset()
    saved = save_configuration(path, harness, quick, deep)
    typer.echo(f"\nSaved default: {_describe(saved.harness)}")
    typer.echo(f"Saved quick: {_describe(saved.quick) if saved.quick else 'use default settings'}")
    typer.echo(f"Saved deep: {_describe(saved.deep) if saved.deep else 'use default settings'}")
    typer.echo(f"Config: {path.expanduser().resolve()}")
    if preset is not None:
        name, overwrite = preset
        target = save_preset(name, saved, overwrite=overwrite)
        typer.echo(f"Preset: {name} ({target})")
