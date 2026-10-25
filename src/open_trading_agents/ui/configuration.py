from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import questionary
import typer
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition, IsDone
from prompt_toolkit.layout import ConditionalContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.processors import Processor, Transformation, TransformationInput
from prompt_toolkit.styles import merge_styles
from questionary.prompts.common import InquirerControl

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
    save_data_configuration,
    save_preset,
    set_default_preset,
    validate_date,
)
from open_trading_agents.dataflows.config import build_data_config, read_data_credentials
from open_trading_agents.dataflows.router import TOOLS_CATEGORIES, VENDOR_METHODS
from open_trading_agents.harnesses.catalog import ModelCatalog, ModelChoice, discover_models, effort_choices

_SOURCE_CATEGORIES = (
    ("core_stock_apis", "Prices"),
    ("technical_indicators", "Technical indicators"),
    ("fundamental_data", "Fundamentals and financial statements"),
    ("news_data", "News and insider data"),
    ("macro_data", "Macroeconomic data"),
    ("prediction_markets", "Prediction markets"),
)
_SOURCE_NAMES = {
    "sec_edgar": "SEC EDGAR",
    "yfinance": "Yahoo Finance",
    "alpha_vantage": "Alpha Vantage",
    "fred": "FRED",
    "polymarket": "Polymarket",
}


_GROUP_DESCRIPTIONS = {
    "Decision manager": "Research Manager weighs the bull/bear debate. Portfolio Manager makes the final recommendation after the risk debate.",
    "Working agents": "Research and trade proposals: market, fundamentals, news and sentiment analysts, bull/bear researchers, trader, three risk analysts, and reflection.",
}
_ROLE_DESCRIPTIONS = {
    "market": "Examines price trends, momentum, volatility and trading volume.",
    "fundamentals": "Examines financial statements, valuation, cash flow and balance-sheet strength.",
    "news": "Reviews company news, macroeconomic events and upcoming catalysts.",
    "social": "Assesses sentiment from news, StockTwits and Reddit evidence.",
    "bull": "Argues the case for investing and challenges the bearish arguments.",
    "bear": "Argues the case against investing and challenges the bullish arguments.",
    "research_manager": "Weighs the bull/bear debate and gives the trader an investment plan.",
    "trader": "Turns the research plan into a trade proposal with entry, stop and sizing guidance.",
    "aggressive": "Examines upside opportunities and the cost of excessive caution in the proposed trade.",
    "conservative": "Challenges downside risk and proposes safeguards to preserve capital.",
    "neutral": "Weighs the aggressive and conservative arguments to balance opportunity and risk.",
    "portfolio_manager": "Combines research, the trade proposal and risk debate into the final recommendation.",
    "reflection": "When memory is enabled, reviews past decisions with measurable outcomes to derive lessons for future runs.",
}


_MENU_STYLE = questionary.Style([
    ("text", "fg:#d9e2ef"),
    ("question", "fg:#bd9bfa bold"),
    ("qmark", "fg:#bd9bfa bold"),
    ("instruction", "fg:#8799af"),
    ("description", "fg:#a9bcd3"),
    ("pointer", "fg:#bd9bfa bold"),
    ("answer", "fg:#78cbd4"),
    ("assignment", "fg:#78cbd4 bold"),
    ("selected", "fg:#d9e2ef bg:#303547 bold"),
    ("highlighted", "bg:#303547 bold"),
    ("forward", "fg:#91d5ac bold"),
    ("configure", "fg:#bd9bfa bold"),
    ("back", "fg:#e8b86d"),
    ("destructive", "fg:#f2858a"),
])


def _menu_tokens(title: str, highlighted: bool) -> list[tuple[str, str]]:
    """Style menu actions and toggle states while retaining the moving row highlight.

    Args:
        title: Original menu label.
        highlighted: Whether the keyboard cursor points to this row.

    Returns:
        Styled text fragments for Questionary.
    """
    action = title.split(" · ", 1)[0]
    style = "class:text"
    if action in {"Continue", "Save", "Yes"}:
        style = "class:forward"
    elif action in {"Back", "Cancel", "No"}:
        style = "class:back"
    elif action == "Configure for this run":
        style = "class:configure"
    elif action == "Remove" or action.startswith("Delete "):
        style = "class:destructive"
    if highlighted:
        style += " class:highlighted"

    if " (selected)" in title:
        before, marker, after = title.partition("(selected)")
        badge_style = "class:assignment" + (" class:highlighted" if highlighted else "")
        return [(style, before), (badge_style, marker), (style, after)]

    return [(style, title)]


def _style_menu(question: questionary.Question, options: list[questionary.Choice], titles: list[str]) -> None:
    """Apply shared semantic colors and a moving highlight to select and checklist menus.

    Args:
        question: Questionary application to style.
        options: Choices whose titles are updated before rendering.
        titles: Original labels in display order.
    """
    def style_rows(application: Application) -> None:
        """Refresh row colors as the keyboard cursor moves.

        Args:
            application: Active menu application before rendering.
        """
        for control in application.layout.find_all_controls():
            if isinstance(control, InquirerControl):
                current = control.get_pointed_at()
                for option, title in zip(options, titles, strict=True):
                    option.title = _menu_tokens(title, option is current)

    question.application.before_render += style_rows


def _choose(label: str, choices: list[str], default: int = 0, *, descriptions: list[str] | None = None) -> int:
    """Choose from a keyboard menu without changing its supplied labels.

    Args:
        label: Name of the selection.
        choices: Human-readable choices in display order.
        default: Initially selected index, accepted by Enter.
        descriptions: Optional explanations shown below the title and updated with the highlighted choice.

    Returns:
        The selected zero-based index.

    Raises:
        typer.Abort: The user cancels or terminal input closes.
    """
    options = [questionary.Choice(title, value=index, description=descriptions[index] if descriptions else None)
               for index, title in enumerate(choices)]
    try:
        question = questionary.select(
            label, choices=options, default=options[default], pointer="›", style=_MENU_STYLE,
            instruction="(↑/↓ select · Enter confirm)",
            use_arrow_keys=True, use_shortcuts=False, use_jk_keys=False, use_emacs_keys=False, use_search_filter=False,
            erase_when_done=True, show_description=False,
        )
        if descriptions:
            control = next(control for control in question.application.layout.find_all_controls() if isinstance(control, InquirerControl))

            def explanation() -> list[tuple[str, str]]:
                """Return the highlighted choice's explanation without an extra label."""
                return [("class:description", control.get_pointed_at().description or "")]

            help_window = Window(FormattedTextControl(explanation), wrap_lines=True, dont_extend_height=True)
            show_help = Condition(lambda: bool(control.get_pointed_at().description)) & ~IsDone()
            question.application.layout.container.children.insert(1, ConditionalContainer(help_window, filter=show_help))

        _style_menu(question, options, choices)
        selected = question.unsafe_ask()
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


def select_configuration_section() -> str:
    """Choose whether to configure agent profiles or data-source access.

    Returns:
        The selected configuration section, agents or data.
    """
    return ("agents", "data")[_choose("Configuration", ["Agents", "Data sources"])]


def _key_preview(value: str) -> str:
    """Show only the final five characters of a sufficiently long API key.

    Args:
        value: Current or saved credential value.

    Returns:
        A fully masked short value or four stars followed by the final five characters.
    """
    return "*" * len(value) if len(value) <= 5 else f"****{value[-5:]}"


class _ApiKeyMask(Processor):
    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        """Render a masked key while preserving cursor positions for the editable buffer.

        Args:
            ti: Current input line and its unmodified document.

        Returns:
            Masked fragments with source/display mappings for the compressed prefix.
        """
        value = ti.document.lines[ti.lineno]
        fragments = [("class:answer", _key_preview(value))]
        if len(value) <= 5:
            return Transformation(fragments)

        hidden = len(value) - 5

        def source_to_display(position: int) -> int:
            """Map the hidden prefix into four cells and retain exact tail positions.

            Args:
                position: Character offset in the original input buffer.

            Returns:
                The corresponding offset in the masked preview.
            """
            position = min(max(position, 0), len(value))
            return position * 4 // hidden if position < hidden else 4 + position - hidden

        def display_to_source(position: int) -> int:
            """Map pointer positions back into the original editable key.

            Args:
                position: Character offset in the masked preview.

            Returns:
                The corresponding offset in the original input buffer.
            """
            position = min(max(position, 0), 9)
            return position * hidden // 4 if position < 4 else hidden + position - 4

        return Transformation(fragments, source_to_display=source_to_display, display_to_source=display_to_source)


def _api_key_prompt(label: str) -> str:
    """Read a new API key with a live masked-tail preview and no completion echo.

    Args:
        label: Data provider's credential prompt.

    Returns:
        The original entered value, with empty input leaving the stored key unchanged.

    Raises:
        typer.Abort: The user cancels or terminal input closes.
    """
    try:
        value = questionary.text(label, default="", style=_MENU_STYLE, is_password=True,
                                 input_processors=[_ApiKeyMask()], erase_when_done=True).unsafe_ask()
    except (KeyboardInterrupt, EOFError) as exc:
        raise typer.Abort() from exc
    if value is None:
        raise typer.Abort()
    return value


def _enabled_sources(category: str, vendors: dict[str, str], overrides: dict[str, str]) -> set[str]:
    """Collect the effective sources used by the category's tools.

    Args:
        category: Registered data category.
        vendors: Category source chains.
        overrides: Tool-specific source chains that take precedence.

    Returns:
        Supported sources currently used by at least one tool in this category.
    """
    enabled = set()
    for tool in TOOLS_CATEGORIES[category]["tools"]:
        chain = overrides.get(tool, vendors[category])
        explicit = {source.strip() for source in chain.split(",") if source.strip() not in {"", "default"}}
        enabled.update(set(VENDOR_METHODS[tool]) & explicit if explicit else VENDOR_METHODS[tool])
    return enabled


def _source_checklist(category: str, enabled: set[str]) -> list[str]:
    """Edit a category's sources and validate them before returning to the category menu.

    Args:
        category: Registered category whose tools must remain covered.
        enabled: Sources initially checked.

    Returns:
        Selected source identifiers in fixed fallback order.

    Raises:
        typer.Abort: The user cancels setup or terminal input closes.
    """
    tools = TOOLS_CATEGORIES[category]["tools"]
    available = set().union(*(VENDOR_METHODS[tool] for tool in tools))
    sources = [source for source in _SOURCE_NAMES if source in available]
    titles = [_SOURCE_NAMES[source] for source in sources]
    options = [questionary.Choice(title, value=source, checked=source in enabled)
               for source, title in zip(sources, titles, strict=True)]

    def validate(selected: list[str]) -> bool | str:
        """Reject empty selections and combinations that leave a category's tools uncovered.

        Args:
            selected: Checked source identifiers.

        Returns:
            True for usable coverage, otherwise a short correction to show in the menu.
        """
        if not selected:
            return "Select at least one source before going back."
        if any(not set(selected).intersection(VENDOR_METHODS[tool]) for tool in tools):
            return "Also select Yahoo Finance or Alpha Vantage for company profiles."
        return True

    style = merge_styles([_MENU_STYLE, questionary.Style([
        ("selected", "fg:#91d5ac bg:default nobold noreverse"),
        ("validation-toolbar", "fg:#f2858a"),
    ])])
    try:
        question = questionary.checkbox(
            dict(_SOURCE_CATEGORIES)[category], choices=options, validate=validate, pointer="›", style=style,
            instruction="(↑/↓ move · Space toggle · Enter back)",
            use_arrow_keys=True, use_jk_keys=False, use_emacs_keys=False, use_search_filter=False,
            show_description=False, erase_when_done=True,
        )
        _style_menu(question, options, titles)
        result = question.unsafe_ask()
    except (KeyboardInterrupt, EOFError) as exc:
        raise typer.Abort() from exc
    if result is None:
        raise typer.Abort()
    return [source for source in sources if source in result]


def _configure_sources(vendors: dict[str, str], overrides: dict[str, str]) -> None:
    """Open source checklists by category, keeping all edits pending until Save.

    Args:
        vendors: Effective category selections, updated in place.
        overrides: Individual tool selections, cleared for explicitly configured categories.
    """
    cursor = 0
    while True:
        selections = {category: _enabled_sources(category, vendors, overrides) for category, _ in _SOURCE_CATEGORIES}
        labels = []
        for category, title in _SOURCE_CATEGORIES:
            names = [_SOURCE_NAMES[source] for source in _SOURCE_NAMES if source in selections[category]]
            labels.append(f"{title} · {', '.join(names) or 'None'}")
        cursor = _choose("Source selection", [*labels, "Back"], cursor)
        if cursor == len(_SOURCE_CATEGORIES):
            return

        category, _ = _SOURCE_CATEGORIES[cursor]
        selected = _source_checklist(category, selections[category])
        tools = TOOLS_CATEGORIES[category]["tools"]
        if any(tool in overrides for tool in tools):
            typer.echo("Replaced individual tool overrides in this category.")
        vendors[category] = ",".join(selected)
        for tool in tools:
            overrides.pop(tool, None)


def _configure_credentials(values: dict[str, str]) -> None:
    """Edit masked credentials in a submenu without saving pending changes.

    Args:
        values: Pending credential fields, updated in place.
    """
    fields = (
        ("fred_api_key", "FRED (macro data)", "FRED API key", True,
         "FRED provides macroeconomic series such as interest rates and inflation. Request an API key at https://fred.stlouisfed.org/docs/api/api_key.html."),
        ("alpha_vantage_api_key", "Alpha Vantage (alternative market/news data)", "Alpha Vantage API key", True,
         "Alpha Vantage is an alternative source for market and news data. Request an API key at https://www.alphavantage.co/support/#api-key."),
        ("sec_edgar_user_agent", "SEC EDGAR (contact identity)", "SEC EDGAR contact identity", False,
         "SEC EDGAR provides company filings. Enter your name or organization and contact email, for example: Research Team research@example.com."),
    )
    cursor = 0
    while True:
        labels = []
        for key, label, _, secret, _ in fields:
            value = values.get(key, "")
            status = _key_preview(value) if secret and value else "Set" if value else "Not set"
            labels.append(f"{label} · {status}")
        cursor = _choose("Credentials", [*labels, "Back"], cursor)
        if cursor == len(fields):
            return

        key, label, prompt, secret, explanation = fields[cursor]
        typer.echo(explanation)
        if values.get(key):
            action = _choose(label, ["Update", "Remove", "Back"])
            if action == 1:
                values[key] = ""
                continue
            if action == 2:
                continue

        value = (_api_key_prompt(prompt) if secret else typer.prompt(prompt, default="", show_default=False)).strip()
        if value:
            values[key] = value


def configure_data_interactively(path: Path) -> None:
    """Edit credentials and source checklists, saving all pending changes together.

    Args:
        path: Application configuration file to create or update.

    Raises:
        typer.Abort: The user cancels before saving.
    """
    values = read_data_credentials(path).copy()
    settings = load_settings(path) if path.exists() else Settings()
    data = build_data_config(settings.data)
    vendors, overrides = data["data_vendors"], data["tool_vendors"]
    original_vendors = vendors.copy()
    cursor = 0
    typer.echo("Leave a value blank to keep it. Ctrl+C cancels without saving.")

    while True:
        cursor = _choose("Data sources", ["Credentials", "Source selection", "Save"], cursor)
        if cursor == 0:
            _configure_credentials(values)
        elif cursor == 1:
            _configure_sources(vendors, overrides)
        else:
            for selections in (vendors, overrides):
                for key, chain in selections.items():
                    sources = [source.strip() for source in chain.split(",")]
                    if "alpha_vantage" in sources and "yfinance" in sources:
                        sources = [source for source in sources if source != "yfinance"]
                        sources.insert(sources.index("alpha_vantage"), "yfinance")
                        selections[key] = ",".join(sources)

            changes = {key: value for key, value in vendors.items() if value != original_vendors[key]}
            target = save_data_configuration(path, values, data_vendors=changes, tool_vendors=overrides)
            typer.echo(f"Saved data sources: {target}")
            return


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
    if current.kind == "claude-code":
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
    kinds = ["pi", "codex", "claude-code"]
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


def _matching_preset(
    settings: HarnessSettings,
    profiles: dict[str, HarnessSettings],
    name: str | None = None,
    default_preset: str | None = None,
) -> str | None:
    """Identify a matching preset while retaining the exact name explicitly chosen.

    Args:
        settings: Effective profile shown in the editor.
        profiles: Available saved profiles.
        name: Exact preset chosen during this editing session.
        default_preset: Saved default to prefer when no explicit name was chosen.

    Returns:
        A matching preset name, or None for a custom runtime.
    """
    identity = settings.profile_identity()
    for candidate in (name, default_preset):
        if candidate is not None and candidate in profiles and profiles[candidate].profile_identity() == identity:
            return candidate

    return next((key for key, profile in profiles.items() if profile.profile_identity() == identity), None)


def _profile_label(
    settings: HarnessSettings,
    profiles: dict[str, HarnessSettings],
    name: str | None = None,
    default_preset: str | None = None,
) -> str:
    """Show a preset name or a compact custom runtime without assignment badges.

    Args:
        settings: Effective profile shown in the editor.
        profiles: Available saved profiles.
        name: Exact preset chosen during this editing session.
        default_preset: Saved default to prefer for matching selections.

    Returns:
        A preset name or harness/model/effort summary.
    """
    matched = _matching_preset(settings, profiles, name, default_preset)
    return matched if matched is not None else _describe(settings)


def _remember_agent_presets(selected: Settings, profiles: dict[str, HarnessSettings], names: dict[str, str | None]) -> Settings:
    """Save role labels alongside the run's effective settings without changing preset files.

    Args:
        selected: Completed run selections.
        profiles: Presets available during the editor session.
        names: Exact preset names explicitly chosen for individual roles.

    Returns:
        The run settings with available preset names captured for display and resume.
    """
    labels = {}
    for key in ROLES:
        preferred = names.get(key, selected.agent_presets.get(key))
        name = _matching_preset(selected.model_settings(role=key), profiles, preferred, selected.default_preset)
        if name is not None:
            labels[key] = name

    selected.agent_presets = labels
    return selected


def _agent_profile(
    label: str,
    current: HarnessSettings,
    profiles: dict[str, HarnessSettings],
    default_preset: str | None,
    catalogs: dict[str, ModelCatalog],
    same_as: HarnessSettings | None = None,
    inherit_default: bool = False,
    current_name: str | None = None,
) -> ProfileChoice:
    """Select a saved profile or configure a runtime for a group or individual role.

    Args:
        label: Group or individual role shown as the menu title.
        current: Existing runtime selection for this group or role.
        profiles: Saved named single-agent profiles.
        default_preset: Saved default used to resolve otherwise identical profile choices.
        catalogs: Shared native discovery results for custom selections.
        same_as: Runtime offered as an explicit inheritance choice.
        inherit_default: Whether inheritance is initially selected.
        current_name: Exact preset name most recently selected for this row.

    Returns:
        The runtime, selected preset name, and whether the inheritance option was chosen.
    """
    labels: list[str] = []
    selections: list[ProfileChoice] = []
    if same_as is not None:
        labels.append("Same as working agents")
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
    default = next((index for index, choice in enumerate(selections) if choice.settings == current and not choice.inherited), 0)
    if preferred is not None:
        default = preferred
    if inherit_default and same_as is not None:
        default = 0

    name, separator, details = labels[default].partition(" · ")
    labels[default] = f"{name} (selected){separator}{details}"
    description = _GROUP_DESCRIPTIONS.get(label) or next((_ROLE_DESCRIPTIONS[key] for key, role in ROLES.items() if role.name == label), "")
    choice = _choose(label, [*labels, "Configure for this run"], default, descriptions=[description] * (len(labels) + 1))
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
    selected.agent_presets = {}
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
            role_names = {key: names.get("deep" if role.deep else "quick") for key, role in ROLES.items()}
            return _remember_agent_presets(selected, profiles, role_names)

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
            return _remember_agent_presets(selected, profiles, names)

        key = roles[cursor]
        role = ROLES[key]
        current = selected.model_settings(role=key)
        choice = _agent_profile(
            role.name,
            current,
            profiles,
            selected.default_preset,
            catalogs,
            current_name=names.get(key),
        )
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
    typer.echo("Ctrl+C cancels setup.")
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
