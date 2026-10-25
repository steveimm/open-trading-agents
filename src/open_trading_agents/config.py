from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import tomllib
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, SerializationInfo, field_serializer, field_validator, model_validator
from tomlkit.exceptions import ParseError
from tomlkit.toml_document import TOMLDocument

from open_trading_agents.analysis.models import HarnessConfig
from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.dataflows.config import DATA_CREDENTIAL_KEYS
from open_trading_agents.dataflows.symbols import normalize_symbol

_PRESET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def application_dir() -> Path:
    """Return the shared directory for application settings, presets, and run data."""
    return Path.home() / ".open-trading-agents"


def default_config_path() -> Path:
    """Return the configuration file used when no explicit path is supplied."""
    return application_dir() / "config.toml"


def preset_path(name: str) -> Path:
    """Resolve a named preset within the application's preset directory.

    Args:
        name: A name containing letters, numbers, dots, underscores, or hyphens.

    Returns:
        The corresponding TOML path.

    Raises:
        ValueError: The name is empty, too long, or contains path characters.
    """
    if not _PRESET_NAME.fullmatch(name):
        raise ValueError("Preset names must start with a letter or number and contain up to 64 letters, numbers, dots, underscores, or hyphens.")
    return application_dir() / "presets" / f"{name}.toml"


def list_presets() -> tuple[str, ...]:
    """Return available preset names in alphabetical order."""
    return tuple(sorted(path.stem for path in (application_dir() / "presets").glob("*.toml")
                        if path.is_file() and _PRESET_NAME.fullmatch(path.stem)))


class HarnessSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["pi", "codex", "claude"] = "pi"
    command: tuple[str, ...] = ()
    model: str | None = None
    provider: str | None = None
    effort: str | None = None
    extra_args: tuple[str, ...] = ()

    @field_validator("model", "provider", "effort")
    @classmethod
    def normalize_selector(cls, value: str | None) -> str | None:
        """Interpret an empty selector as the native harness default.

        Args:
            value: Optional provider, model, or effort selection.

        Returns:
            The trimmed selection, or None for native defaults.
        """
        return (value.strip() or None) if value is not None else None

    @model_validator(mode="after")
    def validate_provider(self) -> HarnessSettings:
        """Keep Claude's provider selection within its native configuration.

        Returns:
            Settings whose selectors can be applied by the chosen adapter.

        Raises:
            ValueError: A provider override is supplied for Claude Code.
        """
        if self.kind == "claude" and self.provider:
            raise ValueError("Claude Code selects its provider through its native configuration.")

        return self

    def runtime(self) -> HarnessConfig:
        """Create the immutable CLI connection configuration.

        Returns:
            Settings passed directly to the selected harness adapter.
        """
        return HarnessConfig(**self.model_dump())

    def profile_identity(self) -> tuple[str, str | None, str | None, str | None]:
        """Return the public selectors that identify a reusable agent profile.

        Returns:
            Harness, provider, model, and effort, without execution configuration.
        """
        return self.kind, self.provider, self.model, self.effort


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    ticker: str
    quantity: float
    average_price: float | None = Field(default=None, ge=0)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Normalize holdings using the same instrument aliases as an analysis.

        Args:
            value: User-supplied holding symbol.

        Returns:
            The canonical instrument symbol.
        """
        return normalize_symbol(value)


class Portfolio(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cash: float | None = None
    currency: str | None = None
    positions: list[Position] = Field(default_factory=list)

    def render(self, ticker: str) -> str:
        """Describe supplied holdings without inferring missing account information.

        Args:
            ticker: The instrument being analyzed.

        Returns:
            A portfolio evidence block for decision agents.
        """
        ticker = normalize_symbol(ticker)
        held = [p for p in self.positions if p.ticker == ticker]
        lines = [f"Portfolio supplied by the user, currency: {self.currency or 'unspecified' }."]
        if not held:
            lines.append(f"No current position in {ticker}.")
        if self.cash is not None:
            lines.append(f"Available cash: {self.cash:g}.")
        for position in self.positions:
            lines.append(f"{position.ticker}: {position.quantity:g} units, average price {position.average_price}.")
        return "\n".join(lines)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    # Runtime storage also records isolated backtest paths for resume.
    state_dir: Path = Field(default_factory=application_dir)
    default_preset: str | None = None
    harness: HarnessSettings = Field(default_factory=HarnessSettings)
    quick: HarnessSettings | None = None
    deep: HarnessSettings | None = None
    agents: dict[str, HarnessSettings] = Field(default_factory=dict)
    agent_presets: dict[str, str] = Field(default_factory=dict)
    analysts: tuple[Literal["market", "social", "news", "fundamentals"], ...] = ("market", "social", "news", "fundamentals")
    research_rounds: int = Field(default=1, ge=1, le=20)
    risk_rounds: int = Field(default=1, ge=1, le=20)
    analyst_concurrency: int = Field(default=4, ge=1, le=4)
    max_tool_rounds: int = Field(default=20, ge=1)
    output_language: str = "English"
    holding_period_days: int = Field(default=5, ge=1)
    benchmark_ticker: str | None = None
    memory_enabled: bool = True
    allow_execution: bool = True
    web_research: bool = True
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data", mode="before")
    @classmethod
    def omit_data_credentials(cls, value: Any) -> Any:
        """Keep file-based credentials and their locator outside persisted analysis settings.

        Args:
            value: Raw data configuration from TOML or a saved run.

        Returns:
            Data settings without authentication values or their source-file locator.

        Raises:
            ValueError: A supplied credential field is not a string.
        """
        if not isinstance(value, Mapping):
            return value
        if any(key in value and not isinstance(value[key], str) for key in DATA_CREDENTIAL_KEYS):
            raise ValueError("Data credential fields must be strings.")

        return {key: item for key, item in value.items() if key not in (*DATA_CREDENTIAL_KEYS, "credentials_file")}

    @field_serializer("data")
    def serialize_data_settings(self, value: dict[str, Any]) -> dict[str, Any]:
        """Exclude credentials even after a caller updates mutable data settings.

        Args:
            value: Current data settings.

        Returns:
            Persistable data preferences without credentials or their file locator.
        """
        return self.omit_data_credentials(value)

    @field_validator("default_preset")
    @classmethod
    def validate_default_preset(cls, value: str | None) -> str | None:
        """Validate a stored preset name without requiring its file during run resume.

        Args:
            value: Optional default preset name.

        Returns:
            The validated name, or None when no default is selected.
        """
        if value is not None:
            value = value.strip() or None
        if value is not None:
            preset_path(value)
        return value

    @model_validator(mode="after")
    def resolve_model_tiers(self) -> Settings:
        """Resolve group selections while retaining sparse role overrides.

        Returns:
            Settings with stable group defaults and explicit role fields preserved.
        """
        for name in ("quick", "deep"):
            tier = getattr(self, name)
            if tier is None:
                continue
            setattr(self, name, _resolve_harness(tier, self.harness))

        return self

    @field_validator("agents", "agent_presets")
    @classmethod
    def validate_agent_roles(cls, value: dict[str, HarnessSettings] | dict[str, str]) -> dict[str, HarnessSettings] | dict[str, str]:
        """Reject role names that the workflow cannot dispatch.

        Args:
            value: Runtime overrides or saved profile labels keyed by canonical role name.

        Returns:
            Role selections or labels with valid workflow keys.

        Raises:
            ValueError: A supplied role is not part of the workflow.
        """
        unknown = value.keys() - ROLES.keys()
        if unknown:
            raise ValueError(f"Unknown agent roles: {', '.join(sorted(unknown))}. Valid roles: {', '.join(ROLES)}.")

        return value

    @field_validator("agent_presets")
    @classmethod
    def validate_agent_preset_names(cls, value: dict[str, str]) -> dict[str, str]:
        """Validate saved display labels without consulting mutable preset files.

        Args:
            value: Selected preset names retained for each role in this run.

        Returns:
            Labels whose preset names follow the supported syntax.
        """
        for name in value.values():
            preset_path(name)

        return value

    @field_serializer("agents")
    def serialize_agent_overrides(self, value: dict[str, HarnessSettings], info: SerializationInfo) -> dict[str, dict[str, Any]]:
        """Keep inherited role fields absent when settings are copied or persisted.

        Args:
            value: Explicit role selections.
            info: Requested Python or JSON serialization mode.

        Returns:
            Role selections containing only explicitly supplied fields.
        """
        return {role: selection.model_dump(mode=info.mode, exclude_unset=True) for role, selection in value.items()}

    @field_validator("analysts")
    @classmethod
    def validate_analysts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty selections and remove duplicate analyst keys.

        Args:
            value: Selected analyst keys.

        Returns:
            The unique keys in input order.
        """
        if not value:
            raise ValueError("select at least one analyst")
        return tuple(dict.fromkeys(value))

    def model_settings(self, deep: bool = False, *, role: str | None = None) -> HarnessSettings:
        """Choose a role override or its manager/working-agent defaults.

        Args:
            deep: Whether to use manager defaults when no role is supplied.
            role: Canonical role whose override and group determine the selection.

        Returns:
            The effective role or group selection, falling back to the shared harness.

        Raises:
            ValueError: The supplied role does not exist.
        """
        if role is not None:
            if role not in ROLES:
                raise ValueError(f"Unknown agent role: {role}.")
            deep = ROLES[role].deep

        baseline = (self.deep if deep else self.quick) or self.harness
        return _resolve_harness(self.agents[role], baseline) if role in self.agents else baseline


def _resolve_harness(selection: HarnessSettings, baseline: HarnessSettings) -> HarnessSettings:
    """Merge explicit selectors, resetting incompatible fields on a harness change.

    Args:
        selection: Partial group or role overrides.
        baseline: Inherited settings for the role or group.

    Returns:
        Fully resolved independent harness settings.
    """
    selected = selection.model_dump(exclude_unset=True)
    kind = selected.get("kind", baseline.kind)
    base = baseline if kind == baseline.kind else HarnessSettings(kind=kind)

    return HarnessSettings.model_validate({**base.model_dump(), **selected})


def load_preset(name: str) -> HarnessSettings:
    """Read one reusable harness, provider, model, and effort profile.

    Args:
        name: Saved profile selected for an agent role or configured as the default.

    Returns:
        A native harness selection without application settings or CLI command overrides.
    """
    selected = preset_path(name)
    if not selected.is_file():
        available = ", ".join(list_presets()) or "none"
        raise ValueError(f"Unknown preset {name!r}. Available presets: {available}")

    with selected.open("rb") as handle:
        selections = tomllib.load(handle)
    if not isinstance(selections.get("harness"), str) or set(selections) - {"harness", "provider", "model", "effort"}:
        raise ValueError(f"Preset {name!r} must contain only harness, provider, model, and effort, with harness set to pi, codex, or claude. Recreate it with 'open-trading-agents configure'.")

    selections["kind"] = selections.pop("harness")
    return HarnessSettings.model_validate(selections)


def _settings_from_config(values: dict[str, Any]) -> Settings:
    """Validate user preferences while reserving storage paths for runtime bookkeeping.

    Args:
        values: Preferences read from an application configuration file.

    Returns:
        Validated preferences using the shared application directory.

    Raises:
        ValueError: Preferences contain an unsupported setting or storage override.
    """
    if "state_dir" in values:
        raise ValueError("Shared storage is fixed at ~/.open-trading-agents. Remove state_dir from the configuration. Use --output-dir for run files.")

    return Settings.model_validate(values)


def load_settings(path: Path | None = None, use_default_preset: bool = False) -> Settings:
    """Read application settings and optionally use the default single-agent profile.

    Args:
        path: An explicit configuration file, otherwise the global default file.
        use_default_preset: Use the configured default profile as the initial agent selection.

    Returns:
        Validated settings using the shared application directory.
    """
    source = (path or default_config_path()).expanduser()
    values: dict[str, Any] = {}
    if path is not None or source.exists():
        try:
            with source.open("rb") as handle:
                values = tomllib.load(handle)
        except tomllib.TOMLDecodeError:
            raise ValueError("Configuration is not valid TOML.") from None

    settings = _settings_from_config(values)
    if use_default_preset and settings.default_preset is not None:
        name = settings.default_preset
        try:
            profile = load_preset(name)
        except ValueError as exc:
            raise ValueError(f"Default preset {name!r} could not be loaded. {exc}. Run 'open-trading-agents configure' to select a default profile.") from exc
        values.update(harness=profile, quick=None, deep=None)
        settings = Settings.model_validate(values)
        settings.agent_presets = {}
        for role in ROLES:
            try:
                selected = settings.model_settings(role=role)
            except ValueError:
                continue
            if selected.profile_identity() == profile.profile_identity():
                settings.agent_presets[role] = name

    return settings


def _configuration_document(path: Path) -> TOMLDocument:
    """Read and validate editable TOML without changing its comments.

    Args:
        path: Configuration file to read, if it exists.

    Returns:
        A validated TOML document.
    """
    try:
        document = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    except ParseError:
        raise ValueError("Configuration is not valid TOML.") from None
    _settings_from_config(document.unwrap())
    return document


def _write_document(path: Path, document: TOMLDocument, overwrite: bool = True) -> None:
    """Persist TOML atomically and restrict credential-bearing files to their owner.

    Args:
        path: Destination configuration or preset file.
        document: Validated TOML content to save.
        overwrite: Whether replacing an existing destination is allowed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(tomlkit.dumps(document))
            handle.flush()
            os.fsync(handle.fileno())
        data = document.get("data", {})
        if any(data.get(key) for key in DATA_CREDENTIAL_KEYS):
            temporary.chmod(0o600)
        elif path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_data_configuration(
    path: Path,
    values: Mapping[str, str],
    *,
    data_vendors: Mapping[str, str] | None = None,
    tool_vendors: Mapping[str, str] | None = None,
) -> Path:
    """Update data access and source selections while retaining unrelated settings and comments.

    Args:
        path: Destination application configuration.
        values: Supported credential fields, with blank values requesting removal.
        data_vendors: Category selections to merge into the existing source table.
        tool_vendors: Complete individual-tool overrides, or None to leave them unchanged.

    Returns:
        The absolute path of the atomically saved configuration.

    Raises:
        ValueError: A supplied field is unknown or its value is not a string.
    """
    if values.keys() - set(DATA_CREDENTIAL_KEYS):
        raise ValueError("Unknown data credential fields.")
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("Data credential fields must be strings.")

    for selections in (data_vendors, tool_vendors):
        if selections is not None and any(not isinstance(key, str) or not isinstance(value, str) for key, value in selections.items()):
            raise ValueError("Data source selections must be strings.")

    path = path.expanduser().resolve()
    document = _configuration_document(path)
    if "data" not in document:
        document["data"] = tomlkit.table()
    for key, value in values.items():
        if cleaned := value.strip():
            document["data"][key] = cleaned
        else:
            document["data"].pop(key, None)

    for name, selections in (("data_vendors", data_vendors), ("tool_vendors", tool_vendors)):
        if selections is None:
            continue
        if name not in document["data"]:
            if not selections:
                continue
            document["data"][name] = tomlkit.table()
        table = document["data"][name]
        if name == "tool_vendors":
            for key in list(table):
                if key not in selections:
                    del table[key]
        table.update(selections)

    _write_document(path, document)

    return path


def _set_harness_tables(
    document: TOMLDocument,
    harness: HarnessSettings,
    quick: HarnessSettings | None,
    deep: HarnessSettings | None,
) -> None:
    """Replace tier choices while retaining comments on existing configuration fields.

    Args:
        document: TOML document to update in memory.
        harness: Shared native harness selections.
        quick: Working-agent selections, or None to inherit shared choices.
        deep: Decision-manager selections, or None to inherit shared choices.
    """
    for name, selection in (("harness", harness), ("quick", quick), ("deep", deep)):
        if selection is None:
            document.pop(name, None)
            continue
        if name not in document:
            document[name] = tomlkit.table()
        values = selection.model_dump(mode="json")
        values.update(model=selection.model or "", provider=selection.provider or "", effort=selection.effort or "")
        document[name].update(values)


def save_configuration(path: Path, harness: HarnessSettings, quick: HarnessSettings | None, deep: HarnessSettings | None) -> Settings:
    """Save all guided setup selections together without changing unrelated preferences.

    Args:
        path: Destination application configuration file.
        harness: Shared native harness selections.
        quick: Working-agent selections, or None to inherit shared choices.
        deep: Decision-manager selections, or None to inherit shared choices.

    Returns:
        The complete validated settings saved to disk.
    """
    path = path.expanduser().resolve()
    document = _configuration_document(path)
    _set_harness_tables(document, harness, quick, deep)
    Settings.model_validate(document.unwrap())
    _write_document(path, document)
    return load_settings(path)


def save_preset(name: str, settings: HarnessSettings, overwrite: bool = False) -> Path:
    """Save a reusable single-agent profile containing only its four native selectors.

    Args:
        name: Name shown in the working-agent and decision-manager menus.
        settings: Native harness, provider, model, and effort selection to retain.
        overwrite: Whether the caller has selected replacement of an existing preset.

    Returns:
        The saved preset file path.

    Raises:
        FileExistsError: The preset exists and replacement was not selected.
        ValueError: The preset name or selections are invalid.
    """
    path = preset_path(name)
    settings = HarnessSettings.model_validate(settings.model_dump())
    document = tomlkit.document()
    document.update(harness=settings.kind, provider=settings.provider or "", model=settings.model or "", effort=settings.effort or "")
    _write_document(path, document, overwrite=overwrite)
    return path


def delete_preset(name: str, path: Path | None = None) -> Path:
    """Delete one saved profile and clear its known default selections.

    Args:
        name: Name of the saved profile to delete.
        path: Selected application config, also checked alongside the global config.

    Returns:
        The path of the deleted profile.

    Raises:
        ValueError: The name, selected configuration, or global configuration is invalid.
        OSError: A file cannot be updated or removed.
        RuntimeError: A failed deletion also prevents restoration of a modified config.
    """
    selected = preset_path(name)
    if not selected.is_file():
        raise ValueError(f"Unknown preset {name!r}.")
    paths = dict.fromkeys(config.expanduser().resolve() for config in (path or default_config_path(), default_config_path()))
    updates: list[tuple[Path, TOMLDocument, TOMLDocument]] = []
    for config in paths:
        if not config.exists():
            continue
        original = _configuration_document(config)
        if original.get("default_preset", "").strip() == name:
            updated = tomlkit.parse(tomlkit.dumps(original))
            updated.pop("default_preset")
            updates.append((config, original, updated))
    changed: list[tuple[Path, TOMLDocument]] = []
    try:
        for config, original, updated in updates:
            _write_document(config, updated)
            changed.append((config, original))
        selected.unlink()
    except OSError as exc:
        failed = []
        for config, original in reversed(changed):
            try:
                _write_document(config, original)
            except OSError:
                failed.append(str(config))
        if failed:
            raise RuntimeError(f"Profile deletion failed and these configurations could not be restored: {', '.join(failed)}") from exc
        raise
    return selected


def set_default_preset(path: Path, name: str | None) -> Settings:
    """Select the preset used by new runs without changing the saved harness choices.

    Args:
        path: Application configuration file to update.
        name: Existing preset name, or None to use the configuration's own choices.

    Returns:
        Updated application settings without applying the selected preset.

    Raises:
        ValueError: The named preset or existing configuration is invalid.
        OSError: The configuration cannot be written.
    """
    path = path.expanduser().resolve()
    document = _configuration_document(path)
    if name is None:
        document.pop("default_preset", None)
    else:
        load_preset(name)
        document["default_preset"] = name
    Settings.model_validate(document.unwrap())
    _write_document(path, document)
    return load_settings(path)


def save_harness_selection(
    path: Path,
    tier: Literal["default", "quick", "deep"] = "default",
    harness: Literal["pi", "codex", "claude"] | None = None,
    model: str | None = None,
    provider: str | None = None,
    effort: str | None = None,
) -> Settings:
    """Persist harness choices while preserving unrelated TOML settings and comments.

    Args:
        path: Application configuration file to create or update.
        tier: Shared defaults, working agents, or decision managers.
        harness: Optional native harness selection.
        model: Optional model selection, with an empty string selecting native defaults.
        provider: Optional provider selection, with an empty string selecting native defaults.
        effort: Optional native reasoning effort, with an empty string selecting native defaults.

    Returns:
        The validated settings saved to disk.

    Raises:
        ValueError: The existing or resulting configuration is invalid.
        OSError: The configuration cannot be written.
    """
    if tier not in ("default", "quick", "deep"):
        raise ValueError(f"unknown configuration tier: {tier}")
    path = path.expanduser().resolve()
    document = _configuration_document(path)
    settings = Settings.model_validate(document.unwrap())
    name = "harness" if tier == "default" else tier
    current = getattr(settings, name) or settings.harness
    if name not in document:
        document[name] = tomlkit.table()
    selected = document[name]

    if harness is not None and harness != current.kind:
        for key in ("model", "provider", "effort", "command", "extra_args"):
            selected.pop(key, None)
        if name != "harness" and harness == settings.harness.kind:
            selected.update(model="", provider="", effort="", command=[], extra_args=[])
        if name == "harness":
            for child_name in ("quick", "deep"):
                child = document.get(child_name)
                if child is not None and "kind" not in child:
                    for key in ("model", "provider", "effort", "command", "extra_args"):
                        child.pop(key, None)
    if harness is not None:
        selected["kind"] = harness
    for key, value in (("model", model), ("provider", provider), ("effort", effort)):
        if value is not None:
            selected[key] = value.strip()
    Settings.model_validate(document.unwrap())

    _write_document(path, document)
    return load_settings(path)


def load_portfolio(path: Path | None) -> Portfolio | None:
    """Read optional caller-supplied portfolio JSON.

    Args:
        path: Portfolio file, or None when holdings are unknown.

    Returns:
        Validated holdings, preserving the difference between absent and flat.
    """
    return Portfolio.model_validate(json.loads(path.read_text())) if path else None


def validate_date(value: str) -> str:
    """Accept a canonical analysis date no later than the local calendar date.

    Args:
        value: Date supplied by the caller.

    Returns:
        The canonical date string.

    Raises:
        ValueError: The date is malformed or in the future.
    """
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value or parsed > date.today():
        raise ValueError("analysis date must be YYYY-MM-DD and cannot be in the future")
    return value
