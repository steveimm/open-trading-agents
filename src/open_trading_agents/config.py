from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from tomlkit.toml_document import TOMLDocument

from open_trading_agents.dataflows.symbols import normalize_symbol
from open_trading_agents.domain import HarnessConfig

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
            value: Optional provider or model selection.

        Returns:
            The trimmed selection, or None for native defaults.
        """
        return (value.strip() or None) if value is not None else None

    def runtime(self) -> HarnessConfig:
        """Create the immutable CLI connection configuration.

        Returns:
            Settings passed directly to the selected harness adapter.
        """
        return HarnessConfig(**self.model_dump())


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
    model_config = ConfigDict(extra="forbid")
    state_dir: Path = Field(default_factory=application_dir)
    default_preset: str | None = None
    harness: HarnessSettings = Field(default_factory=HarnessSettings)
    quick: HarnessSettings | None = None
    deep: HarnessSettings | None = None
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
    social_screening: bool = False
    data: dict[str, Any] = Field(default_factory=dict)

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
        """Resolve partial tier selections against the shared harness configuration.

        Returns:
            Settings with stable, fully resolved tiers for persistence and resume.
        """
        for name in ("quick", "deep"):
            tier = getattr(self, name)
            if tier is None:
                continue
            selected = tier.model_dump(exclude_unset=True)
            kind = selected.get("kind", self.harness.kind)
            base = self.harness if kind == self.harness.kind else HarnessSettings(kind=kind)
            setattr(self, name, HarnessSettings.model_validate({**base.model_dump(), **selected}))
        return self

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

    def model_settings(self, deep: bool = False) -> HarnessSettings:
        """Choose the manager or working-agent harness settings.

        Args:
            deep: Whether this is a decision manager.

        Returns:
            The configured tier, falling back to the shared harness.
        """
        return (self.deep if deep else self.quick) or self.harness


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
    profile = HarnessSettings.model_validate({"kind": selections["harness"], **{key: value for key, value in selections.items() if key != "harness"}})
    if profile.kind == "claude" and profile.provider:
        raise ValueError("Claude Code selects its provider through its native configuration.")
    return profile


def load_settings(path: Path | None = None, use_default_preset: bool = False) -> Settings:
    """Read application settings and optionally use the default single-agent profile.

    Args:
        path: An explicit configuration file, otherwise the global default file.
        use_default_preset: Use the configured default profile as the initial agent selection.

    Returns:
        Validated settings, with relative state paths based on the config file.
    """
    source = (path or default_config_path()).expanduser()
    values: dict[str, Any] = {}
    if path is not None or source.exists():
        with source.open("rb") as handle:
            values = tomllib.load(handle)
    settings = Settings.model_validate(values)
    if use_default_preset and settings.default_preset is not None:
        try:
            profile = load_preset(settings.default_preset)
        except ValueError as exc:
            raise ValueError(f"Default preset {settings.default_preset!r} could not be loaded. {exc}. Run 'open-trading-agents configure' to select a default profile.") from exc
        values.update(harness=profile, quick=None, deep=None)
        settings = Settings.model_validate(values)
    settings.state_dir = settings.state_dir.expanduser()
    if not settings.state_dir.is_absolute():
        settings.state_dir = source.resolve().parent / settings.state_dir
    return settings


def _configuration_document(path: Path) -> TOMLDocument:
    """Read and validate editable TOML without changing its comments.

    Args:
        path: Configuration file to read, if it exists.

    Returns:
        A validated TOML document.
    """
    document = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    Settings.model_validate(document.unwrap())
    return document


def _write_document(path: Path, document: TOMLDocument, overwrite: bool = True) -> None:
    """Persist a complete TOML document atomically, preserving existing permissions.

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
        if path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _set_harness_tables(document: TOMLDocument, harness: HarnessSettings, quick: HarnessSettings | None, deep: HarnessSettings | None) -> None:
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
        if selection.kind == "claude" and selection.provider:
            raise ValueError("Claude Code selects its provider through its native configuration.")
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
    if settings.kind == "claude" and settings.provider:
        raise ValueError("Claude Code selects its provider through its native configuration.")
    document = tomlkit.document()
    document.update(harness=settings.kind, provider=settings.provider or "", model=settings.model or "", effort=settings.effort or "")
    _write_document(path, document, overwrite=overwrite)
    return path


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
    kind = harness or current.kind
    if kind == "claude" and provider is not None and provider.strip():
        raise ValueError("Claude Code selects its provider through its native configuration. Omit --provider.")
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
