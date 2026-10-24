from __future__ import annotations

import json
import os
import stat
import tempfile
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from open_trading_agents.dataflows.symbols import normalize_symbol
from open_trading_agents.domain import HarnessConfig


class HarnessSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["pi", "codex", "claude"] = "pi"
    command: tuple[str, ...] = ()
    model: str | None = None
    provider: str | None = None
    extra_args: tuple[str, ...] = ()

    @field_validator("model", "provider")
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
    state_dir: Path = Field(default_factory=lambda: Path.home() / ".local" / "share" / "open-trading-agents")
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


def load_settings(path: Path | None = None) -> Settings:
    """Read explicit TOML configuration without searching unrelated directories.

    Args:
        path: An optional application configuration file.

    Returns:
        Validated settings, with relative state paths based on the config file.
    """
    if path is None:
        return Settings()
    with path.open("rb") as handle:
        settings = Settings.model_validate(tomllib.load(handle))
    settings.state_dir = settings.state_dir.expanduser()
    if not settings.state_dir.is_absolute():
        settings.state_dir = path.resolve().parent / settings.state_dir
    return settings


def save_harness_selection(
    path: Path,
    tier: Literal["default", "quick", "deep"] = "default",
    harness: Literal["pi", "codex", "claude"] | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Settings:
    """Persist harness choices while preserving unrelated TOML settings and comments.

    Args:
        path: Application configuration file to create or update.
        tier: Shared defaults, working agents, or decision managers.
        harness: Optional native harness selection.
        model: Optional model selection, with an empty string selecting native defaults.
        provider: Optional provider selection, with an empty string selecting native defaults.

    Returns:
        The validated settings saved to disk.

    Raises:
        ValueError: The existing or resulting configuration is invalid.
        OSError: The configuration cannot be written.
    """
    if tier not in ("default", "quick", "deep"):
        raise ValueError(f"unknown configuration tier: {tier}")
    path = path.expanduser().resolve()
    document = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
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
        for key in ("model", "provider", "command", "extra_args"):
            selected.pop(key, None)
        if name != "harness" and harness == settings.harness.kind:
            selected.update(model="", provider="", command=[], extra_args=[])
        if name == "harness":
            for child_name in ("quick", "deep"):
                child = document.get(child_name)
                if child is not None and "kind" not in child:
                    for key in ("model", "provider", "command", "extra_args"):
                        child.pop(key, None)
    if harness is not None:
        selected["kind"] = harness
    for key, value in (("model", model), ("provider", provider)):
        if value is not None:
            selected[key] = value.strip()
    Settings.model_validate(document.unwrap())

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
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
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
