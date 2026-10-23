from __future__ import annotations

import json
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from open_trading_agents.dataflows.symbols import normalize_symbol
from open_trading_agents.domain import HarnessConfig


class HarnessSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["pi", "codex", "claude"] = "pi"
    command: tuple[str, ...] = ()
    model: str | None = None
    provider: str | None = None
    extra_args: tuple[str, ...] = ()

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
