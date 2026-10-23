from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from copy import deepcopy
from pathlib import Path
from typing import Any

_config: dict[str, Any] | None = None
_run_config: ContextVar[dict[str, Any] | None] = ContextVar("open_trading_agents_data_config", default=None)


def default_data_config() -> dict[str, Any]:
    """Return independent defaults for deterministic data adapters.

    Returns:
        A fresh configuration with vendor chains and local cache settings.
    """
    return {
        "data_cache_dir": str(Path.home() / ".local" / "share" / "open-trading-agents" / "cache"),
        "news_article_limit": 20,
        "global_news_article_limit": 10,
        "global_news_lookback_days": 7,
        "global_news_queries": [
            "Federal Reserve interest rates inflation",
            "S&P 500 earnings GDP economic outlook",
            "geopolitical risk trade war sanctions",
            "ECB Bank of England BOJ central bank policy",
            "oil commodities supply chain energy",
        ],
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "sec_edgar,yfinance",
            "news_data": "yfinance",
            "macro_data": "fred",
            "prediction_markets": "polymarket",
        },
        "tool_vendors": {},
    }


def _merge(base: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge scalar settings and one level of nested vendor overrides.

    Args:
        base: Destination defaults.
        config: Overrides to copy into the destination.

    Returns:
        The updated destination.
    """
    for key, value in deepcopy(dict(config)).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value

    return base


def initialize_config() -> None:
    """Initialize process defaults without touching the filesystem or network."""
    global _config
    if _config is None:
        _config = default_data_config()


def set_config(config: Mapping[str, Any]) -> None:
    """Set process defaults while preserving unspecified data settings.

    Args:
        config: Data settings to merge over the existing defaults.
    """
    global _config
    _config = _merge(deepcopy(_config) if _config is not None else default_data_config(), config)


@contextmanager
def run_config(config: Mapping[str, Any]) -> Iterator[None]:
    """Bind an isolated configuration to the current tool execution.

    Args:
        config: Settings for this run.

    Yields:
        Control while the run's settings are active.
    """
    token = _run_config.set(_merge(default_data_config(), config))
    try:
        yield
    finally:
        _run_config.reset(token)


def run_config_context(config: Mapping[str, Any]) -> Context:
    """Create an execution context carrying an isolated data configuration.

    Args:
        config: Settings for calls performed inside the context.

    Returns:
        A context suitable for invoking synchronous data functions.
    """
    context = copy_context()
    context.run(_run_config.set, _merge(default_data_config(), config))
    return context


def get_config() -> dict[str, Any]:
    """Read a copy of the current run's configuration or process defaults.

    Returns:
        Isolated data settings that callers may safely inspect and modify.
    """
    scoped = _run_config.get()
    if scoped is not None:
        return deepcopy(scoped)
    return deepcopy(_config) if _config is not None else default_data_config()
