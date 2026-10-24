from __future__ import annotations

import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from copy import deepcopy
from pathlib import Path
from typing import Any

_config: dict[str, Any] | None = None
_run_config: ContextVar[dict[str, Any] | None] = ContextVar("open_trading_agents_data_config", default=None)
DATA_CREDENTIAL_KEYS = ("fred_api_key", "alpha_vantage_api_key", "sec_edgar_user_agent")


def read_data_credentials(path: Path | None = None) -> dict[str, str]:
    """Read data-source credentials directly without retaining them in runtime settings.

    Args:
        path: Selected configuration file, or the shared application configuration.

    Returns:
        All supported fields, with missing values represented by empty strings.

    Raises:
        ValueError: The file is invalid TOML or credential values are not strings.
    """
    source = (path or Path.home() / ".open-trading-agents" / "config.toml").expanduser()
    values: dict[str, Any] = {}
    if source.exists():
        try:
            with source.open("rb") as handle:
                values = tomllib.load(handle)
        except tomllib.TOMLDecodeError:
            raise ValueError("Data configuration is not valid TOML.") from None

    data = values.get("data", {})
    if not isinstance(data, dict) or any(not isinstance(data.get(key, ""), str) for key in DATA_CREDENTIAL_KEYS):
        raise ValueError("Data credential fields must be strings in the [data] table.")

    return {key: data.get(key, "").strip() for key in DATA_CREDENTIAL_KEYS}


def get_data_credential(name: str) -> str:
    """Read the current configured credential at the time a data request is made.

    Args:
        name: One of the supported data credential fields.

    Returns:
        The selected file's current value, or an empty string when unconfigured.

    Raises:
        ValueError: The credential name or file contents are invalid.
    """
    if name not in DATA_CREDENTIAL_KEYS:
        raise ValueError("Unknown data credential name.")
    source = get_config().get("credentials_file")

    return read_data_credentials(Path(source) if source else None)[name]


def default_data_config() -> dict[str, Any]:
    """Return independent defaults for deterministic data adapters.

    Returns:
        A fresh configuration with vendor chains and local cache settings.
    """
    return {
        "data_cache_dir": str(Path.home() / ".open-trading-agents" / "cache"),
        "news_article_limit": 20,
        "web_search_backend": "yahoo",
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
        if key in DATA_CREDENTIAL_KEYS:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value

    return base


def build_data_config(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Combine data defaults with independently owned override values.

    Args:
        overrides: Scalar settings and optional nested vendor mappings.

    Returns:
        Complete data settings that share no mutable values with the caller.
    """
    return _merge(default_data_config(), overrides)


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
    token = _run_config.set(build_data_config(config))
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
    context.run(_run_config.set, build_data_config(config))
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
