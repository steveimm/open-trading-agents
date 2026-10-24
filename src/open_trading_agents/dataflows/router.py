from __future__ import annotations

# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
import logging
from typing import Any

from open_trading_agents.dataflows.config import get_config
from open_trading_agents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_balance_sheet as get_alpha_vantage_balance_sheet,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_cashflow as get_alpha_vantage_cashflow,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_fundamentals as get_alpha_vantage_fundamentals,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_global_news as get_alpha_vantage_global_news,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_income_statement as get_alpha_vantage_income_statement,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_indicator as get_alpha_vantage_indicator,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_insider_transactions as get_alpha_vantage_insider_transactions,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_news as get_alpha_vantage_news,
)
from open_trading_agents.dataflows.vendors.alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
)
from open_trading_agents.dataflows.vendors.fred import get_macro_data as get_fred_macro_data
from open_trading_agents.dataflows.vendors.polymarket import (
    get_prediction_markets as get_polymarket_prediction_markets,
)
from open_trading_agents.dataflows.vendors.sec_edgar import (
    get_balance_sheet as get_sec_edgar_balance_sheet,
)
from open_trading_agents.dataflows.vendors.sec_edgar import (
    get_cashflow as get_sec_edgar_cashflow,
)
from open_trading_agents.dataflows.vendors.sec_edgar import (
    get_income_statement as get_sec_edgar_income_statement,
)
from open_trading_agents.dataflows.vendors.yahoo.fundamentals import (
    get_balance_sheet as get_yfinance_balance_sheet,
)
from open_trading_agents.dataflows.vendors.yahoo.fundamentals import (
    get_cashflow as get_yfinance_cashflow,
)
from open_trading_agents.dataflows.vendors.yahoo.fundamentals import (
    get_fundamentals as get_yfinance_fundamentals,
)
from open_trading_agents.dataflows.vendors.yahoo.fundamentals import (
    get_income_statement as get_yfinance_income_statement,
)
from open_trading_agents.dataflows.vendors.yahoo.fundamentals import (
    get_insider_transactions as get_yfinance_insider_transactions,
)
from open_trading_agents.dataflows.vendors.yahoo.market import (
    get_stock_stats_indicators_window,
    get_YFin_data_online,
)
from open_trading_agents.dataflows.vendors.yahoo.news import get_global_news_yfinance, get_news_yfinance

logger = logging.getLogger(__name__)


TOOLS_CATEGORIES = {
    "core_stock_apis": {"description": "OHLCV stock price data", "tools": ["get_stock_data"]},
    "technical_indicators": {"description": "Technical analysis indicators", "tools": ["get_indicators"]},
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": ["get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"],
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ],
    },
    "macro_data": {
        "description": "Macroeconomic indicators (rates, inflation, labor, growth)",
        "tools": [
            "get_macro_indicators",
        ],
    },
    "prediction_markets": {
        "description": "Market-implied probabilities for forward-looking events",
        "tools": [
            "get_prediction_markets",
        ],
    },
}


OPTIONAL_CATEGORIES = {"macro_data", "prediction_markets"}


VENDOR_METHODS = {
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "sec_edgar": get_sec_edgar_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "sec_edgar": get_sec_edgar_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "sec_edgar": get_sec_edgar_income_statement,
        "yfinance": get_yfinance_income_statement,
    },
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
    },
    "get_macro_indicators": {
        "fred": get_fred_macro_data,
    },
    "get_prediction_markets": {
        "polymarket": get_polymarket_prediction_markets,
    },
}


def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str | None = None) -> str:
    """Resolve a tool override before its category's configured vendor chain.

    Args:
        category: Financial-data category to resolve.
        method: Optional method whose tool-specific override takes precedence.

    Returns:
        A comma-separated vendor chain, or default when no chain is configured.
    """
    config = get_config()

    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    return config.get("data_vendors", {}).get(category, "default")


def vendor_unavailable(method: str, error: Exception) -> str:
    """What a call returns when every vendor was throttled or unreachable."""
    return (
        f"DATA_UNAVAILABLE: no configured vendor could serve {method} right now ({error}). This says nothing about the instrument; report the data as unavailable and do not estimate or fabricate values."
    )


def no_data_available(error: NoMarketDataError) -> str:
    """What a call returns when every vendor that answered had no usable data."""
    resolved = "" if error.canonical == error.symbol else f" (resolved to '{error.canonical}')"

    reason = f" ({error.detail})" if error.detail else ""
    return (
        f"NO_DATA_AVAILABLE: No usable market data for '{error.symbol}'{resolved} from any configured vendor{reason}. The symbol may be invalid, delisted, not covered, or the vendor returned stale data. Do not estimate or fabricate values — report that data is unavailable for this symbol."
    )


def route_to_vendor(method: str, *args: Any, **kwargs: Any) -> str:
    """Try the configured provider chain without silently adding unconfigured sources.

    Args:
        method: Registered financial-data method to invoke.
        args: Positional arguments accepted by that method's adapters.
        kwargs: Keyword arguments accepted by those adapters.

    Returns:
        Provider evidence, or an explicit notice distinguishing unavailable
        providers from missing instrument data.

    Raises:
        ValueError: The method or its configured provider selection is unsupported.
        Exception: Every required provider fails without a data-availability result.
    """
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(",")]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    all_available_vendors = list(VENDOR_METHODS[method].keys())

    explicit = [v for v in primary_vendors if v and v != "default"]
    if explicit:
        vendor_chain = [v for v in explicit if v in VENDOR_METHODS[method]]
        if not vendor_chain:
            raise ValueError(f"Configured vendor(s) {explicit} not available for '{method}'. Available: {all_available_vendors}.")
    else:
        vendor_chain = all_available_vendors

    last_no_data: NoMarketDataError | None = None
    last_unavailable: VendorUnavailableError | None = None
    failed: Exception | None = None
    first_error: Exception | None = None
    for vendor in vendor_chain:
        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except VendorUnavailableError as e:
            logger.warning("Vendor %r unavailable for %s: %s; trying next vendor.", vendor, method, e)

            last_unavailable = e
            continue
        except VendorNotConfiguredError as e:
            logger.warning("Vendor %r not configured for %s; trying next vendor.", vendor, method)
            if first_error is None:
                first_error = e
            continue
        except NoMarketDataError as e:
            last_no_data = e
            continue
        except Exception as e:
            logger.warning("Vendor %r failed for %s: %s", vendor, method, e)
            if first_error is None:
                first_error = e
            failed = e
            continue

    if last_unavailable is not None:
        return vendor_unavailable(method, last_unavailable)
    if failed is not None and last_no_data is not None:
        return vendor_unavailable(method, failed)

    if last_no_data is not None:
        if first_error is not None:
            logger.warning(
                "Returning NO_DATA for %s, but a vendor errored earlier: %s",
                method,
                first_error,
            )
        return no_data_available(last_no_data)

    if first_error is not None:
        if category in OPTIONAL_CATEGORIES:
            logger.warning("Optional %s unavailable for %s: %s", category, method, first_error)
            return (
                f"DATA_UNAVAILABLE: optional {category} could not be retrieved ({first_error}). Proceed without it; do not fabricate values."
            )
        raise first_error

    raise RuntimeError(f"No available vendor for '{method}'")
