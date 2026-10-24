# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
import logging
from datetime import datetime
from typing import Annotated

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from open_trading_agents.dataflows.errors import NoMarketDataError, VendorError
from open_trading_agents.dataflows.indicators import INDICATOR_DESCRIPTIONS
from open_trading_agents.dataflows.symbols import normalize_symbol
from open_trading_agents.dataflows.vendors.yahoo.common import raise_for_empty, yf_retry
from open_trading_agents.dataflows.vendors.yahoo.ohlcv import _assert_ohlcv_not_stale, load_ohlcv

logger = logging.getLogger(__name__)
MISSING_OBSERVATION = "N/A: No observation available for this date"


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch an inclusive OHLCV window without rounding the source prices.

    Args:
        symbol: Instrument symbol or supported alias.
        start_date: First date to include, in YYYY-MM-DD format.
        end_date: Last date to include, in YYYY-MM-DD format.

    Returns:
        CSV observations with the requested window and record count.

    Raises:
        NoMarketDataError: The source has no usable or sufficiently recent prices.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    canonical = normalize_symbol(symbol)

    end_inclusive = (end_dt + relativedelta(days=1)).strftime("%Y-%m-%d")
    data = yf_retry(lambda: yf.Ticker(canonical).history(start=start_date, end=end_inclusive))
    if data is None or data.empty:
        raise_for_empty(symbol, canonical, f"rows between {start_date} and {end_date}")

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    _assert_ohlcv_not_stale(data, end_date, symbol, canonical)

    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    return f"""# Stock data for {label} from {start_date} to {end_date}
# Total records: {len(data)}

{data.to_csv()}"""


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    as_of_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Report a technical indicator over a trailing date window."""

    if indicator not in INDICATOR_DESCRIPTIONS:
        raise ValueError(f"Indicator {indicator} is not supported. Please choose from: {list(INDICATOR_DESCRIPTIONS.keys())}")

    end_date = as_of_date
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    before = as_of_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, as_of_date)

        current_dt = as_of_dt
        date_values = []

        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")

            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = MISSING_OBSERVATION

            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)

        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"

    except VendorError:
        raise
    except Exception as e:
        logger.warning("Bulk stockstats fetch failed, falling back per-day: %s", e)

        ind_string = ""
        as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
        while as_of_dt >= before:
            indicator_value = get_stockstats_indicator(symbol, indicator, as_of_dt.strftime("%Y-%m-%d"))
            ind_string += f"{as_of_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            as_of_dt = as_of_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + INDICATOR_DESCRIPTIONS.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    as_of_date: Annotated[str, "current date for reference"],
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from stockstats import wrap

    data = load_ohlcv(symbol, as_of_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df[indicator]

    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]

        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)

    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    as_of_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """Compute one technical indicator from the shared cached OHLCV history."""
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    as_of_date = as_of_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = get_stock_stats(
            symbol,
            indicator,
            as_of_date,
        )
    except VendorError:
        raise
    except Exception as e:
        raise NoMarketDataError(symbol, symbol, f"{indicator} could not be read for {as_of_date}: {e}") from e

    return str(indicator_value)


def get_closes(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """Daily closes from ``start_date`` up to, not including, ``end_date``."""
    canonical = normalize_symbol(symbol)
    history = yf_retry(lambda: yf.Ticker(canonical).history(start=start_date, end=end_date))
    return history["Close"] if history is not None and "Close" in history else pd.Series(dtype=float)


def get_stock_stats(
    symbol: Annotated[str, "ticker symbol for the company"],
    indicator: Annotated[str, "quantitative indicators based off of the stock data for the company"],
    as_of_date: Annotated[str, "curr date for retrieving stock price data, YYYY-mm-dd"],
) -> float | str:
    """Read the requested indicator on an exact trading date."""
    data = load_ohlcv(symbol, as_of_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    as_of_str = pd.to_datetime(as_of_date).strftime("%Y-%m-%d")

    df[indicator]
    matching_rows = df[df["Date"].str.startswith(as_of_str)]

    if not matching_rows.empty:
        indicator_value = matching_rows[indicator].values[0]
        return indicator_value
    else:
        return MISSING_OBSERVATION
