# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
import json
import logging

from open_trading_agents.dataflows.errors import NoMarketDataError, VendorError
from open_trading_agents.dataflows.indicators import INDICATOR_DESCRIPTIONS
from open_trading_agents.dataflows.vendors.alpha_vantage.common import _make_api_request

logger = logging.getLogger(__name__)


def get_indicator(
    symbol: str,
    indicator: str,
    as_of_date: str,
    look_back_days: int,
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close",
) -> str:
    """
    Returns Alpha Vantage technical indicator values over a time window.

    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to get the analysis and report of
        as_of_date: The current trading date you are trading on, YYYY-mm-dd
        look_back_days: how many days to look back
        interval: Time interval (daily, weekly, monthly)
        time_period: Number of data points for calculation
        series_type: The desired price type (close, open, high, low)

    Returns:
        String containing indicator values and description
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    supported_indicators = {
        "close_50_sma": ("50 SMA", "close"),
        "close_200_sma": ("200 SMA", "close"),
        "close_10_ema": ("10 EMA", "close"),
        "macd": ("MACD", "close"),
        "macds": ("MACD Signal", "close"),
        "macdh": ("MACD Histogram", "close"),
        "rsi": ("RSI", "close"),
        "boll": ("Bollinger Middle", "close"),
        "boll_ub": ("Bollinger Upper Band", "close"),
        "boll_lb": ("Bollinger Lower Band", "close"),
        "atr": ("ATR", None),
        "vwma": ("VWMA", "close"),
    }


    if indicator not in supported_indicators:
        raise NoMarketDataError(
            symbol, symbol, f"Alpha Vantage does not serve {indicator}; it serves {list(supported_indicators)}"
        )

    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    before = as_of_dt - relativedelta(days=look_back_days)

    _, required_series_type = supported_indicators[indicator]

    if required_series_type:
        series_type = required_series_type

    try:
        if indicator == "close_50_sma":
            data = _make_api_request(
                "SMA",
                {"symbol": symbol, "interval": interval, "time_period": "50", "series_type": series_type, "datatype": "csv"},
            )
        elif indicator == "close_200_sma":
            data = _make_api_request(
                "SMA",
                {"symbol": symbol, "interval": interval, "time_period": "200", "series_type": series_type, "datatype": "csv"},
            )
        elif indicator == "close_10_ema":
            data = _make_api_request(
                "EMA",
                {"symbol": symbol, "interval": interval, "time_period": "10", "series_type": series_type, "datatype": "csv"},
            )
        elif indicator == "macd" or indicator == "macds" or indicator == "macdh":
            data = _make_api_request(
                "MACD", {"symbol": symbol, "interval": interval, "series_type": series_type, "datatype": "csv"}
            )
        elif indicator == "rsi":
            data = _make_api_request(
                "RSI",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "time_period": str(time_period),
                    "series_type": series_type,
                    "datatype": "csv",
                },
            )
        elif indicator in ["boll", "boll_ub", "boll_lb"]:
            data = _make_api_request(
                "BBANDS",
                {"symbol": symbol, "interval": interval, "time_period": "20", "series_type": series_type, "datatype": "csv"},
            )
        elif indicator == "atr":
            data = _make_api_request(
                "ATR", {"symbol": symbol, "interval": interval, "time_period": str(time_period), "datatype": "csv"}
            )
        else:
            raise NoMarketDataError(symbol, symbol, f"Alpha Vantage does not serve the {indicator} indicator")

        lines = data.strip().split("\n")
        if len(lines) < 2:
            return f"Error: No data returned for {indicator}"

        header = [col.strip() for col in lines[0].split(",")]
        try:
            date_col_idx = header.index("time")
        except ValueError:
            return f"Error: 'time' column not found in data for {indicator}. Available columns: {header}"

        col_name_map = {
            "macd": "MACD",
            "macds": "MACD_Signal",
            "macdh": "MACD_Hist",
            "boll": "Real Middle Band",
            "boll_ub": "Real Upper Band",
            "boll_lb": "Real Lower Band",
            "rsi": "RSI",
            "atr": "ATR",
            "close_10_ema": "EMA",
            "close_50_sma": "SMA",
            "close_200_sma": "SMA",
        }

        target_col_name = col_name_map.get(indicator)

        if not target_col_name:
            value_col_idx = 1
        else:
            try:
                value_col_idx = header.index(target_col_name)
            except ValueError:
                return f"Error: Column '{target_col_name}' not found for indicator '{indicator}'. Available columns: {header}"

        result_data = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(",")
            if len(values) > value_col_idx:
                try:
                    date_str = values[date_col_idx].strip()
                    date_dt = datetime.strptime(date_str, "%Y-%m-%d")

                    if before <= date_dt <= as_of_dt:
                        value = values[value_col_idx].strip()
                        result_data.append((date_dt, value))
                except (ValueError, IndexError):
                    continue

        result_data.sort(key=lambda x: x[0])

        ind_string = ""
        for date_dt, value in result_data:
            ind_string += f"{date_dt.strftime('%Y-%m-%d')}: {value}\n"

        if not ind_string:
            ind_string = "No data available for the specified date range.\n"

        period = {"close_50_sma": 50, "close_200_sma": 200, "close_10_ema": 10,
                  "boll": 20, "boll_ub": 20, "boll_lb": 20, "rsi": time_period, "atr": time_period}.get(indicator)
        definition = {"provider": "Alpha Vantage", "interval": interval,
                      "price_basis": "High, Low, previous Close" if indicator == "atr" else series_type,
                      "period_observations": period} if period is not None else {
                          "provider": "Alpha Vantage", "interval": interval, "price_basis": series_type,
                          "periods": "Provider MACD defaults; fast, slow and signal periods were not overridden."}
        result_str = (
            f"""## {indicator.upper()} values from {before.strftime('%Y-%m-%d')} to {as_of_date}:
Calculation parameters: {json.dumps(definition)}
look_back_days selects the displayed history, not the calculation period. Do not assume another provider uses an identical implementation.

"""
            + ind_string
            + "\n\n"
            + INDICATOR_DESCRIPTIONS.get(indicator, "No description available.")
        )

        return result_str

    except VendorError:
        raise
    except Exception as e:
        logger.warning("Alpha Vantage indicator %s failed: %s", indicator, e)
        raise NoMarketDataError(symbol, symbol, f"{indicator} unavailable: {e}") from e
