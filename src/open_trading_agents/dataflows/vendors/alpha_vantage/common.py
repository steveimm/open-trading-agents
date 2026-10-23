# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
import json
import os
from datetime import datetime
from io import StringIO

import pandas as pd

from open_trading_agents.dataflows.errors import VendorNotConfiguredError, VendorUnavailableError
from open_trading_agents.dataflows.net import get_scrubbed

API_BASE_URL = "https://www.alphavantage.co/query"


REQUEST_TIMEOUT = 30


class AlphaVantageNotConfiguredError(VendorNotConfiguredError):
    """Raised when Alpha Vantage is selected but no API key is configured.

    A VendorNotConfiguredError (and thus still a ValueError), so the routing
    layer's "vendor unavailable" handling and existing ValueError callers both
    keep working.
    """

    pass


def get_api_key() -> str:
    """Retrieve the API key for Alpha Vantage from environment variables."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise AlphaVantageNotConfiguredError("ALPHA_VANTAGE_API_KEY environment variable is not set.")
    return api_key


def format_datetime_for_api(date_input: str | datetime, end_of_day: bool = False) -> str:
    """Convert various date formats to the YYYYMMDDTHHMM Alpha Vantage expects.

    A plain date means midnight, which is the start of that day. For the end of
    a window pass ``end_of_day`` so the day itself is inside it, rather than
    dropping everything published on the analysis date.
    """
    if isinstance(date_input, str):
        if len(date_input) == 13 and "T" in date_input:
            return date_input

        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return dt.strftime("%Y%m%dT2359" if end_of_day else "%Y%m%dT0000")
        except ValueError:
            try:
                dt = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
                return dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                raise ValueError(f"Unsupported date format: {date_input}") from None
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")


class AlphaVantageRateLimitError(VendorUnavailableError):
    """Raised when the Alpha Vantage API rate limit is exceeded."""

    pass


def _make_api_request(function_name: str, params: dict) -> dict | str:
    """Helper function to make API requests and handle responses.

    Raises:
        AlphaVantageRateLimitError: When API rate limit is exceeded
    """

    api_key = get_api_key()
    api_params = params.copy()
    api_params.update(
        {
            "function": function_name,
            "apikey": api_key,
        }
    )

    current_entitlement = globals().get("_current_entitlement")
    entitlement = api_params.get("entitlement") or current_entitlement

    if entitlement:
        api_params["entitlement"] = entitlement
    elif "entitlement" in api_params:
        api_params.pop("entitlement", None)

    response = get_scrubbed(API_BASE_URL, params=api_params, timeout=REQUEST_TIMEOUT, secret=api_key)

    response_text = response.text

    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    def scrubbed(text: str) -> str:
        """Remove the API key from provider error messages before returning them."""
        return text.replace(api_key, "***") if api_key else text

    notice = response_json.get("Information") or response_json.get("Note")
    if notice:
        notice = scrubbed(notice)
        low = notice.lower()
        if any(m in low for m in ("rate limit", "requests per day", "call frequency", "premium")):
            raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit exceeded: {notice}")
        if "api key" in low or "apikey" in low:
            raise AlphaVantageNotConfiguredError(f"Alpha Vantage API key invalid or missing: {notice}")

    rejection = response_json.get("Error Message")
    if rejection:
        raise VendorUnavailableError(f"Alpha Vantage rejected the request: {scrubbed(rejection)}")

    return response_text


def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    """
    Filter CSV data to include only rows within the specified date range.

    Args:
        csv_data: CSV string from Alpha Vantage API
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Filtered CSV string
    """
    if not csv_data or csv_data.strip() == "":
        return csv_data

    df = pd.read_csv(StringIO(csv_data))

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    filtered_df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

    return filtered_df.to_csv(index=False)
