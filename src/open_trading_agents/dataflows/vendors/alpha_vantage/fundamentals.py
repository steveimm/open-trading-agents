# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
from open_trading_agents.dataflows.date_window import withhold_live_profile, withhold_undated_statements
from open_trading_agents.dataflows.vendors.alpha_vantage.common import _make_api_request


def get_fundamentals(ticker: str, as_of_date: str | None = None) -> str:
    """Retrieve a current company overview or explain why a historical one is unavailable.

    Args:
        ticker: Company ticker symbol.
        as_of_date: Analysis cutoff, or None for the current view.

    Returns:
        Provider overview data, or a withholding notice because the endpoint
        provides no historical vintage.
    """
    withheld = withhold_live_profile(as_of_date, ticker)
    if withheld:
        return withheld

    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Retrieve current balance sheets from Alpha Vantage.

    Args:
        ticker: Company ticker symbol.
        freq: Requested frequency. This provider returns both annual and quarterly reports.
        as_of_date: Analysis cutoff, or None for the current view.

    Returns:
        Provider statement data, or a notice withholding undated historical data.
    """
    withheld = withhold_undated_statements(as_of_date, ticker, "Balance Sheet")
    if withheld:
        return withheld

    return _make_api_request("BALANCE_SHEET", {"symbol": ticker})


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Retrieve current cash-flow statements from Alpha Vantage.

    Args:
        ticker: Company ticker symbol.
        freq: Requested frequency. This provider returns both annual and quarterly reports.
        as_of_date: Analysis cutoff, or None for the current view.

    Returns:
        Provider statement data, or a notice withholding undated historical data.
    """
    withheld = withhold_undated_statements(as_of_date, ticker, "Cash Flow")
    if withheld:
        return withheld

    return _make_api_request("CASH_FLOW", {"symbol": ticker})


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Retrieve current income statements from Alpha Vantage.

    Args:
        ticker: Company ticker symbol.
        freq: Requested frequency. This provider returns both annual and quarterly reports.
        as_of_date: Analysis cutoff, or None for the current view.

    Returns:
        Provider statement data, or a notice withholding undated historical data.
    """
    withheld = withhold_undated_statements(as_of_date, ticker, "Income Statement")
    if withheld:
        return withheld

    return _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
