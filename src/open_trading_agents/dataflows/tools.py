from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from open_trading_agents.dataflows.artifacts import save_tool_data
from open_trading_agents.dataflows.config import run_config
from open_trading_agents.dataflows.date_window import as_of, as_of_window
from open_trading_agents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from open_trading_agents.dataflows.router import no_data_available, route_to_vendor, vendor_unavailable
from open_trading_agents.dataflows.symbols import safe_ticker_component
from open_trading_agents.dataflows.vendors.reddit import fetch_reddit_posts, subreddits_for
from open_trading_agents.dataflows.vendors.stocktwits import fetch_stocktwits_messages
from open_trading_agents.dataflows.vendors.yahoo.snapshot import build_verified_market_snapshot
from open_trading_agents.dataflows.web import fetch_webpage, search_web

TOOL_NAMES = (
    "get_stock_data",
    "get_indicators",
    "get_verified_market_snapshot",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
    "get_social_sentiment",
    "search_web",
    "fetch_webpage",
)


@dataclass(frozen=True)
class ToolContext:
    ticker: str
    trade_date: str
    asset_type: str = "stock"
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        safe_ticker_component(self.ticker)
        parsed = datetime.strptime(self.trade_date, "%Y-%m-%d").date()
        if parsed.isoformat() != self.trade_date or parsed > date.today():
            raise ValueError("trade_date must be YYYY-MM-DD and no later than today")
        if self.asset_type not in ("stock", "crypto"):
            raise ValueError("asset_type must be stock or crypto")

    @classmethod
    def from_file(cls, path: Path) -> ToolContext:
        """Load the immutable instrument binding used by one agent server.

        Args:
            path: Local JSON file containing ticker, trade_date, asset_type, and config.

        Returns:
            A validated tool context.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ticker=payload["ticker"],
            trade_date=payload["trade_date"],
            asset_type=payload.get("asset_type", "stock"),
            config=payload.get("config", {}),
        )


class FinancialTools:
    def __init__(self, context: ToolContext, artifact_dir: Path | None = None) -> None:
        """Bind financial tools to one instrument and analysis date.

        Args:
            context: Instrument identity and data configuration supplied by the workflow.
            artifact_dir: Optional evidence directory supplied by the application, never a model argument.
        """
        self.context = context
        self.artifact_dir = artifact_dir

    def _result(self, name: str, text: str) -> str:
        """Retain evidence for scripts when the workflow supplies an agent workspace.

        Args:
            name: Internal tool name.
            text: Complete source output.

        Returns:
            Source output with direct file references when artifacts are enabled.
        """
        return save_tool_data(self.artifact_dir, name, text) if self.artifact_dir is not None else text

    def _route(self, name: str, *args: Any) -> str:
        """Invoke the configured vendor chain inside this run's data context.

        Args:
            name: Registered financial method.
            args: Arguments already constrained to the analysis date.

        Returns:
            Vendor evidence or an explicit unavailability notice.
        """
        with run_config(self.context.config):
            result = route_to_vendor(name, *args)
        return self._result(name, result)

    def get_stock_data(self, start_date: str, end_date: str) -> str:
        """Retrieve OHLCV prices for the instrument under analysis.

        Args:
            start_date: Inclusive start date, YYYY-MM-DD.
            end_date: Inclusive end date, clamped to the analysis date.

        Returns:
            Prices and volume as CSV with source context.
        """
        start_date, end_date = as_of_window(start_date, end_date, self.context.trade_date)
        return self._route("get_stock_data", self.context.ticker, start_date, end_date)

    def get_indicators(self, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
        """Retrieve technical indicators for the bound instrument.

        Args:
            indicator: Indicator name such as rsi, macd, or close_50_sma, optionally comma-separated.
            curr_date: Last date to serve, clamped to the analysis date.
            look_back_days: Number of trailing calendar days to report.

        Returns:
            Indicator observations and interpretation guidance.
        """
        curr_date = as_of(curr_date, self.context.trade_date)
        results = []
        for name in (part.strip().lower() for part in indicator.split(",")):
            if not name:
                continue
            try:
                results.append(self._route("get_indicators", self.context.ticker, name, curr_date, look_back_days))
            except ValueError as exc:
                results.append(str(exc))
        return "\n\n".join(results)

    def get_verified_market_snapshot(self, curr_date: str, look_back_days: int = 30) -> str:
        """Verify exact price and indicator claims against a shared OHLCV snapshot.

        Args:
            curr_date: Last date to serve, clamped to the analysis date.
            look_back_days: Number of recent trading rows to include.

        Returns:
            Latest OHLCV, technical values, recent closes, or an unavailability notice.
        """
        try:
            with run_config(self.context.config):
                result = build_verified_market_snapshot(self.context.ticker, as_of(curr_date, self.context.trade_date), look_back_days)
            return self._result("get_verified_market_snapshot", result)
        except VendorUnavailableError as exc:
            return vendor_unavailable("get_verified_market_snapshot", exc)
        except NoMarketDataError as exc:
            return no_data_available(exc)

    def get_fundamentals(self, curr_date: str) -> str:
        """Retrieve a company profile and valuation metrics as of the analysis date.

        Args:
            curr_date: Requested date, clamped to the analysis date.

        Returns:
            Company evidence, with present-day snapshots withheld for historical runs.
        """
        return self._route("get_fundamentals", self.context.ticker, as_of(curr_date, self.context.trade_date))

    def get_balance_sheet(self, freq: str = "quarterly", curr_date: str | None = None) -> str:
        """Retrieve financial position statements known by the analysis date.

        Args:
            freq: Reporting frequency, annual or quarterly.
            curr_date: Requested cutoff, defaulting to the analysis date.

        Returns:
            A balance sheet with its available filing dates and units.
        """
        return self._route("get_balance_sheet", self.context.ticker, freq, as_of(curr_date, self.context.trade_date))

    def get_cashflow(self, freq: str = "quarterly", curr_date: str | None = None) -> str:
        """Retrieve cash-flow statements known by the analysis date.

        Args:
            freq: Reporting frequency, annual or quarterly.
            curr_date: Requested cutoff, defaulting to the analysis date.

        Returns:
            Cash-flow statements with available filing dates and units.
        """
        return self._route("get_cashflow", self.context.ticker, freq, as_of(curr_date, self.context.trade_date))

    def get_income_statement(self, freq: str = "quarterly", curr_date: str | None = None) -> str:
        """Retrieve income statements known by the analysis date.

        Args:
            freq: Reporting frequency, annual or quarterly.
            curr_date: Requested cutoff, defaulting to the analysis date.

        Returns:
            Income statements with available filing dates and units.
        """
        return self._route("get_income_statement", self.context.ticker, freq, as_of(curr_date, self.context.trade_date))

    def get_news(self, start_date: str, end_date: str) -> str:
        """Retrieve company news inside a date window bounded by the run.

        Args:
            start_date: Inclusive start date, YYYY-MM-DD.
            end_date: Inclusive end date, clamped to the analysis date.

        Returns:
            Dated articles or an explicit coverage notice.
        """
        start_date, end_date = as_of_window(start_date, end_date, self.context.trade_date)
        return self._route("get_news", self.context.ticker, start_date, end_date)

    def get_global_news(self, curr_date: str, look_back_days: int | None = None, limit: int | None = None) -> str:
        """Retrieve macroeconomic news through configured topic searches.

        Args:
            curr_date: Last date to serve, clamped to the analysis date.
            look_back_days: Trailing days, or the configured default.
            limit: Maximum articles, or the configured default.

        Returns:
            Dated macro news and coverage notices.
        """
        return self._route("get_global_news", as_of(curr_date, self.context.trade_date), look_back_days, limit)

    def get_insider_transactions(self) -> str:
        """Retrieve disclosed insider transactions known by the analysis date.

        Returns:
            Disclosed transactions, or a notice when publication dates cannot be established.
        """
        return self._route("get_insider_transactions", self.context.ticker, self.context.trade_date)

    def get_macro_indicators(self, indicator: str, curr_date: str, look_back_days: int | None = None) -> str:
        """Retrieve a FRED series using the vintage available by the analysis date.

        Args:
            indicator: Alias such as cpi, core_pce, unemployment, fed_funds_rate, or a FRED series ID.
            curr_date: Last date to serve, clamped to the analysis date.
            look_back_days: Trailing days, defaulting to one year.

        Returns:
            Series units, frequency, values, and change over the window.
        """
        return self._route("get_macro_indicators", indicator, as_of(curr_date, self.context.trade_date), look_back_days)

    def get_prediction_markets(self, topic: str, limit: int | None = None) -> str:
        """Retrieve current Polymarket probabilities for an event topic.

        Args:
            topic: Event keywords such as Fed rate cut or recession.
            limit: Maximum markets, defaulting to six.

        Returns:
            Market probabilities, or an explicit notice for historical runs.
        """
        return self._route("get_prediction_markets", topic, limit, self.context.trade_date)

    def get_social_sentiment(self, start_date: str | None = None, end_date: str | None = None) -> str:
        """Collect dated news, Reddit posts, and StockTwits messages for the sentiment analyst.

        Args:
            start_date: Inclusive start, defaulting to seven days before the analysis date.
            end_date: Inclusive end, defaulting to and bounded by the analysis date.

        Returns:
            Three labelled evidence blocks, including source-availability and coverage notices.
        """
        end_date = end_date or self.context.trade_date
        start_date = start_date or (date.fromisoformat(self.context.trade_date) - timedelta(days=7)).isoformat()
        start_date, end_date = as_of_window(start_date, end_date, self.context.trade_date)
        try:
            news = self.get_news(start_date, end_date)
        except Exception as exc:
            news = f"<news unavailable: {type(exc).__name__}>"
        with run_config(self.context.config):
            stocktwits = fetch_stocktwits_messages(self.context.ticker, start_date=start_date, end_date=end_date)
            reddit = fetch_reddit_posts(
                self.context.ticker,
                subreddits_for(self.context.ticker),
                start_date=start_date,
                end_date=end_date,
            )
        return self._result("get_social_sentiment", f"## News\n{news}\n\n## StockTwits\n{stocktwits}\n\n## Reddit\n{reddit}")

    def search_web(self, query: str, limit: int = 5) -> str:
        """Find additional public sources and return their citeable URLs and snippets.

        Args:
            query: Search terms, optionally including a source domain restriction.
            limit: Maximum results, between one and ten.

        Returns:
            Current search evidence with explicit historical-index limitations.
        """
        with run_config(self.context.config):
            result = search_web(query, self.context.trade_date, limit)
        return self._result("search_web", result)

    def fetch_webpage(self, url: str, max_chars: int = 20000) -> str:
        """Read public-page text with its URL, title, source-reported dates, and limitations.

        Args:
            url: Absolute public HTTP or HTTPS URL.
            max_chars: Maximum visible text length, between one and 20000.

        Returns:
            Source evidence without scripts, images, subresource requests, or unsupported historical claims.
        """
        return self._result("fetch_webpage", fetch_webpage(url, self.context.trade_date, max_chars))


def create_server(context: ToolContext, names: tuple[str, ...] = TOOL_NAMES, *, artifact_dir: Path | None = None) -> MCPServer:
    """Create a stdio-only MCP server with explicitly selected financial tools.

    Args:
        context: Immutable instrument and configuration binding.
        names: Allowed tool names for this agent.
        artifact_dir: Application-selected directory for exact evidence files.

    Returns:
        A local MCP server with no remote-service registration or tracing.
    """
    unknown = set(names) - set(TOOL_NAMES)
    if unknown:
        raise ValueError(f"Unknown financial tools: {', '.join(sorted(unknown))}")
    server = MCPServer("open-trading-agents-data", log_level="WARNING")
    tools = FinancialTools(context, artifact_dir)
    for name in dict.fromkeys(names):
        server.add_tool(getattr(tools, name), name=name)
    return server


def main() -> None:
    """Serve the requested financial tools over local standard input and output."""
    parser = argparse.ArgumentParser(description="Local financial-data MCP server")
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--tools", help="Comma-separated allowed tool names, defaulting to all financial tools")
    parser.add_argument("--artifact-dir", type=Path, help="Agent workspace directory for exact tool evidence")
    arguments = parser.parse_args()
    context = ToolContext.from_file(arguments.context)
    names = tuple(part.strip() for part in arguments.tools.split(",") if part.strip()) if arguments.tools is not None else TOOL_NAMES
    create_server(context, names, artifact_dir=arguments.artifact_dir).run(transport="stdio")


if __name__ == "__main__":
    main()
