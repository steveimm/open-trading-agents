from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from open_trading_agents import tools as module
from open_trading_agents.dataflows.config import get_config
from open_trading_agents.tools import FinancialTools, ToolContext, create_server


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("get_stock_data", {"start_date": "2024-01-01", "end_date": "2099-01-01"}, ("NVDA", "2024-01-01", "2024-03-15")),
        ("get_news", {"start_date": "2024-01-01", "end_date": "2099-01-01"}, ("NVDA", "2024-01-01", "2024-03-15")),
        ("get_fundamentals", {"curr_date": "2099-01-01"}, ("NVDA", "2024-03-15")),
        ("get_balance_sheet", {}, ("NVDA", "quarterly", "2024-03-15")),
        ("get_cashflow", {"curr_date": "2099-01-01"}, ("NVDA", "quarterly", "2024-03-15")),
        ("get_income_statement", {"curr_date": None}, ("NVDA", "quarterly", "2024-03-15")),
        ("get_insider_transactions", {}, ("NVDA", "2024-03-15")),
        ("get_prediction_markets", {"topic": "rates"}, ("rates", None, "2024-03-15")),
        ("get_macro_indicators", {"indicator": "cpi", "curr_date": "2099-01-01"}, ("cpi", "2024-03-15", None)),
        ("get_global_news", {"curr_date": "2099-01-01"}, ("2024-03-15", None, None)),
    ],
)
def test_tool_arguments_cannot_escape_analysis_date(
    monkeypatch: pytest.MonkeyPatch, name: str, arguments: dict[str, Any], expected: tuple[Any, ...]
) -> None:
    """Keep model-provided date arguments inside the workflow's bound date."""
    route = Mock(return_value="evidence")
    monkeypatch.setattr(module, "route_to_vendor", route)
    api = FinancialTools(ToolContext("NVDA", "2024-03-15"))
    assert getattr(api, name)(**arguments) == "evidence"
    route.assert_called_once_with(name, *expected)


def test_snapshot_and_indicators_share_date_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve date restrictions on the direct snapshot and multi-indicator paths."""
    snapshot = Mock(return_value="snapshot")
    route = Mock(return_value="indicator")
    monkeypatch.setattr(module, "build_verified_market_snapshot", snapshot)
    monkeypatch.setattr(module, "route_to_vendor", route)
    api = FinancialTools(ToolContext("BTC-USD", "2024-03-15", asset_type="crypto"))
    api.get_verified_market_snapshot("2099-01-01")
    assert api.get_indicators("rsi, macd", "2099-01-01") == "indicator\n\nindicator"
    snapshot.assert_called_once_with("BTC-USD", "2024-03-15", 30)
    assert [call.args for call in route.call_args_list] == [
        ("get_indicators", "BTC-USD", "rsi", "2024-03-15", 30),
        ("get_indicators", "BTC-USD", "macd", "2024-03-15", 30),
    ]


def test_concurrent_tools_keep_their_vendor_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent server calls must not exchange their configured vendor chains."""

    def vendor(name: str, *args: Any) -> str:
        """Return the currently scoped provider without network access."""
        return get_config()["data_vendors"]["news_data"]

    monkeypatch.setattr(module, "route_to_vendor", vendor)
    first = FinancialTools(ToolContext("AAPL", "2024-03-15", config={"data_vendors": {"news_data": "yfinance"}}))
    second = FinancialTools(ToolContext("NVDA", "2024-03-15", config={"data_vendors": {"news_data": "alpha_vantage"}}))

    async def run() -> list[str]:
        """Exercise the same threaded call style used by MCP tool handlers."""
        return await asyncio.gather(
            asyncio.to_thread(first.get_news, "2024-03-01", "2024-03-15"),
            asyncio.to_thread(second.get_news, "2024-03-01", "2024-03-15"),
        )

    assert asyncio.run(run()) == ["yfinance", "alpha_vantage"]


def test_social_collection_retains_each_source_when_news_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """One unavailable source must not suppress the other deterministic social inputs."""
    monkeypatch.setattr(module, "route_to_vendor", Mock(side_effect=ValueError("secret-should-not-be-echoed")))
    stocktwits = Mock(return_value="<StockTwits unavailable for historical window>")
    reddit = Mock(return_value="A dated Reddit post")
    monkeypatch.setattr(module, "fetch_stocktwits_messages", stocktwits)
    monkeypatch.setattr(module, "fetch_reddit_posts", reddit)
    api = FinancialTools(ToolContext("BTC-USD", "2024-03-15", asset_type="crypto"))
    result = api.get_social_sentiment(end_date="2099-01-01")
    assert "news unavailable: ValueError" in result
    assert "secret-should-not-be-echoed" not in result
    assert "historical window" in result and "A dated Reddit post" in result
    assert stocktwits.call_args.kwargs["end_date"] == "2024-03-15"
    assert reddit.call_args.args[1] == ("Bitcoin", "CryptoCurrency", "CryptoMarkets")


@pytest.mark.asyncio
async def test_mcp_schema_exposes_only_selected_tools_and_no_injected_context() -> None:
    """An agent's MCP schema must not let the model replace its ticker or run date."""
    server = create_server(ToolContext("NVDA", "2024-03-15"), ("get_news", "get_prediction_markets"))
    schemas = await server.list_tools()
    assert {tool.name for tool in schemas} == {"get_news", "get_prediction_markets"}
    for tool in schemas:
        assert not {"ticker", "symbol", "trade_date", "config"} & set(tool.input_schema.get("properties", {}))
    available = await create_server(ToolContext("NVDA", "2024-03-15")).list_tools()
    assert "calculate" not in {tool.name for tool in available}
    with pytest.raises(ValueError, match="Unknown financial tools: calculate"):
        create_server(ToolContext("NVDA", "2024-03-15"), ("calculate",))


@pytest.mark.asyncio
async def test_real_stdio_mcp_process_lists_and_calls_historical_tool(tmp_path: Path) -> None:
    """Run the actual server protocol without making any market or model request."""
    context_file = tmp_path / "context.json"
    context_file.write_text(
        json.dumps({"ticker": "NVDA", "trade_date": "2024-03-15", "config": {"data_vendors": {"fundamental_data": "yfinance"}}}),
        encoding="utf-8",
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "open_trading_agents.tools", "--context", str(context_file), "--tools", "get_fundamentals"],
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams, read_timeout_seconds=10) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert [tool.name for tool in listed.tools] == ["get_fundamentals"]
            result = await session.call_tool("get_fundamentals", {"curr_date": "2099-01-01"})
            assert not result.is_error
            text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
            assert "2024-03-15" in text and "withheld" in text.lower()


def test_social_screening_uses_only_the_explicit_constructor_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send source bodies through the caller's harness callback without exposing it to model arguments."""
    monkeypatch.setattr(module, "route_to_vendor", Mock(return_value="Dated news"))
    stocktwits = Mock(return_value="Screened StockTwits")
    reddit = Mock(return_value="Screened Reddit")
    monkeypatch.setattr(module, "fetch_stocktwits_messages", stocktwits)
    monkeypatch.setattr(module, "fetch_reddit_posts", reddit)
    screen = Mock(return_value=([True, False], "One relevant bullish post"))
    api = FinancialTools(ToolContext("NVDA", "2024-03-15"), screen=screen)
    result = api.get_social_sentiment()
    assert "Screened StockTwits" in result and "Screened Reddit" in result
    assert stocktwits.call_args.kwargs["screen"] is screen
    assert reddit.call_args.kwargs["screen"] is screen
    schemas = asyncio.run(create_server(api.context, ("get_social_sentiment",)).list_tools())
    assert "screen" not in schemas[0].input_schema.get("properties", {})
