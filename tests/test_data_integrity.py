from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from open_trading_agents.dataflows import router
from open_trading_agents.dataflows.config import run_config
from open_trading_agents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from open_trading_agents.dataflows.symbols import normalize_symbol, safe_ticker_component
from open_trading_agents.dataflows.vendors import fred, polymarket, reddit, sec_edgar, stocktwits
from open_trading_agents.dataflows.vendors.alpha_vantage import common as alpha
from open_trading_agents.dataflows.vendors.yahoo import market, ohlcv, snapshot


@pytest.mark.parametrize(
    ("given", "canonical"),
    [("nvda", "NVDA"), ("BTCUSDT", "BTC-USD"), ("ETH-USDC", "ETH-USD"), ("XAUUSD+", "GC=F"), ("0700.HK", "0700.HK")],
)
def test_source_symbol_normalization(given: str, canonical: str) -> None:
    """Preserve exchange symbols, crypto quote aliases, and supported commodity aliases."""
    assert normalize_symbol(given) == canonical


@pytest.mark.parametrize("symbol", ["../escape", "/tmp/escape", "x/y", "..", "x\\y"])
def test_ticker_cannot_escape_local_cache(symbol: str) -> None:
    """Reject symbols that could write a cache outside its configured directory."""
    with pytest.raises(ValueError):
        safe_ticker_component(symbol)


def test_vendor_fallback_never_queries_unconfigured_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly selected source must not silently fall through to another provider."""
    primary = Mock(side_effect=VendorUnavailableError("throttled"))
    unexpected = Mock(return_value="unrequested provider evidence")
    monkeypatch.setitem(router.VENDOR_METHODS, "get_stock_data", {"yfinance": primary, "alpha_vantage": unexpected})
    with run_config({"data_vendors": {"core_stock_apis": "yfinance"}}):
        assert router.route_to_vendor("get_stock_data", "NVDA", "2024-01-01", "2024-03-15").startswith("DATA_UNAVAILABLE")
    unexpected.assert_not_called()
    with run_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}}):
        assert "unrequested provider" in router.route_to_vendor("get_stock_data", "NVDA", "2024-01-01", "2024-03-15")
    unexpected.assert_called_once()


def test_failed_provider_is_distinct_from_absent_instrument(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fallback's missing symbol cannot hide the primary provider's outage."""
    monkeypatch.setitem(
        router.VENDOR_METHODS,
        "get_stock_data",
        {
            "yfinance": Mock(side_effect=VendorUnavailableError("throttled")),
            "alpha_vantage": Mock(side_effect=NoMarketDataError("NVDA")),
        },
    )
    with run_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}}):
        result = router.route_to_vendor("get_stock_data", "NVDA", "2024-01-01", "2024-03-15")
    assert result.startswith("DATA_UNAVAILABLE")
    assert not result.startswith("NO_DATA_AVAILABLE")


def test_sec_uses_filing_date_and_preserves_earlier_restatement() -> None:
    """Fiscal period end alone must not disclose a fact before its filing or restatement."""
    facts = {
        "Assets": {
            "units": {
                "USD": [
                    {"end": "2023-12-31", "filed": "2024-02-15", "val": 100, "form": "10-K"},
                    {"end": "2023-12-31", "filed": "2024-04-01", "val": 120, "form": "10-K/A"},
                ]
            }
        }
    }
    assert sec_edgar._as_of(facts, ("Assets",), "2024-01-31", ((300, 400),))[0] == {}
    assert sec_edgar._as_of(facts, ("Assets",), "2024-03-15", ((300, 400),))[0] == {("2023-12-31", 0): 100}
    assert sec_edgar._as_of(facts, ("Assets",), "2024-04-02", ((300, 400),))[0] == {("2023-12-31", 0): 120}


def test_fred_pins_metadata_and_observations_to_known_vintage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical macro reports must not use revised values unavailable on the run date."""
    request = Mock(
        side_effect=[
            {"seriess": [{"title": "Unemployment", "units": "%", "frequency": "Monthly"}]},
            {"observations": [{"date": "2024-01-01", "value": "4.0"}, {"date": "2024-02-01", "value": "4.1"}]},
        ]
    )
    monkeypatch.setattr(fred, "_request", request)
    result = fred.get_macro_data("unemployment", "2024-03-15", 90)
    assert "4.1" in result
    for call in request.call_args_list:
        assert call.args[1]["realtime_start"] == call.args[1]["realtime_end"] == "2024-03-15"


def test_alpha_request_contains_no_tracking_source_and_scrubs_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Market requests must not add attribution tracking or expose the configured API key."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "fixture-secret")
    response = Mock(text=json.dumps({"Note": "fixture-secret rate limit reached"}))
    request = Mock(return_value=response)
    monkeypatch.setattr(alpha, "get_scrubbed", request)
    with pytest.raises(alpha.AlphaVantageRateLimitError) as error:
        alpha._make_api_request("TIME_SERIES_DAILY", {"symbol": "NVDA"})
    assert "source" not in request.call_args.kwargs["params"]
    assert "fixture-secret" not in str(error.value)


def test_polymarket_historical_run_does_not_request_live_odds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present-day probabilities must not enter historical analysis."""
    request = Mock(side_effect=AssertionError("Live probability fetch forbidden"))
    monkeypatch.setattr(polymarket, "_request", request)
    result = polymarket.get_prediction_markets("recession", 6, "2024-03-15")
    assert "historical" in result.lower() or "unavailable" in result.lower()
    request.assert_not_called()


def test_social_windows_exclude_future_and_undated_posts() -> None:
    """Historical sentiment can only consume items whose timestamps establish eligibility."""
    posts: list[dict[str, Any]] = [
        {"created_utc": datetime(2024, 3, 15, 23, 59, tzinfo=UTC).timestamp(), "title": "past"},
        {"created_utc": datetime(2024, 3, 16, tzinfo=UTC).timestamp(), "title": "future"},
        {"title": "undated"},
    ]
    assert [post["title"] for post in reddit._within_window(posts, "2024-03-01", "2024-03-15")] == ["past"]
    messages = [{"created_at": "2024-03-15T23:59:00Z", "body": "past"}, {"created_at": "2024-03-16T00:00:00Z"}, {}]
    assert stocktwits._within_window(messages, "2024-03-01", "2024-03-15") == [messages[0]]


def test_verified_snapshot_excludes_future_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verification must independently filter even a vendor result containing future rows."""
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-03-14", "2024-03-15", "2024-03-18"]),
            "Open": [99.0, 100.0, 99999.0],
            "High": [101.0, 102.0, 99999.0],
            "Low": [98.0, 99.0, 99999.0],
            "Close": [100.0, 101.0, 99999.0],
            "Volume": [1000, 1200, 1300],
        }
    )
    monkeypatch.setattr(snapshot, "load_ohlcv", Mock(return_value=frame))
    result = snapshot.build_verified_market_snapshot("NVDA", "2024-03-15", indicators=("close_10_ema",))
    assert "Latest trading row used: 2024-03-15" in result
    assert "99999" not in result and "2024-03-18" not in result
    with pytest.raises(NoMarketDataError, match="stale"):
        ohlcv._assert_ohlcv_not_stale(frame, "2024-06-01", "NVDA")


def test_market_evidence_preserves_small_prices_and_signed_indicators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep source precision across price CSV and the verified snapshot used by agents.

    Args:
        monkeypatch: Replace market requests with deterministic local observations.
    """
    frame = pd.DataFrame(
        {
            "Open": [0.004012345678, 0.004123456789],
            "High": [0.0042, 0.0043],
            "Low": [0.0039, 0.0040],
            "Close": [0.0041, 0.004234567891],
            "Volume": [1000, 1200],
        },
        index=pd.DatetimeIndex(["2024-03-14", "2024-03-15"], name="Date"),
    )
    ticker = Mock()
    ticker.history.return_value = frame.copy()
    monkeypatch.setattr(market.yf, "Ticker", Mock(return_value=ticker))

    csv = market.get_YFin_data_online("DOGE-USD", "2024-03-14", "2024-03-15")
    observations = pd.read_csv(StringIO(csv), comment="#", index_col="Date")
    for field in ("Open", "High", "Low", "Close"):
        assert observations[field].tolist() == pytest.approx(frame[field].tolist(), rel=0, abs=1e-15)

    rows = frame.reset_index()
    rows["tiny_indicator"] = [-1.5e-8, -2.5e-8]
    monkeypatch.setattr(snapshot, "load_ohlcv", Mock(return_value=rows))
    report = snapshot.build_verified_market_snapshot("DOGE-USD", "2024-03-15", indicators=("tiny_indicator",))
    assert "| Close | 0.004234567891 |" in report
    assert "| 2024-03-15 | 0.004234567891 |" in report
    assert "| tiny_indicator | -2.5e-08 |" in report


def test_historical_window_without_price_rows_is_explicitly_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a provider window whose observations all fall after the requested cutoff.

    Args:
        tmp_path: Isolated market-data cache.
        monkeypatch: Replace provider requests with a newer observation window.
    """
    frame = pd.DataFrame(
        {"Open": [100.0], "High": [102.0], "Low": [99.0], "Close": [101.0], "Volume": [1000]},
        index=pd.DatetimeIndex(["2024-03-15"], name="Date"),
    )
    ticker = Mock()
    ticker.history.return_value = frame
    monkeypatch.setattr(ohlcv.yf, "Ticker", Mock(return_value=ticker))

    with run_config({"data_cache_dir": str(tmp_path)}), pytest.raises(NoMarketDataError, match="no price rows on or before"):
        ohlcv.load_ohlcv("NVDA", "2020-03-02")


def test_missing_indicator_observation_does_not_claim_a_market_holiday(monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe an absent weekday observation without inventing a calendar explanation.

    Args:
        monkeypatch: Supply an indicator series with a missing weekday.
    """
    monkeypatch.setattr(market, "_get_stock_stats_bulk", Mock(return_value={"2024-03-14": "100.5"}))

    report = market.get_stock_stats_indicators_window("NVDA", "close_50_sma", "2024-03-15", 1)

    assert "2024-03-15: N/A: No observation available for this date" in report
    assert "2024-03-14: 100.5" in report
    assert "weekend" not in report and "holiday" not in report

    frame = pd.DataFrame({"Date": pd.to_datetime(["2024-03-14"]), "Close": [100.5]})
    monkeypatch.setattr(market, "load_ohlcv", Mock(return_value=frame))
    assert market.get_stock_stats("NVDA", "close_50_sma", "2024-03-15") == market.MISSING_OBSERVATION


def test_http_failure_does_not_retain_secret_in_request_or_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query-string credentials must not propagate through a failed HTTP request."""
    from open_trading_agents.dataflows.net import get_scrubbed

    failure = requests.HTTPError("request https://provider.invalid/?key=fixture-secret")
    monkeypatch.setattr(requests, "get", Mock(side_effect=failure))
    with pytest.raises(requests.HTTPError) as error:
        get_scrubbed("https://provider.invalid", params={}, timeout=1, secret="fixture-secret")
    assert "fixture-secret" not in str(error.value)
    assert error.value.request is None and error.value.response is None and error.value.__context__ is None
