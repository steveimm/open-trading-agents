from __future__ import annotations

# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
import contextlib
import html
import http.client
import json
import logging
from datetime import datetime
from typing import Any
from urllib.request import Request, urlopen

from open_trading_agents.dataflows.date_window import coverage_gap, in_window
from open_trading_agents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)

_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_UA = "open-trading-agents/0.1"


def _created_at(message: dict[str, Any]) -> datetime | None:
    """Parse a message's ISO 8601 ``created_at``; None when missing or malformed."""
    raw = message.get("created_at")
    if not raw:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return None


def _within_window(messages: list[dict[str, Any]], start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    """Keep only messages published in [start_date, end_date] (look-ahead safe).

    No window (both None) leaves the list untouched for live callers. A message
    whose ``created_at`` (ISO 8601) is unparseable is dropped in a historical
    window, since we can't prove it isn't from after the as-of date (#1220).
    """
    if not (start_date and end_date):
        return messages
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    return [m for m in messages if in_window(_created_at(m), start_dt, end_dt)]


def _body(message: dict[str, Any]) -> str:
    """The message text; the API serves it HTML-escaped (``&amp;``, ``&#39;``)."""
    return html.unescape(message.get("body") or "")


def _stocktwits_symbol(ticker: str) -> str:
    """Map a crypto pair to StockTwits' ``<BASE>.X`` convention.

    StockTwits lists crypto as ``BTC.X`` (Yahoo's ``BTC-USD`` form 404s), so any
    crypto symbol resolves to its base plus ``.X``; other symbols pass through
    upper-cased.
    """
    base = crypto_base(ticker)
    return f"{base}.X" if base else ticker.strip().upper()


def fetch_stocktwits_messages(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Fetch dated StockTwits messages with their native sentiment labels.

    The public stream only serves recent messages. Historical windows beyond
    its coverage are reported as unavailable rather than as absent discussion.

    Args:
        ticker: Instrument whose symbol stream is requested.
        limit: Maximum eligible messages to render.
        timeout: Network request timeout in seconds.
        start_date: Optional first included date, YYYY-MM-DD.
        end_date: Optional last included date, YYYY-MM-DD.

    Returns:
        Label counts and messages with bodies limited to 700 characters, or a
        notice describing unavailable data or absent messages.
    """
    url = _API.format(ticker=_stocktwits_symbol(ticker))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
        return f"<stocktwits unavailable: {type(exc).__name__}>"

    fetched = data.get("messages", []) if isinstance(data, dict) else []
    messages = _within_window(fetched, start_date, end_date)
    if not messages:
        if start_date and end_date:
            gap = coverage_gap(
                (_created_at(m) for m in fetched),
                start_date,
                end_date,
                "StockTwits",
                f"messages about ${ticker.upper()}",
            )
            return gap or (f"<no StockTwits messages for ${ticker.upper()} within {start_date}..{end_date}>")
        return f"<no StockTwits messages found for ${ticker.upper()}>"

    lines = []
    bullish = bearish = unlabeled = 0
    for m in messages[:limit]:
        created = m.get("created_at", "")
        user = (m.get("user") or {}).get("username", "?")
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        body = _body(m).replace("\n", " ").strip()
        if len(body) > 700:
            body = body[:700] + "…"

        if sentiment == "Bullish":
            bullish += 1
            tag = "Bullish"
        elif sentiment == "Bearish":
            bearish += 1
            tag = "Bearish"
        else:
            unlabeled += 1
            tag = "no-label"
        lines.append(f"[{created} · @{user} · {tag}] {body}")

    total = bullish + bearish + unlabeled
    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0
    summary = (
        f"Bullish: {bullish} ({bull_pct}%) · Bearish: {bearish} ({bear_pct}%) · Unlabeled: {unlabeled} · Total: {total} most-recent messages"
    )
    return summary + "\n\n" + "\n".join(lines)
