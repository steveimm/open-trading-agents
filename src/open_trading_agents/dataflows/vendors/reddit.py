from __future__ import annotations

# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
import html
import http.client
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from open_trading_agents.dataflows.date_window import coverage_gap, in_window
from open_trading_agents.dataflows.symbols import crypto_base

logger = logging.getLogger(__name__)


def _within_window(posts: list[dict[str, Any]], start_date: str | None, end_date: str | None) -> list[dict[str, Any]]:
    """Keep only posts published in [start_date, end_date] (look-ahead safe).

    No window (both None) leaves the list untouched for live callers. A post with
    no ``created_utc`` epoch is dropped in a historical window (#1220).
    """
    if not (start_date and end_date):
        return posts
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    return [p for p in posts if in_window(_posted_at(p), start_dt, end_dt)]


def _posted_at(post: dict[str, Any]) -> datetime | None:
    """A post's ``created_utc`` epoch as a UTC datetime, or None when missing."""
    ts = post.get("created_utc")
    return datetime.fromtimestamp(ts, tz=UTC) if ts else None


def _coverage_dates(posts: list[dict[str, Any]]) -> list:
    """Dates that bound the feed's coverage. The search is limited to the last
    week (``t=week``), so the lookback start bounds it even when nothing came
    back; a full page may have cut older matches off, so then only the posts
    themselves do."""
    dates = [_posted_at(p) for p in posts]
    if len(posts) < _FEED_PAGE:
        dates.append(datetime.now(UTC) - _SEARCH_LOOKBACK)
    return dates


_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"


_UA = "open-trading-agents/0.1"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


CRYPTO_SUBREDDITS = ("CryptoCurrency", "CryptoMarkets")
_COIN_SUBREDDITS = {"BTC": "Bitcoin", "ETH": "ethereum", "SOL": "solana"}


def subreddits_for(ticker: str) -> tuple[str, ...]:
    """The subreddits searched for ``ticker``: crypto communities for a crypto pair."""
    base = crypto_base(ticker)
    if not base:
        return DEFAULT_SUBREDDITS
    coin = _COIN_SUBREDDITS.get(base)
    return ((coin,) if coin else ()) + CRYPTO_SUBREDDITS


_FEED_PAGE = 100
_SCREEN_CHARS = 1000


_SEARCH_LOOKBACK = timedelta(days=7)


def _search_qs(ticker: str, limit: int) -> str:
    """Build the fixed Reddit RSS query parameters for an instrument."""
    return urlencode(
        {
            "q": ticker,
            "restrict_sr": "on",
            "sort": "new",
            "t": "week",
            "limit": limit,
        }
    )


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    """Parse an Atom ``published`` timestamp to a UTC epoch, or None."""
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    """Reduce the HTML body Reddit embeds in an Atom entry to plain text."""
    if not content:
        return ""

    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


_RETRY_FALLBACK_SECONDS = 60.0


def _jitter(seconds: float, frac: float = 0.2) -> float:
    """Return ``seconds`` with +/-``frac`` random jitter, to desynchronize
    concurrent runs pacing against the same per-IP limit."""
    return seconds * (1.0 + random.uniform(-frac, frac))


def _retry_after_seconds(exc: HTTPError) -> float | None:
    """Seconds to wait from a 429's ``Retry-After`` header, capped at 60s.

    The cap matches ``_RETRY_FALLBACK_SECONDS``: honouring less than we would
    wait on our own would spend the one retry on a request we already know is
    too early.

    Returns ``None`` only when the header is absent or unparseable; a valid
    ``Retry-After: 0`` returns ``0.0`` (retry at once), not ``None``.
    """
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 60.0) if val is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


_MAX_FEED_BYTES = 5 * 1024 * 1024


def _read_capped(resp: Any) -> bytes:
    """Read a response body bounded to ``_MAX_FEED_BYTES``, raising on overflow."""
    data = resp.read(_MAX_FEED_BYTES + 1)
    if len(data) > _MAX_FEED_BYTES:
        raise http.client.HTTPException(f"Reddit feed exceeded {_MAX_FEED_BYTES} bytes; refusing to parse")
    return data


def _fetch_subreddit_rss(
    ticker: str,
    sub: str,
    limit: int,
    timeout: float,
    _retry: bool = True,
) -> list[dict] | None:
    """Default path: parse the public Atom search feed for a subreddit.

    ``sub`` may be one subreddit or several joined with ``+``. On a 429 (Reddit's
    per-IP rate limit) we back off once — honouring ``Retry-After`` when
    present — before giving up, so a transient burst doesn't blank the feed.

    Returns ``[]`` when the search ran and matched nothing, and ``None`` when
    the fetch itself failed. The caller must keep these apart: rendering a
    failed fetch as "no posts found" hands the sentiment analyst an absence of
    discussion that was never observed (#1295).
    """
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(_read_capped(resp))
    except HTTPError as exc:
        if exc.code == 429 and _retry:
            retry_after = _retry_after_seconds(exc)
            wait = retry_after if retry_after is not None else _jitter(_RETRY_FALLBACK_SECONDS)
            logger.warning(
                "Reddit RSS 429 for r/%s · %s — backing off %.1fs then retrying once",
                sub,
                ticker,
                wait,
            )
            time.sleep(wait)
            return _fetch_subreddit_rss(ticker, sub, limit, timeout, _retry=False)
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None
    except (OSError, http.client.HTTPException, ET.ParseError) as exc:
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
        return None

    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        category_el = entry.find("atom:category", _ATOM_NS)
        posts.append(
            {
                "title": (title_el.text if title_el is not None else "") or "",
                "created_utc": _iso_to_timestamp(published_el.text if published_el is not None else None),
                "selftext": _strip_html(content_el.text if content_el is not None else ""),
                "subreddit": category_el.get("term") if category_el is not None else (sub if "+" not in sub else ""),
            }
        )
    return posts


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] | None = None,
    *,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    start_date: str | None = None,
    end_date: str | None = None,
    screen: Callable[[list[str]], tuple[list[bool], str]] | None = None,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits (by default ``subreddits_for(ticker)``) and return them as a
    formatted plaintext block.

    All subreddits are searched in one combined feed (``r/a+b+c``): anonymous
    RSS allows about one request per minute per IP, so a request per subreddit
    spent a back-off on almost every run. Each entry names its subreddit, and
    posts are grouped back by it.

    When ``start_date``/``end_date`` (yyyy-mm-dd) are given, posts are trimmed to
    that window so a historical run does not leak current discussion into a
    backtest (#1220).

    ``screen``, when given, takes each post's title and body and returns a keep
    flag per post and a note line that heads the block. It runs before the
    per-subreddit cut, so the posts it keeps fill the slots.
    """
    subreddits = list(subreddits or subreddits_for(ticker))

    ticker = crypto_base(ticker) or ticker
    label = ", ".join(f"r/{s}" for s in subreddits)
    fetched = _fetch_subreddit_rss(ticker, "+".join(subreddits), _FEED_PAGE, timeout)
    if fetched is None:
        return f"<Reddit unavailable: fetch failed ({label}); this is not an absence of discussion>"

    window = bool(start_date and end_date)
    posts = _within_window(fetched, start_date, end_date)
    if not posts:
        gap = window and coverage_gap(
            _coverage_dates(fetched),
            start_date,
            end_date,
            "Reddit search",
            f"discussion of {ticker.upper()}",
        )
        period = f"within {start_date}..{end_date}" if window else "in the past 7 days"
        return gap or f"<no Reddit posts found mentioning {ticker.upper()} across {label} {period}>"

    def sub_of(p: dict[str, Any]) -> str:
        """Read the returned subreddit or the sole requested community."""
        return p.get("subreddit") or (subreddits[0] if len(subreddits) == 1 else "unknown")

    note, screened_out = "", set()
    if screen:
        keep, note = screen([f"{p.get('title') or ''}\n{p.get('selftext') or ''}"[:_SCREEN_CHARS] for p in posts])
        screened_out = {sub_of(p).lower() for p, kept in zip(posts, keep, strict=True) if not kept}
        posts = [p for p, kept in zip(posts, keep, strict=True) if kept]

    by_sub = {s.lower(): (s, []) for s in subreddits}
    for p in posts:
        by_sub.setdefault(sub_of(p).lower(), (sub_of(p), []))[1].append(p)

    page_full = len(fetched) >= _FEED_PAGE
    blocks = []
    for sub, sub_posts in by_sub.values():
        if not sub_posts:
            if sub.lower() in screened_out:
                blocks.append(f"r/{sub}: <no posts about {ticker.upper()} after screening>")
            else:
                blocks.append(
                    f"r/{sub}: <not among the newest {_FEED_PAGE} matches across {label}>"
                    if page_full
                    else f"r/{sub}: <no posts found mentioning {ticker.upper()}>"
                )
            continue
        sub_posts = sub_posts[:limit_per_sub]
        lines = [f"r/{sub} — {len(sub_posts)} recent posts mentioning {ticker.upper()}:"]
        for p in sub_posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            created = p.get("created_utc")
            created_str = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(f"  [{created_str}] {title}" + (f"\n    body excerpt: {selftext}" if selftext else ""))
        blocks.append("\n".join(lines))
    return "\n\n".join(([note] if note else []) + blocks)
