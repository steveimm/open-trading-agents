from __future__ import annotations

import asyncio
import math
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from open_trading_agents.dataflows.config import run_config
from open_trading_agents.dataflows.symbols import normalize_symbol
from open_trading_agents.dataflows.vendors.yahoo.market import get_closes

BENCHMARKS = {
    ".NS": "^NSEI",
    ".BO": "^BSESN",
    ".T": "^N225",
    ".TW": "^TWII",
    ".TWO": "^TWII",
    ".KS": "^KS11",
    ".KQ": "^KQ11",
    ".HK": "^HSI",
    ".SI": "^STI",
    ".L": "^FTSE",
    ".DE": "^GDAXI",
    ".PA": "^FCHI",
    ".AS": "^AEX",
    ".SW": "^SSMI",
    ".MI": "FTSEMIB.MI",
    ".TO": "^GSPTSE",
    ".AX": "^AXJO",
    ".SS": "000001.SS",
    ".SZ": "399001.SZ",
    ".SA": "^BVSP",
    "": "SPY",
}

Reflection = Callable[[dict[str, Any]], Awaitable[str]]
Returns = tuple[float | None, float | None, int | None, str | None]


def resolve_benchmark(ticker: str, config: Mapping[str, Any]) -> str:
    """Resolve an explicit benchmark or the instrument's regional suffix benchmark.

    Args:
        ticker: Instrument symbol, including any exchange suffix.
        config: Optional benchmark_ticker and benchmark_map overrides.

    Returns:
        A normalized benchmark symbol.
    """
    if config.get("benchmark_ticker"):
        return normalize_symbol(config["benchmark_ticker"])
    mapping = {**BENCHMARKS, **config.get("benchmark_map", {})}
    canonical = normalize_symbol(ticker)
    for suffix in sorted(mapping, key=len, reverse=True):
        if suffix and canonical.endswith(suffix.upper()):
            return normalize_symbol(mapping[suffix])
    return normalize_symbol(mapping[""])


def _by_day(closes: pd.Series) -> pd.Series:
    """Normalize daily closes without shifting exchange-local dates to UTC.

    Args:
        closes: Daily price series returned by a data vendor.

    Returns:
        Sorted positive finite prices indexed by local calendar day.
    """
    prices = pd.to_numeric(closes, errors="coerce")
    prices = prices[prices.map(lambda value: pd.notna(value) and math.isfinite(value) and value > 0)]
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices = prices.tz_localize(None)
    prices.index = prices.index.normalize()
    return prices[~prices.index.duplicated(keep="last")].sort_index()


def fetch_returns(ticker: str, trade_date: str, holding_days: int = 5, benchmark: str = "SPY") -> Returns:
    """Measure returns from the first close after analysis over a full holding window.

    Args:
        ticker: Instrument whose decision is being measured.
        trade_date: Analysis date, YYYY-MM-DD.
        holding_days: Trading sessions after the entry close.
        benchmark: Comparison instrument valued at the same calendar boundaries.

    Returns:
        Raw return, alpha, measured sessions, and resolution date. Four None values
        indicate that prices have not yet covered the complete window.
    """
    if isinstance(holding_days, bool) or not isinstance(holding_days, int) or holding_days < 1:
        raise ValueError("holding_days must be a positive integer")
    start = datetime.strptime(trade_date, "%Y-%m-%d")
    end = start + timedelta(days=round((holding_days + 1) * 7 / 5) + 7)
    stock = _by_day(get_closes(ticker, trade_date, end.strftime("%Y-%m-%d")))
    stock = stock[stock.index > start]
    bench_start = (start - timedelta(days=7)).strftime("%Y-%m-%d")
    bench = _by_day(get_closes(benchmark, bench_start, end.strftime("%Y-%m-%d")))
    if len(stock) <= holding_days:
        return None, None, None, None
    entry, exit_date = stock.index[0], stock.index[holding_days]
    if bench.empty or bench.index[0] > entry or bench.index[-1] < exit_date:
        return None, None, None, None
    raw_return = float(stock.iloc[holding_days] / stock.iloc[0] - 1)
    benchmark_return = float(bench.asof(exit_date) / bench.asof(entry) - 1)
    return raw_return, raw_return - benchmark_return, holding_days, exit_date.strftime("%Y-%m-%d")


class MemoryLog:
    def __init__(self, path: Path) -> None:
        """Open a local SQLite decision log without executing model calls.

        Args:
            path: Database path, independent of the workflow's event database.
        """
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    ticker TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    holding_days INTEGER NOT NULL CHECK (holding_days > 0),
                    benchmark TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'settling', 'resolved')),
                    raw_return REAL,
                    alpha_return REAL,
                    resolution_date TEXT,
                    reflection TEXT,
                    claim_token TEXT,
                    claim_until REAL,
                    last_error TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS decisions_ticker_date ON decisions (ticker, trade_date);
                CREATE INDEX IF NOT EXISTS decisions_status ON decisions (status, claim_until);
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Use one short SQLite transaction and always close its connection.

        Yields:
            A connection whose writes are committed on success or rolled back on failure.
        """
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(
        self,
        ticker: str,
        trade_date: str,
        rating: str,
        decision: str,
        run_id: str,
        holding_days: int = 5,
        benchmark: str | None = None,
    ) -> int:
        """Store a finished run once, retaining distinct same-day runs.

        Args:
            ticker: Instrument symbol.
            trade_date: Analysis date, YYYY-MM-DD.
            rating: Final rating or REVIEW when no valid rating was produced.
            decision: Final report stored as text, never parsed as storage records.
            run_id: Stable workflow run identity used for idempotency.
            holding_days: Outcome horizon fixed when the decision is recorded.
            benchmark: Benchmark fixed for this decision, or the regional default.

        Returns:
            The existing or newly inserted decision identifier.
        """
        if not run_id:
            raise ValueError("run_id must not be empty")
        if isinstance(holding_days, bool) or not isinstance(holding_days, int) or holding_days < 1:
            raise ValueError("holding_days must be a positive integer")
        if datetime.strptime(trade_date, "%Y-%m-%d").strftime("%Y-%m-%d") != trade_date:
            raise ValueError("trade_date must be YYYY-MM-DD")
        canonical = normalize_symbol(ticker)
        chosen = normalize_symbol(benchmark) if benchmark else resolve_benchmark(canonical, {})
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO decisions
                   (run_id, ticker, trade_date, rating, decision, holding_days, benchmark, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO NOTHING""",
                (run_id, canonical, trade_date, rating, decision, holding_days, chosen, time.time()),
            )
            row = connection.execute("SELECT id FROM decisions WHERE run_id = ?", (run_id,)).fetchone()
            return int(row["id"])

    def entries(self) -> list[dict[str, Any]]:
        """Read durable decisions and outcomes in insertion order.

        Returns:
            Records containing date, trade_date, status, raw_return, alpha_return,
            resolution_date, and all original decision settings.
        """
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        return [{**dict(row), "date": row["trade_date"], "pending": row["status"] != "resolved"} for row in rows]

    def past_context(self, ticker: str, as_of: str | None = None, n_same: int = 5, n_cross: int = 3) -> str:
        """Read settled lessons whose measured outcomes were known by the cutoff.

        Args:
            ticker: Instrument receiving the lessons.
            as_of: Historical cutoff, or None for all settled lessons.
            n_same: Maximum same-instrument decisions to include.
            n_cross: Maximum other-instrument reflections to include.

        Returns:
            Labelled past decisions and reflections as prompt context.
        """
        if n_same < 0 or n_cross < 0:
            raise ValueError("memory context limits must be nonnegative")
        canonical = normalize_symbol(ticker)
        same, cross = [], []
        for entry in reversed(self.entries()):
            if entry["status"] != "resolved" or (as_of is not None and entry["resolution_date"] > as_of):
                continue
            if entry["ticker"] == canonical and len(same) < n_same:
                same.append(self._format_entry(entry, include_decision=True))
            elif entry["ticker"] != canonical and len(cross) < n_cross:
                cross.append(self._format_entry(entry, include_decision=False))
            if len(same) >= n_same and len(cross) >= n_cross:
                break
        blocks = []
        if same:
            blocks.append(f"Past analyses of {canonical} (most recent first):\n" + "\n\n".join(same))
        if cross:
            blocks.append("Recent cross-ticker lessons:\n" + "\n\n".join(cross))
        return "\n\n".join(blocks)

    @staticmethod
    def _format_entry(entry: Mapping[str, Any], include_decision: bool) -> str:
        """Format a stored outcome without interpreting delimiters in model text.

        Args:
            entry: One resolved database record.
            include_decision: Whether to include the original decision report.

        Returns:
            Plain text describing the measured window and reflection.
        """
        header = (
            f"{entry['trade_date']} | {entry['ticker']} | {entry['rating']} | "
            f"{entry['holding_days']} trading days | raw {entry['raw_return']:+.2%} | "
            f"alpha vs {entry['benchmark']} {entry['alpha_return']:+.2%} | known {entry['resolution_date']}"
        )
        decision = f"\nDecision:\n{entry['decision']}" if include_decision else ""
        return f"{header}{decision}\nReflection:\n{entry['reflection']}"

    def _claim(self, identifier: int, lease_seconds: float) -> str | None:
        """Atomically claim pending work or reclaim an expired worker lease.

        Args:
            identifier: Decision identifier.
            lease_seconds: How long an unrenewed claim remains exclusive.

        Returns:
            A unique claim token, or None when another process owns the work.
        """
        now, token = time.time(), uuid.uuid4().hex
        with self._connection() as connection:
            changed = connection.execute(
                """UPDATE decisions SET status='settling', claim_token=?, claim_until=?
                   WHERE id=? AND (status='pending' OR (status='settling' AND claim_until < ?))""",
                (token, now + lease_seconds, identifier, now),
            ).rowcount
        return token if changed else None

    def _renew(self, identifier: int, token: str, lease_seconds: float) -> bool:
        """Extend a claim only while this process still owns it.

        Args:
            identifier: Decision identifier.
            token: Current claim identity.
            lease_seconds: New lease duration.

        Returns:
            Whether the claim remains owned by this worker.
        """
        with self._connection() as connection:
            return bool(
                connection.execute(
                    "UPDATE decisions SET claim_until=? WHERE id=? AND claim_token=? AND status='settling'",
                    (time.time() + lease_seconds, identifier, token),
                ).rowcount
            )

    async def _heartbeat(self, identifier: int, token: str, lease_seconds: float) -> None:
        """Keep a live worker's lease renewed during slow data or model calls.

        Args:
            identifier: Decision identifier.
            token: Current claim identity.
            lease_seconds: Lease duration to renew periodically.
        """
        while True:
            await asyncio.sleep(lease_seconds / 3)
            if not await asyncio.to_thread(self._renew, identifier, token, lease_seconds):
                return

    def _release(self, identifier: int, token: str, error: str | None = None) -> None:
        """Return an incomplete settlement to pending without disturbing another owner.

        Args:
            identifier: Decision identifier.
            token: Current claim identity.
            error: Optional safe failure description.
        """
        with self._connection() as connection:
            connection.execute(
                """UPDATE decisions SET status='pending', claim_token=NULL, claim_until=NULL, last_error=?
                   WHERE id=? AND claim_token=? AND status='settling'""",
                (error, identifier, token),
            )

    def _complete(self, entry: Mapping[str, Any], token: str, reflection: str) -> bool:
        """Atomically commit a measured outcome and reflection for the owned claim.

        Args:
            entry: Decision and measured return fields.
            token: Current claim identity.
            reflection: Completed harness output.

        Returns:
            Whether this worker still owned the entry and committed its outcome.
        """
        with self._connection() as connection:
            return bool(
                connection.execute(
                    """UPDATE decisions SET status='resolved', raw_return=?, alpha_return=?, resolution_date=?,
                       reflection=?, claim_token=NULL, claim_until=NULL, last_error=NULL
                       WHERE id=? AND claim_token=? AND status='settling'""",
                    (
                        entry["raw_return"],
                        entry["alpha_return"],
                        entry["resolution_date"],
                        reflection,
                        entry["id"],
                        token,
                    ),
                ).rowcount
            )

    @staticmethod
    def _returns(entry: Mapping[str, Any], config: Mapping[str, Any]) -> Returns:
        """Fetch outcome prices inside the caller's scoped data configuration.

        Args:
            entry: Stored decision with its fixed benchmark and horizon.
            config: Data adapter settings for this settlement pass.

        Returns:
            The measured return tuple or four None values for an incomplete window.
        """
        with run_config(config):
            return fetch_returns(entry["ticker"], entry["trade_date"], entry["holding_days"], entry["benchmark"])

    async def settle(self, config: dict[str, Any], reflect: Reflection) -> list[str]:
        """Settle all due decisions through a caller-owned harness reflection callback.

        Args:
            config: Data adapter configuration used for price retrieval.
            reflect: Async callback receiving the decision and measured outcome.

        Returns:
            Notes for failed or superseded settlements. Incomplete windows remain
            pending without a failure note, and resolved decisions are skipped.
        """
        notes = []
        for entry in self.entries():
            if entry["status"] == "resolved":
                continue
            identifier = entry["id"]
            token = self._claim(identifier, lease_seconds=120)
            if token is None:
                continue
            heartbeat = asyncio.create_task(self._heartbeat(identifier, token, lease_seconds=120))
            try:
                raw, alpha, days, resolved = await asyncio.to_thread(self._returns, entry, config)
                if raw is None or alpha is None or days is None or resolved is None:
                    self._release(identifier, token)
                    continue
                measured = {
                    "id": entry["id"],
                    "ticker": entry["ticker"],
                    "date": entry["trade_date"],
                    "decision": entry["decision"],
                    "rating": entry["rating"],
                    "benchmark": entry["benchmark"],
                    "raw_return": raw,
                    "alpha_return": alpha,
                    "holding_days": days,
                    "resolution_date": resolved,
                }
                reflection = await reflect(measured)
                if not isinstance(reflection, str) or not reflection.strip():
                    raise ValueError("reflection returned no text")
                if not self._complete(measured, token, reflection):
                    notes.append(f"{entry['ticker']} {entry['trade_date']}: settlement claim was superseded")
            except asyncio.CancelledError:
                self._release(identifier, token)
                raise
            except Exception as exc:
                error = f"{entry['ticker']} {entry['trade_date']}: settlement failed ({type(exc).__name__})"
                self._release(identifier, token, error)
                notes.append(error)
            finally:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
        return notes
