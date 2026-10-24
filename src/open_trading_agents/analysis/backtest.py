from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from open_trading_agents.analysis.models import AgentEvent, EventSink
from open_trading_agents.analysis.workflow import SessionFactory, Workflow
from open_trading_agents.config import Portfolio, Settings, validate_date
from open_trading_agents.dataflows.files import locked
from open_trading_agents.dataflows.symbols import normalize_symbol, safe_ticker_component
from open_trading_agents.harnesses import open_session
from open_trading_agents.persistence.memory import MemoryLog

DIRECTIONS = {"Buy": 1, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -1}
LIMITATIONS = (
    "Independent decisions, not a portfolio simulation: supplied holdings stay fixed and no fills, costs, or cash ledger are simulated.",
    "News and social feeds are not archived point-in-time datasets. Missing historical coverage is reported, so results are indicative.",
    "Each cell uses one model sampling. Returns use the first close after analysis, and alpha is the instrument return minus its benchmark.",
)


@dataclass(frozen=True)
class RatingScore:
    count: int
    hit_rate: float | None
    mean_alpha: float


@dataclass(frozen=True)
class BacktestSummary:
    resolved: int
    pending: int
    unscored: int
    review: int
    by_rating: dict[str, RatingScore]
    holding_windows: tuple[int, ...]
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        """Render counts and per-rating directional scores with their limitations.

        Returns:
            A concise report that distinguishes pending, REVIEW, and scored outcomes.
        """
        lines = [f"Resolved: {self.resolved} | Pending: {self.pending} | REVIEW: {self.review} | Resolved but unscored: {self.unscored}"]
        for rating, score in self.by_rating.items():
            accuracy = f"direction hit rate {score.hit_rate:.1%}" if score.hit_rate is not None else "no direction claimed"
            lines.append(f"{rating}: n={score.count}, {accuracy}, mean alpha {score.mean_alpha:+.2%}")
        if self.holding_windows:
            lines.append("Measured windows: " + ", ".join(f"{days} trading sessions" for days in self.holding_windows))
        lines.extend(self.notes)
        lines.extend(LIMITATIONS)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


@dataclass
class BacktestResult:
    run_id: str
    directory: Path
    cells_run: int = 0
    skipped: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    summary: BacktestSummary = field(default_factory=lambda: BacktestSummary(0, 0, 0, 0, {}, ()))


def iter_grid(start: str, end: str, every_n_days: int = 1) -> list[str]:
    """Build an inclusive calendar-date grid capped at today's local date.

    Args:
        start: First analysis date, YYYY-MM-DD.
        end: Last requested analysis date, YYYY-MM-DD.
        every_n_days: Positive calendar-day spacing between cells.

    Returns:
        Canonical dates in ascending order, with no future dates.
    """
    first, requested_last = date.fromisoformat(start), date.fromisoformat(end)
    if first.isoformat() != start or requested_last.isoformat() != end:
        raise ValueError("grid dates must use YYYY-MM-DD")
    if isinstance(every_n_days, bool) or not isinstance(every_n_days, int) or every_n_days < 1:
        raise ValueError("every_n_days must be a positive integer")
    if requested_last < first:
        raise ValueError("the grid ends before it starts")
    last = min(requested_last, date.today())
    dates = []
    while first <= last:
        dates.append(first.isoformat())
        first += timedelta(days=every_n_days)
    return dates


def summarize(log: MemoryLog, notes: tuple[str, ...] = ()) -> BacktestSummary:
    """Score durable outcomes without treating holdings as a simulated portfolio.

    Args:
        log: Sweep-local decision memory.
        notes: Additional settlement diagnostics.

    Returns:
        Rating scores based on measured alpha and counts of unresolved or unscored decisions.
    """
    entries = log.entries()
    resolved = [entry for entry in entries if entry["status"] == "resolved"]
    scored = [
        entry
        for entry in resolved
        if entry["rating"] in DIRECTIONS and isinstance(entry["alpha_return"], (float, int)) and math.isfinite(entry["alpha_return"])
    ]
    ratings = {}
    for rating, direction in DIRECTIONS.items():
        alphas = [entry["alpha_return"] for entry in scored if entry["rating"] == rating]
        if alphas:
            hit_rate = sum(alpha * direction > 0 for alpha in alphas) / len(alphas) if direction else None
            ratings[rating] = RatingScore(len(alphas), hit_rate, sum(alphas) / len(alphas))
    return BacktestSummary(
        resolved=len(resolved),
        pending=len(entries) - len(resolved),
        unscored=len(resolved) - len(scored),
        review=sum(entry["rating"] == "REVIEW" for entry in entries),
        by_rating=ratings,
        holding_windows=tuple(sorted({entry["holding_days"] for entry in scored})),
        notes=notes,
    )


def _cell_id(ticker: str, analysis_date: str) -> str:
    """Derive one stable native-workflow run identity for a ticker/date cell.

    Args:
        ticker: Canonical instrument symbol.
        analysis_date: Canonical date.

    Returns:
        A portable path component, independent of grid ordering.
    """
    digest = hashlib.sha256(f"{ticker}\0{analysis_date}".encode()).hexdigest()[:16]
    return f"cell-{analysis_date}-{digest}"


def _manifest(tickers: list[str], dates: list[str], settings: Settings, portfolio: Portfolio | None) -> dict[str, Any]:
    """Build a pinned sweep specification without copying configuration secrets into the manifest.

    Args:
        tickers: Ordered unique canonical instruments.
        dates: Ordered unique analysis dates.
        settings: Original caller settings, before storage isolation overrides.
        portfolio: Fixed holdings supplied for every cell, or None.

    Returns:
        A versioned grid and exact configuration fingerprint.
    """
    inputs = {"settings": settings.model_dump(mode="json"), "portfolio": portfolio.model_dump(mode="json") if portfolio else None}
    if not settings.agents:
        inputs["settings"].pop("agents")

    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "tickers": tickers, "dates": dates, "configuration_fingerprint": digest}


def _save_manifest(path: Path, value: dict[str, Any]) -> None:
    """Atomically establish a sweep specification or reject a changed resume configuration.

    Args:
        path: Manifest path inside the exclusively held sweep directory.
        value: Requested immutable sweep specification.
    """
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != value:
            raise ValueError("backtest inputs differ from the saved manifest; resume with the original grid, settings, and portfolio")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed(directory: Path) -> bool:
    """Read the workflow's authoritative completion status without creating a run.

    Args:
        directory: Deterministic cell workflow directory.

    Returns:
        Whether the workflow finished all stages and saved its artifacts.
    """
    database = directory / "run.sqlite"
    if not database.is_file():
        return False
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
        return row is not None and json.loads(row[0]) == "complete"
    finally:
        connection.close()


async def run_backtest(
    tickers: list[str],
    dates: list[str],
    settings: Settings,
    portfolio: Portfolio | None = None,
    run_id: str | None = None,
    emit: EventSink | None = None,
    *,
    session_factory: SessionFactory = open_session,
) -> BacktestResult:
    """Evaluate and resume sequential ticker/date cells using the normal native-harness workflow.

    Args:
        tickers: Instruments to evaluate.
        dates: Analysis dates, each no later than today.
        settings: Application settings whose live state must remain untouched.
        portfolio: Optional fixed holdings for all independent decisions.
        run_id: Existing sweep identifier to resume, or None to create a new sweep.
        emit: Optional observer receiving workflow and sweep progress events.
        session_factory: Native session factory, injectable for deterministic offline tests.

    Returns:
        Sweep identity, newly completed and skipped cells, failures, and outcome scores.
    """
    settings = settings.model_copy(update={"state_dir": settings.state_dir.expanduser().resolve()})
    tickers = list(dict.fromkeys(safe_ticker_component(normalize_symbol(ticker)) for ticker in tickers))
    dates = list(dict.fromkeys(validate_date(value) for value in dates))
    if not tickers or not dates:
        raise ValueError("a backtest needs at least one ticker and one date")
    identifier = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    safe_ticker_component(identifier, max_len=100)
    directory = settings.state_dir.expanduser().resolve() / "backtests" / identifier
    directory.mkdir(parents=True, exist_ok=True)
    isolated = settings.model_copy(
        update={
            "state_dir": directory,
            "memory_enabled": True,
            "data": {**settings.data, "data_cache_dir": str(directory / "cache")},
        },
        deep=True,
    )
    result = BacktestResult(identifier, directory)
    cells = [(ticker, analysis_date) for ticker in tickers for analysis_date in dates]
    with locked(directory / "sweep", wait=False) as acquired:
        if not acquired:
            raise RuntimeError(f"backtest {identifier} is already running")
        _save_manifest(directory / "manifest.json", _manifest(tickers, dates, settings, portfolio))
        for index, (ticker, analysis_date) in enumerate(cells, 1):
            cell_id = _cell_id(ticker, analysis_date)
            if _completed(directory / "runs" / cell_id):
                result.skipped += 1
                continue
            if emit:
                await emit(
                    AgentEvent(
                        "backtest_cell",
                        "Backtest",
                        "analysis",
                        f"{ticker} on {analysis_date}",
                        {"index": index, "total": len(cells), "ticker": ticker, "date": analysis_date, "run_id": cell_id},
                    )
                )
            try:
                workflow = Workflow(ticker, analysis_date, isolated, portfolio, cell_id, session_factory)
                await workflow.run(emit)
                result.cells_run += 1
            except Exception as exc:
                result.failures.append((ticker, analysis_date, f"{type(exc).__name__}: {exc}"))
        try:
            settlement = Workflow(tickers[0], max(dates), isolated, portfolio, "settlement", session_factory)
            notes = await settlement.settle_pending(emit)
        except Exception as exc:
            notes = [f"Final settlement failed ({type(exc).__name__}); unresolved outcomes remain pending."]
        result.summary = summarize(MemoryLog(directory / "memory.sqlite"), tuple(notes))
        report = {
            "run_id": identifier,
            "cells_run": result.cells_run,
            "skipped": result.skipped,
            "failures": result.failures,
            "summary": result.summary.render(),
        }
        (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (directory / "summary.md").write_text(result.summary.render() + "\n", encoding="utf-8")
    return result
