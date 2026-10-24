from __future__ import annotations

from collections.abc import Iterable

# Adapted from TauricResearch/TradingAgents (Apache-2.0).
# Imports, typing, configuration and tool integration modified for Open Trading Agents.
from datetime import UTC, date, datetime, timedelta


def to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware; a naive value is assumed to be UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def in_window(pub_dt: datetime | None, start_dt: datetime, end_dt: datetime) -> bool:
    """Check publication eligibility through the end of the requested calendar day.

    Args:
        pub_dt: Publication timestamp, or None for an undated item.
        start_dt: First included timestamp, with naive values treated as UTC.
        end_dt: Last included calendar day, with naive values treated as UTC.

    Returns:
        Whether the item falls in the window. Undated items are kept only when
        the window reaches the present.
    """
    end = to_utc(end_dt)
    if pub_dt is not None:
        return to_utc(start_dt) <= to_utc(pub_dt) < end + timedelta(days=1)
    return end >= datetime.now(UTC) - timedelta(days=1)


def get_current_date() -> str:
    """Today's date, YYYY-MM-DD."""
    return date.today().strftime("%Y-%m-%d")


def is_historical(run_date: str | None) -> bool:
    """Whether a run is dated before today, so live-only data would come from after it."""
    return bool(run_date) and str(run_date) < get_current_date()


def coverage_gap(dates: Iterable[datetime | None], start_date: str, end_date: str, source: str, subject: str) -> str | None:
    """Distinguish an unobserved feed window from a verified absence of items.

    Args:
        dates: Timestamps from a continuous newest-first feed, including its fixed
            lookback boundary where known. Pass an empty iterable for merged or
            relevance-ranked results that cannot establish continuous coverage.
        start_date: First requested day in YYYY-MM-DD format.
        end_date: Last requested day in YYYY-MM-DD format.
        source: Source name included in an unavailability notice.
        subject: Kind of evidence whose absence must not be inferred.

    Returns:
        An unavailability notice when coverage starts too late or the window
        extends into the future, otherwise None.

    Raises:
        ValueError: A requested date is not in YYYY-MM-DD format.
    """
    now = datetime.now(UTC)
    oldest = min((to_utc(d) for d in dates if d is not None), default=now)
    if datetime.strptime(end_date, "%Y-%m-%d").date() > now.date():
        reason = "the window extends past today"
    elif oldest.date() > datetime.strptime(start_date, "%Y-%m-%d").date():
        reason = "it only serves recent items"
    else:
        return None

    return f"<{source} unavailable for {start_date}..{end_date}: {reason}, so this is not an absence of {subject}>"


def _parse(date: str | None) -> datetime | None:
    """Parse a canonical date when the input is valid, otherwise return None."""
    try:
        return datetime.strptime(date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def as_of(requested: str | None, trade_date: str) -> str | None:
    """Keep a model-provided date within the run's information cutoff.

    Args:
        requested: Requested date, or None to use the cutoff.
        trade_date: Valid run cutoff in YYYY-MM-DD format. An empty value leaves
            the request unchanged for direct calls outside a workflow.

    Returns:
        The earlier valid date, or the cutoff for a missing or malformed request.
    """
    if not trade_date:
        return requested

    parsed = _parse(requested)
    return requested if parsed is not None and parsed <= _parse(trade_date) else trade_date


def as_of_window(start_date: str, end_date: str, trade_date: str) -> tuple[str, str]:
    """Clamp a requested window to the information cutoff, preserving future-window length.

    Args:
        start_date: First requested day in YYYY-MM-DD format.
        end_date: Last requested day, replaced by the cutoff if malformed.
        trade_date: Valid run cutoff in YYYY-MM-DD format.

    Returns:
        Inclusive start and end dates. A wholly future window moves backward to
        end at the cutoff while retaining its original nonnegative duration.

    Raises:
        ValueError: The start date cannot be parsed.
    """
    start, old_end = _parse(start_date), _parse(end_date)
    if start is None:
        raise ValueError(f"start_date {start_date!r} is not a date; give it as YYYY-MM-DD")

    end = as_of(end_date, trade_date)
    if end == end_date or start <= _parse(end):
        return start_date, end

    span = (old_end - start) if old_end is not None and old_end >= start else timedelta(0)
    return f"{_parse(end) - span:%Y-%m-%d}", end


def withhold_live_profile(as_of_date: str | None, label: str) -> str | None:
    """Notice to serve instead of a live-only company profile, or None to serve it.

    Vendor "company overview" endpoints (yfinance ``Ticker.info``, Alpha Vantage
    ``OVERVIEW``) carry no historical vintage — not even name, sector and
    industry, which move when a company renames or is reclassified — so serving
    one into a run dated in the past leaks post-decision information (#1300).
    Every fundamentals vendor withholds on this rule, so switching between them
    cannot reintroduce the leak.
    """
    if not is_historical(as_of_date):
        return None

    return (
        f"""# Company Fundamentals for {label}
# Point-in-time as of: {as_of_date}

Profile fundamentals are withheld for this date. This vendor serves only present-day values with no historical vintage: market cap, valuation multiples, the 52-week range and TTM income move with today's quote, and even the name, sector and industry reflect today rather than {as_of_date} (companies rename and get reclassified). Serving them would put post-decision information into a {as_of_date} analysis. Statements filed by {as_of_date} are served as filed where the filing date is known (SEC EDGAR, for US filers)."""
    )


def withhold_undisclosed_trades(as_of_date: str | None, label: str) -> str | None:
    """Notice to serve instead of insider trades with no filing date, or None to serve them.

    A vendor that dates an insider trade by when it happened (Yahoo, Alpha
    Vantage) cannot say when it became public: its Form 4 is filed up to two
    business days later. A past run is told so rather than served trades that
    may not yet have been disclosed.
    """
    if not is_historical(as_of_date):
        return None

    return (
        f"""# Insider Transactions for {label}
# Point-in-time as of: {as_of_date}

Insider transactions are withheld for this date. This vendor dates a trade by when it happened and reports no filing date, so it cannot show which trades were public on {as_of_date}."""
    )


def withhold_undated_statements(as_of_date: str | None, label: str, title: str) -> str | None:
    """Notice to serve instead of a statement with no filing date, or None to serve it.

    A vendor that dates a statement by the period it covers (Yahoo, Alpha
    Vantage) cannot say when its figures became public: a company files weeks
    after the period ends, so a run dated in that gap would read figures that
    were not yet known. A past run is told so; SEC EDGAR, which dates every
    filing, serves US filers as filed.
    """
    if not is_historical(as_of_date):
        return None

    return (
        f"""# {title} for {label}
# Point-in-time as of: {as_of_date}

{title} data is withheld for this date. This vendor dates a statement by the period it covers and reports no filing date, so it cannot show which figures were public on {as_of_date}. SEC EDGAR serves US filers as filed."""
    )
