from collections.abc import Iterable, Mapping
from typing import Any

from open_trading_agents.analysis.models import AgentEvent, Usage


def request_usage(event: AgentEvent) -> tuple[tuple[str, str], Usage]:
    """Normalize one authoritative usage snapshot with its deduplication key.

    Args:
        event: A normalized harness usage event.

    Returns:
        The role/request key and token counts, with absent cache fields preserved.
    """
    raw = event.data.get("usage", event.data)
    value = raw if isinstance(raw, Usage) else Usage(**{key: raw[key] for key in Usage.__dataclass_fields__ if key in raw})
    request_id = str(event.data.get("request_id") or event.data.get("message_id") or event.event_id)
    return (event.agent, request_id), value


def _totals(values: list[Usage]) -> dict[str, Any]:
    """Sum distinct requests without adding cache subsets to input a second time.

    Args:
        values: The latest usage snapshot for each request.

    Returns:
        Reported token totals and explicit completeness of optional breakdowns.
    """
    result = {"reported_requests": len(values), "input_tokens": sum(value.input_tokens for value in values),
              "output_tokens": sum(value.output_tokens for value in values)}
    for name in ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
        counts = [getattr(value, name) for value in values]
        known = [value for value in counts if value is not None]
        result[name] = sum(known) if known else None
        result[f"{name}_complete"] = bool(counts) and len(known) == len(counts)
    return result


def summarize_usage(events: Iterable[AgentEvent]) -> dict[str, Any]:
    """Build durable per-role and total usage from the same records consumed by the UI.

    Args:
        events: Recorded workflow and harness events.

    Returns:
        Totals in which snapshots replace previous values for the same request.
    """
    requests: dict[tuple[str, str], Usage] = {}
    for event in events:
        if event.kind == "usage":
            key, value = request_usage(event)
            requests[key] = value
    agents = {agent: _totals([value for (role, _), value in requests.items() if role == agent])
              for agent in dict.fromkeys(agent for agent, _ in requests)}
    return {"totals": _totals(list(requests.values())), "agents": agents,
            "note": "Input includes cache reads and writes. Reasoning is included in output. Counts include only reported usage."}


def format_usage(usages: list[Usage], compact: bool = False) -> str:
    """Format reported usage while omitting an unreported cache-write field.

    Args:
        usages: Latest usage records, deduplicated by request.
        compact: Whether to abbreviate counters for a narrow pane.

    Returns:
        Input and output totals with the available cache breakdown.
    """
    inputs = sum(value.input_tokens for value in usages)
    outputs = sum(value.output_tokens for value in usages)
    read = _format_cache([value.cache_read_tokens for value in usages], compact)
    if compact:
        label = f"In {_short(inputs)} Out {_short(outputs)} Cache R{read}"
    else:
        label = f"In {inputs:,} · Out {outputs:,} · Cache read {read}"
    writes = [value.cache_write_tokens for value in usages]
    if any(value is not None for value in writes):
        write = _format_cache(writes, compact)
        label += f" W{write}" if compact else f" · write {write}"
    return label


def format_run_usage(totals: Mapping[str, Any]) -> str:
    """Render persisted token totals and available cache counts for a completed run.

    Args:
        totals: Aggregate counts and completeness flags from the usage report.

    Returns:
        A readable summary using the same partial-count markers as the interface.
    """
    tokens = f"Reported tokens: {totals['input_tokens']:,} input / {totals['output_tokens']:,} output."
    read = _format_cache_total(totals.get("cache_read_tokens"), totals.get("cache_read_tokens_complete", False), False)
    cache = f"Cache read: {read}"

    if totals.get("cache_write_tokens") is not None:
        write = _format_cache_total(totals["cache_write_tokens"], totals.get("cache_write_tokens_complete", False), False)
        cache += f" / write: {write}"

    return f"{tokens} {cache}."


def _format_cache(values: list[int | None], compact: bool) -> str:
    """Distinguish unknown cache data from reported zero and partial counts.

    Args:
        values: Provider cache counts for the selected requests.
        compact: Whether to abbreviate known counts.

    Returns:
        A known count, a partial count, or an unavailable marker.
    """
    known = [value for value in values if value is not None]
    total = sum(known) if known else None
    return _format_cache_total(total, len(known) == len(values), compact)


def _format_cache_total(value: int | None, complete: bool, compact: bool) -> str:
    """Format a cache total without confusing missing or partial data with zero.

    Args:
        value: Reported cache count, or None when unavailable.
        complete: Whether every counted request reported this cache field.
        compact: Whether to abbreviate the count for a narrow display.

    Returns:
        A known count, partial count, or unavailable marker.
    """
    if value is None:
        return "—"

    label = _short(value) if compact else f"{value:,}"
    return label if complete else f"{label}+?"


def _short(value: int) -> str:
    """Abbreviate large token counts for compact status bars.

    Args:
        value: Exact token count.

    Returns:
        A compact numeric label.
    """
    return f"{value / 1_000_000:.1f}m" if value >= 1_000_000 else f"{value / 1000:.1f}k" if value >= 1000 else str(value)
