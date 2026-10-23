from collections.abc import Iterable
from typing import Any

from open_trading_agents.domain import AgentEvent, Usage


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
