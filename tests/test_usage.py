from open_trading_agents.domain import AgentEvent, Usage
from open_trading_agents.usage import format_usage, summarize_usage


def test_snapshots_replace_counts_and_missing_cache_stays_unknown() -> None:
    """Deduplicate replayed snapshots while retaining incomplete cache information."""
    events = [
        AgentEvent("usage", "Bull", "research", data={"request_id": "a", "input_tokens": 100, "output_tokens": 10}),
        AgentEvent("usage", "Bull", "research", data={"request_id": "a", "input_tokens": 120, "output_tokens": 15,
                                                       "cache_read_tokens": 80, "cache_write_tokens": 0}),
        AgentEvent("usage", "Bear", "research", data={"request_id": "a", "input_tokens": 50, "output_tokens": 5}),
    ]
    summary = summarize_usage(events)
    assert summary["totals"]["reported_requests"] == 2
    assert summary["totals"]["input_tokens"] == 170
    assert summary["totals"]["cache_read_tokens"] == 80
    assert not summary["totals"]["cache_read_tokens_complete"]
    assert summary["agents"]["Bear"]["cache_read_tokens"] is None
    assert summary["agents"]["Bull"]["cache_write_tokens"] == 0


def test_write_cache_is_hidden_until_a_provider_reports_it() -> None:
    """Do not show unavailable write counters, while preserving real writes and zeroes."""
    assert "write" not in format_usage([Usage(input_tokens=100)])
    assert "W" not in format_usage([Usage(input_tokens=100)], compact=True)
    assert "write 0" in format_usage([Usage(cache_write_tokens=0)])
    assert "write 40+?" in format_usage([Usage(cache_write_tokens=40), Usage()])
    assert "write 0+?" in format_usage([Usage(cache_write_tokens=0), Usage()])
