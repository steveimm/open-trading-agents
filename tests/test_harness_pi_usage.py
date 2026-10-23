from pathlib import Path
from typing import Any

import pytest

from open_trading_agents.domain import AgentEvent, AgentSpec, HarnessConfig
from open_trading_agents.harnesses.pi import PiSession, usage_counts


@pytest.mark.parametrize(
    "normalized,raw,expected",
    [
        pytest.param(
            {"input": 70, "output": 10, "cacheRead": 30, "cacheWrite": 0, "reasoning": 4},
            {"promptTokenCount": 100, "cachedContentTokenCount": 30, "candidatesTokenCount": 6, "thoughtsTokenCount": 4},
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 30, "cache_write_tokens": None, "reasoning_tokens": 4},
            id="google-and-vertex-cache-and-reasoning",
        ),
        pytest.param(
            {"input": 100, "output": 10, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0},
            {"promptTokenCount": 100, "cachedContentTokenCount": 0, "thoughtsTokenCount": 0},
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": None, "reasoning_tokens": 0},
            id="google-observed-zero",
        ),
        pytest.param(
            {"input": 100, "output": 10, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0},
            {"promptTokenCount": 100},
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": None, "cache_write_tokens": None, "reasoning_tokens": None},
            id="google-absent-breakdown",
        ),
        pytest.param(
            {"input": 70, "output": 10, "cacheRead": 20, "cacheWrite": 10},
            {"inputTokens": 70, "outputTokens": 10, "cacheReadInputTokens": 20, "cacheWriteInputTokens": 10},
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 20, "cache_write_tokens": 10, "reasoning_tokens": None},
            id="bedrock-cache-read-and-write",
        ),
        pytest.param(
            {"input": 70, "output": 10, "cacheRead": 0, "cacheWrite": 0},
            {"inputTokens": 70, "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0},
            {"input_tokens": 70, "output_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": None},
            id="bedrock-observed-zero",
        ),
        pytest.param(
            {"input": 70, "output": 10, "cacheRead": 20, "cacheWrite": 10, "reasoning": 3},
            {"unrecognized_provider_breakdown": {"cached": 20}},
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 20, "cache_write_tokens": 10, "reasoning_tokens": 3},
            id="positive-native-counts-with-unrecognized-raw-shape",
        ),
        pytest.param(
            {"input": 100, "output": 10, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0},
            {"input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0},
            id="responses-observed-zero",
        ),
        pytest.param(
            {"input": 100, "output": 10, "cacheRead": 0, "cacheWrite": 0, "reasoning": 0},
            {"prompt_tokens_details": None},
            {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": None, "cache_write_tokens": None, "reasoning_tokens": None},
            id="normalized-zero-is-not-observed-zero",
        ),
    ],
)
def test_pi_usage_availability_across_native_provider_shapes(
    normalized: dict[str, Any], raw: dict[str, Any], expected: dict[str, int | None]
) -> None:
    """Preserve actual provider cache evidence without double-counting cached or reasoning tokens.

    Args:
        normalized: Counts emitted by the installed pi provider adapters.
        raw: Provider-specific usage object captured by the native event hook.
        expected: Application token totals and breakdown availability.
    """
    assert usage_counts(normalized, raw) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_read,cache_write,reasoning", [(30, 5, 4), (0, 0, 0)])
async def test_pi_auxiliary_usage_uses_same_positive_fallback(
    tmp_path: Path, cache_read: int, cache_write: int, reasoning: int
) -> None:
    """Keep known auxiliary cache costs without borrowing another request's zero evidence.

    Args:
        tmp_path: Isolated test state directory.
        cache_read: Native aggregate cached-input count.
        cache_write: Native aggregate cache-write count.
        reasoning: Native aggregate reasoning count.
    """
    events: list[AgentEvent] = []

    async def observe(event: AgentEvent) -> None:
        """Collect the auxiliary usage event."""
        events.append(event)

    session = PiSession(HarnessConfig(), AgentSpec("a", "analysis", "", tmp_path, tmp_path), observe)
    await session.auxiliary_usage({"input": 20, "output": 10, "cacheRead": cache_read, "cacheWrite": cache_write,
                                   "reasoning": reasoning}, "native-aggregate", "compaction")
    assert events[0].data["input_tokens"] == 20 + cache_read + cache_write
    assert events[0].data["output_tokens"] == 10
    assert events[0].data["cache_read_tokens"] == (cache_read or None)
    assert events[0].data["cache_write_tokens"] == (cache_write or None)
    assert events[0].data["reasoning_tokens"] == (reasoning or None)
