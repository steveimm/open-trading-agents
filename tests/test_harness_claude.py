import sys
from pathlib import Path

import pytest

from open_trading_agents.analysis.models import AgentEvent, AgentSpec, HarnessConfig
from open_trading_agents.harnesses import open_session


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["claude-code", "claude"])
@pytest.mark.parametrize("streamed_output", [0, 10])
@pytest.mark.parametrize("effort", [None, "high"])
async def test_claude_raw_stream_usage_avoids_final_and_cumulative_duplicates(tmp_path: Path, streamed_output: int, effort: str | None, kind: str) -> None:
    """Count complete provider messages once despite repeated assistant and result counters."""
    script = Path(__file__).with_name("fixtures") / "claude_cli.py"
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        """Collect normalized adapter events."""
        events.append(event)

    spec = AgentSpec("market", "analysis", "system", tmp_path, tmp_path / "sessions")
    session = await open_session(HarnessConfig(kind=kind, command=(sys.executable, str(script)), effort=effort,
                                              environment={"STREAM_OUTPUT": str(streamed_output), "EXPECTED_EFFORT": effort or ""}), spec, emit)
    try:
        assert (await session.prompt("first")).text == "1"
        assert (await session.prompt("second")).text == "2"
    finally:
        await session.close()

    assert sum(bool(event.data.get("prompt_accepted")) for event in events) == 2
    assert [event.data for event in events if event.kind == "model"] == [
        {"harness": "claude-code", "provider": None, "model": "native-default"},
        {"harness": "claude-code", "provider": None, "model": "auxiliary-model"},
    ]
    usage = [event.data for event in events if event.kind == "usage"]
    assert len(usage) == (2 if streamed_output else 4)
    assert sum(row["input_tokens"] for row in usage) == 200
    assert sum(row["output_tokens"] for row in usage) == 20
    assert sum(row["cache_read_tokens"] or 0 for row in usage) == 60
    assert all(row["cache_write_tokens"] is None for row in usage)
    assert all(event.data.get("message_id") == "message1:0" for event in events if event.kind == "text" and event.text == "1")


@pytest.mark.asyncio
async def test_claude_final_text_keeps_stream_index_after_reasoning(tmp_path: Path) -> None:
    """Do not duplicate text when native final block messages restart their indexes."""
    from open_trading_agents.harnesses.claude import ClaudeSession

    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        """Collect normalized content records."""
        events.append(event)

    session = ClaudeSession(HarnessConfig(kind="claude-code"), AgentSpec("a", "analysis", "", tmp_path, tmp_path), emit)
    session.message_id = "native1"
    session.message_text = {"native1:1": "Complete answer"}
    session.stream_blocks = {"native1": {0: "thinking", 1: "text"}}
    session.final_blocks = set()
    await session.handle({"type": "assistant", "message": {
        "id": "native1", "content": [{"type": "text", "text": "Complete answer"}], "usage": {},
    }})
    assert events[0].data["message_id"] == "native1:1"
    assert events[0].data["final"] is True
