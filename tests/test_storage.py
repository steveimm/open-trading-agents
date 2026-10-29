import json
from pathlib import Path

import pytest

from open_trading_agents.analysis.models import AgentEvent
from open_trading_agents.persistence.events import EventRecorder
from open_trading_agents.persistence.storage import RunStore


def test_interrupted_run_reopens_with_session_and_completed_work(tmp_path: Path) -> None:
    """Retain completed steps and native session references across process lifetimes."""
    store = RunStore(tmp_path)
    store.begin_step("market", "Market Analyst", "Analyze synthetic data")
    store.save_session("Market Analyst", "native-session-42")
    store.finish_step("market", "Report with <!-- ENTRY_END --> and arbitrary text")
    event = AgentEvent("text", "Market Analyst", "analysis", "λ\n你好")
    store.append([event, event])
    store.begin_step("news", "News Analyst", "Pending task")
    store.close()

    reopened = RunStore(tmp_path)
    assert reopened.session("Market Analyst") == "native-session-42"
    assert reopened.step("market")["status"] == "complete"
    assert reopened.step("news")["status"] == "pending"
    assert reopened.events() == [event]
    reopened.close()


def test_jsonl_mirror_preserves_full_results_and_recovers_its_prefix(tmp_path: Path) -> None:
    """Mirror complete payloads once and rebuild a damaged text export from SQLite on resume."""
    event = AgentEvent("tool_result", "Market Analyst", "analysis", "head" + "x" * 100_000 + "tail")
    recorder = EventRecorder(tmp_path)
    recorder.append(event)
    recorder.append(event)
    recorder.close()
    path = tmp_path / "transcript.jsonl"
    assert [json.loads(line)["text"] for line in path.read_text().splitlines()] == [event.text]
    path.write_text("interrupted export")
    resumed = EventRecorder(tmp_path)
    next_event = AgentEvent("text", "Market Analyst", "analysis", "Completed output")
    resumed.append(next_event)
    resumed.close()
    assert [json.loads(line)["event_id"] for line in path.read_text().splitlines()] == [event.event_id, next_event.event_id]


def test_event_flush_commits_without_closing_writer(tmp_path: Path) -> None:
    """Make each completed chat turn readable before its long-lived recorder is closed."""
    recorder = EventRecorder(tmp_path)
    events = [AgentEvent("prompt", "Report Assistant", "chat", "Question"), AgentEvent("text", "Report Assistant", "chat", "Answer")]
    try:
        recorder.flush()
        for event in events:
            recorder.append(event)
            recorder.flush()
        assert recorder.worker.is_alive()
        reader = RunStore(tmp_path)
        try:
            assert reader.events() == events
        finally:
            reader.close()
        assert [json.loads(line)["text"] for line in (tmp_path / "transcript.jsonl").read_text().splitlines()] == ["Question", "Answer"]
    finally:
        recorder.close()


def test_event_flush_reports_writer_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop waiting and report failed persistence instead of acknowledging a saved turn."""
    def fail_append(store: RunStore, events: list[AgentEvent]) -> list[str]:
        """Simulate a disk failure in the background writer."""
        raise OSError("Fixture write failure")

    monkeypatch.setattr(RunStore, "append", fail_append)
    recorder = EventRecorder(tmp_path)
    recorder.append(AgentEvent("text", "Report Assistant", "chat", "Answer"))
    with pytest.raises(RuntimeError, match="persistence failed"):
        recorder.flush()
    with pytest.raises(RuntimeError, match="persistence failed"):
        recorder.close()
    assert not recorder.worker.is_alive()
