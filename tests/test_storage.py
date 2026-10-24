import json
from pathlib import Path

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
