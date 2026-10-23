from pathlib import Path

from open_trading_agents.domain import AgentEvent
from open_trading_agents.storage import RunStore


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
