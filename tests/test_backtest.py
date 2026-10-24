from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from open_trading_agents.analysis.backtest import iter_grid, run_backtest, summarize
from open_trading_agents.analysis.models import AgentEvent, AgentResult, AgentSpec, EventSink, HarnessConfig, HarnessSession
from open_trading_agents.config import HarnessSettings, Portfolio, Settings, load_settings, save_preset
from open_trading_agents.dataflows.files import locked
from open_trading_agents.persistence import memory
from open_trading_agents.persistence.memory import MemoryLog
from open_trading_agents.persistence.storage import RunStore


class FixtureSession:
    def __init__(self, spec: AgentSpec, calls: list[tuple[str, str, str]], fail_once: list[bool], emit: EventSink) -> None:
        """Record offline model tasks and optionally interrupt one trader turn.

        Args:
            spec: Real workflow agent specification.
            calls: Shared observed requests.
            fail_once: Mutable flag controlling one simulated interruption.
            emit: Observer receiving the native prompt acknowledgement.
        """
        self.spec = spec
        self.calls = calls
        self.fail_once = fail_once
        self.emit = emit
        self.session_id = spec.session_id or f"fixture:{spec.workspace}"

    async def prompt(self, text: str) -> AgentResult:
        """Return deterministic schema-valid output while preserving session identity.

        Args:
            text: New native-session input from the real workflow.

        Returns:
            A completed synthetic role response.
        """
        self.calls.append((self.spec.agent, text, self.session_id))
        await self.emit(AgentEvent("status", self.spec.agent, self.spec.stage, data={"prompt_accepted": True}))
        if self.spec.agent == "Trader" and self.fail_once[0]:
            self.fail_once[0] = False
            raise RuntimeError("simulated mid-stream interruption")
        outputs = {
            "Research Manager": {
                "recommendation": "Buy",
                "rationale": "Synthetic evidence",
                "strategic_actions": "Small allocation",
            },
            "Trader": {"action": "Buy", "reasoning": "Synthetic evidence", "entry_price": 100, "stop_loss": 90},
            "Portfolio Manager": {
                "rating": "Buy",
                "executive_summary": "Synthetic action",
                "investment_thesis": "Synthetic evidence",
            },
        }
        output = json.dumps(outputs[self.spec.agent]) if self.spec.agent in outputs else "Synthetic evidence and a bounded conclusion."
        return AgentResult(output, self.session_id)

    async def cancel(self) -> None:
        """Cancel the completed fixture without external side effects."""

    async def close(self) -> None:
        """Release the fixture while retaining its recorded session identity."""


def _factory(calls: list[tuple[str, str, str]], fail_once: list[bool] | None = None) -> Callable[..., Awaitable[HarnessSession]]:
    """Create an injectable native-session factory for full workflow tests.

    Args:
        calls: Shared record of submitted prompts.
        fail_once: Optional flag to interrupt one trader call.

    Returns:
        An async factory accepting the real harness adapter contract.
    """
    state = fail_once if fail_once is not None else [False]

    async def create(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> HarnessSession:
        """Construct one synthetic session for the exact real workflow role."""
        return FixtureSession(spec, calls, state, emit)

    return create


def test_grid_uses_calendar_spacing_and_caps_future_end() -> None:
    """Validate requested grid bounds without creating future analysis cells."""
    assert iter_grid("2024-03-01", "2024-03-06", 2) == ["2024-03-01", "2024-03-03", "2024-03-05"]
    today = date.today()
    assert iter_grid(today.isoformat(), (today + timedelta(days=20)).isoformat()) == [today.isoformat()]
    assert iter_grid((today + timedelta(days=1)).isoformat(), (today + timedelta(days=2)).isoformat()) == []
    for spacing in (0, -1, True):
        with pytest.raises(ValueError):
            iter_grid("2024-03-01", "2024-03-06", spacing)
    with pytest.raises(ValueError):
        iter_grid("2024-03-06", "2024-03-01")


@pytest.mark.asyncio
async def test_full_workflow_sweep_is_isolated_and_completed_cells_are_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run two real pipelines and their harness reflections without touching live state."""
    live = MemoryLog(tmp_path / "memory.sqlite")
    live.record("MSFT", "2024-03-01", "Sell", "Live decision must remain untouched", "live")
    original = live.entries()
    settings = Settings(
        state_dir=tmp_path, analysts=("market",), memory_enabled=False, data={"data_cache_dir": str(tmp_path / "live-cache")}
    )
    fetch = Mock(return_value=(0.10, 0.03, 5, "2024-04-01"))
    monkeypatch.setattr(memory, "fetch_returns", fetch)
    calls: list[tuple[str, str, str]] = []
    result = await run_backtest(["nvda", "NVDA", "AAPL"], ["2024-03-15"], settings, run_id="sweep", session_factory=_factory(calls))
    assert result.cells_run == 2 and result.skipped == 0 and result.failures == []
    assert result.summary.resolved == 2 and result.summary.pending == 0
    assert result.summary.by_rating["Buy"].hit_rate == 1
    assert live.entries() == original
    assert not (tmp_path / "runs").exists() and not (tmp_path / "live-cache").exists()
    sweep_log = MemoryLog(result.directory / "memory.sqlite")
    assert {entry["ticker"] for entry in sweep_log.entries()} == {"NVDA", "AAPL"}
    assert sum(agent.startswith("Reflection") for agent, _, _ in calls) == 2
    for context_file in result.directory.glob("runs/*/tool-context.json"):
        config = json.loads(context_file.read_text())["config"]
        assert Path(config["data_cache_dir"]).is_relative_to(result.directory)
    before = len(calls)
    resumed = await run_backtest(["nvda", "NVDA", "AAPL"], ["2024-03-15"], settings, run_id="sweep", session_factory=_factory(calls))
    assert resumed.cells_run == 0 and resumed.skipped == 2 and len(calls) == before
    assert (result.directory / "summary.json").is_file() and (result.directory / "summary.md").is_file()


@pytest.mark.asyncio
async def test_failed_cell_resumes_native_session_without_repeating_completed_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sweep rerun continues the failed role instead of discarding completed research."""
    monkeypatch.setattr(memory, "fetch_returns", Mock(return_value=(None, None, None, None)))
    settings = Settings(state_dir=tmp_path, analysts=("market",))
    calls: list[tuple[str, str, str]] = []
    factory = _factory(calls, [True])
    first = await run_backtest(["NVDA"], ["2024-03-15"], settings, run_id="resume", session_factory=factory)
    assert first.cells_run == 0 and len(first.failures) == 1
    second = await run_backtest(["NVDA"], ["2024-03-15"], settings, run_id="resume", session_factory=factory)
    assert second.cells_run == 1 and second.failures == []
    assert sum(agent == "Market Analyst" for agent, _, _ in calls) == 1
    trader_calls = [(text, session) for agent, text, session in calls if agent == "Trader"]
    assert len(trader_calls) == 2 and trader_calls[0][1] == trader_calls[1][1]
    assert "Continue the unfinished task" in trader_calls[1][0]
    assert second.summary.pending == 1


async def test_named_default_can_resume_legacy_sweep_and_incomplete_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """New display metadata must not change an existing experiment or its pinned role labels.

    Args:
        monkeypatch: Scoped home and offline market-return replacements.
        tmp_path: Isolated presets, configuration, and sweep storage.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(memory, "fetch_returns", Mock(return_value=(None, None, None, None)))
    save_preset("daily", HarnessSettings(model="pinned-model"))
    path = tmp_path / ".open-trading-agents" / "config.toml"
    path.write_text('default_preset = "daily"\nanalysts = ["market"]\n')
    settings = load_settings(path, use_default_preset=True)
    legacy = settings.model_copy(update={"agent_presets": {}})
    calls: list[tuple[str, str, str]] = []
    factory = _factory(calls, [True])
    first = await run_backtest(["NVDA"], ["2024-03-15"], legacy, run_id="legacy", session_factory=factory)
    assert len(first.failures) == 1
    manifest = (first.directory / "manifest.json").read_bytes()

    for directory in (first.directory / "runs").iterdir():
        store = RunStore(directory)
        try:
            inputs = store.metadata("input")
            inputs["settings"].pop("agent_presets")
            store.set_metadata("input", inputs)
        finally:
            store.close()

    current = load_settings(path, use_default_preset=True)
    assert current.agent_presets["market"] == "daily"
    second = await run_backtest(["NVDA"], ["2024-03-15"], current, run_id="legacy", session_factory=factory)

    assert second.cells_run == 1 and second.failures == []
    assert (second.directory / "manifest.json").read_bytes() == manifest
    assert sum(agent == "Market Analyst" for agent, _, _ in calls) == 1
    trader_sessions = [session for agent, _, session in calls if agent == "Trader"]
    assert len(trader_sessions) == 2 and trader_sessions[0] == trader_sessions[1]
    for directory in (second.directory / "runs").iterdir():
        store = RunStore(directory)
        try:
            assert store.metadata("input")["settings"]["agent_presets"] == {}
        finally:
            store.close()
    assert current.agent_presets["market"] == "daily"


@pytest.mark.asyncio
async def test_manifest_rejects_changed_settings_grid_or_portfolio_before_model_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saved native sessions cannot be reused under a different experiment specification."""
    monkeypatch.setattr(memory, "fetch_returns", Mock(return_value=(None, None, None, None)))
    settings = Settings(state_dir=tmp_path, analysts=("market",))
    calls: list[tuple[str, str, str]] = []
    await run_backtest(["NVDA"], ["2024-03-15"], settings, run_id="pinned", session_factory=_factory(calls))
    before = len(calls)
    variants = [
        (["NVDA"], ["2024-03-15"], settings.model_copy(update={"research_rounds": 2}), None),
        (["AAPL"], ["2024-03-15"], settings, None),
        (["NVDA"], ["2024-03-16"], settings, None),
        (["NVDA"], ["2024-03-15"], settings, Portfolio(cash=1000)),
    ]
    for tickers, dates, config, portfolio in variants:
        with pytest.raises(ValueError, match="manifest"):
            await run_backtest(tickers, dates, config, portfolio, "pinned", session_factory=_factory(calls))
    assert len(calls) == before


@pytest.mark.asyncio
async def test_sweep_lock_rejects_a_second_writer_before_model_calls(tmp_path: Path) -> None:
    """Two processes cannot concurrently replay the same native sessions."""
    settings = Settings(state_dir=tmp_path, analysts=("market",))
    directory = tmp_path / "backtests" / "active"
    directory.mkdir(parents=True)
    calls: list[tuple[str, str, str]] = []
    with locked(directory / "sweep", wait=False):
        with pytest.raises(RuntimeError, match="already running"):
            await run_backtest(["NVDA"], ["2024-03-15"], settings, run_id="active", session_factory=_factory(calls))
    assert calls == []


def test_summary_distinguishes_directional_hits_hold_review_and_pending(tmp_path: Path) -> None:
    """Score negative alpha as a correct Sell while leaving Hold and REVIEW without hit rates."""
    log = MemoryLog(tmp_path / "memory.sqlite")
    cases = [("Buy", 0.04), ("Buy", -0.02), ("Sell", -0.03), ("Hold", 0.01), ("REVIEW", 0.20)]
    for index, (rating, alpha) in enumerate(cases):
        identifier = log.record("NVDA", "2024-03-15", rating, "Synthetic report", str(index))
        token = log._claim(identifier, 120)
        assert token is not None
        log._complete({"id": identifier, "raw_return": alpha, "alpha_return": alpha, "resolution_date": "2024-03-25"}, token, "Lesson")
    log.record("AAPL", "2024-03-15", "Buy", "Pending report", "pending")
    result = summarize(log)
    assert result.resolved == 5 and result.pending == 1 and result.unscored == 1 and result.review == 1
    assert result.by_rating["Buy"].hit_rate == 0.5 and result.by_rating["Buy"].mean_alpha == pytest.approx(0.01)
    assert result.by_rating["Sell"].hit_rate == 1 and result.by_rating["Hold"].hit_rate is None
    assert "not a portfolio simulation" in str(result) and "not archived" in str(result)
