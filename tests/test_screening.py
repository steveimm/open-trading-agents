import json
from pathlib import Path

from open_trading_agents.analysis.models import AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.analysis.workflow import Workflow
from open_trading_agents.config import Settings


class ScreeningSession:
    session_id = "native-screen"

    def __init__(self, answer: str) -> None:
        """Keep a synthetic native screening response."""
        self.answer = answer

    async def prompt(self, text: str) -> AgentResult:
        """Return the schema fixture without calling a provider."""
        return AgentResult(self.answer, self.session_id)

    async def close(self) -> None:
        """Close the synthetic session."""

    async def cancel(self) -> None:
        """Cancel the synthetic session."""


async def test_screening_uses_native_role_and_preserves_post_indexes(tmp_path: Path) -> None:
    """Only clearly irrelevant posts are removed and uncertain stance stays explicit."""
    seen = []
    answer = json.dumps({"items": [
        {"index": 1, "relevance": 0.1, "stance": "neutral", "confidence": 0.9},
        {"index": 0, "relevance": 0.9, "stance": "bullish", "confidence": 0.4},
    ]})

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> ScreeningSession:
        """Capture the explicit screening role's harness configuration."""
        seen.append(spec.agent)
        return ScreeningSession(answer)

    workflow = Workflow("NVDA", "2026-01-02", Settings(state_dir=tmp_path), session_factory=factory)
    keep, note = await workflow._screen_posts(["Relevant opinion", "Unrelated promotion"])
    assert keep == [True, False]
    assert "1 unclear" in note
    assert seen[0].startswith("Sentiment Screening")
    workflow.store.close()


async def test_bad_screening_preserves_all_evidence(tmp_path: Path) -> None:
    """A broken optional classifier cannot discard source posts silently."""
    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> ScreeningSession:
        """Return malformed optional screening output."""
        return ScreeningSession('{"items": []}')

    workflow = Workflow("NVDA", "2026-01-02", Settings(state_dir=tmp_path), session_factory=factory)
    keep, note = await workflow._screen_posts(["Post one", "Post two"])
    assert keep == [True, True]
    assert "unavailable" in note
    workflow.store.close()
