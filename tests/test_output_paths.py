import os
from pathlib import Path

import pytest

from open_trading_agents.analysis.models import AgentEvent, AgentResult, AgentSpec, EventSink, HarnessConfig
from open_trading_agents.analysis.workflow import Workflow
from open_trading_agents.config import Settings
from open_trading_agents.persistence.storage import RunStore


class OutputSession:
    def __init__(self, spec: AgentSpec, emit: EventSink, failures: list[str]) -> None:
        """Save a native session fixture in the workspace supplied by the workflow.

        Args:
            spec: Role and storage paths supplied to the harness.
            emit: Workflow event recorder.
            failures: Roles that should fail their next accepted turn.
        """
        self.spec, self.emit, self.failures = spec, emit, failures
        self.session_id = spec.session_id or f"native:{spec.agent}"
        spec.session_dir.mkdir(parents=True, exist_ok=True)
        (spec.session_dir / "session.txt").write_text(self.session_id)

    async def prompt(self, text: str) -> AgentResult:
        """Return synthetic role outputs without data or model requests.

        Args:
            text: Newly submitted application prompt.

        Returns:
            A valid role response carrying the native session reference.
        """
        await self.emit(AgentEvent("prompt", self.spec.agent, self.spec.stage, text))
        await self.emit(AgentEvent("status", self.spec.agent, self.spec.stage, data={"prompt_accepted": True}))
        if self.spec.agent in self.failures:
            self.failures.remove(self.spec.agent)
            raise RuntimeError("interrupted fixture")

        outputs = {
            "Research Manager": '{"recommendation":"Hold","rationale":"Evidence","strategic_actions":"Wait"}',
            "Trader": '{"action":"Hold","reasoning":"Evidence"}',
            "Portfolio Manager": '{"rating":"Hold","executive_summary":"Wait","investment_thesis":"Evidence"}',
        }
        answer = outputs.get(self.spec.agent, f"{self.spec.agent} research")
        await self.emit(AgentEvent("text", self.spec.agent, self.spec.stage, answer))

        return AgentResult(answer, self.session_id)

    async def cancel(self) -> None:
        """Cancel the fixture without external effects."""

    async def close(self) -> None:
        """Keep fixture session files available for resume."""


@pytest.mark.parametrize("custom_output", [False, True])
async def test_output_root_contains_run_artifacts_and_preserves_shared_storage(tmp_path: Path, custom_output: bool) -> None:
    """Export a complete run under its selected root while retaining shared cache and memory.

    Args:
        tmp_path: Isolated application and output locations.
        custom_output: Whether this run supplies an output directory override.
    """
    specs: list[AgentSpec] = []

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> OutputSession:
        """Capture harness storage paths and return an offline session.

        Args:
            config: Selected harness settings.
            spec: Role and storage paths.
            emit: Workflow event recorder.

        Returns:
            A synthetic session with saved native files.
        """
        specs.append(spec)
        return OutputSession(spec, emit, [])

    state_dir = tmp_path / "application"
    output_dir = tmp_path / "exports with spaces" if custom_output else None
    settings = Settings(state_dir=state_dir, analysts=("market",))
    workflow = Workflow("NVDA", "2026-01-02", settings, None, None, factory, output_dir=output_dir)
    result = await workflow.run()

    expected_root = output_dir if output_dir is not None else state_dir / "runs"
    assert result.directory == expected_root / result.run_id
    assert result.rating == "Hold"
    for relative in ("run.sqlite", "tool-context.json", "transcript.jsonl", "reports/complete_report.md",
                     "reports/complete_report.html", "reports/usage.json"):
        assert (result.directory / relative).is_file()
    assert specs
    assert all(spec.workspace.is_relative_to(result.directory / "agents") for spec in specs)
    assert all((spec.session_dir / "session.txt").is_file() for spec in specs)

    assert workflow.data_config["data_cache_dir"] == str(state_dir / "cache")
    assert (state_dir / "memory.sqlite").is_file()
    assert not (result.directory / "memory.sqlite").exists()
    assert settings.state_dir == state_dir
    if custom_output:
        assert not (state_dir / "runs").exists()
        assert not (output_dir / "memory.sqlite").exists()


async def test_resume_reopens_exact_external_directory_and_native_sessions(tmp_path: Path) -> None:
    """Resume an interrupted custom-root run without creating a default-root copy.

    Args:
        tmp_path: Isolated application and output locations.
    """
    specs: list[AgentSpec] = []
    failures = ["Neutral Analyst"]

    async def factory(config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> OutputSession:
        """Record restored session references and interrupt one role once.

        Args:
            config: Selected harness settings.
            spec: Role and storage paths.
            emit: Workflow event recorder.

        Returns:
            A resumable synthetic native session.
        """
        specs.append(spec)
        return OutputSession(spec, emit, failures)

    state_dir = tmp_path / "application"
    settings = Settings(state_dir=state_dir, analysts=("market",), memory_enabled=False)
    workflow = Workflow("NVDA", "2026-01-02", settings, session_factory=factory, output_dir=tmp_path / "exports with spaces")
    original_inputs = workflow.store.metadata("input")
    original_context = workflow.context_file.read_bytes()
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        await workflow.run()

    previous_sessions = len(specs)
    resumed = Workflow.resume(workflow.directory, session_factory=factory)
    assert resumed.directory == workflow.directory
    assert resumed.store.metadata("input") == original_inputs
    assert resumed.context_file.read_bytes() == original_context
    result = await resumed.run()

    restored = specs[previous_sessions:]
    assert [spec.agent for spec in restored] == ["Neutral Analyst", "Portfolio Manager"]
    assert restored[0].session_id == "native:Neutral Analyst"
    assert restored[0].session_dir == specs[previous_sessions - 1].session_dir
    assert result.directory == workflow.directory
    assert not (state_dir / "runs").exists()

    store = RunStore(result.directory)
    try:
        assert store.metadata("status") == "complete"
        assert store.session("Market Analyst") == "native:Market Analyst"
    finally:
        store.close()


@pytest.mark.parametrize("path_kind", ["relative", "tilde"])
def test_output_paths_are_normalized_before_run_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_kind: str) -> None:
    """Accept shell-style paths and keep separate runs in distinct folders.

    Args:
        tmp_path: Isolated working directory and output location.
        monkeypatch: Temporary working-directory control.
        path_kind: Relative or home-relative output path spelling.
    """
    monkeypatch.chdir(tmp_path)
    output_dir = Path("exports") if path_kind == "relative" else Path("~") / os.path.relpath(tmp_path, Path.home()) / "exports"
    settings = Settings(state_dir=tmp_path / "application", memory_enabled=False)
    first = Workflow("NVDA", "2026-01-02", settings, output_dir=output_dir)
    second = Workflow("NVDA", "2026-01-02", settings, output_dir=output_dir)

    try:
        assert first.directory.parent == second.directory.parent == tmp_path / "exports"
        assert first.directory != second.directory
        assert not (settings.state_dir / "runs").exists()
    finally:
        first.store.close()
        second.store.close()
