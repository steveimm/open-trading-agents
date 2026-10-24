import asyncio
import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from open_trading_agents.domain import AgentEvent, AgentSpec, HarnessConfig
from open_trading_agents.harnesses.process import JsonProcess


def child_stopped(pid: int) -> bool:
    """Wait briefly for a killed fixture child to disappear or become nonrunning.

    Args:
        pid: Identifier returned by the owned fixture process.

    Returns:
        Whether the process exited or became a zombie awaiting its system reaper.
    """
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            if stat.read_text().rsplit(")", 1)[1].split()[0] in {"Z", "X"}:
                return True
        except (FileNotFoundError, ProcessLookupError):
            return True
        time.sleep(0.02)
    return False


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix" or not Path("/proc").exists(), reason="Requires Linux process-state inspection")
@pytest.mark.parametrize("parent_ignores_term", [False, True])
async def test_forced_shutdown_stops_only_owned_parent_and_child(tmp_path: Path, parent_ignores_term: bool) -> None:
    """Kill a stubborn native process tree after giving stdin EOF a chance to finish."""
    child_script = Path(__file__).with_name("fixtures") / "stubborn_child.py"
    parent_script = Path(__file__).with_name("fixtures") / "stubborn_parent.py"
    marker = tmp_path / "child.pid"

    async def emit(event: AgentEvent) -> None:
        """Accept diagnostic events from the owned fixture process."""
        return None

    session = JsonProcess(HarnessConfig(), AgentSpec("fixture", "analysis", "", tmp_path, tmp_path), emit)
    unrelated = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time;time.sleep(60)")
    child_pid: int | None = None
    try:
        await session.start_process([sys.executable, str(parent_script), str(child_script), str(marker), str(parent_ignores_term)])
        response = await session.request({"method": "ready"})
        child_pid = response["result"]["child"]
        assert session.process is not None
        assert response["result"]["group"] == session.process.pid
        assert os.getpgid(child_pid) == session.process.pid
        assert os.getpgrp() != session.process.pid
        started = time.monotonic()
        await asyncio.wait_for(session.close(), 10)
        assert time.monotonic() - started < 9
        assert session.process.returncode is not None
        assert await asyncio.to_thread(child_stopped, child_pid)
        assert unrelated.returncode is None
    finally:
        if session.process is not None and session.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(session.process.pid, signal.SIGKILL)
            await session.process.wait()
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
        unrelated.terminate()
        await unrelated.wait()


@pytest.mark.asyncio
async def test_graceful_close_does_not_signal_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow a cooperative native process to finish naturally after stdin closes."""
    async def emit(event: AgentEvent) -> None:
        """Accept events from the cooperative fixture."""
        return None

    def unexpected_signal(force: bool) -> None:
        """Reject any forced signal on the successful graceful shutdown path."""
        pytest.fail(f"Unexpected forced shutdown: {force}")

    session = JsonProcess(HarnessConfig(), AgentSpec("fixture", "analysis", "", tmp_path, tmp_path), emit)
    await session.start_process([sys.executable, "-c", "import sys;sys.stdin.read()"])
    monkeypatch.setattr(session, "stop_process_group", unexpected_signal)
    await session.close()
    assert session.process is not None and session.process.returncode == 0
