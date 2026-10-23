from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from open_trading_agents.domain import AgentEvent, AgentSpec, EventSink, HarnessConfig


class HarnessError(RuntimeError):
    pass


class JsonProcess:
    def __init__(self, config: HarnessConfig, spec: AgentSpec, emit: EventSink) -> None:
        self.config = config
        self.spec = spec
        self.emit = emit
        self.session_id = spec.session_id
        self.process: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.reader: asyncio.Task[None] | None = None
        self.stderr_reader: asyncio.Task[None] | None = None
        self.finished: asyncio.Future[None] | None = None
        self.prompt_lock = asyncio.Lock()
        self.stderr_tail = ""
        self.closing = False
        self.failure: Exception | None = None

    async def event(self, kind: str, text: str = "", **data: Any) -> None:
        """Publish one normalized native harness event.

        Args:
            kind: Event category used by the application.
            text: Human-readable event content.
            data: Structured metadata preserved with the event.
        """
        await self.emit(AgentEvent(kind, self.spec.agent, self.spec.stage, text, data))

    async def start_process(self, arguments: list[str], environment: dict[str, str] | None = None) -> None:
        """Start a native CLI with protocol pipes and an owned process group on POSIX.

        Args:
            arguments: Executable and command-line arguments.
            environment: Adapter-owned environment overrides.
        """
        self.spec.workspace.mkdir(parents=True, exist_ok=True)
        self.spec.session_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **self.config.environment, **(environment or {})}
        try:
            self.process = await asyncio.create_subprocess_exec(
                *arguments, cwd=self.spec.workspace, env=env, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as error:
            raise HarnessError(f"Harness executable not found: {arguments[0]}") from error
        self.reader = asyncio.create_task(self.read_messages())
        self.stderr_reader = asyncio.create_task(self.read_stderr())
        await self.event("system", self.spec.system_prompt, source="application", message_id="system-prompt")

    async def send(self, value: dict[str, Any]) -> None:
        """Write one protocol record without shell interpolation.

        Args:
            value: JSON-compatible command or response.
        """
        if self.failure is not None:
            raise HarnessError(str(self.failure)) from self.failure
        if self.process is None or self.process.stdin is None or self.process.returncode is not None:
            raise HarnessError("Harness process is not running")
        self.process.stdin.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
        await self.process.stdin.drain()

    async def request(self, value: dict[str, Any], acceptance_seconds: float = 60) -> dict[str, Any]:
        """Correlate a protocol response while continuing to consume streamed events.

        Args:
            value: Native command without an identifier.
            acceptance_seconds: Maximum wait for acceptance, not for model completion.

        Returns:
            The native response record.
        """
        identifier = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.send({**value, "id": identifier})
            return await asyncio.wait_for(future, acceptance_seconds)
        finally:
            self.pending.pop(identifier, None)

    async def read_messages(self) -> None:
        """Dispatch native notifications and resolve correlated command responses."""
        assert self.process is not None and self.process.stdout is not None
        failure: Exception | None = None
        try:
            while line := await self.process.stdout.readline():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise HarnessError("Harness emitted an invalid JSON protocol record") from error
                identifier = str(record.get("id", ""))
                if identifier in self.pending and ("result" in record or "error" in record or record.get("type") == "response"):
                    future = self.pending[identifier]
                    if not future.done():
                        if record.get("error") or record.get("success") is False:
                            future.set_exception(HarnessError(str(record.get("error") or record)))
                        else:
                            future.set_result(record)
                else:
                    await self.handle(record)
            if not self.closing:
                failure = HarnessError(f"Harness exited before the session closed. {self.stderr_tail[-2000:]}")
        except asyncio.CancelledError:
            return
        except Exception as error:
            failure = error
        finally:
            if failure is not None:
                self.failure = failure
                for future in [*self.pending.values(), self.finished]:
                    if future is not None and not future.done():
                        future.set_exception(failure)
                await self.event("error", str(failure))

    async def read_stderr(self) -> None:
        """Retain a bounded diagnostic tail without blocking the child process."""
        assert self.process is not None and self.process.stderr is not None
        while chunk := await self.process.stderr.read(4096):
            self.stderr_tail = (self.stderr_tail + chunk.decode(errors="replace"))[-8000:]

    async def handle(self, record: dict[str, Any]) -> None:
        """Handle a native event in a concrete adapter.

        Args:
            record: Parsed native protocol record.
        """
        raise NotImplementedError

    def stop_process_group(self, force: bool) -> None:
        """Signal only this harness and its POSIX process group during forced shutdown.

        Args:
            force: Kill remaining processes after graceful termination has failed.
        """
        process = self.process
        if process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif process.returncode is None:
                if force:
                    process.kill()
                else:
                    process.terminate()

    async def close(self) -> None:
        """Close native input first, then bound forced cleanup of the owned process group."""
        self.closing = True
        if self.finished is not None and not self.finished.done():
            self.finished.cancel()
        process = self.process
        try:
            if process is not None and process.returncode is None:
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), 3)
                except TimeoutError:
                    self.stop_process_group(force=False)
                    try:
                        await asyncio.wait_for(process.wait(), 3)
                    except TimeoutError:
                        pass
                    self.stop_process_group(force=True)
                    await asyncio.wait_for(process.wait(), 3)
        finally:
            for task in (self.reader, self.stderr_reader):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            for future in self.pending.values():
                if not future.done():
                    future.cancel()


async def start_checked(session: JsonProcess, start: Callable[[], Awaitable[None]]) -> None:
    """Close partially initialized native processes on startup failure.

    Args:
        session: Adapter owning the native process.
        start: Adapter initialization routine.
    """
    try:
        await start()
    except BaseException:
        await session.close()
        raise
