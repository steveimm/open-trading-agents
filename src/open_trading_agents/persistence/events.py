import queue
import threading
from pathlib import Path

from open_trading_agents.analysis.models import AgentEvent
from open_trading_agents.persistence.storage import RunStore


class EventRecorder:
    def __init__(self, directory: Path) -> None:
        """Start a batched disk writer independent of streaming and rendering.

        Args:
            directory: The run's database directory.
        """
        self.directory = directory
        self.queue: queue.Queue[AgentEvent | threading.Event | None] = queue.Queue()
        self.error: Exception | None = None
        self.closed = False
        self.worker = threading.Thread(target=self._write, name="trading-event-writer", daemon=True)
        self.worker.start()

    def append(self, event: AgentEvent) -> None:
        """Enqueue an event without performing filesystem work in the caller.

        Args:
            event: The next ordered event.

        Raises:
            RuntimeError: Persistence failed or the writer is closed.
        """
        if self.error:
            raise RuntimeError("event persistence failed") from self.error
        if self.closed:
            raise RuntimeError("event recorder is closed")
        self.queue.put_nowait(event)

    def _write(self) -> None:
        """Drain queued events using a thread-owned SQLite connection."""
        store = None
        try:
            store = RunStore(self.directory)
            with (self.directory / "transcript.jsonl").open("w", encoding="utf-8") as transcript:
                for body in store.event_json():
                    transcript.write(body + "\n")
                transcript.flush()
                done = False
                while not done:
                    first = self.queue.get()
                    if first is None:
                        break
                    if isinstance(first, threading.Event):
                        transcript.flush()
                        first.set()
                        continue
                    barrier = None
                    batch = [first]
                    while len(batch) < 256:
                        try:
                            event = self.queue.get(timeout=0.025)
                        except queue.Empty:
                            break
                        if event is None:
                            done = True
                            break
                        if isinstance(event, threading.Event):
                            barrier = event
                            break
                        batch.append(event)
                    for body in store.append(batch):
                        transcript.write(body + "\n")
                    transcript.flush()
                    if barrier is not None:
                        barrier.set()
        except Exception as exc:
            self.error = exc
        finally:
            if store:
                store.close()

    def flush(self) -> None:
        """Wait until queued events reach SQLite and the transcript.

        Raises:
            RuntimeError: The writer failed or was already closed.
        """
        if self.closed:
            raise RuntimeError("event recorder is closed")
        barrier = threading.Event()
        self.queue.put_nowait(barrier)
        while not barrier.wait(0.05):
            if not self.worker.is_alive():
                raise RuntimeError("event persistence failed") from self.error
        if self.error:
            raise RuntimeError("event persistence failed") from self.error

    def close(self) -> None:
        """Flush all queued events and report any disk-writing failure.

        Raises:
            RuntimeError: Events could not be persisted.
        """
        if not self.closed:
            self.closed = True
            self.queue.put_nowait(None)
            self.worker.join()
        if self.error:
            raise RuntimeError("event persistence failed") from self.error
