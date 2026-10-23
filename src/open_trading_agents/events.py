import queue
import threading
from pathlib import Path

from open_trading_agents.domain import AgentEvent
from open_trading_agents.storage import RunStore


class EventRecorder:
    def __init__(self, directory: Path) -> None:
        """Start a batched disk writer independent of streaming and rendering.

        Args:
            directory: The run's database directory.
        """
        self.directory = directory
        self.queue: queue.Queue[AgentEvent | None] = queue.Queue()
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
            done = False
            while not done:
                first = self.queue.get()
                if first is None:
                    break
                batch = [first]
                while len(batch) < 256:
                    try:
                        event = self.queue.get(timeout=0.025)
                    except queue.Empty:
                        break
                    if event is None:
                        done = True
                        break
                    batch.append(event)
                store.append(batch)
        except Exception as exc:
            self.error = exc
        finally:
            if store:
                store.close()

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
