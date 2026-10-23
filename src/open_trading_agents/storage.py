from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from open_trading_agents.domain import AgentEvent


class RunStore:
    def __init__(self, directory: Path) -> None:
        """Open a run's local event log and completion ledger.

        Args:
            directory: The run's dedicated storage directory.
        """
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.db = sqlite3.connect(directory / "run.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                agent TEXT NOT NULL, stage TEXT NOT NULL, timestamp REAL NOT NULL, body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (agent TEXT PRIMARY KEY, native_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS steps (
                key TEXT PRIMARY KEY, agent TEXT NOT NULL, status TEXT NOT NULL,
                prompt TEXT NOT NULL, output TEXT, updated REAL NOT NULL
            );
        """)

    def set_metadata(self, key: str, value: Any) -> None:
        """Persist a JSON value belonging to this run.

        Args:
            key: Stable metadata name.
            value: JSON-serializable data.
        """
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, json.dumps(value)))

    def metadata(self, key: str, default: Any = None) -> Any:
        """Read one metadata value.

        Args:
            key: Metadata name.
            default: Value used when the key is absent.

        Returns:
            The stored JSON value or the default.
        """
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def append(self, events: list[AgentEvent]) -> None:
        """Append a batch atomically, deduplicating repeated event identifiers.

        Args:
            events: Ordered events drained from the pipeline's write queue.
        """
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO events (event_id, agent, stage, timestamp, body) VALUES (?, ?, ?, ?, ?)",
                [(e.event_id, e.agent, e.stage, e.timestamp, json.dumps(asdict(e), ensure_ascii=False)) for e in events],
            )

    def events(self, after: int = 0) -> list[AgentEvent]:
        """Read recorded events in their original order.

        Args:
            after: Sequence number to start after.

        Returns:
            Events suitable for replaying the UI or exporting a transcript.
        """
        rows = self.db.execute("SELECT body FROM events WHERE sequence>? ORDER BY sequence", (after,))
        return [AgentEvent(**json.loads(row[0])) for row in rows]

    def save_session(self, agent: str, native_id: str) -> None:
        """Keep the explicit harness session identifier for an agent.

        Args:
            agent: Logical role name.
            native_id: Harness-owned session identifier or path.
        """
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?)", (agent, native_id))

    def session(self, agent: str) -> str | None:
        """Look up the native session for a logical role.

        Args:
            agent: Logical role name.

        Returns:
            Its saved identifier, or None for a new role.
        """
        row = self.db.execute("SELECT native_id FROM sessions WHERE agent=?", (agent,)).fetchone()
        return row[0] if row else None

    def step(self, key: str) -> dict[str, Any] | None:
        """Read completion state for one pipeline step.

        Args:
            key: Deterministic step identifier.

        Returns:
            The stored step fields, or None before the first attempt.
        """
        row = self.db.execute("SELECT * FROM steps WHERE key=?", (key,)).fetchone()
        return dict(row) if row else None

    def begin_step(self, key: str, agent: str, prompt: str) -> None:
        """Record new work before submitting it to a harness.

        Args:
            key: Deterministic step identifier.
            agent: Logical role.
            prompt: Exact supplied application input.
        """
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO steps VALUES (?, ?, 'pending', ?, NULL, ?)", (key, agent, prompt, time.time()))

    def finish_step(self, key: str, output: str) -> None:
        """Commit completed output before downstream stages consume it.

        Args:
            key: Deterministic step identifier.
            output: Settled agent output.
        """
        with self.db:
            self.db.execute("UPDATE steps SET status='complete', output=?, updated=? WHERE key=?", (output, time.time(), key))

    def close(self) -> None:
        """Close the local database connection."""
        self.db.close()
