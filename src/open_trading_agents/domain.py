from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    agent: str
    stage: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)


EventSink = Callable[[AgentEvent], Awaitable[None]]


@dataclass(frozen=True)
class HarnessConfig:
    kind: Literal["pi", "codex", "claude"] = "pi"
    command: tuple[str, ...] = ()
    model: str | None = None
    provider: str | None = None
    extra_args: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec:
    agent: str
    stage: str
    system_prompt: str
    workspace: Path
    session_dir: Path
    session_id: str | None = None
    tools: tuple[str, ...] = ()
    tool_server_command: tuple[str, ...] = ()
    tool_environment: dict[str, str] = field(default_factory=dict)
    max_tool_rounds: int = 20
    allow_execution: bool = True


@dataclass(frozen=True)
class AgentResult:
    text: str
    session_id: str
    stop_reason: str = "stop"


class HarnessSession(Protocol):
    session_id: str | None

    async def prompt(self, text: str) -> AgentResult:
        """Submit new input to the native session and wait for settled completion.

        Args:
            text: New application input, without replaying earlier session messages.

        Returns:
            The completed output and native session identifier.
        """
        ...

    async def cancel(self) -> None:
        """Cancel active work and wait for the harness to become idle."""
        ...

    async def close(self) -> None:
        """Release the harness process without discarding its saved session."""
        ...


STAGES = (
    ("analysis", "Analysis"),
    ("research", "Research debate"),
    ("research_decision", "Research decision"),
    ("trader", "Trader"),
    ("risk", "Risk debate"),
    ("portfolio", "Portfolio decision"),
    ("reports", "Reports"),
)

ANALYSTS = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
