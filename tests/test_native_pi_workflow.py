import asyncio
import json
import re
import shutil
import socket
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from open_trading_agents.config import HarnessSettings, Settings
from open_trading_agents.domain import AgentEvent
from open_trading_agents.memory import MemoryLog, Returns
from open_trading_agents.prompts import ROLES
from open_trading_agents.storage import RunStore
from open_trading_agents.tools import FinancialTools
from open_trading_agents.workflow import Workflow

pytestmark = pytest.mark.native


class NativeResponses:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.thoughts: dict[tuple[str, str], str] = {}
        self.calls: dict[str, str] = {}
        self.continuations: set[str] = set()
        self.calculated: Counter[str] = Counter()

    def reply(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Verify native history and produce a synthetic SSE tool turn or report.

        Args:
            payload: Actual provider request constructed by the installed pi CLI.

        Returns:
            Chat-completion chunks with deterministic token usage and reasoning.
        """
        messages = payload["messages"]
        instructions = "\n".join(message.get("content", "") for message in messages if message["role"] == "system")
        match = re.search(r"You are the (.+?) in a financial research workflow", instructions)
        assert match is not None, "Native request lost the role system prompt"
        role = match.group(1)
        tools = {tool["function"]["name"] for tool in payload["tools"]}
        assert "calculate" in tools
        assert not tools.intersection({"read", "write", "edit", "bash"})
        assert not {"temperature", "top_p", "reasoning_effort"}.intersection(payload)
        assert payload["model"] == "fixture-model"

        for message in messages:
            if message["role"] == "assistant":
                calls = message.get("tool_calls", [])
                identity = calls[0]["id"] if calls else message.get("content", "")
                assert message.get("reasoning_content") == self.thoughts[(role, identity)], "Native reasoning was not preserved"
                if calls:
                    assert self.calls[calls[0]["id"]] == role, "Tool call leaked across agent sessions"
            elif message["role"] == "tool":
                assert self.calls[message["tool_call_id"]] == role, "Native tool result lost its call identifier"
                assert json.loads(message["content"])["result"] == 25.0

        turn = sum(message["role"] == "user" for message in messages)
        if turn > 1:
            self.continuations.add(role)
        self.requests.append({"role": role, "turn": turn, "payload": payload})
        identifier = f"fixture-{len(self.requests)}"
        thought = f"Native private calculation check for {role}, turn {turn}, request {identifier}."
        chunks = [{"role": "assistant", "reasoning_content": thought}]
        if messages[-1]["role"] == "user":
            call_id = f"call_{len(self.requests)}"
            self.calls[call_id] = role
            self.thoughts[(role, call_id)] = thought
            arguments = json.dumps({"expression": "(125-100)/100*100"})
            chunks.extend([
                {"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                 "function": {"name": "calculate", "arguments": arguments[:12]}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": arguments[12:]}}]},
            ])
            stop = "tool_calls"
        else:
            assert messages[-1]["role"] == "tool"
            self.calculated[role] += 1
            decisions = {
                "Research Manager": {"recommendation": "Overweight", "rationale": "Verified synthetic change is 25%.",
                                     "strategic_actions": "Use a small synthetic allocation."},
                "Trader": {"action": "Buy", "reasoning": "The supplied synthetic calculation returned 25.",
                           "entry_price": 100, "stop_loss": 95, "position_sizing": "2% synthetic allocation"},
                "Portfolio Manager": {"rating": "Overweight", "executive_summary": "Synthetic workflow validation only.",
                                      "investment_thesis": "The native calculate result is 25.0.", "price_target": 125,
                                      "time_horizon": "5 sessions"},
            }
            answer = json.dumps(decisions[role]) if role in decisions else f"{role}, turn {turn}: verified (125-100)/100*100 = 25.0 using calculate. Synthetic evidence only."
            self.thoughts[(role, answer)] = thought
            chunks.extend([{"content": answer[:20]}, {"content": answer[20:]}])
            stop = "stop"
        result = [{"id": identifier, "object": "chat.completion.chunk", "model": "fixture-model", "created": 1,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in chunks]
        result.append({"id": identifier, "choices": [{"index": 0, "delta": {}, "finish_reason": stop}]})
        result.append({"id": identifier, "choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 4},
        }})
        return result


@contextmanager
def native_model_server() -> Iterator[tuple[str, NativeResponses]]:
    """Serve synthetic OpenAI SSE responses on loopback without outgoing network calls.

    Yields:
        The local API base URL and captured native request state.
    """
    state = NativeResponses()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            """Return streamed fixtures only for the native completion endpoint."""
            try:
                assert self.path == "/v1/chat/completions"
                assert self.headers.get("Authorization") == "Bearer fixture-key"
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with state.lock:
                    chunks = state.reply(payload)
                body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                encoded = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            except Exception as error:
                with state.lock:
                    state.failures.append(f"{type(error).__name__}: {error}")
                encoded = json.dumps({"error": {"message": state.failures[-1]}}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()

        def log_message(self, format: str, *args: Any) -> None:
            """Keep fixture HTTP access logs out of test output."""
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="native-pi-fixture", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def configure_pi(directory: Path, base_url: str) -> None:
    """Create isolated native model settings with a fake key and no auth files.

    Args:
        directory: Temporary pi configuration directory.
        base_url: Loopback-only model server.
    """
    directory.mkdir()
    model = {
        "id": "fixture-model", "name": "Loopback native workflow fixture", "reasoning": True, "input": ["text"],
        "contextWindow": 1048576, "maxTokens": 2048,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False,
                   "supportsUsageInStreaming": True, "maxTokensField": "max_tokens", "thinkingFormat": "openai",
                   "requiresThinkingAsText": False},
    }
    (directory / "models.json").write_text(json.dumps({"providers": {"fixture": {
        "baseUrl": base_url, "api": "openai-completions", "apiKey": "fixture-key", "models": [model],
    }}}))
    (directory / "settings.json").write_text(json.dumps({
        "defaultProvider": "fixture", "defaultModel": "fixture-model", "enableInstallTelemetry": False,
        "cacheWarming": "off", "compaction": {"enabled": False},
        "retry": {"enabled": False, "provider": {"maxRetries": 0}},
    }))


@pytest.mark.asyncio
async def test_native_pi_full_workflow_preserves_history_tools_and_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run all roles and reflection through installed pi against a loopback model fixture."""
    executable = shutil.which("pi")
    if executable is None:
        pytest.skip("The optional native integration test requires an installed pi CLI")

    def social_evidence(self: FinancialTools) -> str:
        """Supply local sentiment evidence without calling social data vendors."""
        return "Synthetic source evidence: fixture headlines and social observations only."

    def outcomes(ticker: str, trade_date: str, holding_days: int = 5, benchmark: str = "SPY") -> Returns:
        """Return an already-known fixture outcome for the seeded prior decision."""
        return 0.05, 0.02, holding_days, "2026-01-09"

    def refuse_python_network(self: socket.socket, address: Any) -> None:
        """Fail if application Python tries to contact a model or external data service."""
        raise AssertionError(f"Application attempted direct network access: {address}")

    monkeypatch.setattr(FinancialTools, "get_social_sentiment", social_evidence)
    monkeypatch.setattr("open_trading_agents.memory.fetch_returns", outcomes)
    settings = Settings(state_dir=tmp_path / "state", harness=HarnessSettings(kind="pi", command=(executable,),
                        model="fixture-model", provider="fixture"), research_rounds=2, risk_rounds=1,
                        allow_execution=False, web_research=False, memory_enabled=True)
    memory = MemoryLog(settings.state_dir / "memory.sqlite")
    prior_id = memory.record("NVDA", "2026-01-02", "Overweight", "Synthetic prior decision", "fixture-prior")
    events: list[AgentEvent] = []

    async def observe(event: AgentEvent) -> None:
        """Retain ordered workflow and native harness evidence."""
        events.append(event)

    with native_model_server() as (base_url, state):
        pi_directory = tmp_path / "pi-config"
        configure_pi(pi_directory, base_url)
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(pi_directory))
        monkeypatch.setenv("PI_OFFLINE", "1")
        monkeypatch.setenv("PI_TELEMETRY", "0")
        monkeypatch.setattr(socket.socket, "connect", refuse_python_network)
        monkeypatch.setattr(socket.socket, "connect_ex", refuse_python_network)
        workflow = Workflow("NVDA", "2026-01-12", settings)
        result = await asyncio.wait_for(workflow.run(observe), 120)
        assert state.failures == []

    expected_roles = {role.name for key, role in ROLES.items() if key != "screening"}
    assert set(state.calculated) == expected_roles
    assert state.calculated["Bull Researcher"] == state.calculated["Bear Researcher"] == 2
    assert sum(state.calculated.values()) == 15
    assert len(state.requests) == 30
    assert state.continuations == {"Bull Researcher", "Bear Researcher"}
    assert result.rating == "Overweight"
    native_auth = pi_directory / "auth.json"
    assert not native_auth.exists() or json.loads(native_auth.read_text()) == {}
    assert [event.stage for event in events if event.kind == "stage_start"] == [
        "analysis", "research", "research_decision", "trader", "risk", "portfolio", "reports",
    ]
    assert [event.stage for event in events if event.kind == "stage_end"] == [
        "analysis", "research", "research_decision", "trader", "risk", "portfolio", "reports",
    ]
    assert [event.agent for event in events if event.kind == "turn_start"] == [
        "Bull Researcher", "Bear Researcher", "Bull Researcher", "Bear Researcher",
        "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst",
    ]
    assert sum(bool(event.data.get("prompt_accepted")) for event in events) == 15
    assert len([event for event in events if event.kind == "usage"]) == 30
    assert len([event for event in events if event.kind == "tool_result" and event.data.get("tool") == "calculate"]) == 15
    assert any(event.kind == "reasoning" and "Native private calculation" in event.text for event in events)
    assert not any(event.kind == "error" for event in events)

    store = RunStore(result.directory)
    try:
        assert store.metadata("status") == "complete"
        rows = store.db.execute("SELECT agent,native_id FROM sessions").fetchall()
        assert len(rows) == len({row["native_id"] for row in rows}) == 13
        for row in rows:
            session_file = Path(row["native_id"])
            contents = await asyncio.to_thread(session_file.read_text)
            native_entries = [json.loads(line) for line in contents.splitlines()]
            assert native_entries[0]["type"] == "session"
            if row["agent"] in {"Bull Researcher", "Bear Researcher"}:
                assert sum(entry.get("message", {}).get("role") == "user" for entry in native_entries) == 2
        assert store.step(f"reflection:{prior_id}")["status"] == "complete"
        assert all(row[0] == "complete" for row in store.db.execute("SELECT status FROM steps"))
    finally:
        store.close()
    resolved = next(entry for entry in memory.entries() if entry["id"] == prior_id)
    assert resolved["status"] == "resolved" and "Reflection" in resolved["reflection"]
    assert any("Reflection, turn 1" in event.text for event in events if event.kind == "prompt" and event.agent == "Portfolio Manager")
    reports = result.directory / "reports"
    assert (reports / "complete_report.md").is_file()
    assert (reports / "complete_report.html").is_file()
    assert (result.directory / "transcript.jsonl").is_file()
    usage = json.loads((reports / "usage.json").read_text())
    assert usage["totals"]["reported_requests"] == 30
    assert usage["totals"]["input_tokens"] == 3000
    assert usage["totals"]["output_tokens"] == 300
    assert usage["totals"]["cache_read_tokens"] == 1800
    assert usage["totals"]["cache_write_tokens"] == 150
    assert usage["totals"]["reasoning_tokens"] == 120
    assert usage["totals"]["cache_read_tokens_complete"] is True
    assert len(usage["agents"]) == 13
