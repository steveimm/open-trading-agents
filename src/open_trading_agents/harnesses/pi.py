from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from open_trading_agents.analysis.models import AgentResult
from open_trading_agents.harnesses.process import HarnessError, JsonProcess

_BREAKDOWN_PATHS = {
    "cacheRead": (("prompt_tokens_details", "cached_tokens"), ("input_tokens_details", "cached_tokens"),
                  ("cache_read_input_tokens",), ("prompt_cache_hit_tokens",), ("cached_tokens",),
                  ("cachedContentTokenCount",), ("cacheReadInputTokens",)),
    "cacheWrite": (("prompt_tokens_details", "cache_write_tokens"), ("input_tokens_details", "cache_write_tokens"),
                   ("cache_creation_input_tokens",), ("cacheWriteInputTokens",)),
    "reasoning": (("completion_tokens_details", "reasoning_tokens"), ("output_tokens_details", "reasoning_tokens"),
                  ("thoughtsTokenCount",), ("reasoning_tokens",)),
}


def usage_counts(usage: dict[str, Any], raw: dict[str, Any] | None = None) -> dict[str, int | None]:
    """Normalize native counts while distinguishing observed zero from missing metadata.

    Args:
        usage: Pi's normalized usage, whose input excludes cache reads and writes.
        raw: Optional provider-reported usage, used to establish breakdown availability.

    Returns:
        Input inclusive of cache, output inclusive of reasoning, and known optional counts.
    """
    result = {
        "input_tokens": usage.get("input", 0) + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0),
        "output_tokens": usage.get("output", 0),
    }
    for target, source in (("cache_read_tokens", "cacheRead"), ("cache_write_tokens", "cacheWrite"),
                           ("reasoning_tokens", "reasoning")):
        count = None
        for path in _BREAKDOWN_PATHS[source]:
            value: Any = raw or {}
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                count = value
                break
        fallback = usage.get(source)
        if count is None and isinstance(fallback, int) and not isinstance(fallback, bool) and fallback > 0:
            count = fallback
        result[target] = count
    return result


class PiSession(JsonProcess):
    async def start(self) -> None:
        """Start native pi RPC with financial tools and request visibility."""
        extension = Path(__file__).with_name("pi_extension.mjs")
        args = [*(self.config.command or ("pi",)), "--mode", "rpc", "--offline", "--no-extensions", "--no-skills",
                "--no-prompt-templates", "--no-themes", "--no-mcp", "--no-context-files", "--no-approve",
                "--extension", str(extension), "--system-prompt", self.spec.system_prompt,
                "--session-dir", str(self.spec.session_dir)]
        native_tools = ["read", "write", "edit", "bash"] if self.spec.allow_execution else []
        active_tools = [*native_tools, *self.spec.tools]
        args.extend(["--tools", ",".join(active_tools)] if active_tools else ["--no-tools"])
        if self.config.model:
            args.extend(["--model", self.config.model])
        if self.config.provider:
            args.extend(["--provider", self.config.provider])
        if self.config.effort:
            args.extend(["--thinking", self.config.effort])
        if self.session_id:
            args.extend(["--session", self.session_id])
        args.extend(self.config.extra_args)
        self.ready = asyncio.get_running_loop().create_future()
        self.spec.session_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.spec.session_dir / "pi_native_events.jsonl"
        self.metadata_path.touch(mode=0o600)
        self.metadata_offset = self.metadata_path.stat().st_size
        self.message_id = ""
        self.output = ""
        self.stop_reason = "stop"
        self.budget_error = ""
        self.raw_usage: dict[str, Any] = {}
        self.request_number = 0
        await self.start_process(args, {
            "PI_OFFLINE": "1", "PI_TELEMETRY": "0", "OTA_TOOL_COMMAND": json.dumps(self.spec.tool_server_command),
            "OTA_TOOL_ENV": json.dumps(self.spec.tool_environment), "OTA_TOOLS": json.dumps(self.spec.tools),
            "OTA_MAX_TOOL_ROUNDS": str(self.spec.max_tool_rounds), "OTA_METADATA_FILE": str(self.metadata_path),
            "OTA_NATIVE_TOOLS": json.dumps(native_tools),
        })
        response = await self.request({"type": "get_state"})
        await self.read_metadata()
        if not self.ready.done():
            raise HarnessError(f"Pi financial extension did not initialize: {self.stderr_tail[-2000:]}")
        await self.save_session(response.get("data", {}))

    async def read_metadata(self) -> None:
        """Read native request metadata written before the corresponding RPC events."""
        with self.metadata_path.open(encoding="utf-8") as source:
            source.seek(self.metadata_offset)
            for line in source:
                await self.handle(json.loads(line))
            self.metadata_offset = source.tell()

    async def save_session(self, state: dict[str, Any]) -> None:
        """Publish the persisted session locator and resolved model exposed by pi.

        Args:
            state: Native get_state response payload.
        """
        identifier = state.get("sessionFile") or state.get("sessionId")
        if identifier and identifier != self.session_id:
            self.session_id = identifier
            await self.event("session", session_id=identifier)
        model = state.get("model")
        if isinstance(model, dict):
            await self.observe_model(model.get("id"), model.get("provider"))

    async def prompt(self, text: str) -> AgentResult:
        """Append input to the native session and wait through automatic recovery.

        Args:
            text: New input for this agent's current stage.

        Returns:
            The final assistant output and persisted session locator.
        """
        async with self.prompt_lock:
            self.output, self.stop_reason, self.budget_error = "", "stop", ""
            self.finished = asyncio.get_running_loop().create_future()
            await self.event("prompt", text, message_id=uuid.uuid4().hex)
            response = await self.request({"type": "prompt", "message": text})
            if response.get("data", {}).get("disposition") == "handled":
                raise HarnessError("Pi handled the input as a command instead of starting an agent rollout")
            await self.event("status", "Prompt accepted by pi", prompt_accepted=True)
            try:
                await self.finished
            except asyncio.CancelledError:
                await self.cancel()
                raise
            state = await self.request({"type": "get_state"})
            await self.save_session(state.get("data", {}))
            if self.budget_error:
                raise HarnessError(self.budget_error)
            if self.stop_reason in {"error", "aborted", "length"}:
                raise HarnessError(f"Pi stopped without a complete response: {self.stop_reason}")
            return AgentResult(self.output, self.session_id or "", self.stop_reason)

    async def cancel(self) -> None:
        """Abort the current pi operation and wait until native work stops."""
        if self.process is not None and self.process.returncode is None:
            await self.request({"type": "abort"})

    async def handle(self, record: dict[str, Any]) -> None:
        """Translate native pi session events without accumulating partial usage.

        Args:
            record: Native RPC event or application extension event.
        """
        kind = record.get("type", "")
        if kind in {"message_start", "message_end", "agent_settled"}:
            await self.read_metadata()
        if kind == "ota_ready":
            if not self.ready.done():
                self.ready.set_result(None)
        elif kind == "ota_budget":
            await self.event("status", "Tool-round budget exhausted; completing response", native=record)
        elif kind == "ota_budget_exceeded":
            self.budget_error = f"Pi exceeded {record['maximumRounds']} tool rounds after its final-response opportunity"
            await self.event("error", self.budget_error, source="tool_budget", native=record)
        elif kind == "ota_provider_request":
            self.raw_usage = {}
            self.request_number += 1
            await self.event("system", json.dumps(record["payload"], ensure_ascii=False), source="native_request",
                             message_id=f"request-{self.request_number}")
        elif kind == "ota_provider_usage":
            self.raw_usage.update(record["usage"])
        elif kind == "message_start" and record.get("message", {}).get("role") == "assistant":
            self.message_id = uuid.uuid4().hex
            await self.observe_model(record["message"].get("model"), record["message"].get("provider"))
        elif kind == "message_update":
            event = record.get("assistantMessageEvent", {})
            event_type = event.get("type")
            if event_type in {"text_delta", "thinking_delta"}:
                await self.event("text" if event_type == "text_delta" else "reasoning", event.get("delta", ""),
                                 message_id=self.message_id)
        elif kind == "message_end" and record.get("message", {}).get("role") == "assistant":
            message = record["message"]
            await self.observe_model(message.get("model"), message.get("provider"))
            text = "".join(part.get("text", "") for part in message.get("content", []) if part.get("type") == "text")
            self.output, self.stop_reason = text, message.get("stopReason", "stop")
            if text:
                await self.event("text", text, message_id=self.message_id, final=True)
            usage = message.get("usage", {})
            if usage:
                await self.event("usage", request_id=self.message_id, **usage_counts(usage, self.raw_usage))
            if message.get("errorMessage"):
                await self.event("error", message["errorMessage"], message_id=self.message_id)
        elif kind in {"tool_execution_start", "tool_execution_update", "tool_execution_end"}:
            starting = kind == "tool_execution_start"
            result = record.get("partialResult") if kind == "tool_execution_update" else record.get("result")
            content = (result or {}).get("content", []) if isinstance(result, dict) else []
            text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
            await self.event("tool_start" if starting else "tool_result",
                             json.dumps(record.get("args"), ensure_ascii=False) if starting else text,
                             tool=record.get("toolName"), tool_call_id=record.get("toolCallId"),
                             is_error=record.get("isError", False), snapshot=kind == "tool_execution_update",
                             final=kind == "tool_execution_end", native=result)
        elif kind == "entry_appended" and record.get("entry", {}).get("type") == "usage":
            entry = record["entry"]
            await self.auxiliary_usage(entry.get("usage", {}), entry.get("id") or uuid.uuid4().hex, entry.get("kind", "auxiliary"))
        elif kind in {"compaction_end", "auto_compaction_end"}:
            await self.read_metadata()
            await self.auxiliary_usage((record.get("result") or {}).get("usage", {}), uuid.uuid4().hex, "compaction")
            await self.event("status", "Compaction finished", native=record)
        elif kind == "agent_settled":
            if record.get("aborted"):
                self.stop_reason = "aborted"
            if self.finished is not None and not self.finished.done():
                self.finished.set_result(None)
        elif kind in {"auto_retry_start", "auto_retry_end", "auto_compaction_start", "compaction_start", "agent_start",
                      "summarization_retry_scheduled", "summarization_retry_attempt_start", "summarization_retry_finished"}:
            await self.event("status", kind.replace("_", " "), native=record)

    async def auxiliary_usage(self, usage: dict[str, Any], identifier: str, source: str) -> None:
        """Count native summarization and other auxiliary model requests separately.

        Args:
            usage: Native aggregate token counts for the auxiliary operation.
            identifier: Stable native entry ID, when available.
            source: Native purpose of the additional model usage.
        """
        if not usage:
            return
        await self.event("usage", request_id=f"{source}:{identifier}", source=source, **usage_counts(usage))
