from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from open_trading_agents.domain import AgentResult
from open_trading_agents.harnesses.process import HarnessError, JsonProcess


class PiSession(JsonProcess):
    async def start(self) -> None:
        """Start native pi RPC with financial tools and request visibility."""
        extension = Path(__file__).with_name("pi_extension.mjs")
        args = [*(self.config.command or ("pi",)), "--mode", "rpc", "--offline", "--no-extensions", "--no-skills",
                "--no-prompt-templates", "--no-themes", "--no-mcp", "--no-context-files", "--no-approve",
                "--extension", str(extension), "--system-prompt", self.spec.system_prompt,
                "--session-dir", str(self.spec.session_dir)]
        args.extend(["--tools", ",".join(self.spec.tools)] if self.spec.tools else ["--no-tools"])
        if self.config.model:
            args.extend(["--model", self.config.model])
        if self.config.provider:
            args.extend(["--provider", self.config.provider])
        if self.session_id:
            args.extend(["--session", self.session_id])
        args.extend(self.config.extra_args)
        self.message_id = ""
        self.output = ""
        self.stop_reason = "stop"
        self.raw_usage: dict[str, Any] = {}
        self.request_number = 0
        await self.start_process(args, {
            "PI_OFFLINE": "1", "PI_TELEMETRY": "0", "OTA_TOOL_COMMAND": json.dumps(self.spec.tool_server_command),
            "OTA_TOOL_ENV": json.dumps(self.spec.tool_environment), "OTA_TOOLS": json.dumps(self.spec.tools),
        })
        response = await self.request({"type": "get_state"})
        await self.save_session(response.get("data", {}))

    async def save_session(self, state: dict[str, Any]) -> None:
        """Publish the persisted session locator exposed by pi.

        Args:
            state: Native get_state response payload.
        """
        identifier = state.get("sessionFile") or state.get("sessionId")
        if identifier and identifier != self.session_id:
            self.session_id = identifier
            await self.event("session", session_id=identifier)

    async def prompt(self, text: str) -> AgentResult:
        """Append input to the native session and wait through automatic recovery.

        Args:
            text: New input for this agent's current stage.

        Returns:
            The final assistant output and persisted session locator.
        """
        async with self.prompt_lock:
            self.output, self.stop_reason = "", "stop"
            self.finished = asyncio.get_running_loop().create_future()
            await self.event("prompt", text, message_id=uuid.uuid4().hex)
            response = await self.request({"type": "prompt", "message": text})
            if response.get("data", {}).get("disposition") == "handled":
                raise HarnessError("Pi handled the input as a command instead of starting an agent rollout")
            try:
                await self.finished
            except asyncio.CancelledError:
                await self.cancel()
                raise
            state = await self.request({"type": "get_state"})
            await self.save_session(state.get("data", {}))
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
        if kind == "ota_provider_request":
            self.raw_usage = {}
            self.request_number += 1
            await self.event("system", json.dumps(record["payload"], ensure_ascii=False), source="native_request",
                             message_id=f"request-{self.request_number}")
        elif kind == "ota_provider_usage":
            self.raw_usage.update(record["usage"])
        elif kind == "message_start" and record.get("message", {}).get("role") == "assistant":
            self.message_id = uuid.uuid4().hex
        elif kind == "message_update":
            event = record.get("assistantMessageEvent", {})
            event_type = event.get("type")
            if event_type in {"text_delta", "thinking_delta"}:
                await self.event("text" if event_type == "text_delta" else "reasoning", event.get("delta", ""),
                                 message_id=self.message_id)
        elif kind == "message_end" and record.get("message", {}).get("role") == "assistant":
            message = record["message"]
            text = "".join(part.get("text", "") for part in message.get("content", []) if part.get("type") == "text")
            self.output, self.stop_reason = text, message.get("stopReason", "stop")
            if text:
                await self.event("text", text, message_id=self.message_id, final=True)
            usage = message.get("usage", {})
            if usage:
                raw = self.raw_usage
                details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
                cache_read = next((details[key] for key in ("cached_tokens",) if key in details), None)
                for key in ("cache_read_input_tokens", "prompt_cache_hit_tokens", "cached_tokens"):
                    if key in raw:
                        cache_read = raw[key]
                cache_write = raw.get("cache_creation_input_tokens", details.get("cache_write_tokens"))
                await self.event("usage", request_id=self.message_id,
                                 input_tokens=usage.get("input", 0) + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0),
                                 output_tokens=usage.get("output", 0), cache_read_tokens=cache_read,
                                 cache_write_tokens=cache_write, reasoning_tokens=usage.get("reasoning"))
            if message.get("errorMessage"):
                await self.event("error", message["errorMessage"], message_id=self.message_id)
        elif kind in {"tool_execution_start", "tool_execution_end"}:
            await self.event("tool_start" if kind.endswith("start") else "tool_result",
                             json.dumps(record.get("args") if kind.endswith("start") else record.get("result"), ensure_ascii=False),
                             tool=record.get("toolName"), tool_call_id=record.get("toolCallId"), is_error=record.get("isError", False))
        elif kind == "agent_settled":
            if self.finished is not None and not self.finished.done():
                self.finished.set_result(None)
        elif kind in {"auto_retry_start", "auto_retry_end", "auto_compaction_start", "auto_compaction_end", "agent_start"}:
            await self.event("status", kind.replace("_", " "), native=record)
