from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from open_trading_agents.domain import AgentResult
from open_trading_agents.harnesses.process import HarnessError, JsonProcess


class ClaudeSession(JsonProcess):
    async def start(self) -> None:
        """Start Claude's native bidirectional stream with only financial MCP tools."""
        self.session_id = self.session_id or str(uuid.uuid4())
        self.output = ""
        self.result_error = ""
        self.message_id = ""
        self.message_usage: dict[str, Any] = {}
        self.message_text: dict[str, str] = {}
        self.stream_blocks: dict[str, dict[int, str]] = {}
        self.final_blocks: set[str] = set()
        self.counted: set[str] = set()
        self.block_tools: dict[int, dict[str, Any]] = {}
        self.turn_number = 0
        self.pending_prompt_id: str | None = None
        self.turn_usage_count = 0
        self.tool_messages: set[str] = set()
        self.previous_cumulative: dict[str, int] = {}
        self.usage_path = self.spec.session_dir / "claude_usage.json"
        if self.spec.session_id and self.usage_path.exists():
            stored = json.loads(self.usage_path.read_text())
            if stored.get("session_id") == self.session_id:
                self.previous_cumulative = stored.get("total", {})
        servers = {}
        if self.spec.tools and self.spec.tool_server_command:
            servers["trading"] = {"command": self.spec.tool_server_command[0], "args": list(self.spec.tool_server_command[1:]),
                                  "env": self.spec.tool_environment}
        args = [*(self.config.command or ("claude",)), "--print", "--input-format", "stream-json", "--output-format", "stream-json",
                "--verbose", "--include-partial-messages", "--replay-user-messages", "--tools", "",
                "--strict-mcp-config", "--mcp-config", json.dumps({"mcpServers": servers}),
                "--system-prompt", self.spec.system_prompt, "--permission-mode", "dontAsk", "--disable-slash-commands",
                "--no-chrome", "--settings", json.dumps({"disableAllHooks": True, "enableAllProjectMcpServers": False}),
                "--setting-sources", "user", "--prompt-suggestions", "false"]
        if self.spec.tools:
            args.extend(["--allowedTools", ",".join(f"mcp__trading__{name}" for name in self.spec.tools)])
        args.extend(["--resume" if self.spec.session_id else "--session-id", self.session_id])
        if self.config.model:
            args.extend(["--model", self.config.model])
        args.extend(self.config.extra_args)
        await self.start_process(args, {
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_ENABLE_TELEMETRY": "0", "OTEL_SDK_DISABLED": "true",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "ENABLE_TOOL_SEARCH": "false",
        })
        await self.event("session", session_id=self.session_id)

    async def prompt(self, text: str) -> AgentResult:
        """Send a new user message to the same native Claude conversation.

        Args:
            text: New application input to append to native history.

        Returns:
            The completed response and persistent session identifier.
        """
        async with self.prompt_lock:
            self.output, self.result_error = "", ""
            self.turn_number += 1
            self.turn_usage_count = 0
            self.tool_messages.clear()
            self.finished = asyncio.get_running_loop().create_future()
            identifier = str(uuid.uuid4())
            self.pending_prompt_id = identifier
            await self.event("prompt", text, message_id=identifier)
            await self.send({"type": "user", "uuid": identifier, "session_id": self.session_id,
                             "message": {"role": "user", "content": text}, "parent_tool_use_id": None})
            try:
                await self.finished
            except asyncio.CancelledError:
                await self.cancel()
                raise
            if self.result_error:
                raise HarnessError(self.result_error)
            return AgentResult(self.output, self.session_id or "")

    async def cancel(self) -> None:
        """Request native interruption without deleting saved conversation state."""
        if self.process is not None and self.process.returncode is None:
            await self.send({"type": "control_request", "request_id": str(uuid.uuid4()), "request": {"subtype": "interrupt"}})

    async def usage(self, identifier: str, raw: dict[str, Any]) -> None:
        """Publish each native model response once, preserving unknown cache fields.

        Args:
            identifier: Stable native model-message identifier.
            raw: Authoritative native usage fields for that message.
        """
        if not raw or identifier in self.counted:
            return
        self.counted.add(identifier)
        self.turn_usage_count += 1
        read, write = raw.get("cache_read_input_tokens"), raw.get("cache_creation_input_tokens")
        await self.event("usage", request_id=identifier, input_tokens=raw.get("input_tokens", 0) + (read or 0) + (write or 0),
                         output_tokens=raw.get("output_tokens", 0), cache_read_tokens=read, cache_write_tokens=write,
                         reasoning_tokens=raw.get("reasoning_tokens"))

    async def stream(self, event: dict[str, Any]) -> None:
        """Translate raw message blocks while retaining provider usage presence.

        Args:
            event: Native streaming API event carried by Claude Code.
        """
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message", {})
            self.message_id = message.get("id") or uuid.uuid4().hex
            self.message_usage = dict(message.get("usage") or {})
            self.stream_blocks[self.message_id] = {}
            self.block_tools = {}
        elif kind == "content_block_start":
            block, index = event.get("content_block", {}), event.get("index", 0)
            self.stream_blocks.setdefault(self.message_id, {})[index] = block.get("type", "")
            if block.get("type") == "tool_use":
                self.tool_messages.add(self.message_id)
                if len(self.tool_messages) > self.spec.max_tool_rounds and not self.result_error:
                    self.result_error = f"Claude exceeded its {self.spec.max_tool_rounds} tool rounds"
                    await self.event("error", self.result_error)
                    await self.cancel()
                self.block_tools[index] = block
                await self.event("tool_start", json.dumps(block.get("input", {})), tool=block.get("name"), tool_call_id=block.get("id"))
        elif kind == "content_block_delta":
            delta, index = event.get("delta", {}), event.get("index", 0)
            identifier = f"{self.message_id}:{index}"
            if delta.get("type") in {"text_delta", "thinking_delta"}:
                channel = "text" if delta["type"] == "text_delta" else "reasoning"
                text = delta.get("text", delta.get("thinking", ""))
                self.message_text[identifier] = self.message_text.get(identifier, "") + text
                await self.event(channel, text, message_id=identifier)
            elif delta.get("type") == "input_json_delta":
                tool = self.block_tools.get(index, {})
                await self.event("tool_start", delta.get("partial_json", ""), tool=tool.get("name"),
                                 tool_call_id=tool.get("id"), delta=True)
        elif kind == "message_delta":
            self.message_usage.update(event.get("usage") or {})
            if event.get("delta", {}).get("stop_reason") == "max_tokens":
                self.result_error = "Claude reached its output token limit before completing"
        elif kind == "message_stop":
            await self.usage(self.message_id, self.message_usage)

    def final_block_id(self, identifier: str, kind: str, text: str, fallback: int) -> str:
        """Match per-block final records to their original streaming positions.

        Args:
            identifier: Native assistant-message identifier.
            kind: Native content block type.
            text: Final block text supplied by the harness.
            fallback: Position in a non-streamed message.

        Returns:
            The stable identifier used by both deltas and their final snapshot.
        """
        positions = self.stream_blocks.get(identifier, {})
        candidates = [index for index, block_kind in positions.items()
                      if block_kind == kind and f"{identifier}:{index}" not in self.final_blocks]
        exact = [index for index in candidates if self.message_text.get(f"{identifier}:{index}") == text]
        index = exact[0] if exact else candidates[0] if candidates else fallback
        result = f"{identifier}:{index}"
        self.final_blocks.add(result)
        return result

    async def handle(self, record: dict[str, Any]) -> None:
        """Normalize Claude native records and finish only on its result message.

        Args:
            record: Native stream-json event, message, or control request.
        """
        kind = record.get("type")
        if kind == "stream_event":
            await self.stream(record.get("event", {}))
        elif kind == "system":
            identifier = record.get("session_id")
            if identifier and identifier != self.session_id:
                self.session_id = identifier
                await self.event("session", session_id=identifier)
            await self.event("status", record.get("subtype", "system"), native=record)
        elif kind == "assistant":
            message = record.get("message", {})
            identifier = message.get("id") or record.get("uuid") or uuid.uuid4().hex
            for index, block in enumerate(message.get("content", [])):
                if block.get("type") == "text":
                    self.output = block.get("text", "")
                    await self.event("text", self.output, message_id=self.final_block_id(identifier, "text", self.output, index), final=True)
                elif block.get("type") == "thinking":
                    thinking = block.get("thinking", "")
                    await self.event("reasoning", thinking,
                                     message_id=self.final_block_id(identifier, "thinking", thinking, index), final=True)
            if identifier != self.message_id:
                await self.usage(identifier, message.get("usage") or {})
            if record.get("error"):
                await self.event("error", str(record["error"]))
        elif kind == "user":
            if self.pending_prompt_id and record.get("uuid") == self.pending_prompt_id:
                await self.event("status", "Prompt accepted by Claude", prompt_accepted=True, prompt_id=self.pending_prompt_id)
                self.pending_prompt_id = None
            for block in record.get("message", {}).get("content", []) if isinstance(record.get("message", {}).get("content"), list) else []:
                if block.get("type") == "tool_result":
                    content = block.get("content", "")
                    await self.event("tool_result", content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
                                     tool_call_id=block.get("tool_use_id"), is_error=block.get("is_error", False))
        elif kind == "result":
            total = record.get("usage") or {}
            if not self.turn_usage_count and total:
                delta = {key: max(0, value - self.previous_cumulative.get(key, 0))
                         for key, value in total.items() if isinstance(value, int)}
                await self.usage(f"{self.session_id}:turn:{self.turn_number}", delta)
            self.previous_cumulative = {key: value for key, value in total.items() if isinstance(value, int)}
            self.usage_path.write_text(json.dumps({"session_id": self.session_id, "total": self.previous_cumulative}))
            self.output = record.get("result") or self.output
            if record.get("is_error") or record.get("subtype") != "success":
                self.result_error = "\n".join(record.get("errors") or []) or record.get("result") or f"Claude ended with {record.get('subtype')}"
            if self.finished is not None and not self.finished.done():
                self.finished.set_result(None)
        elif kind == "control_request":
            request = record.get("request", {})
            if request.get("subtype") == "can_use_tool":
                await self.send({"type": "control_response", "response": {
                    "subtype": "success", "request_id": record.get("request_id"),
                    "response": {"behavior": "deny", "message": "Only explicitly allowed financial tools can run"},
                }})
            else:
                await self.send({"type": "control_response", "response": {
                    "subtype": "error", "request_id": record.get("request_id"), "error": "Unsupported interactive request",
                }})
        elif kind in {"tool_progress", "tool_use_summary", "rate_limit_event"}:
            await self.event("status", kind.replace("_", " "), native=record)
