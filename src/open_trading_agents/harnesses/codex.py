from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from open_trading_agents.domain import AgentResult
from open_trading_agents.harnesses.process import HarnessError, JsonProcess


class CodexSession(JsonProcess):
    async def start(self) -> None:
        """Initialize native app-server and create or resume one durable thread."""
        self.turn_id: str | None = None
        self.output = ""
        self.turn_status = ""
        self.generation = 0
        self.tool_generations: set[int] = set()
        self.budget_error = ""
        self.usage_path = self.spec.session_dir / "codex_usage.json"
        self.last_usage: dict[str, int] = {}
        if self.session_id and self.usage_path.exists():
            stored = json.loads(self.usage_path.read_text())
            if stored.get("session_id") == self.session_id:
                self.last_usage = stored.get("total", {})
        flags = (
            "shell_tool", "unified_exec", "apps", "plugins", "tool_suggest", "multi_agent", "multi_agent_v2",
            "memories", "memory_tool", "hooks", "codex_hooks", "plugin_hooks", "code_mode", "js_repl",
            "browser_use", "computer_use", "image_generation", "view_image", "goals", "sleep_tool",
            "remote_models", "api_key_model_discovery", "runtime_metrics", "search_tool", "tool_search",
        )
        overrides = {
            "analytics.enabled": False, "otel.exporter": "none", "otel.trace_exporter": "none",
            "otel.metrics_exporter": "none", "web_search": "disabled", "tools.update_plan.enabled": False,
            "skills.include_instructions": False, "features.skip_host_skill_discovery": True,
            "project_doc_max_bytes": 0, **{f"features.{name}": False for name in flags},
        }
        args = [*(self.config.command or ("codex",)), "app-server", *self.config.extra_args]
        for key, value in overrides.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        await self.start_process(args, {"OTEL_SDK_DISABLED": "true"})
        await self.request({"method": "initialize", "params": {
            "clientInfo": {"name": "open_trading_agents", "version": "0.1.0"},
            "capabilities": {"experimentalApi": True},
        }})
        await self.send({"method": "initialized"})
        native = await self.request({"method": "config/read", "params": {"includeLayers": False, "cwd": str(self.spec.workspace)}})
        servers = {name: {"enabled": False} for name in native.get("result", {}).get("config", {}).get("mcp_servers", {})}
        if self.spec.tools and self.spec.tool_server_command:
            servers["trading"] = {
                "command": self.spec.tool_server_command[0], "args": list(self.spec.tool_server_command[1:]),
                "env": self.spec.tool_environment, "enabled_tools": list(self.spec.tools), "enabled": True,
                "required": True, "default_tools_approval_mode": "approve", "tool_timeout_sec": 120,
            }
        params: dict[str, Any] = {
            "cwd": str(self.spec.workspace), "baseInstructions": self.spec.system_prompt,
            "developerInstructions": "Use only the supplied financial tools and evidence. Do not read or modify files or execute commands.",
            "approvalPolicy": "never", "sandbox": "read-only", "config": {**overrides, "mcp_servers": servers},
        }
        if self.config.model:
            params["model"] = self.config.model
        if self.config.provider:
            params["modelProvider"] = self.config.provider
        if self.session_id:
            method = "thread/resume"
            params.update(threadId=self.session_id, excludeTurns=True)
        else:
            method = "thread/start"
            params.update(ephemeral=False, environments=[], selectedCapabilityRoots=[])
        response = await self.request({"method": method, "params": params})
        self.session_id = response["result"]["thread"]["id"]
        await self.event("session", session_id=self.session_id)

    async def prompt(self, text: str) -> AgentResult:
        """Append a turn to the native thread and await its terminal status.

        Args:
            text: New role input without replaying prior native history.

        Returns:
            The completed assistant text and native thread identifier.
        """
        async with self.prompt_lock:
            self.output, self.turn_status, self.budget_error = "", "", ""
            self.tool_generations.clear()
            self.finished = asyncio.get_running_loop().create_future()
            await self.event("prompt", text, message_id=uuid.uuid4().hex)
            response = await self.request({"method": "turn/start", "params": {
                "threadId": self.session_id, "input": [{"type": "text", "text": text}], "environments": [],
            }})
            self.turn_id = response["result"]["turn"]["id"]
            await self.event("status", "Prompt accepted by Codex", prompt_accepted=True, turn_id=self.turn_id)
            try:
                await self.finished
            except asyncio.CancelledError:
                await self.cancel()
                raise
            if self.budget_error:
                raise HarnessError(self.budget_error)
            if self.turn_status != "completed":
                raise HarnessError(f"Codex turn ended with status {self.turn_status}")
            return AgentResult(self.output, self.session_id or "")

    async def cancel(self) -> None:
        """Interrupt the active native turn without deleting its thread."""
        if self.turn_id and self.process is not None and self.process.returncode is None:
            await self.request({"method": "turn/interrupt", "params": {"threadId": self.session_id, "turnId": self.turn_id}})

    async def record_usage(self, params: dict[str, Any]) -> None:
        """Convert cumulative native counters into exactly-once incremental usage.

        Args:
            params: Native thread/tokenUsage/updated notification parameters.
        """
        total = params.get("tokenUsage", {}).get("total", {})
        if not total or total == self.last_usage:
            return
        previous = self.last_usage
        if total.get("totalTokens", 0) < previous.get("totalTokens", 0):
            previous = {}
        delta = {key: max(0, value - previous.get(key, 0)) for key, value in total.items() if isinstance(value, int)}
        self.last_usage = total
        self.generation += 1
        self.usage_path.write_text(json.dumps({"session_id": self.session_id, "total": total}))
        await self.event("usage", request_id=f"{self.session_id}:{params.get('turnId')}:{total.get('totalTokens')}",
                         input_tokens=delta.get("inputTokens", 0), output_tokens=delta.get("outputTokens", 0),
                         cache_read_tokens=delta.get("cachedInputTokens"), cache_write_tokens=delta.get("cacheWriteInputTokens"),
                         reasoning_tokens=delta.get("reasoningOutputTokens"))

    async def handle(self, record: dict[str, Any]) -> None:
        """Translate app-server items, reasoning, tool activity, and completion.

        Args:
            record: Native JSON-RPC notification or server request.
        """
        method, params = record.get("method", ""), record.get("params", {})
        if "id" in record:
            await self.send({"id": record["id"], "error": {"code": -32601, "message": "Interactive requests are unavailable"}})
            await self.event("error", f"Native interactive request was rejected: {method}")
        elif method == "item/agentMessage/delta":
            await self.event("text", params.get("delta", ""), message_id=params.get("itemId"))
        elif method in {"item/reasoning/textDelta", "item/reasoning/summaryTextDelta"}:
            suffix = "summary" if "summary" in method else "reasoning"
            await self.event("reasoning", params.get("delta", ""), message_id=f"{params.get('itemId')}:{suffix}")
        elif method in {"item/started", "item/completed"}:
            item, completed = params.get("item", {}), method.endswith("completed")
            kind = item.get("type")
            if kind == "agentMessage" and completed:
                self.output = item.get("text", "")
                await self.event("text", self.output, message_id=item.get("id"), final=True)
            elif kind == "mcpToolCall":
                if not completed:
                    self.tool_generations.add(self.generation)
                    if len(self.tool_generations) > self.spec.max_tool_rounds and not self.budget_error:
                        self.budget_error = f"Codex exceeded its {self.spec.max_tool_rounds} tool rounds"
                        await self.event("error", self.budget_error)
                        await self.send({"id": uuid.uuid4().hex, "method": "turn/interrupt", "params": {
                            "threadId": self.session_id, "turnId": self.turn_id,
                        }})
                await self.event("tool_result" if completed else "tool_start",
                                 json.dumps(item.get("result") or item.get("error") if completed else item.get("arguments"),
                                            ensure_ascii=False),
                                 tool=item.get("tool"), tool_call_id=item.get("id"), is_error=bool(item.get("error")))
            elif kind in {"contextCompaction", "context_compaction"}:
                await self.event("status", "Context compaction completed" if completed else "Compacting context", native=item)
        elif method == "thread/tokenUsage/updated":
            await self.record_usage(params)
        elif method == "turn/started":
            self.turn_id = params.get("turn", {}).get("id")
            await self.event("status", "Generating", turn_id=self.turn_id)
        elif method == "turn/completed":
            turn = params.get("turn", {})
            self.turn_status = turn.get("status", "failed")
            if turn.get("error"):
                await self.event("error", str(turn["error"]))
            if self.finished is not None and not self.finished.done():
                self.finished.set_result(None)
        elif method == "error":
            await self.event("status" if params.get("willRetry") else "error", str(params.get("error", params)),
                             retry=bool(params.get("willRetry")))
        elif method in {"thread/compacted", "item/mcpToolCall/progress"}:
            await self.event("status", method, native=params)
