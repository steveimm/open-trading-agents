import json
import os
import sys

turn_number = 0

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    result = {}
    if method == "initialized":
        continue

    if method == "config/read":
        result = {"config": {"mcp_servers": {"ambient": {}}}}
    if method == "thread/start":
        assert "environments" not in request["params"]
        assert request["params"]["sandbox"] == "workspace-write"
        assert request["params"]["config"]["mcp_servers"]["ambient"]["enabled"] is False
        result = {"thread": {"id": "thread1"}, "model": "native-default", "modelProvider": "native-provider"}
    if method == "turn/start":
        effort = os.environ["EXPECTED_EFFORT"]
        assert ("effort" in request["params"]) == bool(effort)
        if effort:
            assert request["params"]["effort"] == effort

        turn_number += 1
        turn_id = f"turn{turn_number}"
        result = {"turn": {"id": turn_id}}

    print(json.dumps({"id": request["id"], "result": result}), flush=True)

    if method == "turn/start":
        total = {
            "inputTokens": 100 * turn_number,
            "outputTokens": 10 * turn_number,
            "cachedInputTokens": 30 * turn_number,
            "reasoningOutputTokens": 2 * turn_number,
            "totalTokens": 110 * turn_number,
        }
        events = [
            ("turn/started", {"turn": {"id": turn_id}}),
            (
                "model/rerouted",
                {
                    "threadId": "thread1",
                    "turnId": turn_id,
                    "fromModel": "native-default",
                    "toModel": "native-rerouted",
                    "reason": "highRiskCyberActivity",
                },
            ),
            ("item/agentMessage/delta", {"itemId": turn_id, "delta": "hello"}),
            ("item/reasoning/summaryTextDelta", {"itemId": "r", "delta": "consider"}),
            ("item/started", {"item": {"type": "mcpToolCall", "id": "tool", "tool": "get_news", "arguments": {"symbol": "NVDA"}}}),
            ("item/completed", {"item": {"type": "mcpToolCall", "id": "tool", "tool": "get_news", "result": {"content": []}}}),
            ("thread/tokenUsage/updated", {"turnId": turn_id, "tokenUsage": {"total": total}}),
            ("thread/tokenUsage/updated", {"turnId": turn_id, "tokenUsage": {"total": total}}),
            ("item/completed", {"item": {"type": "agentMessage", "id": turn_id, "text": str(turn_number)}}),
            ("turn/completed", {"turn": {"id": turn_id, "status": "completed"}}),
        ]

        for event_method, params in events:
            print(json.dumps({"method": event_method, "params": params}), flush=True)
