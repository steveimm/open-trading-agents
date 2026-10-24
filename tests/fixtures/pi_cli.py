import json
import os
import sys
import time

native_allowed = "--tools" in sys.argv and "bash" in sys.argv[sys.argv.index("--tools") + 1].split(",")
assert native_allowed == (os.environ["ALLOW_EXECUTION"] == "True")
effort = os.environ["EXPECTED_EFFORT"]
assert ("--thinking" in sys.argv) == bool(effort)
if effort:
    assert sys.argv[sys.argv.index("--thinking") + 1] == effort

metadata = os.environ["OTA_METADATA_FILE"]
with open(metadata, "a") as handle:
    handle.write(json.dumps({"type": "ota_ready", "tools": []}) + "\n")

count = 0

for line in sys.stdin:
    command = json.loads(line)
    kind = command["type"]

    if kind == "get_state":
        model = {"id": "native-default", "provider": "native-provider", "baseUrl": "http://private-endpoint", "apiKey": "not-for-report"}
        print(
            json.dumps(
                {
                    "id": command["id"],
                    "type": "response",
                    "success": True,
                    "data": {"sessionFile": "persisted-session.jsonl", "model": model},
                }
            ),
            flush=True,
        )
    elif kind == "prompt":
        count += 1
        print(json.dumps({"id": command["id"], "type": "response", "success": True}), flush=True)
        while not os.path.exists(os.environ["ACCEPTANCE_GATE"]):
            time.sleep(0.01)

        events = [
            {"type": "ota_provider_request", "payload": {"messages": [{"content": command["message"]}]}},
            {"type": "ota_provider_usage", "usage": {"prompt_tokens": 100, "completion_tokens": 5}},
            {"type": "message_start", "message": {"role": "assistant"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "delta": "consider"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "answer"}},
            {"type": "tool_execution_start", "toolName": "get_news", "toolCallId": "t1", "args": {"symbol": "NVDA"}},
            {"type": "tool_execution_end", "toolName": "get_news", "toolCallId": "t1", "result": {"content": []}},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "content": [{"type": "text", "text": str(count)}],
                    "usage": {"input": 100, "output": 5, "cacheRead": 0, "cacheWrite": 0, "reasoning": 2},
                },
            },
            {"type": "agent_end"},
            {"type": "auto_retry_start"},
            {"type": "auto_retry_end"},
            {"type": "agent_settled"},
        ]

        for event in events:
            if event["type"].startswith("ota_"):
                with open(metadata, "a") as handle:
                    handle.write(json.dumps(event) + "\n")
            else:
                print(json.dumps(event), flush=True)
