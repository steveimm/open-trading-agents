import json
import os
import sys

effort = os.environ["EXPECTED_EFFORT"]
assert ("--effort" in sys.argv) == bool(effort)
if effort:
    assert sys.argv[sys.argv.index("--effort") + 1] == effort

turn_number = 0

for line in sys.stdin:
    request = json.loads(line)
    if request["type"] != "user":
        continue

    turn_number += 1
    message_id = f"message{turn_number}"

    events = [
        {"type": "user", "uuid": request["uuid"], "message": request["message"]},
        {"type": "system", "subtype": "init", "session_id": request["session_id"], "model": "native-default"},
        {
            "type": "stream_event",
            "event": {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "model": "native-default",
                    "usage": {"input_tokens": 70, "output_tokens": 0, "cache_read_input_tokens": 30},
                },
            },
        },
        {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}},
        {"type": "stream_event", "event": {"type": "message_delta", "usage": {"output_tokens": int(os.environ["STREAM_OUTPUT"])}}},
        {"type": "stream_event", "event": {"type": "message_stop"}},
        {
            "type": "assistant",
            "message": {"id": message_id, "content": [{"type": "text", "text": str(turn_number)}], "usage": {"input_tokens": 0, "output_tokens": 0}},
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": str(turn_number),
            "usage": {"input_tokens": 70, "output_tokens": 10, "cache_read_input_tokens": 30},
            "modelUsage": {"native-default": {"inputTokens": 70 * turn_number}, "auxiliary-model": {"inputTokens": 10 * turn_number}},
        },
    ]

    for event in events:
        print(json.dumps(event), flush=True)
