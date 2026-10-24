import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    result = {}
    if method == "initialize":
        result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
    if method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "price",
                    "description": "Synthetic price",
                    "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}},
                }
            ]
        }
    if method == "tools/call":
        result = {"content": [{"type": "text", "text": "123.45"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
