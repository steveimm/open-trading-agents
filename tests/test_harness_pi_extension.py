import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from open_trading_agents.harnesses import pi


@pytest.mark.asyncio
async def test_pi_extension_mcp_exchange_and_budget_wrap_up(tmp_path: Path) -> None:
    """Run the real extension against an MCP subprocess without a provider request."""
    if shutil.which("node") is None:
        pytest.skip("Node is required by the native pi harness")
    server = tmp_path / "server.py"
    server.write_text('''import json,sys
for line in sys.stdin:
 request=json.loads(line)
 if 'id' not in request:continue
 method=request.get('method');result={}
 if method=='initialize':result={'protocolVersion':'2025-03-26','capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}
 if method=='tools/list':result={'tools':[{'name':'price','description':'Synthetic price','inputSchema':{'type':'object','properties':{'symbol':{'type':'string'}}}}]}
 if method=='tools/call':result={'content':[{'type':'text','text':'123.45'}]}
 print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}),flush=True)
''')
    extension = Path(pi.__file__).with_name("pi_extension.mjs")
    runner = tmp_path / "test.mjs"
    runner.write_text('''import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
const { default: extension } = await import(pathToFileURL(process.env.EXTENSION));
const handlers = new Map(); const tools = new Map(); const messages = []; let active = [];
const pi = { on: (name, callback) => handlers.set(name, callback), registerTool: (tool) => tools.set(tool.name, tool),
 setActiveTools: (names) => { active = names; }, sendMessage: (message) => messages.push(message) };
await extension(pi);
assert.deepEqual([...tools.keys()], ["price"]);
handlers.get("input")();
assert.deepEqual(active, ["price"]);
const result = await tools.get("price").execute("call1", { symbol: "PROBE" });
assert.equal(result.content[0].text, "123.45");
handlers.get("turn_end")({ toolResults: [result] });
assert.deepEqual(active, []);
assert.match(messages[0].content, /budget is exhausted/);
const payload = { messages: [{ role: "assistant", reasoning: "retained", tool_calls: [{ id: "original" }] }] };
assert.equal(handlers.get("before_provider_request")({ payload }), payload);
assert.equal(handlers.get("cache_warming_decision")().action, "stop");
handlers.get("session_shutdown")();
''')
    metadata = tmp_path / "metadata.jsonl"
    process = await asyncio.create_subprocess_exec(
        "node", str(runner), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "EXTENSION": str(extension), "OTA_TOOL_COMMAND": json.dumps([sys.executable, str(server)]),
             "OTA_TOOLS": '["price"]', "OTA_METADATA_FILE": str(metadata), "OTA_MAX_TOOL_ROUNDS": "1"},
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
    assert process.returncode == 0, stderr.decode()
    assert stdout == b""
    records = [json.loads(line) for line in metadata.read_text().splitlines()]
    assert records[0] == {"type": "ota_ready", "tools": ["price"]}
    assert any(record["type"] == "ota_budget" for record in records)
    assert records[-1]["payload"]["messages"][0]["reasoning"] == "retained"
