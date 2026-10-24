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
    server = Path(__file__).with_name("fixtures") / "mcp_server.py"
    extension = Path(pi.__file__).with_name("pi_extension.mjs")
    runner = Path(__file__).with_name("fixtures") / "pi_extension_runner.mjs"
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
    assert sum(record["type"] == "ota_budget" for record in records) == 1
    assert sum(record["type"] == "ota_budget_exceeded" for record in records) == 1
    requests = [record for record in records if record["type"] == "ota_provider_request"]
    assert requests[0]["payload"]["messages"][0]["reasoning"] == "retained"
    usages = [record["usage"] for record in records if record["type"] == "ota_provider_usage"]
    assert usages == [
        {"promptTokenCount": 100, "cachedContentTokenCount": 0, "thoughtsTokenCount": 0},
        {"promptTokenCount": 100, "cachedContentTokenCount": 30},
        {"inputTokens": 70, "cacheReadInputTokens": 20, "cacheWriteInputTokens": 10},
    ]
