import asyncio
import csv
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from open_trading_agents.dataflows import tools as module
from open_trading_agents.dataflows.artifacts import save_tool_data
from open_trading_agents.dataflows.tools import FinancialTools, ToolContext


def test_price_data_files_preserve_precision_empty_cells_and_source_notes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide directly parseable CSV without requiring the model to copy the returned rows.

    Args:
        tmp_path: Isolated agent evidence directory.
        monkeypatch: Offline vendor replacement.
    """
    source = '# Prices as of 2024-03-15\n# Currency: USD\n\nDate,Close,Volume,Note\n2024-03-14,0.004234567891,,"quoted, value"\n'
    monkeypatch.setattr(module, "route_to_vendor", Mock(return_value=source))
    directory = tmp_path / "agent workspace" / "data"
    api = FinancialTools(ToolContext("BTC-USD", "2024-03-15", asset_type="crypto"), artifact_dir=directory)

    response = api.get_stock_data("2024-03-01", "2024-03-15")

    raw, = directory.glob("*.txt")
    table, = directory.glob("*.csv")
    assert raw.read_text() == source
    with table.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"Date": "2024-03-14", "Close": "0.004234567891", "Volume": "", "Note": "quoted, value"}]
    assert str(raw) in response and str(table) in response
    assert response.endswith(source)


@pytest.mark.parametrize("source,extension", [
    ('{"periods":[{"period_end":"2024-03-31","values":{"Cash":null,"Debt":0}}]}', "json"),
    ("Date,Close\n2024-03-15,123.456789\n", "csv"),
    ("DATA_UNAVAILABLE: no data for this request", "txt"),
    ("Date,Close\n2024-03-15,12,unmatched-column\n", "txt"),
])
def test_evidence_format_detection_never_reconstructs_or_repairs_source_values(tmp_path: Path, source: str, extension: str) -> None:
    """Preserve complete JSON, CSV, unavailable, and malformed-table results exactly.

    Args:
        tmp_path: Isolated evidence location.
        source: Original provider output.
        extension: Format that can safely be claimed for this source.
    """
    response = save_tool_data(tmp_path, "get_stock_data", source)
    artifact, = tmp_path.iterdir()
    assert artifact.suffix == f".{extension}"
    assert artifact.read_text() == source
    assert str(artifact) in response and response.endswith(source)


async def test_parallel_data_calls_keep_distinct_complete_files(tmp_path: Path) -> None:
    """Avoid collisions between simultaneous calls to the same tool.

    Args:
        tmp_path: Shared evidence directory for the agent's parallel tool calls.
    """
    results = await asyncio.gather(*(asyncio.to_thread(save_tool_data, tmp_path, "get_indicators", f"observation {index}")
                                     for index in range(8)))
    contents = await asyncio.to_thread(lambda: [path.read_text() for path in tmp_path.iterdir()])
    assert len(contents) == 8
    assert set(contents) == {f"observation {index}" for index in range(8)}
    assert len(set(results)) == 8


async def test_actual_mcp_server_saves_evidence_in_the_supplied_workspace(tmp_path: Path) -> None:
    """Pass the artifact path through the real subprocess protocol without exposing it as a model argument.

    Args:
        tmp_path: Isolated tool context and agent workspace.
    """
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"ticker": "NVDA", "trade_date": "2024-03-15",
                                   "config": {"data_vendors": {"fundamental_data": "yfinance"}}}))
    directory = tmp_path / "agent" / "data"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "open_trading_agents.dataflows.tools", "--context", str(context),
              "--tools", "get_fundamentals", "--artifact-dir", str(directory)],
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams, read_timeout_seconds=10) as session:
            await session.initialize()
            tool, = (await session.list_tools()).tools
            assert not {"artifact_dir", "context", "ticker"}.intersection(tool.input_schema["properties"])
            result = await session.call_tool("get_fundamentals", {"curr_date": "2024-03-15"})
            assert not result.is_error
            output = "\n".join(block.text for block in result.content if hasattr(block, "text"))

    artifact, = directory.iterdir()
    assert str(artifact) in output and "withheld" in artifact.read_text().lower()


def test_evidence_index_labels_sources_and_excludes_scripts_logs_and_symlinks(tmp_path: Path) -> None:
    """Hand off exact data artifacts with enough metadata to avoid filesystem discovery.

    Args:
        tmp_path: Isolated analyst and downstream directories.
    """
    from open_trading_agents.dataflows.artifacts import write_evidence_index

    market = tmp_path / "Market_Analyst" / "data"
    fundamentals = tmp_path / "Fundamentals_Analyst" / "data"
    save_tool_data(market, "get_stock_data", "Date,High,Low,Close,Volume\n2024-03-15,12,10,11,100\n")
    save_tool_data(market, "get_indicators", "## vwma values from 2024-03-01 to 2024-03-15:\nCalculation: 14-observation typical-price VWMA.\n2024-03-15: 11.5\n")
    save_tool_data(fundamentals, "get_cashflow", json.dumps({"source": "SEC EDGAR", "frequency": "annual", "filing_cutoff": "2024-03-15",
                                                           "units": {"Operating Cash Flow": "USD"}, "periods": [{"values": {"Operating Cash Flow": 123}}]}))
    (market / "verify.py").write_text("raise RuntimeError('do not execute')")
    (market / "transcript.jsonl").write_text("private reasoning")
    outside = tmp_path / "unrelated.json"
    outside.write_text('{"secret": "not evidence"}')
    (market / ("get_stock_data-" + "a" * 32 + ".json")).symlink_to(outside)
    before = {path: path.read_bytes() for path in (*market.iterdir(), *fundamentals.iterdir())}

    index = write_evidence_index(tmp_path / "evidence.json", "NVDA", "2024-03-15", {"Market Analyst": market, "Fundamentals Analyst": fundamentals})
    document = json.loads(index.read_text())
    assert document["ticker"] == "NVDA" and document["analysis_date"] == "2024-03-15"
    assert len(document["files"]) == 3
    rows = {row["tool"]: row for row in document["files"]}
    assert rows["get_indicators"]["indicator"] == "vwma"
    assert rows["get_indicators"]["history_end"] == "2024-03-15"
    assert "14-observation" in rows["get_indicators"]["calculation"]
    assert rows["get_stock_data"]["columns"] == ["Date", "High", "Low", "Close", "Volume"]
    assert rows["get_cashflow"]["frequency"] == "annual"
    assert rows["get_cashflow"]["units"] == {"Operating Cash Flow": "USD"}
    assert "periods" not in rows["get_cashflow"]
    assert all(Path(row["path"]).is_absolute() for row in rows.values())
    assert {path: path.read_bytes() for path in before} == before
    original = index.read_bytes()
    write_evidence_index(index, "NVDA", "2024-03-15", {"Market Analyst": market, "Fundamentals Analyst": fundamentals})
    assert index.read_bytes() == original
    assert write_evidence_index(tmp_path / "absent.json", "NVDA", "2024-03-15", {"Missing": tmp_path / "missing"}) is None
