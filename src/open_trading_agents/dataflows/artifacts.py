import csv
import io
import json
import re
import uuid
from pathlib import Path
from typing import Any

from open_trading_agents.dataflows.files import replace_file

_ARTIFACT_NAME = re.compile(r"(?P<tool>[a-z_]+)-[0-9a-f]{32}\.(?P<format>json|csv|txt)$")



def _csv_body(text: str) -> str | None:
    """Find a complete CSV table after optional source-comment headers.

    Args:
        text: Original tool result.

    Returns:
        The unmodified CSV body when its rows align, otherwise None.
    """
    lines = text.splitlines(keepends=True)
    first = next((index for index, line in enumerate(lines) if line.strip() and not line.startswith("#")), len(lines))
    body = "".join(lines[first:])
    try:
        rows = [row for row in csv.reader(io.StringIO(body), strict=True) if row]
    except csv.Error:
        return None
    if len(rows) < 2 or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        return None

    return body


def save_tool_data(directory: Path, tool: str, text: str) -> str:
    """Save exact evidence for native scripts and attach its paths to the tool response.

    Args:
        directory: Application-selected evidence directory inside the agent workspace.
        tool: Internal financial tool name used as the filename prefix.
        text: Complete result returned by the data source.

    Returns:
        File references followed by the full original result.
    """
    if not tool.isidentifier():
        raise ValueError("Invalid tool name for evidence file")

    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    identifier = f"{tool}-{uuid.uuid4().hex}"
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    extension = "json" if isinstance(parsed, (dict, list)) else "txt"
    body = None
    if extension != "json" and tool in {"get_stock_data", "get_balance_sheet", "get_cashflow", "get_income_statement", "get_insider_transactions"}:
        body = _csv_body(text)
        if body == text:
            extension = "csv"
    original = directory / f"{identifier}.{extension}"
    with original.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)

    references = [f"# Data file ({extension}, exact tool result): {original}"]
    if body is not None and extension != "csv":
        table = directory / f"{identifier}.csv"
        with table.open("x", encoding="utf-8", newline="") as handle:
            handle.write(body)
        references.append(f"# CSV file (header and rows, source notes remain in the exact result): {table}")

    return "\n".join(references) + "\n\n" + text


def _artifact_metadata(path: Path, tool: str) -> dict[str, Any]:
    """Expose small source labels so readers can select an exact file without directory searches.

    Args:
        path: Tool-produced evidence file.
        tool: Original tool name.

    Returns:
        Indicator identity, column names, or source-provided period/unit metadata.
    """
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return {"columns": next(csv.reader(handle), [])}
    text = path.read_text(encoding="utf-8")
    if tool == "get_indicators":
        metadata = {}
        match = re.search(r"^## ([a-z0-9_]+) values from ([0-9-]+) to ([0-9-]+):", text, re.IGNORECASE | re.MULTILINE)
        if match:
            metadata.update(indicator=match[1].lower(), history_start=match[2], history_end=match[3])
        definition = re.search(r"^Calculation(?: parameters)?: (.+)$", text, re.MULTILINE)
        if definition:
            metadata["calculation"] = definition[1]
        return metadata
    if path.suffix == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            keys = ("source", "source_url", "ticker", "analysis_date", "filing_cutoff", "frequency", "units", "value_basis", "period_note")
            return {key: value[key] for key in keys if key in value}
    return {}


def write_evidence_index(path: Path, ticker: str, analysis_date: str, directories: dict[str, Path]) -> Path | None:
    """List only tool-produced analyst evidence for targeted downstream checks.

    Args:
        path: Run-local index destination.
        ticker: Instrument shared by these artifacts.
        analysis_date: Information cutoff enforced by the producing tools.
        directories: Analyst names mapped to their data directories.

    Returns:
        The index path, or None when no source artifacts were produced.
    """
    files = []
    for agent, directory in directories.items():
        if not directory.is_dir() or directory.is_symlink():
            continue
        for artifact in sorted(directory.iterdir()):
            match = _ARTIFACT_NAME.fullmatch(artifact.name)
            if match is None or artifact.is_symlink() or not artifact.is_file():
                continue
            try:
                metadata = _artifact_metadata(artifact, match["tool"])
            except (OSError, UnicodeError, ValueError, csv.Error):
                metadata = {}
            files.append({"agent": agent, "tool": match["tool"], "format": match["format"],
                          "path": str(artifact.resolve()), **metadata})
    if not files:
        return None

    document = {"ticker": ticker, "analysis_date": analysis_date,
                "usage": "Optional source evidence for a specific material discrepancy. Read only relevant files; do not run an exhaustive audit. Filter files by agent, tool, indicator, or frequency before reading exact paths. Source dates, units, and calculation definitions are recorded here when available and in the listed files. Treat them as read-only.",
                "files": files}
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, lambda temporary: Path(temporary).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"))
    return path.resolve() if path.is_file() else None
