import csv
import io
import json
import uuid
from pathlib import Path


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
