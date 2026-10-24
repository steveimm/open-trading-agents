from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from open_trading_agents.config import HarnessSettings

_TIMEOUT_SECONDS = 12
_OUTPUT_LIMIT = 2 * 1024 * 1024
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class ModelChoice:
    id: str
    label: str
    provider: str | None = None
    efforts: tuple[str, ...] | None = None
    is_default: bool = False


@dataclass(frozen=True)
class ModelCatalog:
    models: tuple[ModelChoice, ...] = ()
    notice: str | None = None
    efforts: tuple[str, ...] = ()


async def _close(process: asyncio.subprocess.Process) -> None:
    """Terminate this discovery process and its descendants after every exit path.

    Args:
        process: Native process started in its own process group.
    """
    with contextlib.suppress(ProcessLookupError):
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        elif process.returncode is None:
            process.terminate()
    try:
        await asyncio.wait_for(process.wait(), 0.5)
    except TimeoutError:
        pass
    finally:
        with contextlib.suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        await process.wait()


@contextlib.asynccontextmanager
async def _native_process(args: list[str], environment: dict[str, str]) -> AsyncIterator[asyncio.subprocess.Process]:
    """Bound native metadata discovery and release its process group on every exit path.

    Args:
        args: Native metadata command and its arguments.
        environment: Environment inherited by the native harness.

    Yields:
        The native subprocess within a temporary working directory.
    """
    with tempfile.TemporaryDirectory(prefix="ota-model-catalog-") as directory:
        async with asyncio.timeout(_TIMEOUT_SECONDS):
            process = await asyncio.create_subprocess_exec(
                *args, cwd=directory, env=environment, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                limit=_OUTPUT_LIMIT, start_new_session=os.name == "posix",
            )
            try:
                yield process
            finally:
                await _close(process)


def _effort_values(values: Any) -> tuple[str, ...]:
    """Keep native effort identifiers in their advertised order.

    Args:
        values: Native metadata containing effort strings.

    Returns:
        Unique, nonempty native selectors.
    """
    if not isinstance(values, list):
        return ()
    return tuple(dict.fromkeys(value for value in values if isinstance(value, str) and value.strip()))


async def _read_output(process: asyncio.subprocess.Process) -> str:
    """Read a bounded command result without retaining native stderr or credentials.

    Args:
        process: Native help or catalog command.

    Returns:
        Its decoded standard output.
    """
    assert process.stdout is not None
    result = bytearray()
    while chunk := await process.stdout.read(8192):
        result.extend(chunk)
        if len(result) > _OUTPUT_LIMIT:
            raise ValueError("catalog output exceeded limit")
    if await process.wait() != 0:
        raise ValueError("catalog command failed")
    return _ANSI.sub("", result.decode(errors="replace"))


async def _rpc(
    process: asyncio.subprocess.Process, identifier: int, method: str, params: dict[str, Any], budget: int,
) -> tuple[dict[str, Any], int]:
    """Read one correlated native metadata response with a bounded output budget.

    Args:
        process: Codex app-server process with piped standard input and output.
        identifier: Unique request identifier.
        method: Metadata-only native method.
        params: Native request parameters.
        budget: Remaining output bytes across all discovery requests.

    Returns:
        The successful response payload and remaining output budget.
    """
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write((json.dumps({"id": identifier, "method": method, "params": params}) + "\n").encode())
    await process.stdin.drain()
    while line := await process.stdout.readline():
        budget -= len(line)
        if budget < 0:
            raise ValueError("catalog output exceeded limit")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("invalid catalog response")
        if record.get("id") != identifier:
            continue
        if record.get("error") or not isinstance(record.get("result"), dict):
            raise ValueError("catalog request failed")
        return record["result"], budget
    raise ValueError("catalog process exited")


async def _codex_models(process: asyncio.subprocess.Process, provider: str | None) -> ModelCatalog:
    """Query the installed Codex model catalog without creating a thread or turn.

    Args:
        process: Native app-server process.
        provider: Explicit provider selector, otherwise native provider selection remains unchanged.

    Returns:
        Native model selectors and display names from every catalog page.
    """
    _, budget = await _rpc(process, 1, "initialize", {"clientInfo": {"name": "open_trading_agents", "version": "0.1.0"}}, _OUTPUT_LIMIT)
    assert process.stdin is not None
    process.stdin.write(b'{"method":"initialized"}\n')
    await process.stdin.drain()
    models: list[ModelChoice] = []
    cursors: set[str] = set()
    cursor: str | None = None
    while True:
        response, budget = await _rpc(process, len(cursors) + 2, "model/list",
                                      {"limit": 100, "includeHidden": False, "cursor": cursor}, budget)
        rows = response.get("data")
        if not isinstance(rows, list):
            raise ValueError("invalid model catalog")
        for row in rows:
            if not isinstance(row, dict) or row.get("hidden"):
                continue
            identifier = row.get("model")
            if isinstance(identifier, str) and identifier.strip():
                label = row.get("displayName")
                rows = row.get("supportedReasoningEfforts")
                efforts = _effort_values([option.get("reasoningEffort") for option in rows if isinstance(option, dict)]) if isinstance(rows, list) else None
                models.append(ModelChoice(identifier, label if isinstance(label, str) and label else identifier, provider,
                                          efforts, row.get("isDefault") is True))
        cursor = response.get("nextCursor")
        if not cursor:
            break
        if not isinstance(cursor, str) or cursor in cursors:
            raise ValueError("invalid model catalog cursor")
        cursors.add(cursor)
    return ModelCatalog(tuple(dict.fromkeys(models)))


def _pi_models(output: str) -> ModelCatalog:
    """Parse the installed pi CLI's available-model table using its column boundaries.

    Args:
        output: Native --list-models output with terminal colors removed.

    Returns:
        Exact model and provider selectors reported by pi.
    """
    lines = output.splitlines()
    header = next((index for index, line in enumerate(lines) if line.split() == ["provider", "model", "context", "max-out", "thinking", "images"]), None)
    if header is None:
        return ModelCatalog(notice="Pi did not list available models. Use its native default or enter a configured model.")
    columns = [match.start() for match in re.finditer(r"\S+", lines[header])]
    choices = []
    for line in lines[header + 1:]:
        provider = line[columns[0]:columns[1]].strip()
        model = line[columns[1]:columns[2]].strip()
        if provider and model:
            choices.append(ModelChoice(model, model, provider))
    return ModelCatalog(tuple(dict.fromkeys(choices)))


def _claude_models(output: str) -> ModelCatalog:
    """Offer only model aliases documented by the installed Claude CLI.

    Args:
        output: Native --help output.

    Returns:
        Native aliases, without claiming account-specific model availability.
    """
    section = re.search(r"(?m)^\s*--model\s+<[^>]+>(.*?)(?=^\s*(?:-\w, )?--|\Z)", output, re.DOTALL)
    aliases = re.findall(r"['\"]([a-zA-Z0-9_-]+)['\"]", section[1]) if section else []
    effort_section = re.search(r"(?m)^\s*--effort\s+<[^>]+>(.*?)(?=^\s*(?:-\w, )?--|\Z)", output, re.DOTALL)
    levels = re.search(r"\(([^)]+)\)", effort_section[1]) if effort_section else None
    efforts = tuple(value.strip() for value in levels[1].split(",") if re.fullmatch(r"\s*[a-z][a-z0-9_-]*\s*", value)) if levels else ()
    return ModelCatalog(tuple(ModelChoice(alias, f"{alias} (native alias)") for alias in dict.fromkeys(aliases)),
                        "Claude lists model aliases. Their resolved models and availability are controlled by Claude Code.", efforts)


async def _discover(config: HarnessSettings) -> ModelCatalog:
    """Run a metadata-only native command with bounded time and isolated process cleanup.

    Args:
        config: Selected native executable and provider.

    Returns:
        Discovered models or a descriptive empty native catalog.
    """
    args = list(config.command or (config.kind,))
    environment = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}
    if config.kind == "pi":
        args.extend(["--offline", "--no-extensions", "--no-mcp", "--no-skills", "--no-context-files", "--no-approve", "--list-models"])
        environment["PI_OFFLINE"] = "1"
    elif config.kind == "codex":
        args.append("app-server")
        if config.provider:
            args.extend(["-c", f"model_provider={json.dumps(config.provider)}"])
    else:
        args.append("--help")
    async with _native_process(args, environment) as process:
        if config.kind == "codex":
            return await _codex_models(process, config.provider)
        output = await _read_output(process)
        return _pi_models(output) if config.kind == "pi" else _claude_models(output)


async def _pi_efforts(config: HarnessSettings) -> tuple[str, ...]:
    """Read the selected pi model's supported levels without prompting or persisting a session.

    Args:
        config: Native model/provider selection to resolve.

    Returns:
        Thinking levels advertised by the native session.
    """
    args = [*(config.command or ("pi",)), "--mode", "rpc", "--offline", "--no-session", "--no-tools",
            "--no-extensions", "--no-mcp", "--no-skills", "--no-context-files", "--no-approve"]
    if config.model:
        args.extend(["--model", config.model])
    if config.provider:
        args.extend(["--provider", config.provider])
    async with _native_process(args, {**os.environ, "PI_OFFLINE": "1", "NO_COLOR": "1", "FORCE_COLOR": "0"}) as process:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(b'{"id":"ota-effort","type":"get_available_thinking_levels"}\n')
        await process.stdin.drain()
        budget = _OUTPUT_LIMIT
        while line := await process.stdout.readline():
            budget -= len(line)
            if budget < 0:
                raise ValueError("effort metadata exceeded limit")
            response = json.loads(line)
            if not isinstance(response, dict) or response.get("id") != "ota-effort":
                continue
            if not response.get("success") or not isinstance(response.get("data"), dict):
                raise ValueError("effort metadata unavailable")
            return _effort_values(response["data"].get("levels"))
        raise ValueError("native effort discovery exited")


def effort_choices(config: HarnessSettings, catalog: ModelCatalog) -> tuple[str, ...]:
    """Resolve advertised effort choices while leaving model payload semantics to the harness.

    Args:
        config: Selected native model and provider.
        catalog: Previously discovered native model metadata.

    Returns:
        Supported native effort strings, or an empty tuple when unavailable.
    """
    if config.extra_args:
        return ()
    if config.kind == "pi":
        try:
            return asyncio.run(_pi_efforts(config))
        except (OSError, ValueError, RuntimeError, TimeoutError):
            return ()
    if config.kind == "claude":
        return catalog.efforts
    for model in catalog.models:
        if (model.id == config.model if config.model else model.is_default) and (config.provider is None or model.provider == config.provider):
            return model.efforts or ()
    return ()


def discover_models(config: HarnessSettings) -> ModelCatalog:
    """Ask the selected harness for model choices without sending a model prompt.

    Args:
        config: Selected harness, optionally using a custom native executable.

    Returns:
        Native choices, or a short notice allowing manual selection after discovery fails.
    """
    if config.extra_args:
        return ModelCatalog(notice="Model discovery does not apply extra CLI arguments. Keep the current selection, use native defaults, or enter a model.")
    try:
        catalog = asyncio.run(_discover(config))
    except FileNotFoundError:
        return ModelCatalog(notice=f"{config.kind} is not installed or its configured command was not found.")
    except TimeoutError:
        return ModelCatalog(notice=f"{config.kind} model discovery timed out. Use its native default or enter a model.")
    except (OSError, ValueError, RuntimeError):
        return ModelCatalog(notice=f"{config.kind} could not list models. Use its native default or enter a model.")
    if not catalog.models and not catalog.notice:
        return ModelCatalog(notice=f"{config.kind} did not list models. Use its native default or enter a model.")
    return catalog
