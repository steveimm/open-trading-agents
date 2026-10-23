from __future__ import annotations

import asyncio
import json
import shutil
import sys
from dataclasses import asdict
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from open_trading_agents.config import HarnessSettings, Settings, load_portfolio, load_settings
from open_trading_agents.domain import AgentEvent, EventSink
from open_trading_agents.workflow import RunResult, Workflow

app = typer.Typer(no_args_is_help=True, help="Financial research through native agent harness sessions.")
console = Console(stderr=True)


class HarnessName(StrEnum):
    pi = "pi"
    codex = "codex"
    claude = "claude"


def configured(path: Path | None, harness: HarnessName | None, model: str | None, provider: str | None) -> Settings:
    """Load application choices while leaving provider credentials with the harness.

    Args:
        path: Explicit TOML file or the conventional local configuration.
        harness: Optional CLI harness override.
        model: Optional native model selector.
        provider: Optional native provider selector.

    Returns:
        Validated application settings.
    """
    load_dotenv(Path.cwd() / ".env", override=False)
    path = path or (Path("tradingagents.toml") if Path("tradingagents.toml").is_file() else None)
    settings = load_settings(path)
    overrides = {key: value for key, value in {"kind": harness, "model": model, "provider": provider}.items() if value is not None}
    if overrides:
        settings.harness = HarnessSettings.model_validate({**settings.harness.model_dump(), **overrides})
        for tier in ("quick", "deep"):
            configured_tier = getattr(settings, tier)
            if configured_tier:
                updated = {**configured_tier.model_dump(), **overrides}
                if harness is not None and configured_tier.kind != harness and provider is None:
                    updated["provider"] = None
                setattr(settings, tier, HarnessSettings.model_validate(updated))
    return settings


def run_workflow(workflow: Workflow, headless: bool, events: bool) -> RunResult:
    """Run a common pipeline through either the live terminal UI or a JSON observer.

    Args:
        workflow: A new or resumed run.
        headless: Whether to avoid terminal UI interaction.
        events: Whether to stream complete event JSON to stdout.

    Returns:
        The completed run result.
    """
    if headless or events or not (sys.stdin.isatty() and sys.stdout.isatty()):
        async def observe(event: AgentEvent) -> None:
            """Write explicit JSON events or concise lifecycle progress.

            Args:
                event: The next normalized workflow event.
            """
            if events:
                typer.echo(json.dumps(asdict(event), ensure_ascii=False))
            elif event.kind in ("stage_start", "error"):
                console.print(f"{event.stage}: {event.text or 'started'}", markup=False)

        result = asyncio.run(workflow.run(observe))
    else:
        from open_trading_agents.tui import TradingApp

        selected = tuple(key for key in workflow.settings.analysts if not (workflow.asset_type == "crypto" and key == "fundamentals"))
        ui = TradingApp(workflow.run, selected_analysts=selected, ticker=workflow.ticker, run_dir=workflow.directory)
        ui.run()
        if ui.error:
            raise RuntimeError(str(ui.error)) from ui.error
        if not isinstance(ui.result, RunResult):
            console.print(f"Run interrupted. Resume with: open-trading-agents resume {workflow.directory}", markup=False)
            raise typer.Exit(130)
        result = ui.result
    if not events:
        typer.echo(f"{workflow.ticker}: {result.rating}\nReports: {result.directory / 'reports'}\nRun: {result.run_id}")
        usage = json.loads((result.directory / "reports" / "usage.json").read_text())["totals"]
        typer.echo(f"Reported tokens: {usage['input_tokens']:,} input / {usage['output_tokens']:,} output. Cache details: reports/usage.json")
    return result


@app.command()
def analyze(
    ticker: Annotated[str | None, typer.Argument(help="Ticker or supported alias, for example NVDA or BTC-USD.")] = None,
    analysis_date: Annotated[str | None, typer.Option("--date", help="Information cutoff, YYYY-MM-DD.")] = None,
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    harness: Annotated[HarnessName | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option(help="Native harness model selector.")] = None,
    provider: Annotated[str | None, typer.Option(help="Native pi provider selector.")] = None,
    analysts: Annotated[str | None, typer.Option(help="Comma-separated market,social,news,fundamentals.")] = None,
    research_rounds: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    risk_rounds: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    portfolio: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    state_dir: Annotated[Path | None, typer.Option()] = None,
    headless: Annotated[bool, typer.Option(help="Run without the terminal UI.")] = False,
    events: Annotated[bool, typer.Option(help="Emit normalized JSONL events to stdout.")] = False,
) -> None:
    """Analyze one instrument through the complete research and decision workflow.

    Args:
        ticker: Instrument symbol.
        analysis_date: Analysis cutoff date.
        config: Optional TOML configuration.
        harness: Native harness override.
        model: Native model selector override.
        provider: Pi provider selector override.
        analysts: Selected research roles.
        research_rounds: Number of bull/bear rotations.
        risk_rounds: Number of three-speaker risk rotations.
        portfolio: Optional holdings JSON file.
        state_dir: Application storage override.
        headless: Disable terminal rendering.
        events: Stream JSON events instead of concise progress.
    """
    if ticker is None:
        if not sys.stdin.isatty() or headless or events:
            raise typer.BadParameter("provide a ticker, for example: open-trading-agents analyze NVDA --headless")
        ticker = typer.prompt("Ticker")
        analysis_date = analysis_date or typer.prompt("Analysis date", default=date.today().isoformat())
    try:
        settings = configured(config, harness, model, provider)
        changes = {"analysts": tuple(key.strip() for key in analysts.split(",")) if analysts else None,
                   "research_rounds": research_rounds, "risk_rounds": risk_rounds, "state_dir": state_dir}
        settings = Settings.model_validate({**settings.model_dump(), **{k: v for k, v in changes.items() if v is not None}})
        workflow = Workflow(ticker, analysis_date or date.today().isoformat(), settings, load_portfolio(portfolio))
        run_workflow(workflow, headless, events)
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Analysis failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def resume(directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
           headless: bool = False, events: bool = False) -> None:
    """Continue a saved run using its original inputs and native session identifiers.

    Args:
        directory: Saved run directory containing run.sqlite.
        headless: Disable terminal rendering.
        events: Stream normalized JSON records.
    """
    try:
        load_dotenv(Path.cwd() / ".env", override=False)
        run_workflow(Workflow.resume(directory), headless, events)
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Resume failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def backtest(tickers: Annotated[str, typer.Argument(help="Comma-separated instruments.")],
             start: Annotated[str, typer.Option(help="First analysis date, YYYY-MM-DD.")],
             end: Annotated[str, typer.Option(help="Last analysis date, YYYY-MM-DD.")],
             every: Annotated[int, typer.Option(min=1)] = 1,
             config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
             harness: HarnessName | None = None, model: str | None = None, provider: str | None = None,
             run_id: str | None = None, portfolio: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None) -> None:
    """Evaluate independent decisions over a ticker/date grid without simulating order fills.

    Args:
        tickers: Comma-separated ticker symbols.
        start: First analysis date.
        end: Last analysis date.
        every: Calendar-day spacing between cells.
        config: Optional TOML configuration.
        harness: Native harness override.
        model: Native model override.
        provider: Pi provider override.
        run_id: Existing sweep identifier to resume.
        portfolio: Optional standing holdings used for each cell.
    """
    from open_trading_agents.backtest import iter_grid, run_backtest

    try:
        settings = configured(config, harness, model, provider)
        result = asyncio.run(run_backtest([t.strip() for t in tickers.split(",") if t.strip()], iter_grid(start, end, every),
                                         settings, load_portfolio(portfolio), run_id))
        typer.echo(result.summary)
        typer.echo(f"Backtest: {result.directory}")
        if result.failures:
            raise typer.Exit(1)
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Backtest failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def doctor() -> None:
    """Report installed harness executables without reading credentials or contacting providers."""
    for harness in HarnessName:
        typer.echo(f"{harness.value}: {shutil.which(harness.value) or 'not installed'}")


@app.command()
def replay(directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Browse saved events through the stage interface without running any model.

    Args:
        directory: Saved run directory.
    """
    from open_trading_agents.storage import RunStore
    from open_trading_agents.tui import TradingApp

    store = RunStore(directory)
    inputs = store.metadata("input", {})
    history = store.events()
    store.close()

    async def play(emit: EventSink) -> None:
        """Replay stored events in original order.

        Args:
            emit: Terminal event consumer.
        """
        for event in history:
            await emit(event)

    TradingApp(play, ticker=inputs.get("ticker", ""), run_dir=directory,
               selected_analysts=tuple(inputs.get("settings", {}).get("analysts", ("market", "social", "news", "fundamentals")))).run()
