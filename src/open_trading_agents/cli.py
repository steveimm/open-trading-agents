from __future__ import annotations

import asyncio
import json
import shlex
import sys
import webbrowser
from dataclasses import asdict
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown

from open_trading_agents.analysis.models import AgentEvent
from open_trading_agents.analysis.prompts import ROLES
from open_trading_agents.analysis.workflow import RunResult, Workflow
from open_trading_agents.config import (
    HarnessSettings,
    Settings,
    default_config_path,
    default_portfolio_path,
    list_presets,
    load_portfolio,
    load_settings,
    save_harness_selection,
    validate_date,
)
from open_trading_agents.reporting.usage import format_run_usage

app = typer.Typer(no_args_is_help=False, add_completion=False, help="Financial research through native agent harness sessions.")
configure_app = typer.Typer(no_args_is_help=False, add_completion=False, help="Configure agent profiles, data sources, and portfolio holdings.")
app.add_typer(configure_app, name="configure")
console = Console(stderr=True)


class HarnessName(StrEnum):
    pi = "pi"
    codex = "codex"
    claude = "claude-code"


def configured(
    path: Path | None,
    harness: HarnessName | None,
    model: str | None,
    provider: str | None,
    effort: str | None = None,
) -> Settings:
    """Load application choices while leaving provider credentials with the harness.

    Args:
        path: Explicit TOML file or the shared user configuration.
        harness: Optional CLI harness override.
        model: Optional native model selector.
        provider: Optional native provider selector.
        effort: Optional native reasoning or thinking effort override.

    Returns:
        Validated application settings.
    """
    return runtime_overrides(load_settings(path, use_default_preset=True), harness, model, provider, effort)


def runtime_overrides(
    settings: Settings,
    harness: HarnessName | None,
    model: str | None,
    provider: str | None,
    effort: str | None = None,
) -> Settings:
    """Apply explicit runtime flags to group defaults and individual role selections.

    Args:
        settings: Loaded settings or choices returned by the run wizard.
        harness: Optional native harness override.
        model: Optional native model selector.
        provider: Optional native provider selector.
        effort: Optional native reasoning or thinking effort override.

    Returns:
        Updated settings with backend-specific choices cleared when changing harnesses.
    """
    overrides = {key: value for key, value in {"kind": harness, "model": model, "provider": provider, "effort": effort}.items()
                 if value is not None}
    if overrides:
        named_profiles = {role: settings.model_settings(role=role).profile_identity() for role in settings.agent_presets}

        def apply(selection: HarnessSettings, inherited: HarnessSettings | None = None) -> HarnessSettings:
            """Override one profile while clearing incompatible harness-specific settings.

            Args:
                selection: Existing group or role profile.
                inherited: Group defaults for a partial role profile, or None for a group.

            Returns:
                Validated profile with explicit command-line choices applied.
            """
            updated = selection.model_dump(exclude_unset=inherited is not None)
            kind = updated.get("kind", inherited.kind if inherited is not None else selection.kind)
            if harness is not None and kind != harness:
                resets = {"model": None, "provider": None, "effort": None, "command": (), "extra_args": ()}
                updated.update({key: value for key, value in resets.items() if key in updated})
            updated.update(overrides)
            return HarnessSettings.model_validate(updated)

        agents = {role: apply(selection, settings.model_settings(ROLES[role].deep)) for role, selection in settings.agents.items()}
        for tier in ("harness", "quick", "deep"):
            configured_tier = getattr(settings, tier)
            if configured_tier:
                setattr(settings, tier, apply(configured_tier))
        settings.agents = agents
        settings.agent_presets = {
            role: name for role, name in settings.agent_presets.items()
            if settings.model_settings(role=role).profile_identity() == named_profiles[role]
        }
    return settings


def _offer_report_views(report_dir: Path) -> None:
    """Offer optional browser and terminal views after a completed interactive run.

    Args:
        report_dir: Directory containing the already-exported HTML and Markdown reports.
    """
    from open_trading_agents.ui.configuration import _confirm
    from open_trading_agents.ui.transcript import literal_text

    try:
        if _confirm("Open report in browser?", default=True):
            page = (report_dir / "complete_report.html").resolve()
            try:
                opened = page.is_file() and webbrowser.open_new_tab(page.as_uri())
            except (webbrowser.Error, OSError):
                opened = False
            if not opened:
                typer.echo(f"Could not open a browser. HTML report: {page}")

        if _confirm("Print full report in terminal?", default=True):
            markdown = report_dir / "complete_report.md"
            try:
                content = markdown.read_text(encoding="utf-8")
            except OSError:
                typer.echo(f"Could not read the report: {markdown}")
            else:
                Console().print(Markdown(literal_text(content)))
    except (typer.Abort, KeyboardInterrupt):
        return


def run_workflow(workflow: Workflow, headless: bool, events: bool) -> RunResult:
    """Run a common pipeline through either the live terminal UI or a JSON observer.

    Args:
        workflow: A new or resumed run.
        headless: Whether to avoid terminal UI interaction.
        events: Whether to stream complete event JSON to stdout.

    Returns:
        The completed run result.
    """
    interactive = not headless and not events and sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive:
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
        from open_trading_agents.ui.app import TradingApp

        selected = tuple(key for key in workflow.settings.analysts if not (workflow.asset_type == "crypto" and key == "fundamentals"))
        agent_configs = {role.name: workflow.settings.model_settings(role=key).runtime() for key, role in ROLES.items()}
        agent_presets = {ROLES[key].name: name for key, name in workflow.settings.agent_presets.items()}

        ui = TradingApp(
            workflow.run,
            selected_analysts=selected,
            ticker=workflow.ticker,
            run_dir=workflow.directory,
            agent_configs=agent_configs,
            agent_presets=agent_presets,
            output_language=workflow.settings.output_language,
        )
        ui.run()
        if ui.error:
            raise RuntimeError(str(ui.error)) from ui.error
        if not isinstance(ui.result, RunResult):
            command = shlex.join(["open-trading-agents", "resume", str(workflow.directory)])
            console.print(f"Run cancelled. Resume with: {command}", markup=False, soft_wrap=True)
            raise typer.Exit(130)
        result = ui.result

    if not events:
        report_dir = result.directory / "reports"
        typer.echo(f"{workflow.ticker}: {result.rating}\nRun saved to {report_dir}")
        usage_path = report_dir / "usage.json"
        usage = json.loads(usage_path.read_text(encoding="utf-8"))["totals"]
        typer.echo(format_run_usage(usage))
        if interactive:
            _offer_report_views(report_dir)

    return result


@app.callback(invoke_without_command=True, help="Start a research run, or choose a command to configure profiles or manage saved work.")
def run(
    ctx: typer.Context,
    ticker: Annotated[str | None, typer.Option("--ticker", help="Ticker or supported alias, for example NVDA or BTC-USD.")] = None,
    analysis_date: Annotated[str | None, typer.Option("--date", help="Information cutoff, YYYY-MM-DD. Skips the date prompt.")] = None,
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    harness: Annotated[HarnessName | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option(help="Native harness model selector.")] = None,
    provider: Annotated[str | None, typer.Option(help="Native pi or Codex provider selector.")] = None,
    effort: Annotated[str | None, typer.Option(help="Native reasoning or thinking effort override.")] = None,
    analysts: Annotated[str | None, typer.Option(help="Comma-separated market,social,news,fundamentals.")] = None,
    research_rounds: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    risk_rounds: Annotated[int | None, typer.Option(min=1, max=20)] = None,
    portfolio: Annotated[Path | None, typer.Option(exists=True, dir_okay=False, help="Holdings JSON file. Uses this portfolio without asking.")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", file_okay=False, help="Parent directory for this run's output folder.")] = None,
    headless: Annotated[bool, typer.Option(help="Run without the terminal UI.")] = False,
    events: Annotated[bool, typer.Option(help="Emit normalized JSONL events to stdout.")] = False,
) -> None:
    """Start the research wizard when no maintenance command is selected.

    Args:
        ctx: Command context identifying an optional subcommand.
        ticker: Instrument symbol.
        analysis_date: Analysis cutoff date.
        config: Optional TOML configuration.
        harness: Native harness override.
        model: Native model selector override.
        provider: Native provider selector override.
        effort: Native reasoning or thinking effort override.
        analysts: Selected research roles.
        research_rounds: Number of bull/bear rotations.
        risk_rounds: Number of three-speaker risk rotations.
        portfolio: Optional holdings JSON file.
        output_dir: Parent directory for this run's reports, transcript, and session files.
        headless: Disable terminal rendering.
        events: Stream JSON events instead of concise progress.
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        if not list_presets():
            typer.echo("Run uv run open-trading-agents configure first to create a preset.")
            raise typer.Exit(1)

        interactive = not headless and not events and sys.stdin.isatty() and sys.stdout.isatty()
        if analysis_date is not None:
            analysis_date = validate_date(analysis_date.strip())
        selected_portfolio = load_portfolio(portfolio)
        if interactive and portfolio is None:
            from open_trading_agents.ui.configuration import select_run_portfolio

            selected_portfolio = select_run_portfolio()

        if ticker is None:
            if not interactive:
                raise typer.BadParameter("provide --ticker, for example: open-trading-agents --ticker NVDA --headless")
            ticker = typer.prompt("Ticker")

        if interactive and analysis_date is None:
            from open_trading_agents.ui.configuration import select_analysis_date

            analysis_date = select_analysis_date()
        analysis_date = validate_date(analysis_date.strip() if analysis_date is not None else date.today().isoformat())

        settings = configured(config, harness, model, provider, effort)
        if interactive:
            from open_trading_agents.ui.configuration import select_agent_settings, select_research_depth

            settings = select_research_depth(settings)
            settings = select_agent_settings(settings)
            settings = runtime_overrides(settings, harness, model, provider, effort)
        changes = {"analysts": tuple(key.strip() for key in analysts.split(",")) if analysts else None,
                   "research_rounds": research_rounds, "risk_rounds": risk_rounds}
        settings = Settings.model_validate({**settings.model_dump(), **{k: v for k, v in changes.items() if v is not None}})

        if interactive and output_dir is None:
            from open_trading_agents.ui.configuration import select_output_directory

            output_dir = select_output_directory(settings.state_dir / "runs")
        if output_dir is not None:
            output_dir = output_dir.expanduser().resolve()

        workflow = Workflow(ticker, analysis_date, settings, selected_portfolio, output_dir=output_dir, config_path=config)
        run_workflow(workflow, headless, events)
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Analysis failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def resume(
    directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    headless: bool = False,
    events: bool = False,
) -> None:
    """Continue a saved run using its original inputs and native session identifiers.

    Args:
        directory: Saved run directory containing run.sqlite.
        headless: Disable terminal rendering.
        events: Stream normalized JSON records.
    """
    try:
        run_workflow(Workflow.resume(directory), headless, events)
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Resume failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def backtest(
    tickers: Annotated[str, typer.Argument(help="Comma-separated instruments.")],
    start: Annotated[str, typer.Option(help="First analysis date, YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Last analysis date, YYYY-MM-DD.")],
    every: Annotated[int, typer.Option(min=1)] = 1,
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    harness: HarnessName | None = None,
    model: str | None = None,
    provider: str | None = None,
    effort: Annotated[str | None, typer.Option(help="Native reasoning or thinking effort override.")] = None,
    run_id: str | None = None,
    portfolio: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Evaluate independent decisions over a ticker/date grid without simulating order fills.

    Args:
        tickers: Comma-separated ticker symbols.
        start: First analysis date.
        end: Last analysis date.
        every: Calendar-day spacing between cells.
        config: Optional TOML configuration.
        harness: Native harness override.
        model: Native model override.
        provider: Native provider override.
        effort: Native reasoning or thinking effort override.
        run_id: Existing sweep identifier to resume.
        portfolio: Optional standing holdings used for each cell.
    """
    from open_trading_agents.analysis.backtest import iter_grid, run_backtest

    try:
        settings = configured(config, harness, model, provider, effort)
        result = asyncio.run(run_backtest([t.strip() for t in tickers.split(",") if t.strip()], iter_grid(start, end, every),
                                         settings, load_portfolio(portfolio), run_id, config_path=config))
        typer.echo(result.summary)
        typer.echo(f"Backtest: {result.directory}")
        if result.failures:
            raise typer.Exit(1)
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Backtest failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


def _configuration_path(ctx: typer.Context, path: Path | None) -> Path:
    """Resolve a configuration destination from the command or its parent options.

    Args:
        ctx: Current configuration command context.
        path: Explicit path supplied to this command.

    Returns:
        The selected configuration path, defaulting to the application directory.
    """
    current: typer.Context | None = ctx
    while path is None and current is not None:
        path = current.params.get("config")
        current = current.parent

    return Path(path or default_config_path()).expanduser()


@configure_app.callback(invoke_without_command=True)
def configure(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(dir_okay=False, help="Configuration file. Defaults to ~/.open-trading-agents/config.toml.")] = None,
) -> None:
    """Choose agent profiles, data sources, or portfolio holdings when no subcommand is supplied.

    Args:
        ctx: Command context identifying an optional configuration subcommand.
        config: Configuration file used by the selected editor.
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        if not sys.stdin.isatty():
            raise typer.BadParameter("choose 'configure agents' with selection flags, or run 'configure data' or 'configure portfolio' in a terminal")
        from open_trading_agents.ui.configuration import select_configuration_section

        section = select_configuration_section()
        if section == "agents":
            configure_agents(ctx, config=config)
        elif section == "data":
            configure_data(ctx, config=config)
        else:
            configure_portfolio()
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Configuration failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@configure_app.command("agents", help="Create, update, or delete saved agent profiles.")
def configure_agents(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(dir_okay=False, help="Configuration file. Defaults to ~/.open-trading-agents/config.toml.")] = None,
    harness: Annotated[HarnessName | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option(help='Native model selector. Pass "" to use harness defaults.')] = None,
    provider: Annotated[str | None, typer.Option(help='Native pi or Codex provider selector. Pass "" to use harness defaults.')] = None,
    effort: Annotated[str | None, typer.Option(help='Native reasoning or thinking effort. Pass "" to use harness defaults.')] = None,
) -> None:
    """Manage saved runtime profiles, or apply explicit configuration flags.

    Args:
        ctx: Command context carrying an optional parent configuration path.
        config: Destination TOML configuration file.
        harness: Native harness selection.
        model: Native model selector, or an empty string to reset it.
        provider: Native provider selector, or an empty string to reset it.
        effort: Native reasoning or thinking effort, or an empty string to reset it.
    """
    try:
        config = _configuration_path(ctx, config)
        interactive = harness is None and model is None and provider is None and effort is None
        if interactive and not sys.stdin.isatty():
            raise typer.BadParameter("provide --harness, --model, --provider or --effort when input is not interactive")

        if interactive:
            from open_trading_agents.ui.configuration import configure_interactively

            configure_interactively(config)
            return
        settings = save_harness_selection(config, harness=harness.value if harness is not None else None,
                                          model=model, provider=provider, effort=effort)
        selected = settings.harness
        typer.echo(f"Saved default: {selected.kind} / {selected.model or 'Harness default'} / {selected.effort or 'Harness default'}")
        typer.echo(f"Config: {config.expanduser().resolve()}")
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Configuration failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@configure_app.command("data", help="Choose data sources and configure API keys and SEC contact identity.")
def configure_data(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(dir_okay=False, help="Configuration file. Defaults to ~/.open-trading-agents/config.toml.")] = None,
) -> None:
    """Edit data sources and credentials stored in the selected TOML configuration.

    Args:
        ctx: Command context carrying an optional parent configuration path.
        config: Configuration file to update.
    """
    try:
        if not sys.stdin.isatty():
            raise typer.BadParameter("run 'configure data' in a terminal, or edit the [data] fields in config.toml")
        from open_trading_agents.ui.configuration import configure_data_interactively

        configure_data_interactively(_configuration_path(ctx, config))
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Configuration failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@configure_app.command("portfolio", help="Create or edit saved holdings, cash, and currency.")
def configure_portfolio(
    file: Annotated[Path | None, typer.Option("--file", dir_okay=False, help="Portfolio file. Defaults to ~/.open-trading-agents/portfolio.json.")] = None,
) -> None:
    """Edit the saved portfolio without configuring models or contacting providers.

    Args:
        file: JSON portfolio file to create or update.
    """
    try:
        if not sys.stdin.isatty():
            raise typer.BadParameter("run 'configure portfolio' in a terminal, or provide a portfolio JSON file with --portfolio")
        from open_trading_agents.ui.configuration import configure_portfolio_interactively

        configure_portfolio_interactively((file or default_portfolio_path()).expanduser())
    except (typer.Abort, KeyboardInterrupt):
        typer.echo("Cancelled.", err=True)
        raise typer.Exit(130) from None
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        console.print(f"Portfolio configuration failed: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def doctor(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(dir_okay=False, help="Configuration file. Defaults to ~/.open-trading-agents/config.toml.")] = None,
    portfolio: Annotated[Path | None, typer.Option(dir_okay=False, help="Portfolio file to check. Defaults to ~/.open-trading-agents/portfolio.json.")] = None,
) -> None:
    """Check local harness, agent-profile, data-source, and portfolio configuration.

    Args:
        ctx: Command context carrying an optional parent configuration path.
        config: Configuration file to inspect without contacting model or data providers.
        portfolio: Optional custom portfolio file to validate.
    """
    from open_trading_agents.diagnostics import doctor_checks

    for label, result in doctor_checks(_configuration_path(ctx, config), portfolio):
        typer.echo(f"{label}: {result}")


