# Open Trading Agents

A Python application for financial research with independently managed agent sessions.

The application coordinates analysis, research debate, a research decision, a trader proposal, risk debate, a portfolio decision, and reports. Pi, Codex, and Claude Code CLI adapters own their model connections, authentication, conversation history, and tool loops.

The project is managed with uv. The native harnesses remain responsible for provider authentication, request construction, reasoning history, retries, and context compaction. The application supplies new task messages and records explicit session IDs.

## Run

Install Python dependencies and check which harnesses are available:

```sh
uv sync
uv run open-trading-agents doctor
```

Configure your chosen harness through its normal CLI first. Then start a run:

```sh
uv run open-trading-agents analyze NVDA
uv run open-trading-agents analyze BTC-USD --harness claude
uv run open-trading-agents analyze NVDA --harness codex --date 2026-01-02
```

Omitting the ticker opens a short ticker/date prompt in an interactive terminal. Model and provider selectors are optional. When omitted, the harness uses its own configuration. Native protocol support still matters: a harness that expects Responses or Anthropic Messages is not automatically compatible with an arbitrary Chat Completions endpoint.

Use `configure` for guided harness and model selection:

```sh
uv run open-trading-agents configure
```

Use Up/Down to move through choices and Enter to confirm. Current settings start highlighted, and typing filters the choices. The menus cover shared defaults, working agents (`quick`), and decision managers (`deep`) in one session. Each tier can inherit the shared defaults or select its own harness and model. Providers and models come from native harness metadata where available. Choose `Native default` to delegate a selection to the harness, or enter a custom identifier when needed. Preset confirmations use the same menus. Ctrl+C cancels setup without saving. Setup does not send a model prompt.

Configuration is saved to `~/.open-trading-agents/config.toml` and loaded regardless of the working directory. Use `--config PATH` for another file. Existing comments, data settings, and unrelated preferences are preserved. Connection settings and authentication remain in the harness.

At the end, setup offers to save the default and both tier selections as a named preset under `~/.open-trading-agents/presets/`. It asks before replacing an existing preset. Multiple presets can be kept and selected for an individual analysis or backtest:

```sh
uv run open-trading-agents analyze NVDA --preset local-research
uv run open-trading-agents backtest NVDA --start 2026-01-05 --end 2026-01-09 --preset local-research
```

A preset replaces all three harness selections for that invocation without rewriting the main config. Other application and data preferences still come from the main config. Explicit CLI harness/model/provider options override the preset. Resuming a run keeps the selections saved with that run.

For unattended configuration, selection flags remain available, for example `configure --harness pi --provider my-local-provider --model my-model`. `--tier quick` or `--tier deep` targets one tier, and an empty `--model ""` or `--provider ""` selects native defaults. You can also use `tradingagents.example.toml` as a starting point for the global config or an explicit `--config` file. The `[harness]` section sets shared defaults, with optional `[quick]` and `[deep]` overrides:

```toml
[harness]
kind = "pi"
provider = "my-local-provider"
model = "my-configured-model"

[deep]
kind = "codex"
```

The application does not set temperature, top-p, or thinking effort. A harness may apply its own defaults, so configure those in the harness's model profile when the server must retain particular defaults.

## Interface

The seven stages are Analysis, Research debate, Research decision, Trader, Risk debate, Portfolio decision, and Reports. Running tabs show an animated circular spinner, completed tabs show a check mark, and failed tabs are marked explicitly. Selected analysts run concurrently. Each debater keeps its own native conversation across rounds and receives only newly available public arguments.

Debates show the shared input once, followed by a chat bubble for each participant's turn. The current response streams into its bubble, and its title stays at the top while scrolling through that contribution. Repeated per-agent input messages are omitted from the debate view entirely. System instructions, reasoning, and tool calls remain expandable. Exact prompts remain in the saved event log, and this presentation does not change what is sent to the harness.

| Key | Action |
| --- | --- |
| Tab / Shift+Tab | Next / previous stage |
| Arrow keys | Select between analyst panes in grid view, otherwise scroll the active transcript |
| Enter / f | Expand or restore an analyst pane when multiple analyst panes are present |
| Escape | Return from an analyst fullscreen view to its grid |
| Page Up / Page Down | Scroll a page |
| Home / End | Beginning / jump to the latest content and follow the stream |
| [ / ] | Select the previous / next collapsible section; these keys navigate, rather than expand it |
| Space | Expand or collapse the selected section |
| m | Show supporting memory and screening work |
| q | Exit, cancelling active work while retaining saved progress |

Inputs, emitted reasoning, output, code, tool arguments, and tool results use different colors. Click a disclosure header or use the keyboard to expand it. Reasoning appears only when the harness exposes it. Supplied prompts and tool traffic are recorded in full. Exact native metadata remains in the saved event log rather than a separate payload pane.

Shortcut hints follow the selected stage and focused pane. Fullscreen controls appear only for the analyst grid, and section controls appear only when there are expandable sections. System instructions use the same click or section-selection/Space controls as other disclosures.

Scrolling upward pauses following so new output does not move what you are reading. Scrolling down to the bottom, or pressing End, resumes following automatically. End does not stop the run.

Expanded tool results longer than 10,000 characters show the first 6,000 and last 2,000 characters with an exact omitted-character count. The original events remain in SQLite and the live `transcript.jsonl` file. Ordinary prompts and model output are not shortened by this display rule.

Elapsed clocks update independently of model responses. Usage shows total input, output, and cache reads. Cache writes appear only when a provider reports them, including an explicit zero. Cache counts are subsets of input, not additional input. An em dash means unavailable and `+?` means only part of the usage has cache detail. Reported reasoning tokens are already included in output.

## Calculations and sources

Agents can use native execution tools to write and run calculation scripts in their own workspaces. Interpreters are resolved by the native shell environment, with no application interpreter path prescribed in prompts. The shared `calculate` tool verifies arithmetic and statistics even when a harness cannot execute a script. It interprets a bounded arithmetic syntax without imports or arbitrary code execution. Set `allow_execution = false` to disable native scripts while retaining the calculator.

Financial tools retain Yahoo, SEC EDGAR, Alpha Vantage, FRED, Polymarket, Reddit, and StockTwits integrations. Tools bind the ticker and information cutoff outside the model's arguments. Technical reports have a shared verified price/indicator snapshot. Missing or unavailable data is reported explicitly.

`search_web` and `fetch_webpage` provide additional cited evidence. Search uses the explicitly configured backend (`data.web_search_backend`, default `yahoo`). Page retrieval limits redirects and response size, rejects private network destinations, and extracts inert text. A live page is not an archived historical snapshot. Set `web_research = false` to omit these tools.

Optional social screening (`social_screening = true`) uses a separate native harness session to assess relevance and stance. Failed screening preserves the original posts. No model is called directly from the financial adapters.

Most default sources need no API key. Optional FRED and Alpha Vantage sources use `FRED_API_KEY` and `ALPHA_VANTAGE_API_KEY`. `SEC_EDGAR_USER_AGENT` can identify SEC requests. A `.env` in the current directory is loaded without replacing exported environment values. Provider authentication stays in the harness.

## Saved runs and unattended use

The default state directory is `~/.open-trading-agents`, with runs under `~/.open-trading-agents/runs`. Override it using `state_dir` or `--state-dir`. Each run contains a SQLite completion ledger, native session references, live JSONL events, individual plus combined Markdown/HTML reports, and a separate `reports/usage.json` diagnostic file. Full reports are exported automatically to `RUN_DIR/reports/complete_report.md` and `complete_report.html`. They follow five team sections: analyst reports, research decision, trading plan, risk decision, and portfolio decision. Their headers identify the observed harnesses, providers, and models, with configuration fallbacks explicitly labeled when native identity is unavailable. Token statistics remain in the TUI and diagnostic usage file, not in the human reports. The Reports stage loads the complete saved Markdown after completion. Pressing q closes the interface without removing these files. Reports and transcripts remain available after an interrupted run.

```sh
uv run open-trading-agents analyze NVDA --headless
uv run open-trading-agents analyze NVDA --events > run-events.jsonl
uv run open-trading-agents resume /path/to/state/runs/RUN_ID
```

Resume retains completed steps and the original inputs. An acknowledged unfinished task continues its native session. If delivery was never acknowledged, the stored task is submitted again, so that narrow crash window provides at-least-once delivery. Only one process can drive a saved run at a time.

Use `--portfolio portfolio.json` to supply known holdings and cash:

```json
{"cash": 10000, "currency": "USD", "positions": [{"ticker": "NVDA", "quantity": 10, "average_price": 120}]}
```

An omitted portfolio means unknown holdings. An empty portfolio means a known flat account. No broker connection or order placement is implemented.

## Memory and backtesting

Completed recommendations are stored in SQLite. Later runs settle eligible outcomes using the first close after the analysis date and the configured holding window, five instrument trading sessions by default. Returns are compared with a regional benchmark. Reflections use the configured harness, and historical analysis receives only lessons whose outcomes were known by its cutoff.

```sh
uv run open-trading-agents backtest NVDA,AAPL --start 2024-03-01 --end 2024-03-15 --every 7 --run-id march-evaluation
```

Repeat the same command to resume a sweep. A changed configuration, grid, or portfolio is rejected for an existing sweep identifier. Sweep runs, memory and cache are isolated from live analysis. Results score independent ratings against realized alpha. They do not simulate fills, transaction costs, or evolving holdings. Historical news/social coverage is incomplete and model sampling varies, so these evaluations are not exact historical replay.

## Validation

```sh
uv run pytest -q
uv run pytest -q -m native
uv run ruff check src tests
uv build
```

Default tests use offline fixtures and subprocess protocol doubles. The optional `native` tests use an installed CLI and a local fixture server, without external model requests. Live validation is performed separately with explicitly selected harness configurations. The implementation has been tested with pi 1.3.0, Codex 0.164.0, and Claude Code 2.1.294. Native execution can still be blocked by the harness's own permissions or the host's sandbox support. Such failures stay visible, and the arithmetic tool remains available.

There is no application telemetry, update polling, announcement service, or remote tracing. Dependency and native-harness telemetry are outside this project's scope. Local transcript persistence is part of the requested interface and resume behavior.

See `THIRD_PARTY_NOTICES` and `LICENSE` for the adapted reference components.

## Agreed requirements

- Python and `pyproject.toml`, initialized and managed with uv.
- Preserve the reference workflow, investment prompts, stock and crypto analysis, financial-data sources, portfolio input, feedback, backtesting, reports, and unattended execution.
- Start with fresh application storage. Compatibility with the reference project's files is unnecessary.
- Every model operation uses a harness, including reflection. Do not reconstruct provider request payloads in application code.
- Support pi, Codex, and Claude Code using their structured CLI protocols and explicit session identifiers.
- Keep model settings and credentials in the harness. Models are configurable, with no dependence on a particular model family.
- Enable harness-native script execution for numeric verification and assigned web search/page retrieval tools for additional cited sources. Keep code, inputs and tool output visible in the transcript.
- Include no application telemetry, announcements, update polling, or remote tracing. Dependency telemetry is outside this project's scope.
- Show stages as tabs, with Tab and Shift+Tab navigation. The analysis stage has up to four selectable panes and fullscreen transcripts. Debate stages show the complete shared discussion and the current speaker's live stream.
- Show supplied prompts, tool definitions, tool arguments and results, available reasoning, and output with distinct styling. Avoid repeated bracketed channel markers.
- Keep scrolling and input responsive during concurrent streaming and long transcripts. Update elapsed time independently of model events.
- Report per-agent and total usage, including cache reads and writes when available. Distinguish unavailable cache information from zero.
- Preserve native sessions between debate turns. Persist pipeline progress separately so a completed stage need not rerun after interruption.
- Add meaningful tests during development and commit small, verified increments frequently.

The reference is [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) at revision `e8f5503e9b86d5c2710b0c76b1f686703968c9df`. Its source and the user's development transcript informed this design. No reference checkout is required at runtime.
