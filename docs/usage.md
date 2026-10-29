# Advanced usage

## Unattended runs

Headless runs, JSON-event runs, and redirected input/output skip interactive selection. They use saved settings and explicit flags. A saved preset is still required.

```sh
uv run open-trading-agents --ticker NVDA --headless
uv run open-trading-agents --ticker NVDA --date 2026-01-02 --events > run-events.jsonl
```

Without `--date`, an unattended analysis uses the machine's local date. Interactive runs ask for a date unless the flag is supplied.

Use `--help` to see all options. To check installed harnesses, saved profiles, and data-source setup:

```sh
uv run open-trading-agents doctor
```

`doctor` reports whether the config is readable, the default profile is usable, and API keys or SEC contact identity are configured. It also validates the saved portfolio JSON and reports missing or invalid holdings files. It identifies missing settings without printing API keys or portfolio balances. These are local checks. It does not contact providers to validate API keys. Use `doctor --config PATH` to inspect another configuration file.

## Saved runs and resume

Runs are stored under `~/.open-trading-agents/runs/` by default. The run wizard asks for an output directory after the agent profiles. Press Enter to keep the current default, or enter another parent directory. An explicit flag skips this prompt:

```sh
uv run open-trading-agents --ticker NVDA --output-dir ./research
```

This creates `./research/RUN_ID/` with the reports, complete `transcript.jsonl`, saved progress, and native session files. Each run gets its own folder. Relative paths and `~` are supported. Output choices apply only to the current run and do not change saved configuration. Shared cache and feedback memory remain under `~/.open-trading-agents/`.

When output is unspecified, runs use `~/.open-trading-agents/runs/`, including in unattended runs. The final console summary shows the saved report directory and reported input, output, and available cache counts. Detailed usage remains in `reports/usage.json`. Completed interactive runs enable the **b open in browser** shortcut in the Reports pane’s bottom bar and show chat in a smaller left column beside the report on the right. Left/Right selects a column when the input is not focused; **f** expands that column to fullscreen and restores the split view. While typing, Escape returns focus to the chat before using **f**. Chat uses the full report and its independently configured **Report Assistant** profile, retaining its native conversation history across restarts. Its system prompt is visible before the first question. Enter focuses the input below the conversation, Enter sends a question, and Escape leaves the input. **q** quits from the conversation, as in other panes. Chat has all analyst financial tools, plus web tools and native execution when enabled for the run. Data requests retain the original instrument and analysis-date cutoff. The Reports pane does not display elapsed-time counters. Chat saves its transcript to `RUN_ID/chat/transcript.jsonl`, with events and the native session reference in `RUN_ID/chat/run.sqlite`. Resume the original run directory to restore the chat and continue the same conversation. Partial replies and interruption notices are retained. The saved system prompt and model settings stay with that conversation; only one process can drive it at a time. Quitting does not ask any report-viewing questions.

```sh
uv run open-trading-agents resume ~/.open-trading-agents/runs/RUN_ID
uv run open-trading-agents resume ~/.open-trading-agents/runs/RUN_ID --headless
```

For a custom output directory, pass its run folder instead, for example `resume ./research/RUN_ID`. Resume retains completed steps, original inputs, native sessions, and the same output location. Only one process can drive the same saved run at a time. Pressing `q` stops an active run but retains its progress and files.

HTML reports keep the table of contents visible beside the report on wide screens and highlight the current section as you scroll. On narrow screens, use **Contents** to open or close it. Reports remain self-contained and work offline.

## Reading the interface

Running stage tabs and agent turns show an animated loading indicator that disappears when they finish. Debate prompts occupy the left column, with the conversation on the right. Left/Right selects a debate pane, Up/Down scrolls it, and `f` toggles fullscreen. System prompts are available before the agents' turns begin.

Each agent's panel title shows its selected preset name, or the model name for a custom configuration. Hover the title to see that agent's harness, model, and effort. Preset names are saved with the run and retained on resume.

The terminal tab/window title shows the ticker and an animated activity indicator while a run is active. The activity indicator disappears after completion. Failed or cancelled runs show a status label. On exit, the app clears its title and restores the previous one where the terminal supports title restoration. Headless and JSON-event output contain no title-control sequences.

Click a turn or detail header to collapse it, or select it with `[` / `]` and press Space. Collapsed turns continue receiving output. `m` shows memory activity and reflection work when available.

Scrolling upward pauses stream-following. Scroll to the bottom or press End to follow again. Tool previews longer than 10,000 characters show the beginning and end with an omitted-character count. Complete content remains in the saved transcript.

Usage shows input, output, and cache reads. Cache writes appear only when reported. Cache counts are part of input, and reported reasoning tokens are part of output. A dash means unavailable. `+?` means some requests lack that usage detail.

## Portfolio context

Create or edit your saved portfolio with:

```sh
uv run open-trading-agents configure portfolio
```

The wizard edits currency, available cash, and positions. Select a position to edit or remove it, or choose **Add position**. Enter `unknown` for optional cash or average prices you do not want to provide. **Save** writes `~/.open-trading-agents/portfolio.json`; Ctrl+C discards pending edits. An empty positions list means no holdings.

Interactive runs ask whether to use this portfolio before the ticker question. Declining leaves holdings unknown. If you select Yes but the file is missing or invalid, the run stops with a setup instruction. `doctor` checks the default portfolio even when model configuration is missing.

Use `configure portfolio --file PATH` to edit another JSON file, and `doctor --portfolio PATH` to validate it. An explicit `--portfolio PATH` on a run uses that file without asking. Headless, JSON-event, and redirected runs use a portfolio only when this flag is supplied:

```sh
uv run open-trading-agents --ticker NVDA --headless --portfolio ~/.open-trading-agents/portfolio.json
```

The JSON format remains available for manual editing:

```json
{
  "cash": 10000,
  "currency": "USD",
  "positions": [{"ticker": "NVDA", "quantity": 10, "average_price": 120}]
}
```

Each position requires a ticker and finite quantity; average price is optional and cannot be negative. Cash and currency are optional. Portfolio context reaches the Trader, all risk-debate agents, and the Portfolio Manager. Resumed runs retain their original portfolio snapshot. The application does not connect to a broker or place orders.

## Memory and backtesting

Completed recommendations can be evaluated after their holding window ends. The default measurement starts at the first closing price after the analysis date and ends five trading sessions later. Returns are compared with a regional benchmark. Historical analyses receive only prior lessons whose outcomes were known by their cutoff.

Backtesting reruns the full agent workflow across a ticker/date grid:

```sh
uv run open-trading-agents backtest NVDA,AAPL --start 2024-03-01 --end 2024-03-15 --every 7 --run-id march-evaluation
```

`--every` is the spacing in calendar days, not trading days. Each case makes new model requests. Backtests use configured runtime choices without opening the run wizard.

The summary groups recommendations by rating and reports directional hit rates and mean excess return over the benchmark. Buy/Overweight count as hits when the instrument outperforms. Sell/Underweight count as hits when it underperforms. Hold makes no directional claim.

These are independent recommendation evaluations. Supplied holdings stay fixed, and fills, transaction costs, cash balances, and portfolio evolution are not simulated. Historical source coverage is incomplete, so scores are indicative.

Results are stored under `~/.open-trading-agents/backtests/RUN_ID/`. Repeat the same command with the same `--run-id` to resume. Inputs and configuration must remain unchanged. Backtest memory and caches are separate from ordinary runs.
