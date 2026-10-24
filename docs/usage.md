# Advanced usage

## Unattended runs

Headless runs, JSON-event runs, and redirected input/output skip interactive selection. They use saved settings and explicit flags. A saved preset is still required.

```sh
uv run open-trading-agents --ticker NVDA --headless
uv run open-trading-agents --ticker NVDA --date 2026-01-02 --events > run-events.jsonl
```

Without `--date`, an unattended analysis uses the machine's local date. Interactive runs ask for a date unless the flag is supplied.

Use `--help` to see all options. To check which harness executables are on PATH:

```sh
uv run open-trading-agents doctor
```

## Saved runs and resume

Runs are stored under `~/.open-trading-agents/runs/` by default. The run wizard asks for an output directory after the agent profiles. Press Enter to keep the current default, or enter another parent directory. An explicit flag skips this prompt:

```sh
uv run open-trading-agents --ticker NVDA --output-dir ./research
```

This creates `./research/RUN_ID/` with the reports, complete `transcript.jsonl`, saved progress, and native session files. Each run gets its own folder. Relative paths and `~` are supported. Output choices apply only to the current run and do not change saved configuration. Shared cache and feedback memory remain under the configured `state_dir`.

When output is unspecified, the current `state_dir/runs/` location is retained, including in unattended runs. The final console summary shows reported input, output, and available cache counts, followed by the path to `reports/usage.json` for full details.

```sh
uv run open-trading-agents resume ~/.open-trading-agents/runs/RUN_ID
uv run open-trading-agents resume ~/.open-trading-agents/runs/RUN_ID --headless
```

For a custom output directory, pass its run folder instead, for example `resume ./research/RUN_ID`. Resume retains completed steps, original inputs, native sessions, and the same output location. Only one process can drive the same saved run at a time. Pressing `q` stops an active run but retains its progress and files.

## Reading the interface

Stage tabs show progress. Debate prompts occupy the left column, with the conversation on the right. Left/Right selects a debate pane, Up/Down scrolls it, and `f` toggles fullscreen. System prompts are available before the agents' turns begin.

The terminal tab/window title shows the ticker and an animated activity indicator while a run is active. It switches to a completion mark or failure label when the run finishes. On exit, the app clears its title and restores the previous one where the terminal supports title restoration. Headless and JSON-event output contain no title-control sequences.

Click a turn or detail header to collapse it, or select it with `[` / `]` and press Space. Collapsed turns continue receiving output. `m` shows supporting memory or screening work when available.

Scrolling upward pauses stream-following. Scroll to the bottom or press End to follow again. Tool previews longer than 10,000 characters show the beginning and end with an omitted-character count. Complete content remains in the saved transcript.

Usage shows input, output, and cache reads. Cache writes appear only when reported. Cache counts are part of input, and reported reasoning tokens are part of output. A dash means unavailable. `+?` means some requests lack that usage detail.

## Portfolio context

Use `--portfolio portfolio.json` to supply holdings and cash:

```json
{"cash": 10000, "currency": "USD", "positions": [{"ticker": "NVDA", "quantity": 10, "average_price": 120}]}
```

An omitted portfolio means holdings are unknown. An empty portfolio means a known flat account. The application does not connect to a broker or place orders.

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
