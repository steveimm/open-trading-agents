# Open Trading Agents

Research stocks and crypto with pi, Codex, or Claude Code. Follow the analysts and debates in a terminal interface, then read the generated Markdown and HTML reports. The application produces research and recommendations without placing trades.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and at least one installed harness with working model access.

```sh
uv sync
uv run open-trading-agents configure
uv run open-trading-agents
```

Configuration saves a **named preset** containing your harness, provider, model, and reasoning effort. At least one saved preset is required to start an analysis. Run `configure` again to create, update, or delete profiles.

The run wizard asks for the ticker, analysis date, research depth, agent configuration, and output directory. Shallow, Medium, and Deep use 1, 3, and 5 debate rounds. Use arrow keys to select and Enter to confirm. `(default)` identifies your saved default preset.

Choose **Simple** to edit the working-agent and decision-manager profiles, or **Advanced** to assign profiles directly to individual agents. Select a row to change it, then choose **Continue** at the bottom when finished. Rows show the selected preset name or a compact model summary.

To supply the ticker and date directly:

```sh
uv run open-trading-agents --ticker NVDA --date 2026-01-02
```

Settings and profiles are stored under `~/.open-trading-agents/`. See [configuration](docs/configuration.md) for model settings, output language, and analyst selection.

## Controls

| Key | Action |
| --- | --- |
| Tab / Shift+Tab | Switch stages |
| Arrow keys | Select panes or scroll the focused pane |
| f | Toggle fullscreen for an analyst or debate pane |
| Page Up / Page Down | Scroll a page |
| Home / End | Go to the beginning / follow the latest output |
| [ / ] | Select the previous / next expandable item |
| Space or click its header | Expand or collapse an item |
| Hover an agent title | Show that agent's model configuration |
| q | Quit, cancelling active work and retaining progress |

## Reports and saved runs

Reports are exported automatically under `~/.open-trading-agents/runs/RUN_ID/reports/` by default:

- `complete_report.md`
- `complete_report.html`

Press Enter at the output-directory prompt to keep this default, or choose another parent directory. You can also pass `--output-dir ./research`; reports then appear under `./research/RUN_ID/reports/`.

Resume an interrupted run with:

```sh
uv run open-trading-agents resume ~/.open-trading-agents/runs/RUN_ID
```

## Documentation

- [Configuration and profiles](docs/configuration.md)
- [Headless runs, portfolios, and backtesting](docs/usage.md)
- [Data sources and coverage](docs/data-sources.md)
- [Development and architecture](docs/development.md)

## Credits

Adapted from [TradingAgents](https://github.com/TauricResearch/TradingAgents/tree/e8f5503e9b86d5c2710b0c76b1f686703968c9df) by Tauric Research and TradingAgents contributors, under the [Apache License 2.0](LICENSE).
