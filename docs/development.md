# Development

## Source layout

The package root contains the CLI entry points and shared application configuration.

| Directory | Responsibility |
| --- | --- |
| `analysis/` | Workflow orchestration, role prompts, decision schemas, backtesting, and shared models |
| `harnesses/` | Native CLI adapters and the pi tool extension |
| `dataflows/` | Financial sources, source configuration, and the local MCP tool server |
| `persistence/` | Run storage, event recording, and decision memory |
| `reporting/` | Markdown/HTML reports and usage summaries |
| `ui/` | Terminal interface, transcript rendering, and configuration menus |

## Workflow and harnesses

The workflow runs parallel analysts, a bull/bear debate, the research manager, the trader, a three-participant risk debate, the portfolio manager, and report export. Optional memory evaluation and social screening use the same harness interface.

Harnesses manage model connections, authentication, request construction, conversation history, and tool loops. The application supplies role instructions and new task messages, then records events and native session identifiers.

Claude Code and Codex connect to the application's data tools through an MCP server named `trading`. Pi uses an extension that registers wrappers around the same MCP tools. The server entry point is:

```sh
uv run python -m open_trading_agents.dataflows.tools --help
```

Structured decision roles receive their validation schema in the initial system instructions. A malformed response gets one correction request in the same native session. Validated fields are converted into the readable report output.

## Persistence and rendering

Each run stores its completion ledger and events in SQLite, with a JSONL transcript and native session references. Completed steps are reused on resume. An acknowledged unfinished task continues its native session. An unacknowledged task is submitted again, giving that interruption window at-least-once delivery.

The UI renders queued events independently of model generation. Conversation sections retain their full source text while indexing and caching visible rows. Large tool previews are bounded in the viewport without shortening stored events.

## Validation

```sh
uv sync
uv run pytest -q
uv run pytest -q -m native
uv run ruff check src tests
uv build
```

The default suite uses offline fixtures and subprocess protocol doubles. Native tests require an installed pi CLI and use a local fixture server, with no external model requests. They exercise native script execution, history retention, effort selection, usage, and tool budgets.

When changing package layout, also test the built wheel outside the checkout. Check the CLI, grouped imports, `pi_extension.mjs` asset, and MCP subprocess entry point.

## Reference

The adapted source is [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents/tree/e8f5503e9b86d5c2710b0c76b1f686703968c9df), revision `e8f5503e9b86d5c2710b0c76b1f686703968c9df` (0.7.0). Attribution is in the [README](../README.md#credits), and the license is in [LICENSE](../LICENSE).

See the [detailed reference comparison](reference-comparison.md) for preserved features, implementation changes, superseded designs, remaining differences, and the corresponding commit history.
