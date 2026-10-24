# Changes from the TradingAgents reference

This document records the implemented changes discussed in this chat session, including intermediate designs that were later replaced. It compares the final implementation with the pinned reference rather than assuming every feature in this repository is new.

## Scope and evidence

- **Reference:** [TauricResearch/TradingAgents][ref], revision `e8f5503e9b86d5c2710b0c76b1f686703968c9df`, version 0.7.0. The local reference checkout was verified at that revision.
- **Implementation snapshot:** this repository through `c45ff37`, covering the work recorded on October 23–24, 2026. This comparison is a documentation addition after that snapshot.
- **Evidence:** the messages visible in this conversation, the implementation commit history, and both codebases. Short commit IDs below refer to this repository's history and can be inspected with `git show COMMIT`.
- **Classification:** “preserved” means the reference already had the capability. “Replaced” means a different implementation now supplies it. “Added” means the current project introduced it. “Revised” includes corrections to intermediate versions of this project.

Permissions to test a model server, authentication troubleshooting, and observations from particular runs are not treated as product features. The discussion of possible payload/history problems in the reference motivated the new architecture, but this document does not claim those suspected upstream problems were proven.

## 1. Overall comparison

| Area | Pinned reference | Current implementation |
| --- | --- | --- |
| Application structure | Python 3.11+, setuptools, `tradingagents` and `cli` packages, LangChain/LangGraph dependencies | Python 3.12+, uv-managed project, `uv_build`, and the grouped `open_trading_agents` package |
| Model execution | Application-created provider clients and graph-managed calls | Native pi, Codex, or Claude Code sessions behind a shared adapter interface |
| Model requests and history | Provider/LangChain wrappers and graph state | Harness-owned request construction, tool loops, history, reasoning retention, retries, and compaction |
| Research pipeline | Parallel analysts, research debate, managers, trader, risk debate, final report | Same main stages, reimplemented with an async coordinator and native sessions |
| Financial tools | LangChain tools with hidden graph-state ticker/date injection | Local MCP tools with immutable application context, exposed through each harness |
| Script execution | Analyst data-tool workflow was the primary tool interface | Harness-native script and file tools are enabled for numeric verification |
| Extra web research | Reference financial/news sources and provider-specific model behavior | Shared application `search_web` and `fetch_webpage` tools assigned to analysts |
| Terminal interface | Rich streaming panes with selection, fullscreen, scrolling, and timing | Textual stage tabs, chronological debate bubbles, system-prompt columns, collapsible content, and incremental row rendering |
| Configuration | Reference config/environment variables and remembered CLI selections | Global TOML config, explicit named single-agent profiles, and per-run working/manager selection |
| Run persistence | Graph checkpoints and reference result/log files | Per-run SQLite ledger, normalized events, JSONL transcript, and native session references |
| Reports | Individual Markdown, combined Markdown, and self-contained HTML in five team sections | Same report family and grouping, adapted presentation, automatic run-directory exports, and observed harness/model provenance |
| Feedback memory | Stored decisions, outcome evaluation, reflections, and lessons for the Portfolio Manager | Preserved feedback concept with SQLite storage, run-ID identity, fixed per-decision evaluation settings, and harness-based reflections |
| Backtesting | Independent ticker/date evaluations with benchmark-relative scoring | Preserved scoring, plus resumable native workflows, pinned manifests, and isolated run/cache/memory state |

Sources: [reference dependencies][ref-project], [reference graph][ref-graph], [current dependencies](../pyproject.toml), [current workflow](../src/open_trading_agents/analysis/workflow.py).

## 2. What was deliberately preserved

### Research roles and stage order

The main analytical workflow was retained. It was not invented during this rewrite.

| Stage | Participants and responsibility |
| --- | --- |
| Analysis | Up to four selected analysts: Market, Sentiment, News, and Fundamentals. They work concurrently. Fundamentals is omitted for crypto. |
| Research Debate | Bull Researcher and Bear Researcher exchange evidence-based arguments. |
| Research Decision | Research Manager produces an investment recommendation and implementation strategy. |
| Trader | Trader turns the research plan and supplied portfolio context into a transaction proposal. |
| Risk Debate | Aggressive, Conservative, and Neutral analysts evaluate the proposal. |
| Portfolio Decision | Portfolio Manager produces the final rating, executive summary, and investment thesis. |
| Reports | Available outputs are assembled into human-readable reports. |

Research depth retains the reference's Shallow/Medium/Deep choices, mapped to 1/3/5 research and risk rounds. The initial analysts have a concurrency limit of four. Optional reflection or screening sessions are separate supporting work, so that limit is not a claim that the entire application can never have more than four sessions.

The role instructions were adapted into a central role catalog rather than copied as an unchanged collection of graph-node implementations. The financial responsibilities, rating scale, evidence requirements, portfolio uncertainty, and opposing debate viewpoints were retained.

Sources: [reference graph setup][ref-setup], [reference agents][ref-agents], [current role catalog](../src/open_trading_agents/analysis/prompts.py), [current orchestration](../src/open_trading_agents/analysis/workflow.py).

### Financial data, symbols, and historical guards

The following were already present in the pinned reference and were ported or adapted:

- Yahoo Finance, SEC EDGAR, Alpha Vantage, FRED, Polymarket, Reddit, and StockTwits integrations.
- Symbol normalization for stock exchange suffixes, supported crypto aliases, and supported commodity/index/forex aliases.
- Regional benchmark selection. International ticker support was not newly added in response to the later question about non-US stocks.
- Hidden ticker and analysis-date binding. The implementation mechanism changed from `InjectedState` to `ToolContext`, but preventing the model from replacing the run's context was already a reference feature.
- Date-window clamping and exclusion of known future observations.
- Withholding live-only company profiles, undated historical statements, and insider trades without disclosure dates when they cannot establish historical availability.
- SEC filing-date selection, FRED historical vintages, and withholding live prediction-market odds from historical analysis.
- Stale-price checks, local trading-date handling, and verified market snapshots.
- Explicit distinctions between missing data and a provider being unavailable.

The later review corrected inherited precision and missing-observation problems, described under [code review corrections](#code-review-corrections). Data availability remains uneven by market, and the application does not add dedicated local-filing integrations for every country.

Two reference behaviors are not fully reproduced. The reference performs a best-effort upfront company-identity lookup before constructing agent context; the current common context supplies the ticker, asset type, date, and an instruction to support identity claims with evidence. Also, the reference CLI classifies crypto by quote suffix, while the current workflow recognizes a bounded set of bases through `crypto_base`. Unrecognized bases, or quotes such as `ETH-BTC`, can therefore take the stock path. This describes classification behavior, not a guarantee that Yahoo supplies data for every symbol. Sources: [reference instrument context][ref-context], [reference ticker handling][ref-cli-prompts], [current workflow](../src/open_trading_agents/analysis/workflow.py).

Sources: [reference tools][ref-tools], [reference historical guards][ref-dates], [current data sources](data-sources.md), [current symbol handling](../src/open_trading_agents/dataflows/symbols.py).

### Portfolio, reports, and evaluation concepts

Optional cash, currency, and position input was retained, including the distinction between unknown holdings and a known flat portfolio. The current models additionally normalize holding symbols and validate finite values, unsupported fields, and average prices. Neither codebase is a broker integration or order-execution system.

The reference already had five-section Markdown/HTML reports, partial report saving, feedback memory, and independent backtests. The changes to their storage, invocation, presentation, and recovery are described below rather than credited as entirely new features.

## 3. Native harness architecture

### Replaced application-owned model clients

The graph/provider-client execution layer was replaced by a shared harness contract:

- `HarnessConfig` describes the chosen native CLI and selectors.
- `AgentSpec` supplies a role, system prompt, workspace, native session location, permitted application tools, execution permission, and tool-round budget.
- A session supports prompting, cancellation, and closure while retaining its native identity.
- `AgentEvent` normalizes lifecycle, input, output, exposed reasoning, tool activity, usage, and model identity for persistence and display.

The application submits new task messages. It does not reconstruct a provider's conversation payload for each turn. Each logical agent has its own workspace and session. Debate participants reuse their session across rounds and receive only newly available public arguments after their initial context.

Visibility depends on what each native protocol exposes. The app records its supplied prompts and normalized native events. Pi additionally exposes request metadata through the extension. This is not a claim that every adapter exposes every internal provider request, hidden reasoning field, or harness-generated instruction.

The project remains Python. Node.js was discussed early in the conversation, but the implemented Node component is the pi extension, not a replacement Node.js application.

### Three implemented adapters

| Harness | Native interface | Session and tool integration |
| --- | --- | --- |
| pi | Persistent RPC mode | Saves native session IDs/files, streams native events, and loads the project's extension for data tools, request visibility, usage, and budgets |
| Codex | Native app-server JSON-RPC | Starts/resumes a thread and submits turns. Registers the application MCP server in the thread's configuration |
| Claude Code | Bidirectional `stream-json` CLI mode | Uses a stable session UUID and native resume. Registers the application MCP server as `trading` |

This is an adapter abstraction with three implementations, not automatic compatibility with every agent CLI. Local-model support depends on the selected harness and its provider protocol. The application does not translate an arbitrary Chat Completions server into Responses or Anthropic Messages.

Provider credentials and model connection details stay with the harness. The app can select a model/provider where supported, but it does not add a competing model-authentication system. The example Qwen and DeepSeek models mentioned during development are not hardcoded application requirements.

### Native settings and reasoning effort

Temperature and top-p are not set by the application. An omitted effort selector leaves effort to the native harness. An explicit effort is passed through the corresponding native mechanism:

- pi: `--thinking`.
- Codex: native turn `effort`.
- Claude Code: `--effort`.

Configuration discovery uses pi's available-model listing and model-specific thinking-level metadata, Codex's model catalog and supported effort metadata, and aliases/effort values documented by the installed Claude CLI. Discovery does not submit a generation prompt. Extra native CLI arguments that make catalog matching uncertain result in a manual-selection fallback.

Claude aliases such as those shown by its installed CLI are selectors, not pinned version numbers. Manual model identifiers are supported. The run header shows selected configuration; observed native model IDs are recorded separately for report provenance.

### Process lifecycle and budgets

The adapters stream through owned subprocesses, report startup/process failures, and use explicit cancellation paths. Forced shutdown targets the owned POSIX process group, including descendants, rather than unrelated processes. Completed stage sessions are released instead of accumulating idle processes throughout the run.

Tool-round limits are enforced at native response/tool-turn boundaries. Pi provides one final tool-free report opportunity, then aborts if the native loop continues. Codex and Claude interrupt when their tracked tool-generating rounds exceed the limit. These mechanisms are protocol-specific, not one identical implementation applied to all CLIs.

Protocol refinements made during implementation include waiting for pi's settled state through native retry activity, reconciling Codex cumulative usage and streamed/final tool output, and matching Claude final text/thinking blocks to their original stream positions. These fix local adapter behavior and should not be read as evidence that the reference's direct-provider clients had the same defects.

Sources: [shared contracts](../src/open_trading_agents/analysis/models.py), [process management](../src/open_trading_agents/harnesses/process.py), [pi](../src/open_trading_agents/harnesses/pi.py), [Codex](../src/open_trading_agents/harnesses/codex.py), [Claude](../src/open_trading_agents/harnesses/claude.py), [catalog discovery](../src/open_trading_agents/harnesses/catalog.py).

## 4. Tools, calculations, and source gathering

### Financial tools through MCP

Financial adapters are exposed through a local stdio MCP server. Its context binds the ticker, asset type, analysis date, and data configuration outside the model-visible arguments. The workflow assigns a role-specific tool list.

- Claude sees names such as `mcp__trading__get_news` and `mcp__trading__search_web` because `trading` is the MCP server name.
- Codex connects through its native MCP support.
- Pi loads an extension that lists the same MCP tool schemas, registers them with `pi.registerTool`, and forwards executions to the local server. Pi displays the underlying tool names without Claude's namespace prefix.

The MCP entry point is now `python -m open_trading_agents.dataflows.tools` after the module reorganization.

### Native scripts replace the temporary calculator

Native file/script tools are supplied by the harness, not reimplemented by the application. Pi exposes `read`, `write`, `edit`, and `bash`; Claude exposes its native Read/Write/Edit/Bash tools; Codex supplies native execution subject to its configured workspace permissions.

The rewrite temporarily added a dedicated `calculate` tool as a bounded arithmetic fallback. It was later removed at the user's request. The final code has no calculator registration or implementation. Prompts now ask agents to write and execute scripts, batch related calculations where possible, and state unverified numerical claims if execution is unavailable.

The earlier prompt hint directing agents to the application's virtualenv Python was also removed. The application's interpreter still starts its own MCP server, while an agent's script interpreter is selected through the native shell environment.

The shared prompt tells agents to keep scripts in their workspace and prohibits package installation, credential inspection, placing trades, and launching other AI clients from scripts. These are role instructions alongside each harness's permissions, not a claim that every native shell has an identical sandbox.

### Shared web research

The project added application-owned `search_web` and `fetch_webpage` tools for analysts when `web_research` is enabled:

- Search uses DDGS with an explicitly configured backend, defaulting to Yahoo. The current default does not require a search API key.
- Search returns URLs, titles, snippets, and available date information. It reports unavailable searches rather than substituting fabricated evidence.
- Page retrieval performs an HTTP request and static text extraction. It removes executable/hidden content and collects titles, links, and reported publication/modification dates.
- Fetching has response-size, redirect, time, and public-destination checks. Extracted text is capped at 20,000 characters. It is not a JavaScript browser, PDF reader, or article-focused summarization service.
- Known post-cutoff publications are excluded or withheld. Live pages and search results are not represented as historical archives.

The application currently disables Codex's native `web_search` and does not enable Claude's native WebSearch in its built-in tool list. The similarly named search observed in Claude runs is the application MCP tool. Pi obtains equivalent search functionality through its extension.

Market, fundamental, and news analysts gather role-specific financial evidence through their tools. Sentiment evidence is collected from news, StockTwits, and Reddit before the sentiment turn. Debate and decision roles consume supplied reports and arguments; they are not assigned the analysts' financial/web tool list, although native calculation scripts remain available when enabled.

### Social screening

Optional social-post screening was retained conceptually, but the reference's separate screening-service path was replaced by a native-harness session. The current role returns validated relevance/stance/confidence entries tied to the original post indices. Screening failures retain the source posts instead of silently discarding them.

Sources: [reference tool binding][ref-tools], [reference screening][ref-screening], [current MCP tools](../src/open_trading_agents/dataflows/tools.py), [pi bridge](../src/open_trading_agents/harnesses/pi_extension.mjs), [web retrieval](../src/open_trading_agents/dataflows/web.py), [workflow tool assignment](../src/open_trading_agents/analysis/workflow.py).

## 5. CLI, configuration, and presets

### Final entry points

The application originally introduced an `analyze` command during this rewrite. The final interaction is:

```sh
uv run open-trading-agents configure
uv run open-trading-agents
```

The root command starts analysis. `--ticker` and `--date` can bypass their questions. `configure`, `resume`, `backtest`, and `doctor` remain separate commands.

Changes made during the session:

- Removed the `analyze` subcommand after moving its options to the root command.
- Removed the standalone `replay` command. Event replay still exists internally for resume and display recovery.
- Disabled the generated shell-completion install/show options.
- Added a configure-first gate: without any saved preset, a new analysis exits with an instruction before asking for a ticker or launching work. Help and maintenance commands remain accessible.
- Kept headless and JSON-event runs. They skip the startup menus and use saved settings plus explicit flags.

### Final run wizard

The final sequence is ticker, analysis date, research depth, working-agent profile, and decision-manager profile. The date defaults to the machine's local date, accepts historical dates, and rejects invalid/future dates. `--date` skips that question. The date prompt was accidentally removed during the wizard refactor and subsequently restored.

Choices use arrow keys and Enter, with typing to filter. Numeric menu entry was replaced with Questionary, which the reference also used. The initial selection carries a `(default)` suffix, while a pointer indicates the current cursor position. Reverse-video white highlighting was removed. The menu label `Configure for this run` has no trailing ellipsis.

Working agents and decision managers can select different saved profiles or custom settings for the current run. The manager can use the working-agent selection. Custom run choices do not rewrite saved profiles.

### Final configuration model

Defaults moved from the rewrite's initial current-directory `tradingagents.toml` and `~/.local/share/open-trading-agents` storage to:

| Location | Purpose |
| --- | --- |
| `~/.open-trading-agents/config.toml` | Application defaults |
| `~/.open-trading-agents/presets/NAME.toml` | Saved profiles |
| `~/.open-trading-agents/runs/RUN_ID/` | Analysis artifacts |
| `~/.open-trading-agents/memory.sqlite` | Ordinary-run decision memory |
| `~/.open-trading-agents/backtests/RUN_ID/` | Isolated backtest state |

The pinned reference used its own `~/.tradingagents` configuration/log/cache conventions. Storage compatibility with the reference was explicitly not required. Changing the default path did not implement an automatic relocation of all older run directories.

A saved profile now contains only `harness`, `provider`, `model`, and `effort`. An earlier three-tier preset design and the run-time `--preset` option were replaced. Profiles are now selected in the run wizard. Old intermediate preset shapes are not automatically migrated.

The configure flow edits one profile, optionally saves it by name, and can mark it as the default. Existing profiles can be deleted through the same menu. Deleting the default clears matching markers in the selected and global config, preserves unrelated files, and attempts to restore config changes if deletion fails. Invalid profile files can also be removed through this path.

TOML updates preserve unrelated values, comments, and existing permissions. Writes use temporary files and replacement rather than partially rewriting the live file. Default profile selection and explicit CLI override precedence are handled consistently. Claude provider overrides are rejected because Claude manages that setting natively.

### Remaining setup differences from the reference

The current wizard does not reproduce every reference setup question:

- Output language is configured with `output_language`, rather than a startup language menu.
- Analyst selection comes from config or `--analysts`, rather than a startup checkbox menu.
- The reference automatically remembered prior language, team, depth, provider, and model selections in CLI preferences. Current per-run choices are temporary; explicit config and profiles supply future defaults.

These are current differences, not implemented features waiting to be discovered in the UI.

Sources: [reference startup][ref-selections], [reference preferences][ref-prefs], [reference defaults][ref-defaults], [current CLI](../src/open_trading_agents/cli.py), [configuration storage](../src/open_trading_agents/config.py), [wizard](../src/open_trading_agents/ui/configuration.py).

## 6. Terminal interface redesign and refinements

The reference already had selected analyst panes, fullscreen, scrolling, timers, and streamed reasoning. The following describe the Textual replacement and subsequent refinements made in this session.

### Stage and pane layout

- Added seven persistent stage tabs, navigable with Tab/Shift+Tab without losing another stage's scroll position.
- Kept the initial analyst grid, supporting one through four selected analysts.
- Changed debates to a shared chronological conversation instead of repeatedly rendering a participant's accumulated input.
- Added a narrow left column of stacked participant system-prompt panes and a substantially wider right transcript column.
- Populate all five debate system prompts immediately, using the same prompt builder and configured language as the sessions. Actual recorded system events remain authoritative on resume.
- Left/Right selects debate panes. Up/Down scrolls the focused pane. Mouse focus is also supported.
- `f` toggles fullscreen for an analyst, a debate prompt, or the debate transcript. Enter/Esc fullscreen aliases and separate restore-grid hints were removed.
- Fullscreen state is independent between debate stages. Restoring panes preserves reading and collapse state.

### Conversation rendering

- Shared debate evidence appears once. Subsequent participant contributions are appended as separate role/round/turn bubbles.
- Whole turns, inputs, reasoning, and tool details can collapse. Hidden content continues to receive streaming updates.
- Headers remain visible while scrolling inside a contribution, including their collapse control.
- Supplied inputs, exposed reasoning, output, tool arguments, and tool results have distinct presentation. The interface does not invent reasoning that the harness did not expose.
- Text is rendered literally rather than letting tool/model strings introduce unintended terminal markup.
- Stream fragments, snapshots, final native messages, and durable completed outputs are reconciled to prevent duplicate or truncated contributions. Reused native IDs are scoped to their role turn.
- Full text carried by recorded prompt, model, and tool events is retained. Viewport shortening does not alter that stored source. Native CLIs can limit their own output, and diagnostic stderr is bounded separately; the app does not claim to recover content omitted by a harness.

### Controls, status, and spacing

- Tab titles use Title Case.
- Running tabs and active turns use animated circular dots instead of `[run]` or the word `streaming`.
- Completed tabs, agent turns, and analyst status use checkmarks.
- The run header uses a dot before the ticker and shows total elapsed time plus the selected working/manager harness, provider, model, and effort. Identical configurations are shown once.
- Removed the duplicated `Running: <stage>` header text because the active stage is already indicated.
- Renamed section navigation to `prev` and `next`. Brackets select a disclosure; Space toggles it.
- Removed the `d` request/payload view, `s` control, full-text-path hints, and filler preview messages.
- Shortcut hints appear only where applicable. End is described as following the latest output, not as ending the run.
- Removed the final trailing separator and redundant debate border/margin space. Scrollbar space remains available where needed, with no outer-screen scrollbar or clipped footer at tested compact sizes.

### Scroll and responsiveness corrections

- Scrolling upward pauses following. Reaching the bottom through ordinary scrolling, or pressing End, resumes it.
- Disclosure changes and width/height changes preserve logical reading anchors.
- Transcript indexing and row caches avoid re-rendering full history on each token. Clock and spinner updates do not reindex conversation bodies.
- Tool results over 10,000 characters display the first 6,000 and last 2,000 with an exact omitted count. This is a viewport preview rule, not deletion of stored tool output.
- Event recording, large replay, and report export were moved off the immediate rendering path where needed.
- Later fixes corrected height-only follow timing, refresh callbacks during shutdown, lifecycle status/clock refresh timing, and collapse hit-testing after debate borders shifted widget coordinates.

Sources: [reference streaming UI][ref-streaming], [reference controls][ref-controls], [current app](../src/open_trading_agents/ui/app.py), [chat rendering](../src/open_trading_agents/ui/chat.py), [transcript rendering](../src/open_trading_agents/ui/transcript.py).

## 7. Usage and native model provenance

The reference already displayed input/output totals, calls, and time. The current project adds a common per-request usage representation across the three native protocols.

- Usage snapshots replace earlier snapshots for the same role/request instead of being added repeatedly.
- Cumulative totals are converted to increments where needed, and streamed/final duplicate records are reconciled.
- Input totals include cache reads and writes. Cache counts are subsets, not extra input to add again.
- Reported reasoning tokens are already part of output totals.
- Unknown cache fields are distinct from explicitly reported zero. Unreported cache-write fields are omitted from the UI. Partial information is marked separately.
- Native auxiliary activity, including exposed compaction usage, can be included when reported.
- Per-agent and total usage is saved to `reports/usage.json`.

Native model/provider identities are captured from the adapters, deduplicated per agent, retained across resume, and kept when a model changes during a run. Missing identities are not replaced with guessed model names. Reports distinguish observed runtime values from configured fallbacks, and provenance avoids copying credentials or endpoint fields.

Sources: [usage normalization](../src/open_trading_agents/reporting/usage.py), [model event capture](../src/open_trading_agents/harnesses/process.py), [report metadata](../src/open_trading_agents/reporting/reports.py).

## 8. Decisions and report export

### Structured decisions

The reference already defined structured decision schemas and attempted provider-native structured output, with a free-text fallback. The current implementation validates the final text returned through a native harness instead of binding a provider-specific output mode in the application.

- Research Manager: `recommendation`, `rationale`, `strategic_actions`.
- Trader: `action`, `reasoning`, and optional entry, stop, and sizing fields.
- Portfolio Manager: `rating`, `executive_summary`, `investment_thesis`, and optional target/horizon fields.
- The five-level final rating scale and three-level trader action scale are retained.
- Optional absolute prices are normalized without treating percentages or ranges as valid prices.

An early rewrite prompt named decision fields but supplied the exact schema only after validation failed. The final code appends the authoritative schema to the initial system instructions. A malformed result gets one correction request in the same session, including specific JSON/field errors and an instruction to preserve the substantive conclusion. If validation still fails, the result is marked for human review rather than extracting a rating from arbitrary prose.

### Human reports

The reference's five groups are retained:

1. Analyst Team Reports.
2. Research Team Decision.
3. Trading Team Plan.
4. Risk Management Team Decision.
5. Portfolio Manager Decision.

Individual Markdown outputs and `complete_report.md` / `complete_report.html` are exported under each run's `reports/` directory. The HTML presentation is adapted from the reference, with this project's branding and fixes for narrow-screen numeric tables. The reference's inert HTML rendering approach is retained.

Changes from the reference interaction include automatic combined export during cleanup, available partial outputs after interruption, and loading the saved complete Markdown into the Reports tab. No explicit export button, save-destination question, HTML toggle, or browser-opening offer was added to the TUI.

Chronological debate entries use recorded speaker/round/turn events when they reconstruct the full debate exactly. Otherwise the full combined discussion is retained, avoiding content loss or guessed speaker attribution.

The reference's human reports were already free of token statistics. Token counts temporarily appeared in this project's early report output, then were removed again at the user's request. The final human reports include model/harness/vendor provenance, while token details remain in the TUI and `usage.json`.

Sources: [reference schemas][ref-schemas], [reference structured calls][ref-structured], [reference reports][ref-reports], [reference HTML][ref-html], [current decisions](../src/open_trading_agents/analysis/decisions.py), [report writer](../src/open_trading_agents/reporting/reports.py), [HTML renderer](../src/open_trading_agents/reporting/html.py).

## 9. Persistence, interruption, and feedback

### Run recovery

The reference's graph checkpoints were optional and disabled by default, with checkpoint and clear-checkpoint CLI controls. Current analyses always create a run ledger containing metadata, normalized events, completed step outputs, and native session references. There is no corresponding checkpoint toggle. Events are mirrored to `transcript.jsonl` through a batched background writer.

Completed steps are reused. An unfinished task already acknowledged by the native harness continues its saved session. If acknowledgement was never recorded, the original task is submitted again. That interruption window is explicitly at-least-once, not an exactly-once guarantee.

Resume validates the original inputs, protects the saved tool context from incompatible changes, and holds a per-run execution lock. Finalization closes sessions and flushes persistence before declaring completion. UI replay yields control rather than blocking on the entire history, and completed public debate outputs can rebuild bubbles after an interrupted event flush.

### Outcome memory

The reference already had an outcome/reflection loop, including settlement alongside analysts and historical-cutoff filtering. The current implementation replaces its Markdown-oriented decision storage with SQLite:

- Decisions are identified by run ID, so distinct runs on the same ticker/date remain distinct.
- Each record retains the original rating/report and its evaluation horizon and benchmark.
- Numeric returns, benchmark-relative returns, resolution date, reflection, status, and retry information are stored directly.
- Renewable per-record claims prevent concurrent processes from generating the same reflection, and failed or interrupted evaluations remain retryable.
- Reflection uses a native harness session instead of a direct model client.

The scope of feedback is preserved: the Portfolio Manager receives up to five same-ticker records and three cross-ticker lessons whose outcomes were known by the analysis cutoff. It is prompt-context feedback, not model training or automatic rewriting of the application prompts. Analysts and debaters do not receive this memory context.

`pending` means the saved decision has not completed outcome evaluation and reflection. Later analysis runs check pending records. There is no background scheduler or separately added settlement CLI command. The reference's optional memory-log rotation setting has not been reproduced in the SQLite implementation.

### Backtests

The reference already evaluated independent recommendations over calendar-date grids and compared outcomes with benchmarks. That model remains in place, including fixed supplied holdings and no simulation of fills, costs, or evolving portfolios.

The rewrite adds deterministic per-cell run IDs, full native-session/workflow resume for interrupted cells, pinned input/configuration manifests, a sweep lock, isolated caches as well as memory, and saved JSON/Markdown summaries. Repeating a sweep requires matching inputs. New or unfinished cells use model calls; completed analysis cells are skipped. Pending outcome reflections can still be evaluated during settlement.

Regional benchmark mappings were retained, and a run can set `benchmark_ticker`. The reference also exposed a configurable `benchmark_map`. Although the current low-level benchmark helper accepts such a map, ordinary application runs do not wire a custom map through `Settings` into decision recording. That configuration capability has not been reproduced.

Sources: [reference memory][ref-memory], [reference settlement][ref-settlement], [reference backtests][ref-backtest], [run ledger](../src/open_trading_agents/persistence/storage.py), [event writer](../src/open_trading_agents/persistence/events.py), [current memory](../src/open_trading_agents/persistence/memory.py), [current backtests](../src/open_trading_agents/analysis/backtest.py).

## 10. Testing, maintenance, and project structure

The requested test emphasis became installed-pi integration, with Codex/Claude supported through protocol/contract tests and metadata checks. This does not claim equivalent end-to-end certification across every harness version and model endpoint.

The current validation includes:

- Offline workflow, data-integrity, persistence, UI, configuration, reporting, and adapter tests.
- Subprocess protocol fixtures for all three adapters.
- Installed pi tests against a loopback fixture server, including native default/explicit effort, preserved history and reasoning, isolated sessions, script execution, usage, and tool-budget enforcement.
- Actual batched Python calculations through pi's native `bash`, after the dedicated calculator was removed.
- Wide/compact UI geometry, mouse and keyboard disclosures, sticky titles, fullscreen restoration, scroll anchors, and bounded rendering checks.
- A built-wheel check outside the source checkout for imports, CLI, the pi extension asset, and a real local MCP exchange.

The latest full validation recorded before this document was 285 default-suite tests and three native pi tests, plus lint/build and installed-wheel checks. These are snapshot results, not a promise about future changes.

### Code review corrections

A fresh-context no-slop review produced the following implemented corrections:

| Finding | Implemented correction | Origin |
| --- | --- | --- |
| Two-decimal formatting could turn small prices/indicators into zero | Removed OHLCV rounding and retained scalar precision in verified snapshots | Inherited adapter behavior |
| Empty historical windows and missing dates could be called weekends/holidays | Reject empty post-cutoff price frames and report absent observations without inventing calendar explanations | Inherited adapter behavior |
| Claude provider overrides could be accepted but ignored | Centralized the restriction in `HarnessSettings` validation | Local configuration inconsistency |
| Tool-specific vendor overrides disappeared from reports | Exported `tool_vendors` alongside category vendors | Local provenance omission |
| Indicator descriptions were duplicated across providers | Extracted shared vendor-independent descriptions | Inherited duplication |
| Workflow data merging duplicated the data layer and could share mutable values | Added one defaults-plus-overrides helper with independent values | Local assembly duplication |
| Optional-date annotations and complex contracts were inaccurate or unclear | Corrected nullable types and Google-style contract documentation | Mostly adapted modules |
| Dense signatures, split literals, and unseparated phases reduced readability | Expanded long signatures, consolidated literals, and separated logical code phases | Both adapted and new modules |
| Embedded subprocess fixtures were compressed scratch-style strings | Extracted readable standalone fixtures without replacing their distinct protocols | Local test code |

Validation also exposed and fixed a UI lifecycle timing issue so failure/completion status does not depend on the next clock tick.

### Module organization

The initially flat package was regrouped by responsibility. The root now contains `__init__.py`, `__main__.py`, `cli.py`, and `config.py`.

| Group | Modules or responsibility |
| --- | --- |
| `analysis/` | Workflow, prompts, decisions, backtests, and shared models formerly in `domain.py` |
| `ui/` | App formerly in `tui.py`, chat, transcripts, and configuration menus |
| `persistence/` | Run storage, event recording, and memory |
| `reporting/` | Report writer, HTML renderer, and usage summaries |
| `harnesses/` | Native adapters, shared process code, model discovery, and pi extension |
| `dataflows/` | Existing provider groups, shared data helpers, and the MCP tool server |

Imports, string-based test patches, and executable MCP module paths were updated together. The CLI and serialized run formats were preserved. Unnecessary package initializers and compatibility wrappers for the old module layout were not added.

## 11. Intermediate choices that were superseded

These changes occurred in the conversation but should not be mistaken for current functionality.

| Intermediate state | Final state |
| --- | --- |
| Considered moving the application to Node.js | Python application retained; Node used for the pi extension |
| `analyze` subcommand | Bare root command starts the run wizard |
| Standalone `replay` command | Removed; recovery still replays events internally |
| Shell completion install/show options | Removed |
| Current-directory config and `.local/share` run default | Global `~/.open-trading-agents/` config, profiles, and runs |
| Presets contained shared/quick/deep bundles | Each profile contains one harness/provider/model/effort selection |
| `--preset NAME` on run commands | Removed; profiles selected for working agents and managers in the wizard |
| Tier assignment inside configure | Moved to the run wizard; configure manages individual profiles |
| Numeric menu choices | Arrow/Enter menus with filtering |
| White default highlight and a trailing menu ellipsis | Explicit `(default)` suffix, movable pointer, plain `Configure for this run` label |
| Date silently defaulted during one wizard revision | Date prompt restored after ticker, with today as default and `--date` bypass |
| Dedicated calculator and verbose calculator results | Calculator removed; native script execution with batching guidance |
| Prompt supplied the app virtualenv Python path | Interpreter choice left to the native shell |
| Repeated debate inputs and inline debate system sections | Shared input once, chronological turns, and separate prompt panes |
| `[run]` / `streaming` status text and text completion labels | Animated dots and completion checkmarks where applicable |
| `d`, `s`, Enter/Esc fullscreen aliases, verbose path/preview hints | Contextual controls with `f` as the fullscreen toggle |
| Debate prompts waited for each role's first event | All prompt panes populated immediately, reconciled with recorded events |
| Only prompt panes could fullscreen within a debate | Both prompt and transcript panes can fullscreen |
| Tokens appeared in early human reports | Human reports omit them; TUI and `usage.json` retain usage |
| `tradingagents.example.toml` | Unused template deleted |
| `THIRD_PARTY_NOTICES` | Credit consolidated into README, retaining LICENSE and source notices |
| Long README with agreed requirements and implementation history | Short user README plus configuration, usage, data-source, and development guides |

## 12. Source lineage, documentation, and boundaries

Apache-2.0 attribution was retained while moving the visible credit into the README. The license text and existing adapted-source notices remain. The source archive includes the separate guides under `docs/`.

The reference's update/announcement mechanism was not carried into this application. Alpha Vantage requests also no longer add the reference-specific `source=trading_agents` query tag. That request-tag change is separate from source-code attribution, which is retained. No application telemetry or direct model-client layer was introduced. Dependency and native-harness telemetry remained outside the requested scope; this is not a claim that every external dependency is telemetry-free.

The following remain outside the final implementation or are known limits discussed in the session:

- No model training, automatic strategy optimizer, or background memory scheduler.
- No broker/order execution and no portfolio-trading simulation in backtests.
- No universal compatibility with all local-model API protocols.
- No guaranteed complete historical news, social, or company-fundamental coverage.
- No claim that all international markets have equally complete research sources. Price snapshot currency metadata remains incomplete.
- No reproduction of the reference's automatic CLI-preference persistence or memory-log rotation.
- No equivalent centralized upfront instrument-identity lookup, complete crypto-suffix classification parity, or application-configurable benchmark map.
- No startup language/team picker beyond the existing config/flag controls.
- No use of hidden reasoning that a harness does not expose.
- No claim that the suspected reference history/payload behavior was conclusively diagnosed or repaired in the original codebase.

The later questions about web-search API keys, page postprocessing, native write/bash tools, Claude aliases, non-US symbols, backtest scoring, and pending memory clarified the existing implementation. They did not independently add new capabilities unless a corresponding change is described above.

## 13. Commit guide

This is a feature-oriented index rather than a dump of every commit.

| Change group | Representative implementation commits |
| --- | --- |
| Project, contracts, native pi, storage, and orchestration | `1061cf8`, `d24cd4f`, `2a731e8`, `c53bb56`, `797cf5d` |
| Ported sources, roles, MCP, Codex, and Claude | `6f0b050`, `85d6a5e`, `85ecbc7`, `82b9979` |
| Feedback, isolated backtests, social screening | `4902f50`, `34070f7`, `524687d` |
| Native scripts, web tools, and process lifecycle | `79c4277`, `10b83f1`, `46ba54e`, `4307b1c` |
| Delivery acknowledgement and interrupted-run recovery | `6a570ff`, `87cb8ce`, `7f9ee40` |
| Chat bubbles, shared turns, scroll anchors, and terminal errors | `895c6ff`, `53c0fb3`, `d33395a`, `2b8f2da`, `0ce315b`, `27e1fdc` |
| Usage reconciliation and bounded previews | `5ed028f`, `52dda2d`, `ea50d41`, `f636280`, `12f8e5b` |
| Contextual controls, stage indicators, and sticky titles | `8170ce4`, `f26ee58`, `f66e933`, `0a017d6`, `7d2e9a1` |
| Report grouping, HTML, and observed model provenance | `eedbf04`, `50fbd39`, `81bac86`, `2d46dbc`, `309f899` |
| Global config, catalog discovery, and keyboard menus | `45351ee`, `c81b6d5`, `2a42043`, `3206752`, `6c3268e` |
| Single-agent profiles, effort, and run-time role selection | `568faeb`, `315d476`, `41f996b`, `9b9af6e`, `ef75b96` |
| Root CLI, preset gate, and calculator removal | `093d8ca`, `9df0449`, `5e709ca` |
| Manager schema, debate columns, and collapsible turns | `0d56eac`, `82a8f20`, `4da3d7c`, `bff07aa` |
| Date, header, click targets, prompt prefill, and transcript fullscreen | `5175c65`, `fff58d8`, `a9874bb`, `35903d1` |
| Default suffixes, profile deletion, and unused-file cleanup | `96c39d7`, `62dc060`, `24231eb`, `acb1f94`, `30fad91` |
| Review-driven data/configuration fixes and shared logic | `6e1bc53`, `223ebb2`, `2fcf781`, `f2ba2f0`, `e365367` |
| Maintainability, grouped modules, and concise documentation | `a9474ce`, `d7cc56c`, `c45ff37` |

[ref]: https://github.com/TauricResearch/TradingAgents/tree/e8f5503e9b86d5c2710b0c76b1f686703968c9df
[ref-project]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/pyproject.toml
[ref-graph]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/graph/trading_graph.py
[ref-setup]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/graph/setup.py
[ref-agents]: https://github.com/TauricResearch/TradingAgents/tree/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/agents
[ref-tools]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/agents/tools.py
[ref-dates]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/dataflows/date_window.py
[ref-context]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/agents/context.py
[ref-cli-prompts]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/cli/prompts.py
[ref-screening]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/agents/post_screen.py
[ref-selections]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/cli/selections.py
[ref-prefs]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/cli/prefs.py
[ref-defaults]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/default_config.py
[ref-streaming]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/cli/streaming.py
[ref-controls]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/cli/pane_controls.py
[ref-schemas]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/agents/schemas.py
[ref-structured]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/agents/structured.py
[ref-reports]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/reporting.py
[ref-html]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/report_html.py
[ref-memory]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/memory/log.py
[ref-settlement]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/memory/settlement.py
[ref-backtest]: https://github.com/TauricResearch/TradingAgents/blob/e8f5503e9b86d5c2710b0c76b1f686703968c9df/tradingagents/backtest.py
