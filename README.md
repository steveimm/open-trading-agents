# Open Trading Agents

A Python application for financial research with independently managed agent sessions.

The application coordinates analysis, research debate, a research decision, a trader proposal, risk debate, a portfolio decision, and reports. Pi, Codex, and Claude Code CLI adapters own their model connections, authentication, conversation history, and tool loops.

Development is in progress. The project is managed with uv.

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

The reference is `/home/steve/Git/TradingAgents`. Its source and development transcript informed this design. The application produces research and recommendations rather than placing orders.
