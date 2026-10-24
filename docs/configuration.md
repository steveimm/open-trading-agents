# Configuration and profiles

## Create and manage profiles

Set up model access through your chosen harness first. Then run:

```sh
uv run open-trading-agents configure
```

Choose **Create or update profile**, select the harness, provider, model, and reasoning effort, then save a named preset. The first setup goes directly to profile creation. The wizard can also mark a preset as the default.

Use Up/Down to move the pointer, type to filter choices, and Enter to confirm. The initial choice has a `(default)` suffix. Ctrl+C cancels setup without saving.

To remove a preset, choose **Delete saved profile**, select its name, and confirm. Deleting the default clears its default selection in the selected and global configuration. Other presets and saved runs are retained.

## Model selection

- **Harness default** uses the harness's own selection. In pi, choosing the harness default for the model also resets the provider to its harness default.
- Providers, models, and effort levels are listed when the harness exposes them. Custom model identifiers can be entered manually.
- Claude's listed model choices are aliases. To select a specific version, enter its full model identifier manually. Claude manages its provider through its native settings.
- Effort is passed to the harness only when selected explicitly. Temperature and top-p remain controlled by the harness and provider.

Native protocol support matters when connecting to a local model server. A harness expecting Responses or Anthropic Messages is not automatically compatible with a Chat Completions endpoint.

## Files and precedence

| File or directory | Purpose |
| --- | --- |
| `~/.open-trading-agents/config.toml` | Application defaults |
| `~/.open-trading-agents/presets/NAME.toml` | Saved agent profiles |
| `~/.open-trading-agents/runs/` | Default analysis run and report location |

A profile contains only four selectors:

```toml
harness = "pi"
provider = "my-local-provider"
model = "my-model"
effort = "high"
```

Profiles are selected independently for the working agents and decision managers. Run-specific choices apply only to that run. A named default profile supplies the initial selection for both groups and supersedes the main config's harness/tier settings. Explicit CLI harness, provider, model, effort, and round options take precedence over the wizard's choices.

`--config PATH` selects another application configuration file. Profiles, shared cache, and feedback memory remain under `~/.open-trading-agents/`. The run wizard's output-directory choice or `--output-dir PATH` changes where that run's folder is created. Without an override, runs are saved under `~/.open-trading-agents/runs/`. Resume uses the choices saved with that run.

Selection flags can update the main config without opening the wizard:

```sh
uv run open-trading-agents configure --harness pi --provider local --model my-model --effort high
```

These flags do not create a named preset. Use the interactive wizard to save one before the first analysis. Empty `--model ""`, `--provider ""`, or `--effort ""` values reset the corresponding selector to its harness default.

## Application preferences

Edit `config.toml` for settings outside the profile picker:

| Setting | Default | Effect |
| --- | --- | --- |
| `output_language` | `"English"` | Language of agent reports |
| `analysts` | `["market", "social", "news", "fundamentals"]` | Analyst team, also overridable with `--analysts` |
| `research_rounds` | `1` | Bull/bear debate rounds |
| `risk_rounds` | `1` | Risk debate rounds |
| `memory_enabled` | `true` | Use prior evaluated decisions as context |
| `holding_period_days` | `5` | Trading sessions used to evaluate outcomes |
| `benchmark_ticker` | Regional default | Reference instrument for outcome scoring |
| `allow_execution` | `true` | Permit native script execution |
| `web_research` | `true` | Offer search and page retrieval to analysts |
| `social_screening` | `false` | Use an additional agent to screen social posts |

The fundamentals analyst is omitted for crypto. Language and analyst selection are configured here rather than asked in the run wizard.

Without a named default profile, `[harness]` sets shared runtime choices. Optional `[quick]` and `[deep]` sections override the working agents and decision managers:

```toml
output_language = "Korean"
analysts = ["market", "news", "fundamentals"]

[harness]
kind = "pi"
provider = "local"
model = "working-model"

[deep]
kind = "claude"
model = "opus"
effort = "high"
```

Provider credentials remain in the native harness. Financial-data credentials are described in [data sources](data-sources.md).
