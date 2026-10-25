# Configuration and profiles

## Create and manage profiles

Set up model access through your chosen harness first. Then run:

```sh
uv run open-trading-agents configure agents
```

Choose **Create or update profile**, select the harness, provider, model, and reasoning effort, then enter a preset name. Interactive configuration always saves a named preset. The final question asks whether to use it by default. The first setup goes directly to profile creation.

Running `configure` without a subcommand offers **Agents**, **Data sources**, and **Portfolio**. Use `configure data` to enter data-source API keys and SEC contact information; these settings are stored in the same `config.toml`. See [data-source setup](data-sources.md#source-configuration). Use `configure portfolio` to edit the saved holdings described in [portfolio context](usage.md#portfolio-context).

If the name already exists, confirm replacement or enter a different name. Declining to make a new preset the default still saves it and retains your existing default.

Use Up/Down to move the pointer and Enter to confirm. Inside a group or role’s profile menu, the colored `(selected)` badge identifies its current assignment. Moving the pointer does not move that badge. Selecting a profile for a run does not change your saved default. Ctrl+C cancels setup without saving.

To remove a preset, choose **Delete saved profile**, select its name, and confirm. Deleting the default clears its default selection in the selected and global configuration. Other presets and saved runs are retained.

## Model selection

- When creating a profile, **Harness default** uses the harness's own selection. In pi, choosing the harness default for the model also resets the provider to its harness default.
- Providers, models, and effort levels are listed when the harness exposes them. Custom model identifiers can be entered manually.
- `claude-code` launches the native `claude` executable. Existing presets using `claude` still load. Claude Code's listed model choices are aliases. To select a specific version, enter its full model identifier manually. Claude manages its provider through its native settings.
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

Profiles are selected independently for the working agents and decision managers. Run-specific choices apply only to that run. A named default profile supplies the initial selection for both groups and supersedes the main config's harness/tier settings. Individual agent assignments override their group selection. Explicit CLI harness, provider, model, effort, and round options take precedence over the wizard's choices, including individual assignments.

`--config PATH` selects another application configuration file. Profiles, shared cache, and feedback memory remain under `~/.open-trading-agents/`. The run wizard's output-directory choice or `--output-dir PATH` changes where that run's folder is created. Without an override, runs are saved under `~/.open-trading-agents/runs/`. Resume uses the choices saved with that run.

Selection flags can update the main config without opening the wizard:

```sh
uv run open-trading-agents configure agents --harness pi --provider local --model my-model --effort high
```

These flags do not create a named preset. Use the interactive wizard to save one before the first analysis. Empty `--model ""`, `--provider ""`, or `--effort ""` values reset the corresponding selector to its harness default.

## Run profile selection

Choose **Simple** or **Advanced** before selecting profiles. Each mode opens an editable list. Select a row to change its profile, return to the list, and choose **Continue** at the bottom when finished. Rows display the selected preset name, or the harness, model, and effort when no saved preset matches. After opening a group or role to choose its profile, a short explanation appears directly below the title and wraps to the terminal width. The overview lists have no descriptions.

**Simple** has two rows: **Working agents** and **Decision manager**. It uses only those group assignments for the current run, clearing individual overrides from that run's settings. When no separate manager profile is set, the decision manager follows the working profile. Choosing an independent profile fixes that assignment. **Same as working agents** restores the link.

**Advanced** opens the individual-agent list directly. Edit any analyst, bull or bear researcher, trader, risk participant, manager, or reflection agent. Choose a saved profile or **Configure for this run**. The current assignment is marked `(selected)` inside the profile menu.

These choices are saved with the run for resume and do not modify your presets or configuration file. Cancelling either editor discards its changes.

To retain individual assignments across runs, including unattended runs, add role sections to `config.toml`:

```toml
[agents.bull]
kind = "claude-code"
model = "sonnet"
effort = "high"

[agents.bear]
kind = "codex"
model = "my-model"

[agents.portfolio_manager]
model = "my-manager-model"
```

Omitted fields inherit from the role's group profile when using the same harness. Role keys are `market`, `social`, `news`, `fundamentals`, `bull`, `bear`, `research_manager`, `trader`, `aggressive`, `conservative`, `neutral`, `portfolio_manager`, and `reflection`. Configuring an analyst or supporting role does not enable it if it is disabled elsewhere.

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
kind = "claude-code"
model = "opus"
effort = "high"
```

Provider credentials remain in the native harness. Financial-data credentials are described in [data sources](data-sources.md).
