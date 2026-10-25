# Data sources and coverage

## Financial and web sources

Default data sources include Yahoo Finance for prices, indicators, profiles, and news; SEC EDGAR for eligible filings; FRED for macroeconomic series; Polymarket for prediction-market evidence; and Reddit/StockTwits for social evidence. Alpha Vantage is available as an alternative for supported financial tools.

Analysts can also use `search_web` and `fetch_webpage`. Search uses the configured backend, with Yahoo as the default. Page retrieval extracts text, titles, links, and reported dates from static HTML. It does not run JavaScript or read PDFs.

Agents use native harness tools to write and run scripts for calculations. The prompt asks them to batch related calculations where possible. Scripts use the harness's environment. Set `allow_execution = false` to disable them, or `web_research = false` to omit the additional search/page tools.

Data-tool results include paths to exact evidence files in the agent's `data/` directory. CSV results with source-comment headers also get a clean CSV file for direct parsing. Agents can read these files from calculation scripts instead of transcribing tool output. The complete evidence remains in the tool response.

SEC financial statements use JSON records labelled by period and metric. Units are explicit, values retain their original precision, and missing observations are `null`. Quarter and year-to-date periods are identified separately.

## Source configuration

Open the data-source editor with:

```sh
uv run open-trading-agents configure data
```

Open **Credentials** to edit FRED, Alpha Vantage, or SEC EDGAR access. Open **Source selection**, choose a category such as **Prices**, then use **Space** to check or uncheck providers. **Enter** returns to the category list after validation. At least one source must remain enabled. Fundamentals also require Yahoo Finance or Alpha Vantage for company profiles, because SEC EDGAR only covers statements. Enabled sources use a fixed fallback order: SEC EDGAR, Yahoo Finance, then Alpha Vantage where supported. Unchecked providers are not used. Choose **Back**, then **Save** to apply your changes. API keys appear as `****abcde`, showing only their last five characters. Short keys remain fully masked. Blank input keeps the existing value. **Remove** clears it. Ctrl+C discards unsaved changes.

To enable Alpha Vantage, enter its API key and check it in the categories you want to use. Confirming a category replaces any individual tool overrides in that category. Saving puts Yahoo Finance ahead of Alpha Vantage when both are enabled, including individual tool overrides. Other settings are preserved.

These settings live in `~/.open-trading-agents/config.toml`:

```toml
[data]
fred_api_key = "your-fred-key"
alpha_vantage_api_key = "your-alpha-vantage-key"
sec_edgar_user_agent = "Your Name you@example.com"
```

| Field | Purpose |
| --- | --- |
| `fred_api_key` | Required for FRED macroeconomic queries |
| `alpha_vantage_api_key` | Required only for tools configured to use Alpha Vantage |
| `sec_edgar_user_agent` | Your name and contact email for SEC requests, not an API key |

Request keys through [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) or [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

Unset fields can be omitted. `configure data --config PATH` edits another configuration file. Runs started with `--config PATH` use that file's credentials, and resumed runs retain the same file location. Key changes are read on subsequent data requests.

The application no longer reads `.env` or the former data-source environment variables. Model authentication remains configured in the native harness. API keys stay in the selected configuration file and are excluded from run settings, tool-context files, transcripts, and reports.

Run `uv run open-trading-agents doctor` to see which data credentials are configured or missing. It checks local setup without displaying keys or making provider requests.

The wizard saves category selections in `[data.data_vendors]`. You can also edit custom fallback chains or individual tool overrides in `config.toml`:

```toml
[data]
web_search_backend = "yahoo"

[data.data_vendors]
fundamental_data = "sec_edgar,yfinance"
news_data = "yfinance"

[data.tool_vendors]
get_news = "alpha_vantage"
```

Comma-separated vendors are tried in the configured order. A tool-specific override takes precedence over its category. Fallback stays within the explicitly configured supported vendors.

## Coverage limits

Use Yahoo Finance ticker formats, including exchange suffixes for non-US listings. Source coverage varies by instrument and market. The default social and macro sources are more US-centric than the price feed.

The Sentiment Analyst receives up to 30 StockTwits messages and five Reddit posts per subreddit within the requested date window. Post bodies include their first 700 characters, with an omission marker for longer text. Reddit titles are retained in full. These excerpts go directly to the analyst.

Company profiles can distinguish quote and reporting currencies, but price snapshots currently lack explicit quote-currency metadata.

Analysis dates are information cutoffs. Tools exclude data known to be after the cutoff and report unavailable data explicitly. Live news, social feeds, search results, and web pages are not historical archives. A reported publication date alone does not prove that today's page content was unchanged on a past date.

Yahoo company profiles and financial statements are withheld for historical runs when the provider cannot establish when the information became available. SEC statements use filing dates for supported US filers. Yahoo indicator/snapshot history currently covers five years relative to today.

The verified market snapshot preserves source numeric precision, but prices and indicators can still be unavailable when the source is missing, stale, or unreachable. Missing observations do not establish that a market was closed.
