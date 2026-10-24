# Data sources and coverage

## Financial and web sources

Default data sources include Yahoo Finance for prices, indicators, profiles, and news; SEC EDGAR for eligible filings; FRED for macroeconomic series; Polymarket for prediction-market evidence; and Reddit/StockTwits for social evidence. Alpha Vantage is available as an alternative for supported financial tools.

Analysts can also use `search_web` and `fetch_webpage`. Search uses the configured backend, with Yahoo as the default. Page retrieval extracts text, titles, links, and reported dates from static HTML. It does not run JavaScript or read PDFs.

Agents use native harness tools to write and run scripts for calculations. The prompt asks them to batch related calculations where possible. Scripts use the harness's environment. Set `allow_execution = false` to disable them, or `web_research = false` to omit the additional search/page tools.

## Source configuration

Set category defaults or individual tool overrides in `config.toml`:

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

Optional source settings:

| Variable | Use |
| --- | --- |
| `FRED_API_KEY` | FRED macroeconomic data |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage tools |
| `SEC_EDGAR_USER_AGENT` | Identify requests to SEC EDGAR |

A `.env` in the current working directory is loaded without overriding exported environment variables. Model-provider authentication is configured in the harness separately.

## Coverage limits

Use Yahoo Finance ticker formats, including exchange suffixes for non-US listings. Source coverage varies by instrument and market. The default social and macro sources are more US-centric than the price feed.

Company profiles can distinguish quote and reporting currencies, but price snapshots currently lack explicit quote-currency metadata.

Analysis dates are information cutoffs. Tools exclude data known to be after the cutoff and report unavailable data explicitly. Live news, social feeds, search results, and web pages are not historical archives. A reported publication date alone does not prove that today's page content was unchanged on a past date.

Yahoo company profiles and financial statements are withheld for historical runs when the provider cannot establish when the information became available. SEC statements use filing dates for supported US filers. Yahoo indicator/snapshot history currently covers five years relative to today.

The verified market snapshot preserves source numeric precision, but prices and indicators can still be unavailable when the source is missing, stale, or unreachable. Missing observations do not establish that a market was closed.
