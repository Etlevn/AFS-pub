# AFS

**Agentic Financial System** is an LLM-agent system for automated financial research and trading decisions. It starts from [TradingAgents](https://github.com/TauricResearch/TradingAgents) and replaces the original multi-agent debate workflow with a single tool-using agent that can run the same research task across many tickers.

Traditional coverage splits prices, indicators, fundamentals, news, and trading activity across separate sources, then relies on a person to combine them. That process does not stay consistent when the universe grows and the same study has to be repeated. AFS turns those sources into tools the agent can call, then writes a machine-readable decision instead of a free-text note.

```text
Multi-source financial data
        |
        v
Agent tool planning and research
        |
        v
Evidence integration
        |
        v
Forecast or trading decision
        |
        v
Structured JSON
        |
        v
Concurrent batch output
```

Two task modes share the same graph and data tools. Only the prompt and output contract change:

- **`livetrading`** produces a BUY, SELL, or HOLD plan with confidence, position sizing, entry and exit conditions, stop-loss and take-profit levels, and risk flags.
- **`nowcasting`** produces 1-day, 5-day, and 20-day scenarios with bull, base, and bear probabilities, expected volatility, drivers, and monitoring triggers.

Despite the `livetrading` name, this project does not place broker orders. The research and decision pipeline is runnable. A paper portfolio, automatic rebalancing, and a daily execution loop are still proposed work in [plan.md](plan.md). Output is research material, not investment advice.

## Architecture

TradingAgents organizes fundamental, sentiment, news, and technical analysts together with bull and bear researchers, a trader, risk management, and a portfolio manager. That layout shows how financial roles can debate. For repeated batch research it also repeats context across many nodes and makes state and model cost harder to control.

AFS keeps the LangGraph foundation and compresses that chain into one **agent ↔ tool** loop. The agent decides which evidence to fetch. When the model returns no further `tool_calls`, the graph ends and emits the structured result. Each ticker gets its own graph instance. A shared `AgentState` carries the symbol, research date, message history, tool results, and the final decision or forecast.

Legacy multi-agent modules remain in the package. The active path is the single-agent workflow. The upstream interactive CLI and Docker setup are not part of this version.

## Research infrastructure

**Nine financial tools.** The agent can call historical prices, technical indicators, company overview, balance sheet, cash flow, income statement, company news, global news, and insider transactions. Calls go through a tool → provider router → provider implementation stack, so the research logic does not depend on one vendor. Each `data_vendors` category can use Alpha Vantage or yfinance. yfinance and stockstats provide the alternate path, including retry on Yahoo Finance rate limits. Indicator names and parameters are constrained, and exchange suffixes such as `.TO`, `.L`, and `.HK` are preserved.

**Research-date controls.** The analysis date is system state, not only a prompt field. Price history is cut off at that date before indicators are computed, Alpha Vantage daily series are filtered to the requested window, yfinance statements drop periods that end after the research date, and news is limited to a historical window. Cached raw prices stay separate from that filtering, so one cache can serve more than one research date. These controls reduce look-ahead risk. They are not a strict point-in-time snapshot: a fiscal period end is not a disclosure date, and company overview plus some insider and fundamental responses still cannot guarantee a full historical snapshot.

**Structured decisions.** Model text passes through JSON extraction, field normalization, validation, and a safe fallback. Extraction accepts raw JSON, fenced code blocks, and responses with extra prose. Unknown trading actions become `HOLD`, and confidence is clamped to 0–1. The same workflow can target OpenAI, Anthropic, Google, DeepSeek, xAI, Qwen, GLM, OpenRouter, NovAIAPI, Ollama, and Azure through one client factory.

**Batch runs and traces.** `ThreadPoolExecutor` runs one full graph per ticker. CLI flags override `config.py`. Trace levels are `basic` (latest message), `full` (all messages), and `audit` (tool-call and message metadata). Terminal progress stays off the log file. Each run writes the structured decisions to a batch JSON file. With debug enabled, the log records the agent messages for later review; the JSON file does not store the full graph state.

The main chain has been exercised on multi-ticker runs. It is a research prototype, not a production trading system. Known limits:

- Alpha Vantage rate limits can leave a batch with incomplete data.
- Some indicator fallback errors still propagate poorly.
- Historical fundamentals are not a fully strict point-in-time snapshot.
- `audit` traces grow quickly on long runs.
- Failed tickers are counted in the terminal summary, but their error details are not yet stored in the batch file.

Near-term work is provider fallback, HTTP timeouts, stricter point-in-time data, failure persistence, and offline tests. Storage, a paper portfolio, target weights, risk controls, rebalancing, and scheduling come after that infrastructure is stable.

## How it works

```text
config.py + command-line overrides + .env
                    |
                    v
          main.py concurrent batch runner
                    |
                    v
      TradingAgentsGraph.propagate(ticker, date)
                    |
                    v
       Single agent <-> market data tools
                    |
                    v
     Structured trading plan or scenario forecast
                    |
                    v
       output/<YYYYMMDD_HHMMSS>.json
       log/<YYYYMMDD_HHMMSS>.log (optional)
```

`main.py` loads `config.py`, applies command-line overrides, and calls `TradingAgentsGraph.propagate(ticker, date)` once per symbol. LangGraph runs the agent and tool loop inside that call.

## Quick start

Requires **Python 3.10 or newer**. Run these commands from the repository root:

```bash
git clone <repository-url>
cd AFS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
cp .env.example .env
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Edit `.env` with credentials for your selected LLM provider and data sources. Both `main.py` and `test.py` load it automatically. The file is ignored by Git; keep credentials out of tracked files.

Edit `MAIN_CONFIG` in [config.py](config.py), then run:

```bash
python main.py
```

The checked-in configuration selects DeepSeek, Alpha Vantage for all data categories, six tickers, three workers, and enabled debug and log output. It also contains a fixed analysis date: update the date and choose a model available through your provider before running. These defaults require `DEEPSEEK_API_KEY` and `ALPHA_VANTAGE_API_KEY`.

To use yfinance instead, set all four `data_vendors` entries in `config.py` to `"yfinance"`. That backend does not require an Alpha Vantage key.

## Configuration

`main.py` reads `MAIN_CONFIG` from `config.py`; the legacy name `CONFIG` is also accepted. Select a custom file with `--config path/to/config.py`. Required keys are `tickers`, `max_workers`, `provider`, and `model`.

| Setting | Purpose |
| --- | --- |
| `tickers` | A symbol list or comma-separated string, such as `["AAPL", "MSFT"]` |
| `date` | Analysis date in `YYYY-MM-DD` format; defaults to the local current date if omitted |
| `agent_mode` | `livetrading` or `nowcasting` |
| `max_workers` | Concurrent ticker analyses; the runner uses at least one worker |
| `provider` | LLM provider identifier |
| `model` | Model identifier for the single agent |
| `trace_level` | `basic`, `full`, or `audit` |
| `debug` | Enable graph trace output |
| `log` | Mirror terminal output to `log/<run_id>.log` |
| `data_vendors` | Backend for each market-data category |

Command-line arguments override the corresponding runtime settings in `config.py`. Framework settings start from [DEFAULT_CONFIG](tradingagents/default_config.py), then the runner applies its resolved options. The selected model is assigned to both `quick_think_llm` and `deep_think_llm`; the single agent uses the deep-thinking client. The default output language is English.

For example:

```bash
python main.py \
  --tickers AAPL,MSFT \
  --date 2026-03-31 \
  --agent-mode nowcasting \
  --max-workers 2 \
  --trace-level basic \
  --no-debug \
  --no-log \
  --output-dir output
```

Use `--provider` and `--model` to override the LLM selection, and `--debug` or `--log` to enable those features. Run `python main.py --help` for all arguments. `--max-debate-rounds` is retained as a compatibility setting; the active graph does not run the legacy investment debate.

## LLM providers and credentials

The client factory supports these provider identifiers. Configure only the credentials you need; [.env.example](.env.example) lists all environment variables.

| Provider | Environment variables |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `qwen` | `DASHSCOPE_API_KEY` |
| `glm` | `ZHIPU_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `novaiapi` | `NOVAIAPI_API_KEY` |
| `ollama` | No API key; run Ollama at `http://localhost:11434` |
| `azure` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `OPENAI_API_VERSION` |

Use a model that supports the tool calls required by the workflow. Provider-specific behavior is defined in [tradingagents/llm_clients/](tradingagents/llm_clients/).

## Market data

Set a backend independently for each category in `MAIN_CONFIG["data_vendors"]`:

```python
"data_vendors": {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
},
```

Supported category backends are `yfinance` and `alpha_vantage`. Alpha Vantage requires `ALPHA_VANTAGE_API_KEY`. Coverage and availability depend on the vendor, instrument, and endpoint.

Shared tools include `get_stock_data`, `get_indicators`, `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_news`, `get_global_news`, and `get_insider_transactions`. Agent prompts require stock data before technical indicators and instruct the model to report missing evidence and uncertainty.

## Reports

Each run writes a **JSON array** to `output/<run_id>.json`, or the directory supplied through `--output-dir`. Entries are sorted by ticker. Only successful results containing a structured decision are included. Failed analyses are excluded; the terminal summary reports total, successful, and failed counts. A run with no successful decisions writes an empty array.

These examples illustrate the normalized report shape, not actual market recommendations.

### Trading plan (`livetrading`)

```json
[
  {
    "mode": "livetrading",
    "ticker": "AAPL",
    "date": "2026-03-31",
    "rating": "HOLD",
    "action": "HOLD",
    "confidence": 0.5,
    "position_size_hint": "No new position",
    "entry_condition": "Wait for confirmation with complete data",
    "exit_condition": "Reassess if the research thesis changes",
    "stop_loss": null,
    "take_profit": null,
    "risk_flags": ["Insufficient evidence for a new trade"],
    "data_used": ["Price history", "Company news"],
    "rationale": "Remain on hold until the required evidence is available."
  }
]
```

The agent's raw JSON uses `symbol` and `as_of_date`; the report normalizes them to `ticker` and `date` and adds `rating`. Invalid or missing actions fall back to HOLD, and confidence is clamped to 0–1. JSON extraction and fallback handling do not guarantee that a plan is suitable for execution.

### Scenario forecast (`nowcasting`)

```json
[
  {
    "mode": "nowcasting",
    "symbol": "AAPL",
    "as_of_date": "2026-03-31",
    "forecast_horizon": "1D/5D/20D",
    "directional_view": "neutral",
    "probabilities": {"bull": 0.3, "base": 0.4, "bear": 0.3},
    "expected_volatility": "Moderate",
    "bull_case": "Positive catalysts support an upside move",
    "base_case": "Price remains range-bound without a new catalyst",
    "bear_case": "Negative catalysts drive a downside move",
    "key_drivers": ["Company news", "Market conditions"],
    "watchlist_triggers": ["A confirmed break outside the recent range"],
    "data_used": ["Price history", "Company news"],
    "limitations": ["Forecasts depend on available evidence"]
  }
]
```

Nowcasting reports preserve `symbol` and `as_of_date`. Probabilities are individually clamped to 0–1; the implementation does not normalize their sum. Reports contain the forecast object rather than the internal HOLD wrapper used by the graph.

## Debugging and verification

`--debug` enables graph traces. `basic` shows the latest message, `full` shows all messages, and `audit` includes metadata. `--log` captures stdout and stderr in `log/<run_id>.log`; progress bars stay on the terminal, and the log stream filters progress artifacts. Both `output/` and `log/` are ignored by Git.

Framework cache and historical state directories can be overridden with:

- `TRADINGAGENTS_CACHE_DIR` (default: `~/.tradingagents/cache`)
- `TRADINGAGENTS_RESULTS_DIR` (default: `~/.tradingagents/logs`)

For a single-ticker smoke run:

```bash
python test.py
```

`test.py` is a live smoke script, not an isolated unit test. It has its own hard-coded model, provider, ticker, and date settings, and uses yfinance. Edit those values before running; it calls external services and requires LLM credentials.

## Repository layout

```text
main.py                    Concurrent batch runner and JSON report writer
config.py                  Runtime defaults (MAIN_CONFIG)
test.py                    Live single-ticker smoke script
.env.example               Credential and directory settings template
plan.md                    Proposed daily research and paper-trading pipeline
pyproject.toml             Package metadata and dependencies
uv.lock                    Dependency lockfile
tradingagents/
    agents/                Mode-specific agents, legacy roles, and tools
    graph/                 LangGraph setup, propagation, and signal processing
    dataflows/             Market-data integrations and vendor routing
    llm_clients/           Provider client factory and adapters
    default_config.py      Framework defaults
output/                    Generated batch reports (ignored)
log/                       Captured terminal logs (ignored)
```

## Roadmap

The runnable system is the research pipeline above: two single-agent modes, nine data tools, structured JSON, concurrent batches, and graded traces. The next engineering work is provider fallback, request timeouts, stricter point-in-time fundamentals, persisted failure records, and offline tests.

[plan.md](plan.md) then outlines SQLite report storage, a simulated portfolio with target weights and rebalancing, and a scheduled `run_daily.py` pipeline. That document is a design sketch. Those modules are not part of the current application.

## Attribution

This project builds on TradingAgents by TauricResearch. If you use it in research, please acknowledge the upstream project and paper:

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework},
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138},
}
```
