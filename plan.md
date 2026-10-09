# Plan: Automated Daily Research + Trading Pipeline

## Context

This is a development roadmap with illustrative code, not an implemented daily trading pipeline. See [README.md](README.md) for the current runnable v2 workflow.

Build a fully automated daily quantitative research and paper-trading system on the TradingAgents framework, with the interactive CLI removed.

- **Researcher**: Analyze all Russell 1000 companies on a daily schedule and produce structured research reports with forecasts across multiple time horizons.
- **Trader**: Consume Researcher output, make investment decisions, and execute trades in a simple paper-trading portfolio.
- **Automation**: Trigger runs through local cron scheduling without manual interaction.

Reuse `TradingAgentsGraph.propagate(ticker, date)` from the existing framework. When this plan was drafted, structured output, a paper-trading engine, and persistent storage still needed to be implemented. The v2 workflow now provides structured JSON output; the pipeline components below remain proposed work.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Cron      │────▶│  run_daily   │────▶│  Researcher │
│  (daily)    │     │  .py         │     │  Pipeline   │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  SQLite      │
                                         │  reports.db  │
                                         └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │   Trader     │
                                         │   Engine     │
                                         └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  SQLite      │
                                         │  portfolio.db│
                                         └──────────────┘
```

---

## Implementation Steps

### Step 1: Structured Researcher Output

**Goal**: Make `propagate()` return structured JSON instead of free text, including forecasts across multiple horizons.

**Files**:
- `tradingagents/graph/signal_processing.py` — Extend `process_signal()` or add `process_structured_output()`.
- `tradingagents/agents/managers/portfolio_manager.py` — Update the prompt to require JSON output through `json_mode` or explicit prompt constraints.

**Schema** (Researcher output):
```json
{
  "ticker": "AAPL",
  "date": "2026-04-24",
  "rating": "BUY",
  "horizons": {
    "1w": {"rating": "BUY", "confidence": 0.8, "price_target": 220},
    "1m": {"rating": "OVERWEIGHT", "confidence": 0.7, "price_target": 230},
    "3m": {"rating": "HOLD", "confidence": 0.6, "price_target": 225}
  },
  "thesis": "...",
  "key_risks": ["..."],
  "analyst_reports": {
    "market": "...",
    "fundamentals": "...",
    "sentiment": "...",
    "news": "..."
  }
}
```

**How**: Append a JSON schema requirement to the Portfolio Manager prompt so it produces formatted JSON. Prefer `response_format={"type": "json_object"}` when the provider supports it; otherwise use explicit prompt constraints and parse the response with `json.loads()`.

---

### Step 2: Parallel Pipeline Runner

**Goal**: Analyze 1,000 companies concurrently each day and save the results to SQLite.

**New File**: `pipeline/researcher.py`

```python
import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

async def run_research(tickers: list[str], trade_date: str, config: dict, max_workers: int = 10):
    """Run analysis for all tickers in parallel."""
    # Use ThreadPool because TradingAgentsGraph uses sync LangGraph
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, _analyze_one, ticker, trade_date, config)
            for ticker in tickers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

def _analyze_one(ticker, trade_date, config):
    ta = TradingAgentsGraph(debug=False, config=config)
    final_state, signal = ta.propagate(ticker, trade_date)
    # Extract structured JSON from final_state["final_trade_decision"]
    return {"ticker": ticker, "state": final_state, "signal": signal}
```

**New File**: `pipeline/storage.py`

```python
import sqlite3
import json
from datetime import datetime

DB_PATH = "data/research.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_reports (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            date TEXT,
            rating TEXT,
            horizon_1w_rating TEXT,
            horizon_1m_rating TEXT,
            horizon_3m_rating TEXT,
            full_report TEXT,  -- JSON
            created_at TEXT
        )
    """)
    conn.commit()
    return conn

def save_report(conn, ticker: str, date: str, report: dict):
    conn.execute(
        "INSERT INTO research_reports ... VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ticker, date, report["rating"], ...)
    )
    conn.commit()
```

---

### Step 3: Simple Portfolio Simulation

**Goal**: Build a lightweight paper-trading portfolio with long positions, end-of-day rebalancing, and simple stop-loss and take-profit rules.

**New File**: `pipeline/portfolio.py`

```python
from dataclasses import dataclass
from typing import Optional
import sqlite3
import yfinance as yf

@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    entry_date: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class Portfolio:
    def __init__(self, initial_cash: float = 1_000_000.0, db_path: str = "data/portfolio.db"):
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT PRIMARY KEY,
                shares REAL,
                entry_price REAL,
                entry_date TEXT,
                stop_loss REAL,
                take_profit REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                date TEXT,
                reason TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_current_prices(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest close prices via yfinance."""
        prices = {}
        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(period="1d")
                prices[ticker] = float(hist["Close"].iloc[-1])
            except Exception:
                prices[ticker] = None
        return prices

    def buy(self, ticker: str, amount: float, date: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """Buy with a fixed dollar amount."""
        price = self.get_current_prices([ticker]).get(ticker)
        if not price or price <= 0:
            return False
        shares = amount / price
        if self.cash < amount:
            return False
        self.cash -= amount
        self.positions[ticker] = Position(ticker, shares, price, date, stop_loss, take_profit)
        self._log_trade(ticker, "BUY", shares, price, date, "rebalance")
        return True

    def sell(self, ticker: str, date: str, reason: str = "rebalance"):
        """Sell entire position."""
        if ticker not in self.positions:
            return False
        pos = self.positions.pop(ticker)
        price = self.get_current_prices([ticker]).get(ticker)
        if not price:
            return False
        self.cash += pos.shares * price
        self._log_trade(ticker, "SELL", pos.shares, price, date, reason)
        return True

    def check_stop_loss_take_profit(self, date: str):
        """Check and execute SL/TP."""
        prices = self.get_current_prices(list(self.positions.keys()))
        for ticker, pos in list(self.positions.items()):
            price = prices.get(ticker)
            if not price:
                continue
            if pos.stop_loss and price <= pos.stop_loss:
                self.sell(ticker, date, "stop_loss")
            elif pos.take_profit and price >= pos.take_profit:
                self.sell(ticker, date, "take_profit")

    def rebalance(self, targets: dict[str, float], date: str, max_positions: int = 20):
        """
        targets: dict of ticker -> target portfolio weight (0.0-1.0)
        Sell positions no longer in targets, then buy top max_positions.
        """
        # 1. Check SL/TP first
        self.check_stop_loss_take_profit(date)

        # 2. Sell positions not in targets
        for ticker in list(self.positions.keys()):
            if ticker not in targets:
                self.sell(ticker, date, "not_in_targets")

        # 3. Calculate how much to allocate to each target
        total_value = self.cash + sum(
            pos.shares * self.get_current_prices([pos.ticker]).get(pos.ticker, 0)
            for pos in self.positions.values()
        )

        # Sort targets by weight, take top max_positions
        sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)[:max_positions]

        for ticker, weight in sorted_targets:
            target_value = total_value * weight
            current_value = 0
            if ticker in self.positions:
                price = self.get_current_prices([ticker]).get(ticker, 0)
                current_value = self.positions[ticker].shares * price
            delta = target_value - current_value
            if delta > 0:
                self.buy(ticker, delta, date)
            elif delta < -100:  # threshold to avoid micro-trades
                self.sell(ticker, date, "rebalance_down")
                self.buy(ticker, target_value, date)

    def get_portfolio_value(self) -> float:
        prices = self.get_current_prices(list(self.positions.keys()))
        position_value = sum(
            pos.shares * prices.get(ticker, pos.entry_price)
            for ticker, pos in self.positions.items()
        )
        return self.cash + position_value

    def _log_trade(self, ticker, action, shares, price, date, reason):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO trades (ticker, action, shares, price, date, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (ticker, action, shares, price, date, reason)
        )
        conn.commit()
        conn.close()
```

---

### Step 4: Trader Logic

**Goal**: Read researcher reports, select stocks, generate target weights, and call portfolio rebalancing.

**New File**: `pipeline/trader.py`

```python
from pipeline.storage import get_latest_reports
from pipeline.portfolio import Portfolio

def run_trader(trade_date: str, portfolio: Portfolio):
    """
    1. Fetch all latest researcher reports for the date.
    2. Filter: only BUY/OVERWEIGHT ratings, sort by confidence/horizon.
    3. Generate equal-weight targets for top N picks.
    4. Call portfolio.rebalance().
    """
    reports = get_latest_reports(trade_date)

    # Filter and score
    scored = []
    for r in reports:
        rating = r["rating"]
        if rating not in ("BUY", "OVERWEIGHT"):
            continue
        # Simple score: confidence weighted by horizon
        score = r.get("horizons", {}).get("1w", {}).get("confidence", 0.5)
        scored.append((r["ticker"], score))

    # Sort by score, take top 20
    scored.sort(key=lambda x: x[1], reverse=True)
    top_picks = scored[:20]

    # Equal weight
    weight = 1.0 / len(top_picks) if top_picks else 0
    targets = {ticker: weight for ticker, _ in top_picks}

    portfolio.rebalance(targets, trade_date, max_positions=20)
```

---

### Step 5: Entry Point + Cron

**New File**: `run_daily.py`

```python
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from tradingagents.default_config import DEFAULT_CONFIG
from pipeline.researcher import run_research
from pipeline.storage import init_db, save_report
from pipeline.portfolio import Portfolio
from pipeline.trader import run_trader

load_dotenv()

# Russell 1000 tickers (or load from a CSV/JSON file)
TICKERS = json.load(open("data/russell1000.json"))

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = os.getenv("LLM_PROVIDER", "openai")
    config["deep_think_llm"] = os.getenv("DEEP_THINK_LLM", "gpt-4o")
    config["quick_think_llm"] = os.getenv("QUICK_THINK_LLM", "gpt-4o-mini")
    config["output_language"] = "English"

    # Step 1: Research
    print(f"[{today}] Starting research for {len(TICKERS)} tickers...")
    results = asyncio.run(run_research(TICKERS, today, config, max_workers=10))

    # Step 2: Save to DB
    conn = init_db()
    for result in results:
        if isinstance(result, Exception):
            print(f"ERROR: {result}")
            continue
        save_report(conn, result["ticker"], today, result["state"])
    conn.close()

    # Step 3: Trade
    portfolio = Portfolio(initial_cash=1_000_000.0)
    run_trader(today, portfolio)

    print(f"[{today}] Done. Portfolio value: ${portfolio.get_portfolio_value():,.2f}")

if __name__ == "__main__":
    main()
```

**Cron setup** (documented in README):
```bash
# crontab -e
0 6 * * 1-5 cd /path/to/project && uv run python run_daily.py >> logs/daily.log 2>&1
```

---

## File Changes Summary

| File | Action | Purpose |
|------|--------|---------|
| `tradingagents/graph/signal_processing.py` | Modify | Add structured JSON extraction |
| `tradingagents/agents/managers/portfolio_manager.py` | Modify | Update prompt for JSON output |
| `pipeline/researcher.py` | **New** | Parallel analysis runner |
| `pipeline/storage.py` | **New** | SQLite storage for reports |
| `pipeline/portfolio.py` | **New** | Portfolio simulation engine |
| `pipeline/trader.py` | **New** | Trading decision logic |
| `run_daily.py` | **New** | Daily pipeline entry point |
| `data/russell1000.json` | **New** | Ticker list |
| `README.md` | Update | Document cron setup and usage |

---

## Reusable Components from Existing Framework

| Component | File | How Used |
|-----------|------|----------|
| `TradingAgentsGraph` | `tradingagents/graph/trading_graph.py` | `graph.propagate(ticker, date)` |
| `DEFAULT_CONFIG` | `tradingagents/default_config.py` | Base config, override provider/model |
| `LLM Client Factory` | `tradingagents/llm_clients/factory.py` | Auto via config |
| Data fetchers | `tradingagents/dataflows/y_finance.py` | Internal to agents |
| Technical indicators | `tradingagents/dataflows/stockstats_utils.py` | Internal to agents |

---

## Verification Plan

1. **Unit test researcher**: Pick 5 tickers (AAPL, MSFT, TSLA, NVDA, GOOGL), run `run_research()`, verify:
   - All 5 complete without error
   - Output contains valid JSON with `rating` and `horizons` fields
   - Results saved to SQLite

2. **Unit test portfolio**: Create a Portfolio with $100k, simulate:
   - Buy AAPL for $10k
   - Check SL triggers (mock price below SL)
   - Check rebalance logic

3. **Integration test end-to-end**:
   ```bash
   uv run python run_daily.py
   ```
   - Verify `research.db` has reports
   - Verify `portfolio.db` has trades
   - Check logs for errors

4. **Stress test**: Run on 50 tickers with `max_workers=10`, verify total runtime < 30 min.