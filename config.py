from datetime import date

# Public example defaults. Store credentials in .env, never in this file.
MAIN_CONFIG = {
    "tickers": ["AAPL", "MSFT"],
    "date": date.today().isoformat(),
    "agent_mode": "nowcasting",  # livetrading, nowcasting
    "max_workers": 2,
    "provider": "openai",
    "model": "gpt-4.1-mini",  # Choose a tool-capable model available to your account.
    "debug": False,
    "log": False,
    "trace_level": "basic",  # basic, full, audit
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
}
