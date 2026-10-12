MAIN_CONFIG = {
    "tickers": ["AAPL","MSFT","NVDA","AMZN","TSLA","JPM"], # AAPL, MSFT, NVDA, AMZN, TSLA, JPM
    "date": "2026-03-31",
    "agent_mode": "livetrading",  # livetrading, nowcasting
    "max_workers": 3,
    "provider": "deepseek",  # deepseek, ollama
    "model": "deepseek-v4-flash",  # single model for single agent workflow
    "debug": True,
    "log": True,
    "trace_level": "audit",  # basic, full, audit
    "data_vendors": { #alpha_vantage, yfinance
        "core_stock_apis": "alpha_vantage",
        "technical_indicators": "alpha_vantage",
        "fundamental_data": "alpha_vantage",
        "news_data": "alpha_vantage",
    },
}
