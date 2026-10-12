from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Load environment variables from .env file
load_dotenv()

# Create a custom config for single-case test run
test_config = DEFAULT_CONFIG.copy()
test_config["llm_provider"] = "deepseek"  # deepseek, ollama, novaiapi
test_config["quick_think_llm"] = "deepseek-v4-flash"  # deepseek-v4-flash, qwen3:latest
test_config["deep_think_llm"] = "deepseek-v4-flash"  # deepseek-v4-pro, qwen3:latest
test_config["max_debate_rounds"] = 1

# Configure data vendors (default uses yfinance, no extra API keys needed)
test_config["data_vendors"] = {
    "core_stock_apis": "yfinance",  # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",  # Options: alpha_vantage, yfinance
    "news_data": "yfinance",  # Options: alpha_vantage, yfinance
}

# Initialize with custom config
test_graph = TradingAgentsGraph(debug=True, config=test_config)

# Run a single propagation for smoke testing
_, test_decision = test_graph.propagate("AAPL", "2026-03-31")
print(test_decision)

# Memorize mistakes and reflect
# test_graph.reflect_and_remember(1000)  # parameter is the position returns
