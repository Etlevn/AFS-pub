"""Live single-ticker smoke run using the same defaults as main.py."""
from dotenv import load_dotenv

from main import build_runtime_config, load_main_defaults, parse_args, parse_tickers, resolve_runtime_options
from pathlib import Path
from tradingagents.graph.trading_graph import TradingAgentsGraph


def main():
    load_dotenv()
    args = parse_args()
    defaults = load_main_defaults(Path(args.config))
    options = resolve_runtime_options(args, defaults)
    tickers = parse_tickers(options["tickers"])
    if not tickers:
        raise ValueError("Provide at least one ticker in config or --tickers.")
    config = build_runtime_config(options, defaults)
    graph = TradingAgentsGraph(debug=options["debug"], config=config)
    state, decision = graph.propagate(tickers[0], options["date"])
    print(state.get("forecast_decision") or state.get("final_trade_decision") or decision)


if __name__ == "__main__":
    main()
