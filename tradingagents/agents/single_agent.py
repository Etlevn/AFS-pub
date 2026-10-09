from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.managers.portfolio_manager import _extract_json_object
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_single_agent_tools,
)


ALLOWED_ACTIONS = {"BUY", "SELL", "HOLD"}
ACTION_TO_RATING = {
    "BUY": "BUY",
    "SELL": "SELL",
    "HOLD": "HOLD",
}


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _coerce_float(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _ensure_list(value: Any, default: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or (default or [])
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return default or []


def _normalize_action(value: Any) -> str:
    if not isinstance(value, str):
        return "HOLD"
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    return normalized if normalized in ALLOWED_ACTIONS else "HOLD"


def _build_livetrading_decision(
    parsed: dict[str, Any] | None,
    fallback_text: str,
    ticker: str,
    trade_date: str,
) -> dict[str, Any]:
    parsed = parsed or {}
    action = _normalize_action(parsed.get("action"))
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = fallback_text.strip()

    return {
        "mode": "livetrading",
        "ticker": str(parsed.get("symbol") or parsed.get("ticker") or ticker).upper(),
        "date": str(parsed.get("as_of_date") or parsed.get("date") or trade_date),
        "rating": ACTION_TO_RATING[action],
        "action": action,
        "confidence": _coerce_float(parsed.get("confidence")),
        "position_size_hint": str(parsed.get("position_size_hint") or "Not specified"),
        "entry_condition": str(parsed.get("entry_condition") or "Not specified"),
        "exit_condition": str(parsed.get("exit_condition") or "Not specified"),
        "stop_loss": parsed.get("stop_loss"),
        "take_profit": parsed.get("take_profit"),
        "risk_flags": _ensure_list(parsed.get("risk_flags"), ["Not explicitly provided."]),
        "data_used": _ensure_list(parsed.get("data_used"), ["Not explicitly provided."]),
        "rationale": rationale,
    }


def _build_forecast_decision(
    parsed: dict[str, Any] | None,
    fallback_text: str,
    ticker: str,
    trade_date: str,
) -> dict[str, Any]:
    parsed = parsed or {}
    probabilities = parsed.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = {}

    return {
        "mode": "nowcasting",
        "symbol": str(parsed.get("symbol") or parsed.get("ticker") or ticker).upper(),
        "as_of_date": str(parsed.get("as_of_date") or parsed.get("date") or trade_date),
        "forecast_horizon": str(parsed.get("forecast_horizon") or "1D/5D/20D"),
        "directional_view": str(parsed.get("directional_view") or "neutral"),
        "probabilities": {
            "bull": _coerce_float(probabilities.get("bull"), 0.33),
            "base": _coerce_float(probabilities.get("base"), 0.34),
            "bear": _coerce_float(probabilities.get("bear"), 0.33),
        },
        "expected_volatility": str(parsed.get("expected_volatility") or "Not specified"),
        "bull_case": str(parsed.get("bull_case") or "Not specified"),
        "base_case": str(parsed.get("base_case") or fallback_text.strip()),
        "bear_case": str(parsed.get("bear_case") or "Not specified"),
        "key_drivers": _ensure_list(parsed.get("key_drivers"), ["Not explicitly provided."]),
        "watchlist_triggers": _ensure_list(
            parsed.get("watchlist_triggers"), ["Not explicitly provided."]
        ),
        "data_used": _ensure_list(parsed.get("data_used"), ["Not explicitly provided."]),
        "limitations": _ensure_list(parsed.get("limitations"), ["Not explicitly provided."]),
    }


def _create_single_agent(llm, mode: str):
    tools = get_single_agent_tools()
    mode = str(mode).strip().lower()

    if mode == "livetrading":
        role_prompt = """You are the sole livetrading agent for this system.
Your job is to perform the full research-to-trade workflow and produce a trader-ready,
machine-readable decision. This first v2 implementation does not place broker orders;
it outputs an executable trading plan for a downstream paper-trading or execution layer.

Required workflow:
1. Establish context for the exact ticker and date.
2. Call get_stock_data first before any technical indicator request.
3. Call get_indicators for a compact, complementary set of indicators chosen from:
   close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi,
   boll, boll_ub, boll_lb, atr, vwma.
4. Use company news, global news, and fundamentals/statements when they are relevant.
5. Synthesize market, fundamentals, news/sentiment, and risk in one internal workflow.
6. Apply risk gates: confidence, position size, stop-loss, take-profit, data gaps,
   macro/event risk, and explicit no-trade conditions.
7. Final answer must be valid JSON only, with no markdown or prose outside JSON.

Use this exact JSON schema:
{
  "mode": "livetrading",
  "symbol": "<ticker>",
  "as_of_date": "<YYYY-MM-DD>",
  "action": "BUY|SELL|HOLD",
  "confidence": 0.0,
  "position_size_hint": "string",
  "entry_condition": "string",
  "exit_condition": "string",
  "stop_loss": "string or number or null",
  "take_profit": "string or number or null",
  "risk_flags": ["string"],
  "data_used": ["string"],
  "rationale": "string"
}"""
    elif mode == "nowcasting":
        role_prompt = """You are the sole nowcasting agent for this system.
Your job is to produce a short-horizon forecast and scenario update. Do not issue
broker-style trade instructions. Output a probability-weighted forecast that can be
used by a livetrading agent or a human decision maker.

Required workflow:
1. Define short-horizon forecast targets for 1D, 5D, and 20D where evidence allows.
2. Call get_stock_data first before any technical indicator request.
3. Call get_indicators for a compact, complementary set of indicators chosen from:
   close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi,
   boll, boll_ub, boll_lb, atr, vwma.
4. Use company news and global news to detect fresh catalysts or macro regime shifts.
5. Use fundamentals/statements when they materially constrain the forecast.
6. Produce base, bull, and bear scenarios with explicit probabilities and triggers.
7. Final answer must be valid JSON only, with no markdown or prose outside JSON.

Use this exact JSON schema:
{
  "mode": "nowcasting",
  "symbol": "<ticker>",
  "as_of_date": "<YYYY-MM-DD>",
  "forecast_horizon": "1D/5D/20D",
  "directional_view": "bullish|neutral|bearish|mixed",
  "probabilities": {"bull": 0.0, "base": 0.0, "bear": 0.0},
  "expected_volatility": "string",
  "bull_case": "string",
  "base_case": "string",
  "bear_case": "string",
  "key_drivers": ["string"],
  "watchlist_triggers": ["string"],
  "data_used": ["string"],
  "limitations": ["string"]
}"""
    else:
        raise ValueError(f"Unsupported single-agent mode: {mode}")

    def single_agent_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        instrument_context = build_instrument_context(ticker)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a disciplined financial AI agent with access to REST-backed "
                    "market data tools. Use tools to gather evidence before final output. "
                    "Never invent missing data; mark uncertainty in the required JSON fields. "
                    "Available tools: {tool_names}.\n\n"
                    "{role_prompt}\n\n"
                    "Current date: {current_date}. {instrument_context}"
                    "{language_instruction}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(role_prompt=role_prompt)
        prompt = prompt.partial(current_date=trade_date)
        prompt = prompt.partial(instrument_context=instrument_context)
        prompt = prompt.partial(language_instruction=get_language_instruction())

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        output: dict[str, Any] = {"messages": [result], "sender": f"{mode}_agent"}
        tool_calls = getattr(result, "tool_calls", None) or []
        if tool_calls:
            return output

        response_text = _as_text(result.content)
        parsed = _extract_json_object(response_text)
        if mode == "livetrading":
            structured = _build_livetrading_decision(parsed, response_text, ticker, trade_date)
            output.update(
                {
                    "investment_plan": response_text,
                    "trader_investment_plan": response_text,
                    "final_trade_decision": structured,
                }
            )
        else:
            forecast = _build_forecast_decision(parsed, response_text, ticker, trade_date)
            output.update(
                {
                    "nowcast_report": response_text,
                    "forecast_decision": forecast,
                    "final_trade_decision": {
                        "mode": "nowcasting",
                        "ticker": forecast["symbol"],
                        "date": forecast["as_of_date"],
                        "rating": "HOLD",
                        "forecast": forecast,
                    },
                }
            )

        return output

    return single_agent_node


def create_livetrading_agent(llm):
    return _create_single_agent(llm, "livetrading")


def create_nowcasting_agent(llm):
    return _create_single_agent(llm, "nowcasting")
