import json
from typing import Any

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


ALLOWED_RATINGS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}
DEFAULT_HORIZON_CONFIDENCE = 0.5
DEFAULT_HORIZONS = ("1w", "1m", "3m")


def _normalize_rating(value: Any, default: str = "HOLD") -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    return normalized if normalized in ALLOWED_RATINGS else default


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    if not raw_text:
        return None
    text = raw_text.strip()
    candidates = [text]
    if "```" in text:
        candidates.append(text.replace("```json", "").replace("```", "").strip())
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _coerce_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_HORIZON_CONFIDENCE
    return max(0.0, min(1.0, parsed))


def _coerce_price_target(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_structured_decision(
    parsed: dict[str, Any] | None,
    fallback_text: str,
    ticker: str,
    trade_date: str,
    market_research_report: str,
    fundamentals_report: str,
    sentiment_report: str,
    news_report: str,
) -> dict[str, Any]:
    parsed = parsed or {}
    top_rating = _normalize_rating(parsed.get("rating"), default="HOLD")
    parsed_horizons = parsed.get("horizons") if isinstance(parsed.get("horizons"), dict) else {}
    horizons: dict[str, dict[str, Any]] = {}
    for horizon in DEFAULT_HORIZONS:
        source = parsed_horizons.get(horizon)
        source = source if isinstance(source, dict) else {}
        horizons[horizon] = {
            "rating": _normalize_rating(source.get("rating"), default=top_rating),
            "confidence": _coerce_confidence(source.get("confidence")),
            "price_target": _coerce_price_target(source.get("price_target")),
        }

    thesis = parsed.get("thesis")
    if not isinstance(thesis, str) or not thesis.strip():
        thesis = fallback_text.strip()

    key_risks = parsed.get("key_risks")
    if not isinstance(key_risks, list):
        key_risks = []
    normalized_risks = [str(item).strip() for item in key_risks if str(item).strip()]
    if not normalized_risks:
        normalized_risks = ["Not explicitly provided."]

    analyst_reports = parsed.get("analyst_reports")
    if not isinstance(analyst_reports, dict):
        analyst_reports = {}

    return {
        "ticker": str(parsed.get("ticker") or ticker).upper(),
        "date": str(parsed.get("date") or trade_date),
        "rating": top_rating,
        "horizons": horizons,
        "thesis": thesis,
        "key_risks": normalized_risks,
        "analyst_reports": {
            "market": str(analyst_reports.get("market") or market_research_report),
            "fundamentals": str(
                analyst_reports.get("fundamentals") or fundamentals_report
            ),
            "sentiment": str(analyst_reports.get("sentiment") or sentiment_report),
            "news": str(analyst_reports.get("news") or news_report),
        },
    }


def create_portfolio_manager(llm, memory):
    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
- Lessons from past decisions: **{past_memory_str}**

**Output Contract (CRITICAL):**
- Return valid JSON ONLY. No markdown, no code fences, no prose outside JSON.
- Use uppercase ratings from this fixed enum only: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL.
- Confidence must be a float in [0, 1].
- Use this exact schema and keys:
{{
  "ticker": "{ticker}",
  "date": "{trade_date}",
  "rating": "BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL",
  "horizons": {{
    "1w": {{"rating": "...", "confidence": 0.0, "price_target": 0.0}},
    "1m": {{"rating": "...", "confidence": 0.0, "price_target": 0.0}},
    "3m": {{"rating": "...", "confidence": 0.0, "price_target": 0.0}}
  }},
  "thesis": "Detailed rationale grounded in debate evidence",
  "key_risks": ["risk 1", "risk 2"],
  "analyst_reports": {{
    "market": "...",
    "fundamentals": "...",
    "sentiment": "...",
    "news": "..."
  }}
}}

---

**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.
Respond in English.
{get_language_instruction()}"""

        response = llm.invoke(prompt)
        response_text = response.content if isinstance(response.content, str) else str(response.content)
        parsed_decision = _extract_json_object(response_text)
        structured_decision = _build_structured_decision(
            parsed_decision,
            response_text,
            ticker,
            trade_date,
            market_research_report,
            fundamentals_report,
            sentiment_report,
            news_report,
        )

        new_risk_debate_state = {
            "judge_decision": response_text,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": structured_decision,
        }

    return portfolio_manager_node
