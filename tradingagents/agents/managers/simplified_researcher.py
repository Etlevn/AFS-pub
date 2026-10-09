from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.managers.portfolio_manager import (
    _build_structured_decision,
    _extract_json_object,
)


def create_simplified_researcher(llm, memory):
    """Create a single researcher node for simple mode.

    This node replaces the full debate/trader/risk pipeline by combining
    research synthesis + trading recommendation + final JSON decision output.
    """

    def researcher_node(state) -> dict:
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        instrument_context = build_instrument_context(ticker)

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = (
            f"{market_research_report}\n\n{sentiment_report}\n\n"
            f"{news_report}\n\n{fundamentals_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are the sole Researcher and Trader in simplified mode.
Your job is to synthesize all analyst reports, make one actionable trading decision,
and output the final machine-readable decision object.

{instrument_context}

Available evidence:
- Market report: {market_research_report}
- Sentiment report: {sentiment_report}
- World/news report: {news_report}
- Fundamentals report: {fundamentals_report}
- Lessons from past similar situations: {past_memory_str}

Output Contract (CRITICAL):
- Return valid JSON ONLY. No markdown, no code fences, no prose outside JSON.
- Use uppercase ratings from this enum only: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL.
- Confidence must be a float in [0, 1].
- Use this exact schema:
{{
  "ticker": "{ticker}",
  "date": "{trade_date}",
  "rating": "BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL",
  "horizons": {{
    "1w": {{"rating": "...", "confidence": 0.0, "price_target": 0.0}},
    "1m": {{"rating": "...", "confidence": 0.0, "price_target": 0.0}},
    "3m": {{"rating": "...", "confidence": 0.0, "price_target": 0.0}}
  }},
  "thesis": "Detailed rationale grounded in analyst evidence",
  "key_risks": ["risk 1", "risk 2"],
  "analyst_reports": {{
    "market": "...",
    "fundamentals": "...",
    "sentiment": "...",
    "news": "..."
  }}
}}

Be decisive and provide a trader-ready view in the thesis (entry/exit posture,
positioning logic, and risk controls). Respond in English.
{get_language_instruction()}"""

        response = llm.invoke(prompt)
        response_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
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

        return {
            "investment_plan": response_text,
            "trader_investment_plan": response_text,
            "final_trade_decision": structured_decision,
        }

    return researcher_node
