# TradingAgents/graph/signal_processing.py

import json
import re
from typing import Any


ALLOWED_RATINGS = {"BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"}


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def _normalize_rating(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized in ALLOWED_RATINGS:
            return normalized
        for rating in ALLOWED_RATINGS:
            if re.search(rf"\b{rating}\b", normalized):
                return rating
        return None

    def _extract_structured_rating(self, full_signal: Any) -> str | None:
        if isinstance(full_signal, dict):
            return self._normalize_rating(full_signal.get("rating"))

        if not isinstance(full_signal, str):
            return None

        text = full_signal.strip()
        if not text:
            return None

        for candidate in (text, text.replace("```json", "").replace("```", "").strip()):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return self._normalize_rating(parsed.get("rating"))
        return None

    def process_signal(self, full_signal: Any) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, or SELL)
        """
        structured_rating = self._extract_structured_rating(full_signal)
        if structured_rating:
            return structured_rating

        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        response = self.quick_thinking_llm.invoke(messages).content
        return self._normalize_rating(response) or "HOLD"
