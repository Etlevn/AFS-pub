# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
import time
from typing import Dict, Any, Tuple, List, Optional
import threading

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
    get_single_agent_tools,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.agent_mode = str(self.config.get("agent_mode", "livetrading")).strip().lower()
        self.trace_level = str(self.config.get("trace_level", "basic")).strip().lower()
        if self.trace_level not in {"basic", "full", "audit"}:
            self.trace_level = "basic"
        self.callbacks = callbacks or []
        self.selected_analysts = selected_analysts
        self.progress_map = self._build_progress_map(
            self.selected_analysts, self.agent_mode
        )
        self._thread_code_by_name: Dict[str, str] = {}
        self._thread_code_lock = threading.Lock()
        self._next_thread_code = 1

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
            self.agent_mode,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(self.selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "deepseek":
            reasoning_effort = self.config.get("deepseek_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
            "single": ToolNode(get_single_agent_tools()),
        }

    @staticmethod
    def _normalize_agent_label(raw: str) -> str:
        """Normalize agent labels for robust matching."""
        return "".join(ch for ch in raw.lower() if ch.isalnum())

    def _build_progress_map(
        self, selected_analysts: List[str], agent_mode: str
    ) -> Dict[str, Dict[str, Any]]:
        """Build stage/substage mapping for progress display."""
        progress_map: Dict[str, Dict[str, Any]] = {}
        if agent_mode in {"livetrading", "nowcasting"}:
            total_big_stages = 1
        else:
            raise ValueError(
                f"Unsupported agent_mode '{agent_mode}'. Use 'livetrading' or 'nowcasting'."
            )
        self.total_big_stages = total_big_stages

        analyst_alias = {
            "market": ["market", "marketanalyst", "analystmarket"],
            "social": [
                "social",
                "socialmedia",
                "socialmediaanalyst",
                "socialanalyst",
                "analystsocial",
            ],
            "news": ["news", "newsanalyst", "analystnews"],
            "fundamentals": [
                "fundamentals",
                "fundamental",
                "fundamentalsanalyst",
                "fundamentalanalyst",
                "analystfundamentals",
            ],
        }
        analyst_display = {
            "market": "Analyst_Market",
            "social": "Analyst_Social",
            "news": "Analyst_News",
            "fundamentals": "Analyst_Fundamentals",
        }

        display = (
            "Livetrading_Agent" if agent_mode == "livetrading" else "Nowcasting_Agent"
        )
        aliases = (
            ["livetradingagent", "livetrading", "liveagent"]
            if agent_mode == "livetrading"
            else ["nowcastingagent", "nowcasting", "nowcastagent"]
        )
        stage_definitions = [(1, 1, [(1, display, aliases)])]

        for big_stage_idx, sub_total, stage_agents in stage_definitions:
            for sub_idx, display_name, aliases in stage_agents:
                for alias in aliases:
                    progress_map[alias] = {
                        "progress": f"{big_stage_idx}/{total_big_stages}_{sub_idx}/{sub_total}",
                        "agent_label": display_name,
                    }

        return progress_map

    def _resolve_progress_info(
        self,
        message: Any,
        chunk: Dict[str, Any],
        trace_context: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Resolve progress marker and normalized agent label."""
        context = trace_context if trace_context is not None else {}
        candidates: List[str] = []

        inferred_alias = self._infer_active_agent_alias(chunk, context)
        if inferred_alias:
            candidates.append(inferred_alias)

        for candidate in (
            getattr(message, "name", None),
            getattr(message, "sender", None),
            chunk.get("sender"),
            getattr(message, "role", None),
        ):
            if candidate:
                candidates.append(str(candidate))

        for candidate in candidates:
            normalized = self._normalize_agent_label(candidate)
            if normalized in self.progress_map:
                resolved = self.progress_map[normalized]
                context["last_progress"] = resolved["progress"]
                context["last_agent_label"] = resolved["agent_label"]
                context["last_agent_alias"] = normalized
                return resolved

        msg_type = str(getattr(message, "type", "")).lower()
        if msg_type == "tool" and context.get("last_progress"):
            return {
                "progress": context["last_progress"],
                "agent_label": context.get("last_agent_label", "Tool"),
            }

        return {
            "progress": context.get(
                "last_progress", f"?/?_?/{getattr(self, 'total_big_stages', 5)}"
            ),
            "agent_label": context.get("last_agent_label", "UnknownAgent"),
        }

    def _infer_active_agent_alias(
        self, state: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[str]:
        """Infer active agent from full graph state to label tool/human messages."""
        if self.agent_mode == "livetrading":
            return "livetradingagent"

        if self.agent_mode == "nowcasting":
            return "nowcastingagent"

        return str(context.get("last_agent_alias") or "livetradingagent")

    def _get_thread_code(self, trace_context: Optional[Dict[str, Any]]) -> str:
        """Return deterministic 001..999 code for each runtime thread."""
        context = trace_context or {}
        explicit = context.get("thread_code")
        if explicit:
            return str(explicit)

        thread_name = str(context.get("thread_name") or threading.current_thread().name)
        with self._thread_code_lock:
            cached = self._thread_code_by_name.get(thread_name)
            if cached:
                return cached
            code = f"{self._next_thread_code:03d}"
            self._thread_code_by_name[thread_name] = code
            self._next_thread_code += 1
            if self._next_thread_code > 999:
                self._next_thread_code = 1
            return code

    def _resolve_trace_ticker(
        self, chunk: Dict[str, Any], trace_context: Optional[Dict[str, Any]]
    ) -> str:
        """Resolve ticker for trace header display."""
        if trace_context and trace_context.get("ticker"):
            return str(trace_context["ticker"])
        if chunk.get("company_of_interest"):
            return str(chunk["company_of_interest"])
        if self.ticker:
            return str(self.ticker)
        return "UNKNOWN"

    def _emit_progress_update(
        self, stage_progress: str, trace_context: Optional[Dict[str, Any]]
    ) -> None:
        """Emit one-time progress updates for each unique small stage per flow."""
        if trace_context is None:
            return
        callback = trace_context.get("progress_callback")
        if not callable(callback):
            return
        if stage_progress.startswith("?/?"):
            return

        seen = trace_context.setdefault("seen_stage_progress", set())
        if stage_progress in seen:
            return
        seen.add(stage_progress)
        callback(stage_progress)

    @staticmethod
    def _format_elapsed_from_start(trace_context: Optional[Dict[str, Any]]) -> str:
        """Format elapsed time since program start as +HH:MM:SS."""
        if trace_context is None:
            return "+00:00:00"
        start_ts = trace_context.get("run_start_monotonic")
        if not isinstance(start_ts, (int, float)):
            return "+00:00:00"

        elapsed_seconds = max(0, int(time.perf_counter() - float(start_ts)))
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        return f"+{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_trace_header(
        self,
        message: Any,
        chunk: Dict[str, Any],
        trace_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build a compact debug header for streamed messages."""
        thread_code = self._get_thread_code(trace_context)
        ticker = self._resolve_trace_ticker(chunk, trace_context)

        msg_type_raw = str(getattr(message, "type", "message")).lower()
        if msg_type_raw == "human":
            msg_type = "Human"
        elif msg_type_raw == "tool":
            msg_type = "Tool"
        else:
            msg_type = "AI"
        border = "=" * 60
        elapsed = self._format_elapsed_from_start(trace_context)
        return (
            "\n"
            f"{border}\n"
            f"[ {thread_code} | {ticker} | {msg_type} ] {elapsed}\n"
            f"{border}\n"
        )

    @staticmethod
    def _format_message_body(message: Any) -> str:
        """Format message content for terminal output."""
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content)
        return content if content else "<empty message>"

    def _build_audit_details(self, message: Any) -> str:
        """Build compact metadata lines for trace_level=audit."""
        details: list[str] = []
        msg_type = str(getattr(message, "type", "")).lower()

        msg_name = getattr(message, "name", None)
        if msg_name:
            details.append(f"name={msg_name}")

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            details.append(f"tool_call_id={tool_call_id}")

        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list) and tool_calls:
            call_summaries = []
            for call in tool_calls:
                if isinstance(call, dict):
                    call_name = call.get("name") or call.get("type") or "unknown"
                    call_id = call.get("id")
                    if call_id:
                        call_summaries.append(f"{call_name}#{call_id}")
                    else:
                        call_summaries.append(str(call_name))
                else:
                    call_summaries.append(str(call))
            details.append(f"tool_calls=[{', '.join(call_summaries)}]")

        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict) and additional_kwargs:
            extra_keys = sorted(additional_kwargs.keys())
            details.append(f"additional_keys={','.join(extra_keys)}")

        body = self._format_message_body(message)
        details.append(f"body_chars={len(body)}")
        details.append(f"body_lines={body.count(chr(10)) + 1}")
        details.append(f"message_type={msg_type or 'unknown'}")
        return "[AUDIT] " + " | ".join(details)

    def _emit_trace_text(
        self, text: str, trace_context: Optional[Dict[str, Any]], logger: Any
    ) -> None:
        if callable(logger):
            logger(text)
        else:
            print(text)

    def _print_debug_chunk(
        self, chunk: Dict[str, Any], trace_context: Optional[Dict[str, Any]]
    ) -> None:
        """Print debug stream chunk with standardized trace markers."""
        messages = chunk.get("messages") or []
        if not messages:
            return

        if self.trace_level == "basic":
            stream_messages = [messages[-1]]
        else:
            stream_messages = list(messages)

        logger = None
        if trace_context is not None:
            logger = trace_context.get("log_callback")

        for idx, message in enumerate(stream_messages, start=1):
            header = self._format_trace_header(message, chunk, trace_context)
            if self.trace_level in {"full", "audit"} and len(stream_messages) > 1:
                header += f"[MSG] {idx}/{len(stream_messages)}\n"
            body = self._format_message_body(message)

            self._emit_trace_text(header, trace_context, logger)
            self._emit_trace_text(body, trace_context, logger)
            if self.trace_level == "audit":
                self._emit_trace_text(
                    f"\n{self._build_audit_details(message)}",
                    trace_context,
                    logger,
                )

    def propagate(self, company_name, trade_date, trace_context: Optional[Dict[str, Any]] = None):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                self._print_debug_chunk(chunk, trace_context)
                if len(chunk["messages"]) > 0:
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        if self.agent_mode == "nowcasting":
            return final_state, "NOWCAST"

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "agent_mode": self.agent_mode,
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
            "nowcast_report": final_state.get("nowcast_report", ""),
            "forecast_decision": final_state.get("forecast_decision", ""),
        }

        # Save to file
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
