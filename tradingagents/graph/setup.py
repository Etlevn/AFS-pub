# TradingAgents/graph/setup.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        conditional_logic: ConditionalLogic,
        agent_mode: str = "livetrading",
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.conditional_logic = conditional_logic
        self.agent_mode = str(agent_mode or "livetrading").strip().lower()

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the single-agent workflow graph."""
        return self._setup_single_agent_graph()

    def _setup_single_agent_graph(self):
        """Set up workflow with one mode-specific agent and one shared ToolNode."""
        workflow = StateGraph(AgentState)
        if self.agent_mode not in {"livetrading", "nowcasting"}:
            raise ValueError(
                f"Unsupported agent_mode '{self.agent_mode}'. Use 'livetrading' or 'nowcasting'."
            )
        node_name = (
            "Livetrading Agent"
            if self.agent_mode == "livetrading"
            else "Nowcasting Agent"
        )
        agent_node = (
            create_livetrading_agent(self.deep_thinking_llm)
            if self.agent_mode == "livetrading"
            else create_nowcasting_agent(self.deep_thinking_llm)
        )

        workflow.add_node(node_name, agent_node)
        workflow.add_node("tools_single", self.tool_nodes["single"])
        workflow.add_edge(START, node_name)
        workflow.add_conditional_edges(
            node_name,
            self.conditional_logic.should_continue_single_agent,
            {
                "tools_single": "tools_single",
                "__end__": END,
            },
        )
        workflow.add_edge("tools_single", node_name)
        return workflow.compile()
