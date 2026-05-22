"""ReAct Agent subgraph — replaces the fixed DAG planner→tool→summarize pipeline."""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_neo4j import Neo4jGraph
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from gustobot.application.agents.react_agent.prompts import REACT_AGENT_SYSTEM_PROMPT
from gustobot.application.agents.react_agent.state import ReactAgentState
from gustobot.application.agents.react_agent.tools import build_react_tools


def create_react_agent(
    neo4j_graph: Optional[Neo4jGraph],
    llm: BaseChatModel,
    predefined_cypher_dict: Dict[str, str],
    *,
    max_iterations: int = 10,
) -> CompiledStateGraph:
    """Build a ReAct agent subgraph that replaces the fixed multi-tool DAG.

    The agent receives a question and iteratively: thinks → acts →
    observes → thinks → acts → … until it decides it has enough
    information to answer.

    Parameters
    ----------
    neo4j_graph : Neo4jGraph or None
        Neo4j connection used by cypher / predefined_cypher / text2sql tools.
    llm : BaseChatModel
        The LLM used for the agent brain (must support `bind_tools`).
    predefined_cypher_dict : dict
        Mapping of predefined Cypher query names → Cypher statements.
    max_iterations : int
        Maximum number of tool-calling rounds before forcing a stop (default 10).

    Returns
    -------
    CompiledStateGraph
        A runnable LangGraph subgraph that conforms to the ReAct pattern.
    """
    tools = build_react_tools(neo4j_graph, llm, predefined_cypher_dict)
    tool_node = ToolNode(tools)
    agent_llm = llm.bind_tools(tools)

    # Provide the agent with a structured observation block so it can reason
    # about tool results without the model being confused by raw ToolMessages.
    def _format_tool_results(state: ReactAgentState) -> Dict[str, Any]:
        """Insert a summary `AIMessage` after every batch of tool results.

        This helps models that prefer a coherent "observation" over a long
        sequence of plain ToolMessages.
        """
        msgs = state.messages
        # Find the most recent ToolMessage batch
        tool_msgs: List[ToolMessage] = []
        for m in reversed(msgs):
            if isinstance(m, ToolMessage):
                tool_msgs.append(m)          # pragma: no cover – covered by integration
            else:
                break
        if not tool_msgs:
            return {}

        parts: List[str] = []
        for tm in reversed(tool_msgs):
            name = getattr(tm, "name", "unknown_tool")
            content = getattr(tm, "content", "") or ""
            content_str = str(content)[:800]
            parts.append(f"[{name}] 返回: {content_str}")
        observation = "工具执行结果:\n" + "\n\n".join(parts)
        return {"messages": [AIMessage(content=observation)]}

    async def agent_node(state: ReactAgentState) -> Dict[str, Any]:
        iterations = state.iterations + 1
        logger.info(f"ReAct agent iteration {iterations}/{max_iterations}")

        if iterations > max_iterations:
            stop_msg = AIMessage(
                content=(
                    "已达到最大工具调用次数。请基于目前已有的信息，"
                    "直接给出最完整的回答。如果信息不足，请诚实告知用户。"
                )
            )
            return {"messages": [stop_msg], "iterations": iterations}

        messages = state.messages
        if not messages:
            return {"messages": [AIMessage(content="请提供您的问题。")], "iterations": iterations}

        response = await agent_llm.ainvoke(messages)
        return {"messages": [response], "iterations": iterations}

    def should_continue(state: ReactAgentState) -> str:
        last_msg = state.messages[-1] if state.messages else None
        if last_msg is None:
            return END
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            if state.iterations >= max_iterations:
                return END
            return "tools"
        return END

    graph_builder = StateGraph(ReactAgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("format", _format_tool_results)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "format")
    graph_builder.add_edge("format", "agent")

    compiled = graph_builder.compile()
    logger.info("ReAct agent subgraph compiled successfully")
    return compiled
