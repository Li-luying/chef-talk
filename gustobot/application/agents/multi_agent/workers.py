"""Worker Agent subgraphs — each is a mini ReAct agent with domain-specific tools."""

from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from gustobot.application.agents.multi_agent.prompts import (
    KG_AGENT_PROMPT,
    KNOWLEDGE_AGENT_PROMPT,
    SQL_AGENT_PROMPT,
)
from gustobot.application.agents.multi_agent.state import WorkerState
from gustobot.application.agents.react_agent.tools import (
    _extract_text_from_cypher_result,
    build_react_tools,
)
from gustobot.infrastructure.knowledge import KnowledgeService


# ═══════════════════════════════════════════════════════════════════════
# Helper: build a mini ReAct worker graph
# ═══════════════════════════════════════════════════════════════════════

def _build_worker(
    name: str,
    system_prompt: str,
    tools: List,
    llm: BaseChatModel,
    max_iterations: int = 2,
) -> CompiledStateGraph:
    """Build a single worker subgraph: agent ↔ tools loop."""

    agent_llm = llm.bind_tools(tools)
    worker_logger = logger.bind(worker=name)

    async def agent_node(state: WorkerState) -> Dict[str, Any]:
        iterations = state.iterations + 1
        worker_logger.info(f"Iteration {iterations}/{max_iterations}")

        if iterations > max_iterations:
            return {
                "messages": [
                    AIMessage(content=f"[{name}] 已达到最大查询次数，请基于现有结果给出回答。")
                ],
                "iterations": iterations,
            }

        msgs = state.messages
        if not msgs:
            return {"messages": [AIMessage(content="请提供具体问题。")], "iterations": iterations}

        response = await agent_llm.ainvoke(msgs)
        return {"messages": [response], "iterations": iterations}

    def should_continue(state: WorkerState) -> str:
        if state.iterations >= max_iterations:
            return END
        last_msg = state.messages[-1] if state.messages else None
        if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    graph_builder = StateGraph(WorkerState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "agent")

    compiled = graph_builder.compile()
    worker_logger.info("Worker compiled ({})".format(len(tools)))
    return compiled


# ═══════════════════════════════════════════════════════════════════════
# Factory: create all 4 worker agents
# ═══════════════════════════════════════════════════════════════════════

def create_worker_agents(
    neo4j_graph: Optional[Neo4jGraph],
    llm: BaseChatModel,
    predefined_cypher_dict: Dict[str, str],
    knowledge_service: Optional[KnowledgeService] = None,
) -> Dict[str, CompiledStateGraph]:
    """Build all 4 domain-specific worker agent subgraphs.

    Returns a dict mapping worker_name → CompiledStateGraph.
    """

    # ── Tools shared across workers ──
    all_tools = build_react_tools(neo4j_graph, llm, predefined_cypher_dict)
    tool_map: Dict[str, Any] = {t.name: t for t in all_tools}

    # ── KB Agent 专属工具 ──
    kb_service = knowledge_service or KnowledgeService()

    @tool
    async def kb_search(query: str) -> str:
        """知识库向量检索。查询菜谱的历史典故、文化背景、长文本知识。

        适用场景：历史文化知识、菜谱小传、典故故事。
        参数 query：用中文描述要检索的内容。
        """
        try:
            docs = await kb_service.search(query=query, top_k=5)
            if not docs:
                return "向量知识库中未找到相关内容。"
            parts: List[str] = []
            for i, doc in enumerate(docs[:5]):
                content = doc.get("content", "") or doc.get("document", "") or ""
                score = doc.get("score") or doc.get("rerank_score") or 0.0
                source = doc.get("source") or doc.get("id", "") or ""
                parts.append(f"[结果{i+1}] (相似度: {score:.3f})\n{content[:500]}")
                if source:
                    parts[-1] += f"\n来源: {source}"
            return "\n\n".join(parts)
        except Exception as exc:
            logger.error(f"kb_search failed: {exc}")
            return f"知识库检索失败: {exc}"

    # ── Build workers ──
    workers: Dict[str, CompiledStateGraph] = {}

    workers["kg_agent"] = _build_worker(
        name="kg_agent",
        system_prompt=KG_AGENT_PROMPT,
        tools=[tool_map["cypher_query"], tool_map["predefined_cypher"]],
        llm=llm,
    )

    workers["knowledge_agent"] = _build_worker(
        name="knowledge_agent",
        system_prompt=KNOWLEDGE_AGENT_PROMPT,
        tools=[tool_map["graphrag_query"], kb_search],
        llm=llm,
    )

    workers["sql_agent"] = _build_worker(
        name="sql_agent",
        system_prompt=SQL_AGENT_PROMPT,
        tools=[tool_map["text2sql_query"]],
        llm=llm,
    )

    return workers
