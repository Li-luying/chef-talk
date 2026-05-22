"""Supervisor Agent — orchestrates 4 worker agents in a multi-agent system.

Pattern: Supervisor → (fan-out) → Worker(s) → (fan-in) → Supervisor → ...
The supervisor LLM outputs a structured decision (SupervisorDecision) via
with_structured_output, forcing chain-of-thought before agent selection.
"""

import asyncio
from typing import Any, Dict, List, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_neo4j import Neo4jGraph
from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from loguru import logger

from gustobot.application.agents.multi_agent.prompts import SUPERVISOR_SYSTEM_PROMPT
from gustobot.application.agents.multi_agent.state import (
    SupervisorDecision,
    SupervisorDecisionRecord,
    SupervisorState,
)
from gustobot.application.agents.multi_agent.workers import create_worker_agents
from gustobot.infrastructure.knowledge import KnowledgeService

# ── 自动扩增阈值 ──
_AUTO_FANOUT_AGENTS: Dict[str, List[str]] = {
    # 如果只选了某个 Agent 且置信度低，自动补充这些 Agent
    "kg_agent":        ["knowledge_agent"],  # 图谱查询 → 补充知识推理
    "knowledge_agent": ["kg_agent"],         # 知识推理 → 补充图谱数据
    "sql_agent":       ["kg_agent"],         # 统计查询 → 补充图谱详情
}


def create_supervisor_agent(
    neo4j_graph: Optional[Neo4jGraph],
    llm: BaseChatModel,
    predefined_cypher_dict: Dict[str, str],
    *,
    max_iterations: int = 5,
    knowledge_service: Optional[KnowledgeService] = None,
) -> CompiledStateGraph:
    """Build the Supervisor-orchestrated multi-agent graph.

    Returns a compiled StateGraph that can be invoked with:
        {"messages": [HumanMessage(content="用户问题")], "question": "用户问题"}
    """

    workers = create_worker_agents(
        neo4j_graph=neo4j_graph,
        llm=llm,
        predefined_cypher_dict=predefined_cypher_dict,
        knowledge_service=knowledge_service,
    )

    # ═══════════════════════════════════════════════════════════════
    # Supervisor 节点 —— 结构化决策，强制 chain-of-thought
    # ═══════════════════════════════════════════════════════════════

    supervisor_llm = llm.with_structured_output(SupervisorDecision)

    async def supervisor_node(state: SupervisorState) -> Dict[str, Any]:
        iterations = state.iterations + 1
        logger.info(f"[Supervisor] Round {iterations}/{max_iterations}")

        if iterations > max_iterations:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "已进行多轮专家咨询。现在请基于所有已收集的信息，"
                            "为用户生成一份完整、融合的最终回答。"
                        )
                    )
                ],
                "iterations": iterations,
                "pending_workers": [],
            }

        msgs = state.messages
        decision: SupervisorDecision = await supervisor_llm.ainvoke(msgs)

        selected = list(decision.selected_agents or [])
        confidence = decision.confidence or "medium"
        reasoning = decision.reasoning or ""

        logger.info(
            f"[Supervisor Decision] round={iterations} agents={selected} "
            f"confidence={confidence} reasoning={reasoning[:100]}"
        )

        # ── 单选自动扩增 ──
        if confidence != "high" and len(selected) == 1:
            primary = selected[0]
            supplement = _AUTO_FANOUT_AGENTS.get(primary, ["knowledge_agent"])[0]
            if supplement not in selected and supplement in workers:
                selected.append(supplement)
                logger.info(
                    f"[Supervisor Auto-Fanout] 单选({primary}, confidence={confidence}) → 自动补充 {supplement}"
                )

        # ── 结构化决策记录 ──
        record = SupervisorDecisionRecord(
            round=iterations,
            selected_agents=selected,
            reasoning=reasoning,
            confidence=confidence,
            fallback_plan=(
                f"如果 {selected[0]} 无结果，自动补充 {_AUTO_FANOUT_AGENTS.get(selected[0], ['rag_agent'])[0]}"
                if len(selected) == 1 else ""
            ),
            question_snapshot=(state.question or "")[:100],
        )

        history = list(state.decision_history or [])
        history.append(record.model_dump())

        # 将决策 reasoning 作为 AIMessage 写入对话历史，供下一轮 Supervisor 参考
        decision_msg = AIMessage(
            content=f"[决策] {reasoning}\n[选中] {', '.join(selected) if selected else '无需调用专家'}\n[置信度] {confidence}"
        )

        return {
            "messages": [decision_msg],
            "iterations": iterations,
            "decision_history": history,
            "pending_workers": selected,
        }

    # ═══════════════════════════════════════════════════════════════
    # Worker 并行调用节点 —— 用 asyncio.gather 同时执行多个 Worker
    # ═══════════════════════════════════════════════════════════════

    async def _run_one_worker(
        worker_name: str, question: str, state: SupervisorState
    ) -> tuple[str, str, Optional[str]]:
        """Execute a single worker and return (worker_name, answer, tool_call_id)."""
        worker = workers.get(worker_name)
        if not worker:
            return worker_name, f"未知专家: {worker_name}", None

        logger.info(f"[Supervisor] → invoking {worker_name} with: {question[:80]}...")
        try:
            result = await worker.ainvoke({
                "messages": [HumanMessage(content=question)],
                "question": question,
            })
            final_msgs = result.get("messages", [])
            answer = ""
            for m in reversed(final_msgs):
                if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                    answer = m.content
                    break
            if not answer and final_msgs:
                answer = str(final_msgs[-1].content)
            return worker_name, answer, None
        except Exception as exc:
            logger.error(f"[Supervisor] Worker {worker_name} failed: {exc}")
            return worker_name, f"[{worker_name}] 执行失败: {exc}", None

    async def invoke_workers_node(state: SupervisorState) -> Dict[str, Any]:
        """Execute all pending workers in parallel via asyncio.gather.

        Each worker receives the original user question directly — the worker's
        own prompt tells it which domain it specializes in, so it will naturally
        focus on its own angle of the question.
        """
        pending = state.pending_workers or []
        if not pending:
            return {}

        question = state.question or ""
        logger.info(f"[Supervisor] ⚡ 并行执行 {len(pending)} 个 Worker: {pending}")

        # Parallel execution — all workers get the same user question
        tasks = [
            _run_one_worker(name, question, state)
            for name in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results as ToolMessage for conversation history
        tool_msgs: List[ToolMessage] = []
        new_worker_results = dict(state.worker_results or {})

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[Supervisor] Worker task exception: {result}")
                continue
            worker_name, answer, _ = result
            new_worker_results[worker_name] = answer
            tool_msgs.append(
                ToolMessage(
                    content=f"[{worker_name} 的报告]\n\n{answer}",
                    tool_call_id=worker_name,
                    name=f"handoff_to_{worker_name}",
                )
            )

        return {
            "messages": tool_msgs,
            "worker_results": new_worker_results,
            "pending_workers": [],
        }

    # ═══════════════════════════════════════════════════════════════
    # 条件路由 —— 解读 Supervisor 的决策并分发
    # ═══════════════════════════════════════════════════════════════

    def route_supervisor_decision(
        state: SupervisorState,
    ) -> Literal["invoke_workers", "final_answer"] | str:
        """Route to workers if there are pending workers, otherwise final_answer."""
        pending = state.pending_workers or []
        if pending:
            return "invoke_workers"
        return "final_answer"

    # ═══════════════════════════════════════════════════════════════
    # Final Answer 节点
    # ═══════════════════════════════════════════════════════════════

    async def final_answer_node(state: SupervisorState) -> Dict[str, Any]:
        """Synthesise final answer from all worker results and conversation."""
        logger.info("[Supervisor] Generating final answer")

        # ── 汇总调度决策日志 ──
        decision_history = state.decision_history or []
        agents_used: set = set()
        for d in decision_history:
            agents_used.update(d.get("selected_agents", []))
        logger.info(
            f"[Supervisor Summary] total_rounds={len(decision_history)} "
            f"agents_used={list(agents_used)} "
            f"worker_results_keys={list(state.worker_results.keys()) if state.worker_results else []}"
        )

        synthesis_prompt = SystemMessage(
            content=(
                "你是菜谱客服的最终答案生成器。请综合以下对话中所有专家的意见，"
                "生成一份完整、连贯、友好的中文回答。"
                "要融合各位专家的贡献，而非简单拼接。"
                "如果信息不足，诚实告知用户。"
            )
        )

        msgs = state.messages
        synthesis_input = [synthesis_prompt] + list(msgs) + [
            HumanMessage(content="请综合以上所有信息，给出最终回答。")
        ]

        response = await llm.ainvoke(synthesis_input)
        content = getattr(response, "content", "") or str(response)
        return {
            "messages": [AIMessage(content=content)],
            "final_decision_log": {
                "total_rounds": len(decision_history),
                "agents_used": list(agents_used),
                "decision_history": decision_history,
            },
        }

    # ═══════════════════════════════════════════════════════════════
    # 构建图
    # ═══════════════════════════════════════════════════════════════

    graph_builder = StateGraph(SupervisorState)

    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("invoke_workers", invoke_workers_node)
    graph_builder.add_node("final_answer", final_answer_node)

    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges(
        "supervisor",
        route_supervisor_decision,
        {"invoke_workers": "invoke_workers", "final_answer": "final_answer"},
    )
    graph_builder.add_edge("invoke_workers", "supervisor")
    graph_builder.add_edge("final_answer", END)

    compiled = graph_builder.compile()
    logger.info("[Supervisor] Multi-agent orchestration graph compiled")
    return compiled
