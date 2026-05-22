"""State definitions for the multi-agent supervisor orchestration."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """LLM 结构化输出的决策模型 —— 直接驱动 Agent 调度。

    关键设计：reasoning 字段排在 selected_agents 前面，
    LLM 自回归生成时必须先输出 reasoning（思考过程）再输出 selected_agents（结论），
    强制形成 chain-of-thought，提高决策准确率。
    """
    reasoning: str = Field(
        description="选择理由，逐条说明为什么选择每个 Agent 以及为什么排除其他 Agent"
    )
    selected_agents: List[Literal["kg_agent", "knowledge_agent", "sql_agent"]] = Field(
        default_factory=list,
        description="选中的专家 Agent 列表。简单问题选 1 个，跨领域问题选 2 个。无需调用专家时空列表。"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="对本次决策的置信度"
    )


class SupervisorDecisionRecord(BaseModel):
    """单次 Supervisor 调度决策的结构化记录，用于可观测性回溯。"""
    round: int
    selected_agents: List[str] = Field(description="本轮选择的 Agent 列表")
    reasoning: str = Field(default="", description="选择理由")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    fallback_plan: str = Field(default="", description="如果无结果时的备选方案")
    question_snapshot: str = Field(default="", description="本轮传给 Agent 的问题摘要")


@dataclass(kw_only=True)
class WorkerState:
    """State used inside each worker subgraph."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str = ""
    iterations: int = 0


@dataclass(kw_only=True)
class SupervisorState:
    """Top-level state for the supervisor-orchestrated multi-agent system."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str = ""
    worker_results: Dict[str, str] = field(default_factory=dict)
    iterations: int = 0
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    final_decision_log: Optional[Dict[str, Any]] = field(default=None)
    pending_workers: List[str] = field(default_factory=list)
