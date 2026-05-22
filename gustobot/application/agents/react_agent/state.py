from dataclasses import dataclass, field
from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class ReactAgentState:
    messages: Annotated[List[BaseMessage], add_messages]
    question: str = ""
    iterations: int = 0
    final_answer: str = ""
