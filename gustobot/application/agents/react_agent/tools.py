"""LangChain Tools wrapping existing GustoBot tool implementations for ReAct Agent."""

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from loguru import logger

from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.cypher_tools.node import (
    create_cypher_query_node,
)
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.predefined_cypher.node import (
    create_predefined_cypher_node,
)
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.customer_tools.node import (
    get_lightrag_api,
)
from gustobot.application.agents.kg_sub_graph.agentic_rag_agents.components.text2cypher.text2sql_tool import (
    create_text2sql_tool_node,
)


def _extract_text_from_cypher_result(result: Dict[str, Any]) -> str:
    """Extract readable text from a cypher/lightrag query result dict."""
    cyphers = result.get("cyphers", [])
    if not cyphers:
        return "查询未返回任何结果。"

    parts: List[str] = []
    for c in cyphers:
        errors = c.get("errors", [])
        if errors:
            parts.append(f"错误: {'; '.join(errors)}")

        records = c.get("records", {})
        if not records:
            continue

        # 兼容两种格式：{"result": [...]} 或裸 list
        if isinstance(records, list):
            parts.append(_format_neo4j_records(records))
            continue

        if isinstance(records, dict):
            content = records.get("result", "")
            if isinstance(content, list):
                parts.append(_format_neo4j_records(content))
            elif isinstance(content, str) and content.strip():
                parts.append(content)
            elif isinstance(content, dict) and content:
                parts.append(str(content))

    return "\n".join(parts) if parts else "查询完成，但无有效数据返回。"


def _format_neo4j_records(records: List[Any]) -> str:
    """Format Neo4j query result records into readable text."""
    if not records:
        return "无数据。"
    lines: List[str] = []
    for i, record in enumerate(records[:20]):
        if hasattr(record, "data"):
            lines.append(str(record.data()))
        elif isinstance(record, dict):
            lines.append(str(record))
        else:
            lines.append(str(record))
    if len(records) > 20:
        lines.append(f"... 共 {len(records)} 条结果，仅展示前 20 条")
    return "\n".join(lines)


def build_react_tools(
    neo4j_graph: Optional[Neo4jGraph],
    llm: ChatOpenAI,
    predefined_cypher_dict: Dict[str, str],
) -> List:
    """Build the list of LangChain Tools for the ReAct agent."""

    # Pre-create the node functions (each is async, takes Dict[str, Any] state)
    cypher_node = create_cypher_query_node()
    predefined_node = create_predefined_cypher_node(
        graph=neo4j_graph, predefined_cypher_dict=predefined_cypher_dict
    )
    text2sql_node = create_text2sql_tool_node(neo4j_graph)

    @tool
    async def predefined_cypher(task: str, query_name: str = "") -> str:
        """使用预定义的 Cypher 查询模板快速查询菜谱知识图谱。比 cypher_query 更快更准，优先使用。

        可选模板 (query_name)：
        - dish_complete_info: 查菜品的综合信息（做法、耗时、口味、工艺、类型）
        - dish_instructions: 查菜品的完整做法文本
        - dish_flavor: 查菜品的口味标签
        - dish_cooking_method: 查菜品的烹饪工艺
        - ingredients_of_dish: 查菜品的所有食材和用量
        - cooking_steps: 查菜品的分步烹饪步骤
        - dishes_by_flavor: 按口味筛选菜品列表
        - dishes_by_main_ingredient: 按主食材反查菜品
        - similar_dishes: 找口味相似的菜品
        - ingredient_nutrition: 查食材的营养价值
        - dishes_by_method: 按烹饪工艺筛选菜品

        参数 task：用中文描述查询意图，例如 '鱼香肉丝的完整信息'。
        参数 query_name：指定要使用的模板名称（从上面列表选），留空则自动匹配。
        """
        if not task or not task.strip():
            return "错误: 请提供具体的查询描述。"
        try:
            state: Dict[str, Any] = {"task": task, "steps": []}
            if query_name:
                state["query_name"] = query_name
            logger.info(f"[predefined_cypher] task={task[:80]} query_name={query_name}")
            result = await predefined_node(state)
            text = _extract_text_from_cypher_result(result)
            logger.info(f"[predefined_cypher] result_len={len(text)} preview={text[:150]}")
            return text
        except Exception as exc:
            logger.error(f"predefined_cypher tool failed: {exc}")
            return f"预定义查询失败: {exc}"

    @tool
    async def cypher_query(task: str) -> str:
        """对 Neo4j 菜谱知识图谱执行动态 Text2Cypher 查询。

        适用场景：菜品做法、食材搭配、口味特征、烹饪技法、菜系归属等结构化图谱查询。
        参数 task：用中文详细描述你要查询的内容，例如 '红烧肉需要哪些食材和调料'。
        """
        if not task or not task.strip():
            return "错误: 请提供具体的查询描述。"
        try:
            logger.info(f"[cypher_query] task={task[:80]}")
            result = await cypher_node({"task": task, "steps": []})
            text = _extract_text_from_cypher_result(result)
            logger.info(f"[cypher_query] result_len={len(text)} preview={text[:150]}")
            return text
        except Exception as exc:
            logger.error(f"cypher_query tool failed: {exc}")
            return f"图谱查询失败: {exc}"

    @tool
    async def graphrag_query(query: str) -> str:
        """调用 LightRAG 进行深度图谱推理和知识检索。

        适用场景：菜谱历史典故、文化背景、多跳推理、复杂关联问题。
        参数 query：完整的中文问题，例如 '宫保鸡丁的历史典故和命名由来是什么'。
        """
        if not query or not query.strip():
            return "错误: 请提供具体的查询内容。"
        try:
            lightrag_api = get_lightrag_api()
            logger.info(f"[graphrag_query] 开始 LightRAG 查询: {query[:100]}")
            result = await lightrag_api.query(query)
            response = result.get("response", "")
            mode = result.get("mode", "hybrid")
            logger.info(
                f"[graphrag_query] LightRAG 返回: mode={mode} len={len(response)} "
                f"preview={response[:150] if response else '(empty)'}"
            )
            return response if response else f"LightRAG ({mode}) 未返回结果。"
        except Exception as exc:
            logger.error(f"graphrag_query tool failed: {exc}", exc_info=True)
            return f"图谱推理查询失败: {exc}"

    @tool
    async def text2sql_query(task: str) -> str:
        """将自然语言转换为 SQL 并对菜谱数据库执行统计查询。

        适用场景：统计数量、排名、占比、平均值等数值聚合分析。
        参数 task：用中文描述统计需求，例如 '统计各菜系的菜品数量并按降序排列'。
        """
        if not task or not task.strip():
            return "错误: 请提供具体的统计查询描述。"
        try:
            result = await text2sql_node({"task": task, "steps": []})
            return _extract_text_from_cypher_result(result)
        except Exception as exc:
            logger.error(f"text2sql_query tool failed: {exc}")
            return f"SQL 统计查询失败: {exc}"

    tools = [cypher_query, predefined_cypher, graphrag_query, text2sql_query]
    logger.info(f"Built {len(tools)} ReAct agent tools")
    return tools
