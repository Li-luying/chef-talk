"""
Unified Chat API with Agent Integration

Provides a single endpoint for chat interactions with automatic routing.
"""
import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from gustobot.application.agents.lg_builder import graph
from gustobot.application.agents.lg_states import InputState
from gustobot.application.services.redis_cache import RedisSemanticCache
from gustobot.config import settings
from gustobot.infrastructure.core.database import get_db
from gustobot.infrastructure.persistence.crud import chat_message, chat_session
from gustobot.interfaces.http.models.chat_message import ChatMessageCreate, ChatMessageResponse
from gustobot.interfaces.http.models.chat_session import ChatSessionCreate, ChatSessionResponse
from sqlalchemy.orm import Session

router = APIRouter()

# ── Progress messages for streaming node transitions ──
_NODE_PROGRESS_MAP: Dict[str, str] = {
    "analyze_and_route_query": "正在分析您的问题...",
    "respond_to_general_query": "正在生成回复...",
    "create_research_plan": "正在查询菜谱知识图谱...",
    "create_kb_query": "正在检索知识库...",
    "get_additional_info": "正在确认需求细节...",
    "create_image_query": "正在处理图片...",
    "create_file_query": "正在处理文件...",
}

# ── 语义缓存单例（模块级，全局复用）──
_semantic_cache: Optional[RedisSemanticCache] = None


def _get_semantic_cache() -> RedisSemanticCache:
    """惰性初始化语义缓存单例。异常时返回一个 disabled 的缓存实例，不阻塞主链路。"""
    global _semantic_cache
    if _semantic_cache is not None:
        return _semantic_cache
    try:
        _semantic_cache = RedisSemanticCache()
        logger.info("Semantic cache initialized")
    except Exception as exc:
        logger.warning("Semantic cache unavailable, caching disabled: {}", exc)
        _semantic_cache = RedisSemanticCache(
            score_threshold=1.0,  # 阈值设为 1.0 确保永不命中
            ttl=1,
        )
    return _semantic_cache


async def _update_semantic_cache(
    cache: RedisSemanticCache,
    messages: List[Dict[str, Any]],
    response: str,
) -> None:
    """后台任务：将问答对写入全局语义缓存。异常不影响主链路。"""
    try:
        await cache.update(messages, response)
    except Exception as exc:
        logger.warning("Semantic cache update failed (non-critical): {}", exc)


def _stringify_content(content: object) -> str:
    """Convert streamed message content into printable text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message", min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field("default_user", description="User identifier")
    stream: bool = Field(False, description="Enable streaming response")
    image_path: Optional[str] = Field(None, description="Path to uploaded image file")
    file_path: Optional[str] = Field(None, description="Path to uploaded file")
    ingest_incremental: Optional[bool] = Field(
        None,
        description="Override whether Excel ingestion uses incremental mode (defaults to server setting)",
    )


class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    session_id: str
    message_id: str
    route: Optional[str] = None
    route_logic: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatStreamChunk(BaseModel):
    """Streaming response chunk"""
    type: str = Field(..., description="Chunk type: 'message', 'metadata', 'error', 'done'")
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    route: Optional[str] = None


def get_or_create_session(db: Session, session_id: Optional[str], user_id: str) -> str:
    """Get existing session or create new one"""
    if session_id:
        session = chat_session.get(db, id=session_id)
        if session:
            return session_id

    # Create new session
    new_session_id = str(uuid.uuid4())
    session_data = ChatSessionCreate(
        id=new_session_id,
        user_id=user_id,
        title=f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    chat_session.create(db, obj_in=session_data)
    return new_session_id


async def save_message(db: Session, session_id: str, message: str, is_user: bool,
                       route: Optional[str] = None, metadata: Optional[Dict] = None):
    """Save message to database"""
    try:
        last_message = chat_message.get_latest_by_session(db, session_id=session_id)
        next_order_index = (last_message.order_index + 1) if last_message else 1

        message_metadata: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            # Avoid persisting huge/unstable objects (e.g. full agent graph state).
            message_metadata.update({k: v for k, v in metadata.items() if k != "agent_state"})
        elif metadata is not None:
            message_metadata["metadata"] = str(metadata)

        if route:
            message_metadata["route"] = route

        message_data = ChatMessageCreate(
            session_id=session_id,
            message_type="user_query" if is_user else "agent_response",
            content=message,
            message_metadata=message_metadata or None,
            order_index=next_order_index,
        )
        created = chat_message.create(db, obj_in=message_data)

        # Update session activity timestamp so session list ordering stays correct.
        chat_session.update_activity(db, session_id=session_id)
        return str(created.id)
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        return None


async def process_agent_query(message: str, session_id: str,
                            image_path: Optional[str] = None,
                            file_path: Optional[str] = None,
                            ingest_incremental: Optional[bool] = None) -> Dict[str, Any]:
    """Process query through agent system"""
    incremental_flag = (
        settings.INGEST_INCREMENTAL_DEFAULT if ingest_incremental is None else bool(ingest_incremental)
    )
    config = {
        "configurable": {
            "thread_id": session_id,
            "image_path": image_path,
            "file_path": file_path,
            "incremental": incremental_flag,
        }
    }

    input_state = InputState(messages=[HumanMessage(content=message)])

    try:
        # Invoke agent graph with timeout protection
        result = await asyncio.wait_for(
            graph.ainvoke(input_state, config=config),
            timeout=settings.AGENT_TIMEOUT,
        )

        # Extract response and metadata
        response_text = ""
        agent_trace = {}
        if result.get("messages"):
            last_msg = result["messages"][-1]
            response_text = last_msg.content
            agent_trace = getattr(last_msg, "additional_kwargs", {}).get("agent_trace", {})

        # Extract route information
        router_info = result.get("router", {})
        route = router_info.get("type")
        route_logic = router_info.get("logic")

        # Extract sources if available
        sources_raw = result.get("sources", [])

        # Convert sources to expected format (list of dicts)
        sources = []
        if sources_raw:
            # If sources is a list of strings, convert to list of dicts
            if isinstance(sources_raw[0], str):
                for src in sources_raw:
                    sources.append({"document_id": src, "source": src})
            else:
                # Already in correct format
                sources = sources_raw

        return {
            "message": response_text,
            "route": route,
            "route_logic": route_logic,
            "sources": sources,
            "metadata": {
                "session_id": session_id,
                "agent_trace": agent_trace,
            },
        }
    except asyncio.TimeoutError:
        logger.warning(f"Agent query timed out after {settings.AGENT_TIMEOUT}s")
        return {
            "message": "抱歉，查询超时了。您的问题比较复杂，请尝试缩小范围后重试。",
            "route": "error",
            "route_logic": "timeout",
            "sources": [],
            "metadata": {"error": "timeout"},
        }
    except Exception as e:
        logger.error(f"Agent query failed: {e}", exc_info=True)
        return {
            "message": "抱歉，处理您的请求时出现了错误。请稍后重试。",
            "route": "error",
            "route_logic": f"Error: {str(e)}",
            "sources": [],
            "metadata": {"error": str(e)},
        }


async def stream_agent_response(message: str, session_id: str,
                               image_path: Optional[str] = None,
                               file_path: Optional[str] = None,
                               ingest_incremental: Optional[bool] = None) -> AsyncGenerator[str, None]:
    """Stream agent response using real graph.astream for token-by-token output."""
    incremental_flag = (
        settings.INGEST_INCREMENTAL_DEFAULT if ingest_incremental is None else bool(ingest_incremental)
    )
    config = {
        "configurable": {
            "thread_id": session_id,
            "image_path": image_path,
            "file_path": file_path,
            "incremental": incremental_flag,
        }
    }
    input_state = InputState(messages=[HumanMessage(content=message)])

    # Send initial metadata
    metadata_chunk = ChatStreamChunk(
        type="metadata",
        metadata={"status": "streaming"},
        session_id=session_id,
    )
    yield f"data: {metadata_chunk.model_dump_json()}\n\n"

    response_text = ""
    route_sent = False
    last_node = ""

    try:
        async for chunk, metadata in graph.astream(
            input=input_state,
            stream_mode="messages",
            config=config,
        ):
            current_node = metadata.get("langgraph_node", "")

            # Emit progress event when node transitions
            if current_node and current_node != last_node:
                last_node = current_node
                progress_msg = _NODE_PROGRESS_MAP.get(current_node)
                if progress_msg:
                    progress_chunk = ChatStreamChunk(
                        type="metadata",
                        metadata={"status": "progress", "node": current_node, "message": progress_msg},
                        session_id=session_id,
                    )
                    yield f"data: {progress_chunk.model_dump_json()}\n\n"

            # Send route info once on first user-visible content
            if not route_sent:
                state_snapshot = graph.get_state(config)
                router_info = {}
                if state_snapshot and state_snapshot.values:
                    router_info = state_snapshot.values.get("router", {})
                route = getattr(router_info, "type", None) or router_info.get("type", "")
                route_chunk = ChatStreamChunk(
                    type="metadata",
                    metadata={"route": route},
                    session_id=session_id,
                    route=route,
                )
                yield f"data: {route_chunk.model_dump_json()}\n\n"
                route_sent = True

            # Skip internal sub-agent messages (ReAct agent tool calls, format nodes)
            tags = metadata.get("tags", [])
            if "research_plan" in tags:
                continue

            # Skip ToolMessage and internal AIMessage chunks (tool_calls only, no user text)
            if chunk.additional_kwargs.get("tool_calls"):
                continue

            text = _stringify_content(chunk.content)
            if text:
                response_text += text
                message_chunk = ChatStreamChunk(
                    type="message",
                    content=text,
                    session_id=session_id,
                )
                yield f"data: {message_chunk.model_dump_json()}\n\n"

        # Extract agent_trace from final state for frontend display
        agent_trace = {}
        final_state = graph.get_state(config)
        if final_state and final_state.values:
            msgs = final_state.values.get("messages", [])
            if msgs:
                agent_trace = getattr(msgs[-1], "additional_kwargs", {}).get("agent_trace", {})

        # Send done signal
        done_chunk = ChatStreamChunk(
            type="done",
            metadata={
                "sources": [],
                "full_response": response_text,
                "agent_trace": agent_trace,
            },
            session_id=session_id,
        )
        yield f"data: {done_chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Streaming failed: {e}", exc_info=True)
        error_chunk = ChatStreamChunk(
            type="error",
            content=f"处理请求时出错: {str(e)}",
            session_id=session_id,
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Unified chat endpoint with automatic agent routing

    - Automatically routes queries to appropriate agents
    - Maintains conversation history
    - Supports file uploads and images
    """
    # Get or create session
    session_id = get_or_create_session(db, request.session_id, request.user_id)

    # Save user message
    await save_message(db, session_id, request.message, is_user=True)

    # ── 语义缓存查找 ──
    cache = _get_semantic_cache()
    cached_response: Optional[str] = None
    # 仅纯文本问题走缓存（图片/文件请求跳过）
    if not request.image_path and not request.file_path:
        cached_response = await cache.lookup(
            [{"role": "user", "content": request.message}],
        )

    if cached_response:
        logger.info("Semantic cache hit for session={}", session_id)
        result = {
            "message": cached_response,
            "route": "cache",
            "route_logic": "semantic_cache_hit",
            "sources": [],
            "metadata": {"cached": True},
        }
    else:
        # Process through agent
        effective_incremental = (
            request.ingest_incremental
            if request.ingest_incremental is not None
            else settings.INGEST_INCREMENTAL_DEFAULT
        )

        result = await process_agent_query(
            request.message,
            session_id,
            request.image_path,
            request.file_path,
            effective_incremental,
        )

        # ── 语义缓存更新（后台任务，不阻塞响应）──
        if not request.image_path and not request.file_path and result.get("message"):
            background_tasks.add_task(
                _update_semantic_cache,
                cache,
                [{"role": "user", "content": request.message}],
                result["message"],
            )

    # Save assistant message
    message_id = await save_message(
        db,
        session_id,
        result["message"],
        is_user=False,
        route=result["route"],
        metadata=result.get("metadata")
    )

    return ChatResponse(
        message=result["message"],
        session_id=session_id,
        message_id=message_id or str(uuid.uuid4()),
        route=result["route"],
        route_logic=result["route_logic"],
        sources=result.get("sources"),
        metadata=result.get("metadata")
    )


# ---------------------------------------------------------------------------
# Legacy alias routes (backwards compatibility)
#
# Older docs/scripts use `/api/v1/chat/chat` and `/api/v1/chat/chat/stream`.
# Keep them working to reduce migration friction.
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse, include_in_schema=False)
async def chat_legacy(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ChatResponse:
    return await chat(request, background_tasks, db)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """
    Streaming chat endpoint with automatic agent routing

    Returns responses in Server-Sent Events (SSE) format
    """
    # Get or create session
    session_id = get_or_create_session(db, request.session_id, request.user_id)

    # Save user message
    await save_message(db, session_id, request.message, is_user=True)

    effective_incremental = (
        request.ingest_incremental
        if request.ingest_incremental is not None
        else settings.INGEST_INCREMENTAL_DEFAULT
    )

    # Return streaming response
    return StreamingResponse(
        stream_agent_response(
            request.message,
            session_id,
            request.image_path,
            request.file_path,
            effective_incremental,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/chat/stream", include_in_schema=False)
async def chat_stream_legacy_post(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    return await chat_stream(request, db)


@router.get("/chat/stream", include_in_schema=False)
async def chat_stream_legacy_get(
    message: str = Query(..., min_length=1, max_length=5000),
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query("default_user"),
    image_path: Optional[str] = Query(None),
    file_path: Optional[str] = Query(None),
    ingest_incremental: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    request = ChatRequest(
        message=message,
        session_id=session_id,
        user_id=user_id,
        stream=True,
        image_path=image_path,
        file_path=file_path,
        ingest_incremental=ingest_incremental,
    )
    return await chat_stream(request, db)


@router.get("/history/{session_id}", response_model=List[ChatMessageResponse])
async def get_chat_history(
    session_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
) -> List[ChatMessageResponse]:
    """
    Get chat history for a session
    """
    # Verify session exists
    session = chat_session.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Get messages
    messages = chat_message.get_by_session(
        db,
        session_id=session_id,
        skip=offset,
        limit=limit
    )

    return messages


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Clear all messages in a session
    """
    session = chat_session.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Delete all messages in session
    chat_message.delete_by_session(db, session_id=session_id)

    return {"message": "Session cleared successfully", "session_id": session_id}


@router.get("/routes")
async def get_route_info() -> Dict[str, Any]:
    """
    Get information about available routes and their purposes
    """
    return {
        "routes": {
            "general-query": {
                "name": "日常对话",
                "description": "处理问候、寒暄等日常对话",
                "examples": ["你好", "谢谢", "今天天气不错"]
            },
            "additional-query": {
                "name": "补充信息",
                "description": "当问题模糊时，询问更多信息",
                "examples": ["我想做菜", "帮我推荐一道菜"]
            },
            "culture-query": {
                "name": "知识库查询",
                "description": "查询宽泛饮食文化、菜系发展史（极少使用）",
                "examples": ["中国茶文化的发展", "鲁菜的演变历史"]
            },
            "recipe-query": {
                "name": "图谱查询",
                "description": "查询具体菜品的做法、食材、烹饪技巧、历史典故、口味特点",
                "examples": ["鱼香肉丝的历史典故", "红烧肉怎么做", "川菜有什么特点"]
            },
            "stats-query": {
                "name": "统计查询",
                "description": "统计分析、计数、排名",
                "examples": ["有多少道菜", "最受欢迎的菜"]
            },
            "image-query": {
                "name": "图片处理",
                "description": "生成或分析图片",
                "examples": ["生成一张红烧肉的图片"]
            },
            "file-query": {
                "name": "文件处理",
                "description": "处理上传的菜谱文件",
                "examples": ["分析这个菜谱文档"]
            }
        },
        "auto_routing": "系统会根据您的问题自动选择合适的处理方式"
    }
