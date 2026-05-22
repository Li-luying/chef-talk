"""Redis-backed LangGraph checkpointer for persistent graph state.

Compatible with langgraph==0.2.60 / langgraph-checkpoint==2.0.8.
Stores checkpoint tuples as JSON blobs with configurable TTL.
Falls back to in-memory storage if Redis is unreachable.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)
from langgraph.checkpoint.memory import MemorySaver
from redis.asyncio import Redis
from loguru import logger

from gustobot.config import settings


def _serialize_checkpoint_tuple(
    config: dict,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
) -> bytes:
    """Serialize a checkpoint triple into a JSON byte payload."""
    payload = {
        "config": {k: v for k, v in config.items() if k != "configurable"},
        "thread_id": (config.get("configurable") or {}).get("thread_id", "default"),
        "checkpoint": checkpoint,
        "metadata": metadata,
    }
    return json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")


def _deserialize_checkpoint_tuple(
    raw: bytes,
    config: dict,
) -> Optional[CheckpointTuple]:
    """Deserialize a JSON byte payload back into a CheckpointTuple."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    cp = data.get("checkpoint")
    meta = data.get("metadata")
    if cp is None:
        return None
    return CheckpointTuple(
        config=config,
        checkpoint=cp,
        metadata=meta,
    )


class RedisCheckpointer(BaseCheckpointSaver):
    """Async LangGraph checkpointer backed by Redis with MemorySaver fallback."""

    def __init__(
        self,
        *,
        redis_client: Optional[Redis] = None,
        redis_url: Optional[str] = None,
        db: Optional[int] = None,
        key_prefix: str = "checkpoint",
        ttl: int = 3600,
    ) -> None:
        super().__init__()
        self.key_prefix = key_prefix
        self.ttl = ttl
        self._redis: Optional[Redis] = None
        self._memory = MemorySaver()
        self._redis_available = False

        try:
            self._redis = redis_client or Redis.from_url(
                redis_url or settings.REDIS_URL,
                db=db if db is not None else settings.CHECKPOINT_REDIS_DB,
                decode_responses=False,
            )
        except Exception:
            logger.warning(
                "RedisCheckpointer: Redis 不可用，降级为 MemorySaver"
            )

    async def _ensure_redis(self) -> Optional[Redis]:
        """Lazy-verify Redis connectivity. Returns None if unavailable."""
        if self._redis_available:
            return self._redis
        if self._redis is None:
            return None
        try:
            await self._redis.ping()
            self._redis_available = True
            return self._redis
        except Exception as exc:
            logger.warning(
                "RedisCheckpointer: Redis ping 失败 ({})，使用 MemorySaver 回退", exc
            )
            return None

    def _thread_key(self, thread_id: str) -> str:
        return f"{self.key_prefix}:{thread_id}"

    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        redis = await self._ensure_redis()
        if redis is None:
            return await self._memory.aget_tuple(config)

        thread_id = (config.get("configurable") or {}).get(
            "thread_id", "default"
        )
        key = self._thread_key(thread_id)

        try:
            raw = await redis.get(key)
            if raw is None:
                return None
            result = _deserialize_checkpoint_tuple(raw, config)
            if result is not None:
                logger.debug("RedisCheckpointer: cache hit thread={}", thread_id)
            return result
        except Exception as exc:
            logger.error(
                "RedisCheckpointer: aget_tuple 失败 ({}), 回退", exc
            )
            return await self._memory.aget_tuple(config)

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        redis = await self._ensure_redis()
        if redis is None:
            return await self._memory.aput(
                config, checkpoint, metadata, new_versions
            )

        thread_id = (config.get("configurable") or {}).get(
            "thread_id", "default"
        )
        key = self._thread_key(thread_id)

        try:
            payload = _serialize_checkpoint_tuple(config, checkpoint, metadata)
            await redis.set(key, payload, ex=self.ttl)
            logger.debug("RedisCheckpointer: saved thread={}", thread_id)
        except Exception as exc:
            logger.error(
                "RedisCheckpointer: aput 失败 ({}), 回退 MemorySaver", exc
            )
            return await self._memory.aput(
                config, checkpoint, metadata, new_versions
            )

        return config

    async def alist(
        self,
        config: Optional[dict] = None,
        *,
        filter: Optional[dict] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints — delegates to MemorySaver (Redis stores only latest)."""
        async for item in self._memory.alist(
            config=config, filter=filter, before=before, limit=limit
        ):
            yield item

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for interrupted graph execution."""
        redis = await self._ensure_redis()
        if redis is None:
            await self._memory.aput_writes(config, writes, task_id)
            return

        thread_id = (config.get("configurable") or {}).get(
            "thread_id", "default"
        )
        key = f"{self._thread_key(thread_id)}:writes:{task_id}"

        try:
            payload = json.dumps(
                [(w[0], w[1]) for w in writes],
                default=str,
                ensure_ascii=False,
            ).encode("utf-8")
            await redis.set(key, payload, ex=self.ttl)
        except Exception:
            await self._memory.aput_writes(config, writes, task_id)
