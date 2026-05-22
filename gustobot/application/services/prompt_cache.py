"""Prompt caching utilities for LLM cost optimisation.

OpenAI-compatible APIs (including Qwen, DeepSeek, vLLM) automatically cache
the longest common prefix of the messages array across requests when the
prompt exceeds ~1024 tokens.  This module provides a thin helper to ensure
that fixed system prompts are always placed at the very front of the
messages list so the API can identify and cache the shared prefix.

For Anthropic (future): add ``cache_control: {"type": "ephemeral"}`` to the
first content block of each system message.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages import HumanMessage as LangHumanMessage
from loguru import logger

from gustobot.config import settings


def _is_enabled() -> bool:
    return getattr(settings, "PROMPT_CACHE_ENABLED", True)


def prepend_system_prompt(
    system_content: str,
    messages: Sequence[Any],
    *,
    provider: Optional[str] = None,
) -> List[Any]:
    """Return a new messages list with *system_content* inserted at index 0.

    Placing the fixed system prompt first in the array allows the API
    provider to cache the shared prefix across requests, reducing per-request
    token costs by up to 90 % for the system-prompt portion.

    Parameters
    ----------
    system_content:
        The system-prompt text (must be identical across requests).
    messages:
        Existing conversation messages (history + latest user message).
    provider:
        LLM provider hint.  For Anthropic a ``cache_control`` block is
        injected; for other providers the prompt is left as-is since
        caching is handled server-side.
    """
    if not _is_enabled():
        return [{"role": "system", "content": system_content}, *list(messages)]

    effective_provider = (provider or settings.LLM_PROVIDER).lower()

    if effective_provider == "anthropic":
        # Anthropic requires explicit cache_control markers on content blocks.
        sys_msg: Dict[str, Any] = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        return [sys_msg, *list(messages)]

    # OpenAI / Qwen / vLLM / DeepSeek — automatic prefix caching.
    # The key invariant: the system prompt MUST be the first message
    # and MUST be byte-for-byte identical across every request.
    return [{"role": "system", "content": system_content}, *list(messages)]


def build_system_message(
    content: str,
    *,
    provider: Optional[str] = None,
) -> SystemMessage:
    """Build a cache-optimised ``SystemMessage`` for LangGraph workflows.

    When the provider is Anthropic, the returned message includes
    ``cache_control`` markers in its ``additional_kwargs`` so that
    LangChain's Anthropic integration can forward them correctly.
    """
    effective_provider = (provider or settings.LLM_PROVIDER).lower()

    if _is_enabled() and effective_provider == "anthropic":
        return SystemMessage(
            content=content,
            additional_kwargs={
                "cache_control": {"type": "ephemeral"},
            },
        )

    return SystemMessage(content=content)


logger.debug("Prompt caching utility loaded (provider={})", settings.LLM_PROVIDER)
