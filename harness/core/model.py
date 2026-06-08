"""
Thin async wrapper over the Anthropic API. No agent logic, no parsing — just
a single `generate()` that takes (system, messages, tools, tool_choice) and
returns the raw Message.

Prompt caching: a 1-hour ephemeral cache_control marker is placed on the
system block and on the last reference-image content block, so the (large)
references are cached across frames.
"""
from __future__ import annotations

import os
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-opus-4-6"
MAX_TOKENS = 2048

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def with_cache(block: dict[str, Any] | str, ttl: str = "1h") -> dict[str, Any]:
    """Attach an ephemeral cache_control marker to a content block."""
    if isinstance(block, str):
        block = {"type": "text", "text": block}
    return {**block, "cache_control": {"type": "ephemeral", "ttl": ttl}}


def image_block(b64: str) -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


async def generate(
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: dict[str, Any] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    thinking: dict[str, Any] | None = None,
    effort: str | None = None,
) -> anthropic.types.Message:
    """Single API call. Agent loop lives in core/agent.py, not here.

    `thinking` / `effort` default to None (off) so one-shot parity solvers
    reproduce the old hybrid@4.6 baseline byte-for-byte. Agentic solvers opt in
    via Solver(thinking="adaptive", effort="high").
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [with_cache(system)],
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice or {"type": "auto"},
    }
    if thinking is not None:
        kwargs["thinking"] = thinking
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}
    return await _get_client().messages.create(**kwargs)
