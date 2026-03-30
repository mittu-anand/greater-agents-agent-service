"""
Abstract base class that every framework runner must implement.
All runners must produce a unified SSE stream format:

  data: {"type": "text_delta",  "content": "..."}        # partial text
  data: {"type": "tool_call",   "tool": "...", "args": {...}}
  data: {"type": "tool_result", "tool": "...", "result": "..."}
  data: {"type": "error",       "content": "..."}
  data: {"type": "final",       "content": "...", "session_id": "..."}
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any
from app import config


class BaseAgentRunner(ABC):
    """Every framework runner inherits this and implements `stream()`."""

    def __init__(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        llm: str,
        api_key: str,
        temperature: float,
        tools: list,
        extra_config: dict[str, Any],
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.llm = llm or config.DEFAULT_MODEL
        self.api_key = api_key
        self.temperature = temperature
        self.tools = tools
        self.extra_config = extra_config

    @abstractmethod
    async def stream(
        self,
        message: str,
        session_id: str,
        max_retries: int,
        base_backoff: int,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted strings."""
        ...  # pragma: no cover

    # ── Helpers for consistent SSE output ────────────────────────────────────

    @staticmethod
    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def _text_delta(text: str) -> str:
        return BaseAgentRunner._sse({"type": "text_delta", "content": text})

    @staticmethod
    def _tool_call(name: str, args: dict) -> str:
        return BaseAgentRunner._sse({"type": "tool_call", "tool": name, "args": args})

    @staticmethod
    def _tool_result(name: str, result: str) -> str:
        return BaseAgentRunner._sse({"type": "tool_result", "tool": name, "result": result})

    @staticmethod
    def _error(msg: str) -> str:
        return BaseAgentRunner._sse({"type": "error", "content": msg})

    @staticmethod
    def _final(text: str, session_id: str) -> str:
        return BaseAgentRunner._sse({"type": "final", "content": text or "Done.", "session_id": session_id})
