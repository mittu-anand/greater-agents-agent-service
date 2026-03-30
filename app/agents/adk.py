"""
ADK (Google Agent Development Kit) runner.

Migrated from the original main.py _stream_adk function.
All ADK-specific imports are at module level — this module is only imported
when the factory resolves framework == "adk".
"""
from __future__ import annotations
import os
import asyncio
import logging
import re
from typing import AsyncGenerator, Any

from app.agents.base import BaseAgentRunner
from app import config

log = logging.getLogger("greater-agent-service.adk")


def _patch_google_api_key(api_key: str, llm: str) -> None:
    """Inject the correct provider API key into the environment."""
    if not api_key:
        return
    if llm.startswith("groq/"):
        os.environ["GROQ_API_KEY"] = api_key
    elif llm.startswith("openai/"):
        os.environ["OPENAI_API_KEY"] = api_key
    elif llm.startswith("anthropic/"):
        os.environ["ANTHROPIC_API_KEY"] = api_key
    else:
        os.environ["GOOGLE_API_KEY"] = api_key


def _resolve_adk_model(llm: str, api_key: str):
    """Return a LiteLlm wrapper for non-Gemini models, plain string for Gemini."""
    _patch_google_api_key(api_key, llm)
    if llm.startswith(("groq/", "anthropic/", "ollama/", "openai/", "azure/")):
        from google.adk.models.lite_llm import LiteLlm
        return LiteLlm(model=llm)
    return llm


def _parse_retry_delay(msg: str) -> float:
    m = re.search(r"(?:retry|try again) in (\d+(?:\.\d+)?)s", str(msg), re.IGNORECASE)
    return float(m.group(1)) + 0.5 if m else 0.0


def _build_session_service():
    if config.REDIS_URL:
        try:
            from google.adk.sessions import RedisSessionService
            svc = RedisSessionService(redis_url=config.REDIS_URL)
            log.info("ADK: Using Redis session service: %s", config.REDIS_URL)
            return svc
        except (ImportError, Exception) as e:
            log.warning("ADK: RedisSessionService unavailable (%s), falling back to InMemory", e)
    log.info("ADK: REDIS_URL not set — using InMemorySessionService (dev only)")
    from google.adk.sessions import InMemorySessionService
    return InMemorySessionService()


# Singleton session service for the ADK runner
_session_service = None


def get_session_service():
    global _session_service
    if _session_service is None:
        _session_service = _build_session_service()
    return _session_service


def build_adk_toolsets(tools: list, extra_config: dict) -> list:
    """Convert ToolConfig objects into ADK McpToolset instances."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams

    toolsets = []
    for t in tools:
        if t.type == "mcp" and t.url:
            # Allow per-tool sse_options override via extra_config
            sse_opts = t.extra or {}
            toolsets.append(McpToolset(connection_params=SseConnectionParams(url=t.url, **sse_opts)))
        else:
            log.warning("ADK: Skipping unsupported tool type '%s' / empty url", t.type)

    # Fall back to global MCP_SSE_URL if no tools explicitly provided
    if not toolsets and config.MCP_SSE_URL and "localhost:8080" not in config.MCP_SSE_URL:
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams
        toolsets.append(McpToolset(connection_params=SseConnectionParams(url=config.MCP_SSE_URL)))

    return toolsets


class AdkAgentRunner(BaseAgentRunner):
    """Google ADK-powered streaming agent."""

    async def stream(
        self,
        message: str,
        session_id: str,
        max_retries: int,
        base_backoff: int,
    ) -> AsyncGenerator[str, None]:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.genai import types

        if not self.system_prompt.strip():
            yield self._error("system_prompt is required.")
            return

        model = _resolve_adk_model(self.llm, self.api_key)
        toolsets = build_adk_toolsets(self.tools, self.extra_config)

        # Allow extra_config to override runner app_name and user_id
        app_name = self.extra_config.get("app_name", config.APP_NAME)
        user_id  = self.extra_config.get("user_id",  "portal_user")

        runner = Runner(
            agent=LlmAgent(
                model=model,
                name=self.agent_name.replace(" ", "_"),
                instruction=self.system_prompt,
                tools=toolsets or [],
            ),
            app_name=app_name,
            session_service=get_session_service(),
            auto_create_session=True,
        )
        user_message = types.Content(role="user", parts=[types.Part(text=message)])
        final_text = ""

        for attempt in range(max_retries + 1):
            run_error = None
            final_text = ""
            try:
                async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=user_message,
                ):
                    if event.get_function_calls():
                        for call in event.get_function_calls():
                            yield self._tool_call(call.name, dict(call.args))
                            await asyncio.sleep(0)

                    if event.get_function_responses():
                        for resp in event.get_function_responses():
                            yield self._tool_result(resp.name, str(resp.response))
                            await asyncio.sleep(0)

                    if not event.is_final_response() and event.content and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, "text") and part.text:
                                yield self._text_delta(part.text)
                                await asyncio.sleep(0)

                    if event.is_final_response():
                        if event.content and event.content.parts:
                            final_text = "".join(
                                p.text for p in event.content.parts if hasattr(p, "text")
                            )

            except TypeError as e:
                if "NoneType" not in str(e):
                    run_error = e
            except Exception as e:
                run_error = e

            if run_error is None:
                break

            err_str = str(run_error)
            is_rate = any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED", "RateLimitError", "rate_limit_exceeded"])
            is_exhausted = "limit: 0" in err_str or "GenerateRequestsPerDay" in err_str
            is_tool_fail = "tool_use_failed" in err_str or "failed_generation" in err_str

            if is_tool_fail and attempt < max_retries:
                wait = 2 * (attempt + 1)
                yield self._tool_result("retry", f"Tool call format error — retrying ({attempt + 1}/{max_retries})…")
                await asyncio.sleep(wait)
                continue

            if is_tool_fail:
                yield self._error("The model failed to format a tool call correctly after multiple retries. Try rephrasing or switching model.")
                return

            if is_rate and is_exhausted:
                yield self._error("Daily quota exhausted. Switch model or API key.")
                return

            if is_rate and attempt < max_retries:
                suggested = _parse_retry_delay(err_str)
                wait = max(suggested + 3, base_backoff * (2 ** attempt))
                yield self._tool_result("rate_limit_backoff", f"Rate limit — retrying in {wait:.0f}s ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(wait)
                continue

            if is_rate:
                yield self._error(f"Rate limit exceeded after {max_retries} retries.")
                return

            yield self._error(err_str)
            return

        yield self._final(final_text, session_id)
