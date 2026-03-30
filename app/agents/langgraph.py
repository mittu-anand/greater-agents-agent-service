"""
LangGraph runner.

Libraries lazily imported — only required when framework == "langgraph".
Install: pip install langgraph langchain-core langchain-openai langchain-google-genai langchain-groq

Supports:
  - Any LangChain-compatible ChatModel (via model string resolution)
  - MCP tools wrapped as LangChain tools
  - Full streaming of text deltas, tool calls, tool results
  - Configurable graph state via extra_config["state_schema"]
  - Configurable checkpointer backend via extra_config["checkpointer"] ("memory" | "redis")
"""
from __future__ import annotations
import os
import asyncio
import logging
import uuid
from typing import AsyncGenerator, Any

from app.agents.base import BaseAgentRunner
from app import config

log = logging.getLogger("greater-agent-service.langgraph")


def _resolve_langchain_model(llm: str, api_key: str, temperature: float):
    """Return an appropriate LangChain ChatModel based on the llm string."""
    if api_key:
        if llm.startswith("groq/"):
            os.environ["GROQ_API_KEY"] = api_key
        elif llm.startswith("openai/") or llm.startswith("gpt"):
            os.environ["OPENAI_API_KEY"] = api_key
        elif llm.startswith("anthropic/") or llm.startswith("claude"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            os.environ["GOOGLE_API_KEY"] = api_key

    model_name = llm or config.DEFAULT_MODEL

    if model_name.startswith("groq/"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name.replace("groq/", ""), temperature=temperature)

    if model_name.startswith("openai/") or model_name.startswith("gpt"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name.replace("openai/", ""), temperature=temperature)

    if model_name.startswith("anthropic/") or model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name.replace("anthropic/", ""), temperature=temperature)

    if model_name.startswith("ollama/"):
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name.replace("ollama/", ""), temperature=temperature)

    # Default: Google Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _build_mcp_tools(tools: list, extra_config: dict) -> list:
    """Convert ToolConfig list to LangChain-compatible tool objects."""
    lc_tools = []
    for t in tools:
        if t.type == "mcp" and t.url:
            try:
                # Use langchain_mcp_adapters if available
                from langchain_mcp_adapters.client import MultiServerMCPClient
                client = MultiServerMCPClient({t.name or t.url: {"url": t.url, "transport": "sse"}})
                # load_tools is a sync call here — wrap in thread
                import asyncio as _as
                _loop = _as.get_event_loop()
                adapted = _loop.run_until_complete(client.get_tools())
                lc_tools.extend(adapted)
            except ImportError:
                log.warning("LangGraph: langchain_mcp_adapters not installed, skipping MCP tool %s", t.url)
            except Exception as e:
                log.warning("LangGraph: Failed to load MCP tool %s: %s", t.url, e)
    return lc_tools


def _build_checkpointer(extra_config: dict):
    """Build a LangGraph checkpointer from extra_config."""
    backend = extra_config.get("checkpointer", "memory")
    if backend == "redis" and config.REDIS_URL:
        try:
            from langgraph.checkpoint.redis import RedisSaver
            return RedisSaver.from_conn_string(config.REDIS_URL)
        except ImportError:
            log.warning("LangGraph: RedisSaver not available, falling back to MemorySaver")
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


class LangGraphAgentRunner(BaseAgentRunner):
    """LangGraph ReAct agent with streaming."""

    async def stream(
        self,
        message: str,
        session_id: str,
        max_retries: int,
        base_backoff: int,
    ) -> AsyncGenerator[str, None]:
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            yield self._error(
                "LangGraph is not installed on this agent service. "
                "Add 'langgraph langchain-core' to requirements.txt and redeploy."
            )
            return

        if not self.system_prompt.strip():
            yield self._error("system_prompt is required.")
            return

        try:
            model = _resolve_langchain_model(self.llm, self.api_key, self.temperature)
            tools = _build_mcp_tools(self.tools, self.extra_config)
            checkpointer = _build_checkpointer(self.extra_config)

            # Allow custom system message prefix via extra_config
            react_graph = create_react_agent(
                model,
                tools=tools,
                state_modifier=self.system_prompt,
                checkpointer=checkpointer,
            )

        except Exception as e:
            yield self._error(f"LangGraph agent build failed: {e}")
            return

        thread_id = self.extra_config.get("thread_id", session_id)
        run_config = {"configurable": {"thread_id": thread_id}, **self.extra_config.get("run_config", {})}

        final_text = ""
        try:
            async for event in react_graph.astream_events(
                {"messages": [{"role": "user", "content": message}]},
                config=run_config,
                version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    yield self._text_delta(block["text"])
                                    final_text += block["text"]
                        elif isinstance(content, str):
                            yield self._text_delta(content)
                            final_text += content
                    await asyncio.sleep(0)

                elif kind == "on_tool_start":
                    tool_input = event.get("data", {}).get("input", {})
                    yield self._tool_call(name, tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)})
                    await asyncio.sleep(0)

                elif kind == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")
                    yield self._tool_result(name, str(tool_output))
                    await asyncio.sleep(0)

        except Exception as e:
            yield self._error(f"LangGraph execution error: {e}")
            return

        yield self._final(final_text, session_id)
