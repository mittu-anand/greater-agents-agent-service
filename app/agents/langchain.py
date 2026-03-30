"""
LangChain AgentExecutor runner.

Libraries lazily imported — only required when framework == "langchain".
Install: pip install langchain langchain-core langchain-openai langchain-google-genai langchain-groq

Supports:
  - Any LangChain-compatible ChatModel
  - MCP tools via langchain_mcp_adapters
  - Custom agent type via extra_config["agent_type"]: "openai-tools" (default) | "react" | "structured-chat"
  - Streaming via astream_events
"""
from __future__ import annotations
import os
import asyncio
import logging
from typing import AsyncGenerator

from app.agents.base import BaseAgentRunner
from app import config

log = logging.getLogger("greater-agent-service.langchain")


def _resolve_langchain_model(llm: str, api_key: str, temperature: float):
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
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _build_mcp_tools(tools: list) -> list:
    lc_tools = []
    for t in tools:
        if t.type == "mcp" and t.url:
            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient
                import asyncio as _as
                client = MultiServerMCPClient({t.name or t.url: {"url": t.url, "transport": "sse"}})
                adapted = _as.get_event_loop().run_until_complete(client.get_tools())
                lc_tools.extend(adapted)
            except ImportError:
                log.warning("LangChain: langchain_mcp_adapters not installed, skipping MCP tool %s", t.url)
            except Exception as e:
                log.warning("LangChain: Failed to load MCP tool %s: %s", t.url, e)
    return lc_tools


class LangChainAgentRunner(BaseAgentRunner):
    """LangChain AgentExecutor with streaming."""

    async def stream(
        self,
        message: str,
        session_id: str,
        max_retries: int,
        base_backoff: int,
    ) -> AsyncGenerator[str, None]:
        try:
            from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.messages import SystemMessage
        except ImportError as e:
            log.error("LangChain import failed: %s", e)
            yield self._error(
                f"LangChain is not installed or incomplete on this agent service ({e}). "
                "Ensure standard 'langchain' and 'langchain-core' are in requirements.txt and redeploy with --no-cache."
            )
            return

        if not self.system_prompt.strip():
            yield self._error("system_prompt is required.")
            return

        try:
            model = _resolve_langchain_model(self.llm, self.api_key, self.temperature)
            tools = _build_mcp_tools(self.tools)

            agent_type = self.extra_config.get("agent_type", "openai-tools")
            max_iters  = self.extra_config.get("max_iterations", 10)
            verbose    = self.extra_config.get("verbose", False)

            if agent_type == "react":
                from langchain import hub
                prompt = self.extra_config.get("prompt") or hub.pull("hwchase17/react")
                agent = create_react_agent(model, tools, prompt)
            else:
                # openai-tools agent (default) — works with any model that supports tool_calls
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ])
                agent = create_openai_tools_agent(model, tools, prompt)

            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=max_iters,
                verbose=verbose,
                handle_parsing_errors=True,
                return_intermediate_steps=True,
            )

        except Exception as e:
            yield self._error(f"LangChain agent build failed: {e}")
            return

        final_text = ""
        try:
            async for event in executor.astream_events(
                {"input": message},
                version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        text = chunk.content
                        if isinstance(text, str):
                            yield self._text_delta(text)
                            final_text += text
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
            yield self._error(f"LangChain execution error: {e}")
            return

        yield self._final(final_text, session_id)
