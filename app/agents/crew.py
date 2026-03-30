"""
CrewAI runner.

Libraries lazily imported — only required when framework == "crewai".
Install: pip install crewai crewai-tools

Supports:
  - Single-agent mode: one Crew with one Agent and one Task (default)
  - Multi-agent mode: pass extra_config["agents"] list to define the crew
  - Dynamic task via the incoming message (fully overridable via extra_config["task"])
  - Process: "sequential" (default) | "hierarchical"
  - Streaming simulation via step_callback — CrewAI itself is sync/blocking,
    so we run it in a thread executor and forward live step events.
  - LLM resolved from any LangChain-compatible provider
"""
from __future__ import annotations
import os
import asyncio
import logging
from typing import AsyncGenerator, Any

from app.agents.base import BaseAgentRunner
from app import config

log = logging.getLogger("greater-agent-service.crewai")


def _resolve_crewai_llm(llm: str, api_key: str, temperature: float):
    """Return a CrewAI LLM wrapper."""
    if api_key:
        if llm.startswith("groq/"):
            os.environ["GROQ_API_KEY"] = api_key
        elif llm.startswith("openai/") or llm.startswith("gpt"):
            os.environ["OPENAI_API_KEY"] = api_key
        elif llm.startswith("anthropic/") or llm.startswith("claude"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            os.environ["GOOGLE_API_KEY"] = api_key

    from crewai import LLM
    model_name = llm or config.DEFAULT_MODEL
    return LLM(model=model_name, temperature=temperature)


def _build_crew_tools(tools: list) -> list:
    """Map generic ToolConfig list to specific CrewAI tool instances."""
    try:
        from crewai_tools import SerperDevTool, WebsiteSearchTool, FileReadTool, ScrapeWebsiteTool
    except ImportError:
        log.warning("CrewAI Tools: crewai-tools not installed. Skipping tool mapping.")
        return []

    crew_tools = []
    # Simple mapping of type strings to classes
    mapping = {
        "serper":           SerperDevTool,
        "crewai/serper":    SerperDevTool,
        "web_search":       WebsiteSearchTool,
        "crewai/web_search": WebsiteSearchTool,
        "file_read":        FileReadTool,
        "crewai/file_read": FileReadTool,
        "scrape":           ScrapeWebsiteTool,
        "crewai/scrape":    ScrapeWebsiteTool,
    }

    for t in tools:
        tool_cls = mapping.get(t.type) or mapping.get(t.name)
        if tool_cls:
            try:
                # Instantiate with extra kwargs (e.g. n_results, file_path, etc.)
                crew_tools.append(tool_cls(**t.extra))
                log.info("CrewAI: Added tool %s", t.type or t.name)
            except Exception as e:
                log.warning("CrewAI: Failed to initialize tool %s: %s", t.type or t.name, e)
    return crew_tools


class CrewAgentRunner(BaseAgentRunner):
    """CrewAI-powered agent runner."""

    async def stream(
        self,
        message: str,
        session_id: str,
        max_retries: int,
        base_backoff: int,
    ) -> AsyncGenerator[str, None]:
        try:
            from crewai import Agent, Task, Crew, Process
        except ImportError:
            yield self._error(
                "CrewAI is not installed on this agent service. "
                "Add 'crewai' to requirements.txt and redeploy."
            )
            return

        if not self.system_prompt.strip():
            yield self._error("system_prompt is required.")
            return

        # ── Live event queue ─────────────────────────────────────────────────
        # CrewAI is synchronous. We use a queue + thread executor to simulate
        # streaming without blocking the event loop.
        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _step_callback(step_output):
            """Called by CrewAI after each agent step."""
            try:
                text = str(step_output)
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    self._tool_result("crew_step", text[:1000])
                )
            except Exception:
                pass

        def _task_callback(task_output):
            """Called by CrewAI after each task completes."""
            try:
                text = str(task_output)
                loop.call_soon_threadsafe(
                    event_queue.put_nowait,
                    self._tool_result("task_complete", text[:1000])
                )
            except Exception:
                pass

        # ── Build LLM ────────────────────────────────────────────────────────
        try:
            llm = _resolve_crewai_llm(self.llm, self.api_key, self.temperature)
        except Exception as e:
            yield self._error(f"CrewAI LLM init failed: {e}")
            return

        # ── Build agents ─────────────────────────────────────────────────────
        # Supports single-agent (default) or multi-agent via extra_config["agents"]
        process_str = self.extra_config.get("process", "sequential")
        process     = Process.hierarchical if process_str == "hierarchical" else Process.sequential
        memory      = self.extra_config.get("memory", False)
        verbose     = self.extra_config.get("verbose", False)

        agents_cfg: list[dict] = self.extra_config.get("agents", [])
        tasks_cfg:  list[dict] = self.extra_config.get("tasks",  [])
        global_tools = _build_crew_tools(self.tools)

        try:
            if agents_cfg:
                # ── Multi-agent / custom crew ──────────────────────────────
                crew_agents = []
                for ag in agents_cfg:
                    crew_agents.append(Agent(
                        role        = ag.get("role", self.agent_name),
                        goal        = ag.get("goal", self.system_prompt),
                        backstory   = ag.get("backstory", ""),
                        llm         = llm,
                        verbose     = ag.get("verbose", verbose),
                        memory      = ag.get("memory", memory),
                        tools       = _build_crew_tools(ag.get("tools", [])) or global_tools,
                        step_callback = _step_callback,
                    ))

                crew_tasks = []
                for i, task in enumerate(tasks_cfg):
                    assigned_agent = crew_agents[task.get("agent_index", 0)]
                    crew_tasks.append(Task(
                        description     = task.get("description", message),
                        expected_output = task.get("expected_output", "A detailed and complete response."),
                        agent           = assigned_agent,
                        callback        = _task_callback,
                    ))

                # If no tasks defined in config, add a default task using the message
                if not crew_tasks:
                    crew_tasks.append(Task(
                        description     = message,
                        expected_output = self.extra_config.get("expected_output", "A detailed and complete response."),
                        agent           = crew_agents[0],
                        callback        = _task_callback,
                    ))

            else:
                # ── Single-agent mode (default) ────────────────────────────
                primary_agent = Agent(
                    role        = self.extra_config.get("role", self.agent_name),
                    goal        = self.extra_config.get("goal", self.system_prompt),
                    backstory   = self.extra_config.get("backstory", ""),
                    llm         = llm,
                    verbose     = verbose,
                    memory      = memory,
                    tools       = global_tools,
                    step_callback = _step_callback,
                )
                crew_agents = [primary_agent]
                crew_tasks = [Task(
                    description     = message,
                    expected_output = self.extra_config.get("expected_output", "A detailed and complete response."),
                    agent           = primary_agent,
                    callback        = _task_callback,
                )]

            crew = Crew(
                agents    = crew_agents,
                tasks     = crew_tasks,
                process   = process,
                memory    = memory,
                verbose   = verbose,
            )

        except Exception as e:
            yield self._error(f"CrewAI crew build failed: {e}")
            return

        # ── Run crew in thread, stream events ────────────────────────────────
        _SENTINEL = object()

        def _run_crew():
            try:
                result = crew.kickoff()
                loop.call_soon_threadsafe(event_queue.put_nowait, result)
            except Exception as e:
                loop.call_soon_threadsafe(event_queue.put_nowait, Exception(str(e)))
            finally:
                loop.call_soon_threadsafe(event_queue.put_nowait, _SENTINEL)

        executor_future = loop.run_in_executor(None, _run_crew)

        final_text = ""
        try:
            while True:
                item = await event_queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    yield self._error(f"CrewAI execution error: {item}")
                    return
                if isinstance(item, str) and item.startswith("data:"):
                    # Already formatted SSE from callbacks
                    yield item
                else:
                    # Final CrewAI result object
                    final_text = str(item)
                    yield self._text_delta(final_text)
        finally:
            await executor_future

        yield self._final(final_text, session_id)
