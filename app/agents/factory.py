"""
Agent factory — resolves the correct runner class from the framework string.

All framework imports are lazy: the runner module is only imported when it's
actually needed. This means the service starts fine even if only some of the
framework libraries are installed.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from app.agents.base import BaseAgentRunner

if TYPE_CHECKING:
    from app.schemas import DeployRequest, ChatRequest

log = logging.getLogger("greater-agent-service.factory")

# Registry maps framework name -> (module path, class name)
_FRAMEWORK_REGISTRY: dict[str, tuple[str, str]] = {
    "adk":       ("app.agents.adk",       "AdkAgentRunner"),
    "langgraph": ("app.agents.langgraph", "LangGraphAgentRunner"),
    "langchain": ("app.agents.langchain", "LangChainAgentRunner"),
    "crewai":    ("app.agents.crew",      "CrewAgentRunner"),
}


def get_runner(
    framework: str,
    *,
    agent_name: str,
    system_prompt: str,
    llm: str,
    api_key: str,
    temperature: float,
    tools: list,
    extra_config: dict,
) -> BaseAgentRunner:
    """
    Lazily instantiate the correct agent runner.

    Raises ValueError for unknown frameworks.
    The runner itself raises ImportError (caught in stream()) if its library
    is missing, so the service stays alive and returns a helpful error message.
    """
    framework = (framework or "adk").lower().strip()

    if framework not in _FRAMEWORK_REGISTRY:
        known = ", ".join(_FRAMEWORK_REGISTRY.keys())
        raise ValueError(
            f"Unknown framework '{framework}'. "
            f"Supported frameworks: {known}. "
            f"Override via AGENT_FRAMEWORK env var or the 'framework' field in the request."
        )

    module_path, class_name = _FRAMEWORK_REGISTRY[framework]

    import importlib
    module = importlib.import_module(module_path)
    runner_cls = getattr(module, class_name)

    log.info("Factory: creating %s runner for agent '%s'", framework, agent_name)

    return runner_cls(
        agent_name=agent_name,
        system_prompt=system_prompt,
        llm=llm,
        api_key=api_key,
        temperature=temperature,
        tools=tools,
        extra_config=extra_config,
    )


def register_framework(name: str, module_path: str, class_name: str) -> None:
    """
    Extend the registry at runtime with a custom framework.
    Allows third-party plugins without modifying this file.

    Example:
        from app.agents.factory import register_framework
        register_framework("autogen", "my_plugin.autogen_runner", "AutoGenRunner")
    """
    _FRAMEWORK_REGISTRY[name.lower()] = (module_path, class_name)
    log.info("Factory: registered custom framework '%s' -> %s.%s", name, module_path, class_name)


def list_frameworks() -> list[str]:
    """Return registered framework names."""
    return list(_FRAMEWORK_REGISTRY.keys())
