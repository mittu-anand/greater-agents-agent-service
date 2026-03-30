"""
Centralised Pydantic schemas shared across all framework runners.
Every field has an env-overridable default via the config module.
"""
from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field
from app import config


class ToolConfig(BaseModel):
    type: str = "mcp"        # "mcp" | "openapi" | "function" (future)
    name: str = ""
    url: str = ""
    credential_ref: str = ""
    # Arbitrary extra kwargs forwarded to the tool adapter
    extra: dict[str, Any] = Field(default_factory=dict)


class DeployRequest(BaseModel):
    agent_id: str
    agent_name: str
    system_prompt: str
    framework: Literal["adk", "langgraph", "langchain", "crewai"] = Field(
        default_factory=lambda: config.DEFAULT_FRAMEWORK
    )
    llm: str = Field(default_factory=lambda: config.DEFAULT_MODEL)
    api_key: str = ""
    temperature: float = Field(default_factory=lambda: config.AGENT_TEMPERATURE)
    tools: list[ToolConfig] = Field(default_factory=list)
    # Arbitrary framework-specific overrides (e.g. graph state schema, crew roles)
    config: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str
    agent_id: str = ""
    agent_name: str = "Agent"
    framework: Literal["adk", "langgraph", "langchain", "crewai"] = Field(
        default_factory=lambda: config.DEFAULT_FRAMEWORK
    )
    llm: str = ""
    api_key: str = ""
    system_prompt: str = ""
    temperature: float = Field(default_factory=lambda: config.AGENT_TEMPERATURE)
    session_id: str = ""
    max_retries: int = Field(default_factory=lambda: config.MAX_RETRIES)
    base_backoff: int = Field(default_factory=lambda: config.BASE_BACKOFF)
    tools: list[ToolConfig] = Field(default_factory=list)
    # Arbitrary framework-specific overrides
    config: dict[str, Any] = Field(default_factory=dict)
