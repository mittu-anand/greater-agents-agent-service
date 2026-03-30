"""
Dynamic configuration — every value is overridable via environment variables.
No hard-coded defaults that can't be changed at runtime.
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

# ── Core ──────────────────────────────────────────────────────────────────────

PORT            = int(os.getenv("PORT",              "8001"))
LOG_LEVEL       = os.getenv("LOG_LEVEL",             "INFO")
APP_NAME        = os.getenv("APP_NAME",              "greater-agents")
CORS_ORIGINS    = os.getenv("CORS_ORIGINS",          "*")

# ── LLM defaults ──────────────────────────────────────────────────────────────

DEFAULT_MODEL   = os.getenv("MODEL",                 "gemini-1.5-flash")
DEFAULT_FRAMEWORK = os.getenv("AGENT_FRAMEWORK",     "adk")  # adk | langgraph | langchain | crewai

# ── Retry / backoff ───────────────────────────────────────────────────────────

MAX_RETRIES     = int(os.getenv("MAX_RETRIES",       "4"))
BASE_BACKOFF    = int(os.getenv("BASE_BACKOFF",      "20"))

# ── Session ────────────────────────────────────────────────────────────────────

REDIS_URL       = os.getenv("REDIS_URL",             "")

# ── MCP ───────────────────────────────────────────────────────────────────────

MCP_SSE_URL     = os.getenv("MCP_SSE_URL",           "")

# ── API keys (read once; individual runners may also set env vars) ─────────────

GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY",        "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",          "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",        "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY",   "")

# ── Single-agent auto-deploy (container / k8s env-var driven) ────────────────

AGENT_ID        = os.getenv("AGENT_ID",              "")
AGENT_NAME      = os.getenv("AGENT_NAME",            "")
SYSTEM_PROMPT   = os.getenv("SYSTEM_PROMPT",         "")
AGENT_LLM       = os.getenv("AGENT_LLM",             "")
AGENT_API_KEY   = os.getenv("AGENT_API_KEY",         "")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", str(MAX_RETRIES)))
AGENT_BASE_BACKOFF = int(os.getenv("AGENT_BASE_BACKOFF", str(BASE_BACKOFF)))
AGENT_FRAMEWORK = os.getenv("AGENT_FRAMEWORK",       DEFAULT_FRAMEWORK)
AGENT_CONFIG    = json.loads(os.getenv("AGENT_CONFIG", "{}"))
MCP_SERVER_URLS_RAW = os.getenv("MCP_SERVER_URLS",   "")

# ── Callbacks ─────────────────────────────────────────────────────────────────

BACKEND_URL     = os.getenv("BACKEND_URL",           "")
