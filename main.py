import os
import json
import asyncio
import logging
import uuid
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
load_dotenv()

# ADK imports google.genai at import time and checks for GOOGLE_API_KEY.
# Inject a placeholder when only GROQ_API_KEY is set so the import succeeds.
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GROQ_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "placeholder-not-used"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx  # noqa: F401 — kept for future use

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams
from google.adk.runners import Runner
from google.genai import types

log = logging.getLogger("greater-agent-service")
logging.basicConfig(level=logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────

MAX_RETRIES = 4
BASE_BACKOFF = 20
MCP_SSE_URL = os.getenv("MCP_SSE_URL", "http://localhost:8080/mcp/sse")
DEFAULT_MODEL = os.getenv("MODEL", "gemini-1.5-flash")
REDIS_URL = os.getenv("REDIS_URL", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


def _parse_retry_delay(msg: str) -> float:
    m = re.search(r"(?:retry|try again) in (\d+(?:\.\d+)?)s", str(msg), re.IGNORECASE)
    return float(m.group(1)) + 0.5 if m else 0.0


# ── Session service ───────────────────────────────────────────────────────────

def _build_session_service():
    if REDIS_URL:
        try:
            from google.adk.sessions import RedisSessionService
            svc = RedisSessionService(redis_url=REDIS_URL)
            log.info("Using Redis session service: %s", REDIS_URL)
            return svc
        except (ImportError, Exception) as e:
            log.warning("RedisSessionService unavailable (%s), falling back to InMemory", e)
    log.info("REDIS_URL not set — using InMemorySessionService (dev only)")
    from google.adk.sessions import InMemorySessionService
    return InMemorySessionService()


# ── Deployed agents registry ──────────────────────────────────────────────────

class DeployedAgent:
    def __init__(self, agent_id: str, config: "DeployRequest"):
        self.agent_id = agent_id
        self.config = config
        self.status = "running"
        self.deployed_at = asyncio.get_event_loop().time()


_deployed_agents: dict[str, DeployedAgent] = {}
_adk_toolset: Optional[McpToolset] = None
session_service = _build_session_service()

# ── Single-agent mode env vars ────────────────────────────────────────────────
AGENT_ID          = os.getenv("AGENT_ID", "")
AGENT_NAME        = os.getenv("AGENT_NAME", "")
SYSTEM_PROMPT     = os.getenv("SYSTEM_PROMPT", "")
AGENT_LLM         = os.getenv("AGENT_LLM", "")
AGENT_API_KEY     = os.getenv("AGENT_API_KEY", "")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
MCP_SERVER_URLS_RAW = os.getenv("MCP_SERVER_URLS", "")
BACKEND_URL       = os.getenv("BACKEND_URL", "")  # e.g. http://your-backend:3001


# ── Schemas (defined before lifespan so they can be used there) ───────────────

class ToolConfig(BaseModel):
    type: str = "mcp"
    name: str = ""
    url: str = ""
    credential_ref: str = ""


class DeployRequest(BaseModel):
    agent_id: str
    agent_name: str
    system_prompt: str
    llm: str = ""
    api_key: str = ""
    temperature: float = 0.7
    tools: list[ToolConfig] = []


class ChatRequest(BaseModel):
    message: str
    agent_id: str = ""
    agent_name: str = "Agent"
    llm: str = ""
    api_key: str = ""
    system_prompt: str = ""
    temperature: float = 0.7
    session_id: str = ""
    max_retries: int = MAX_RETRIES
    base_backoff: int = BASE_BACKOFF
    tools: list[ToolConfig] = []


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _adk_toolset
    # Only connect to MCP if a real URL is configured and not the default placeholder
    if MCP_SSE_URL and "localhost:8080" not in MCP_SSE_URL:
        _adk_toolset = McpToolset(connection_params=SseConnectionParams(url=MCP_SSE_URL))
        log.info("Default ADK toolset initialised for %s", MCP_SSE_URL)
    else:
        log.info("No MCP server configured — skipping default toolset")

    # Auto-deploy if running in single-agent container mode
    if AGENT_ID and AGENT_NAME and SYSTEM_PROMPT:
        # Build tool configs from MCP_SERVER_URLS env var
        tools = []
        if MCP_SERVER_URLS_RAW:
            for url in MCP_SERVER_URLS_RAW.split(","):
                url = url.strip()
                if url:
                    tools.append(ToolConfig(type="mcp", url=url))
                    log.info("Auto-wiring MCP server: %s", url)

        req = DeployRequest(
            agent_id=AGENT_ID,
            agent_name=AGENT_NAME,
            system_prompt=SYSTEM_PROMPT,
            llm=AGENT_LLM or DEFAULT_MODEL,
            api_key=AGENT_API_KEY,
            temperature=AGENT_TEMPERATURE,
            tools=tools,
        )
        _deployed_agents[AGENT_ID] = DeployedAgent(AGENT_ID, req)
        log.info("Auto-deployed agent %s (%s) with %d MCP server(s)", AGENT_ID, AGENT_NAME, len(tools))

    yield
    log.info("Agent service shutting down. %d agents were deployed.", len(_deployed_agents))


app = FastAPI(title="Greater Agents — Agent Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model resolution ──────────────────────────────────────────────────────────

def _resolve_model(llm: str, api_key: str = ""):
    """Return LiteLlm for non-Gemini models, plain string for Gemini."""
    if not llm:
        llm = DEFAULT_MODEL

    # Inject API key into environment for LiteLLM
    if api_key:
        if llm.startswith("groq/"):
            os.environ["GROQ_API_KEY"] = api_key
        elif llm.startswith("openai/"):
            os.environ["OPENAI_API_KEY"] = api_key
        elif llm.startswith("anthropic/"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            # Assume Gemini / Google
            os.environ["GOOGLE_API_KEY"] = api_key

    if llm.startswith(("groq/", "anthropic/", "ollama/", "openai/", "azure/")):
        return LiteLlm(model=llm)
    return llm


def _build_toolsets(tools: list[ToolConfig]) -> list:
    if not tools:
        return [_adk_toolset] if _adk_toolset else []
    toolsets = []
    for t in tools:
        if t.type == "mcp" and t.url:
            toolsets.append(McpToolset(connection_params=SseConnectionParams(url=t.url)))
        else:
            log.warning("Skipping unsupported tool type '%s' / empty url", t.type)
    return toolsets if toolsets else ([_adk_toolset] if _adk_toolset else [])


# ── Core streaming runner ─────────────────────────────────────────────────────

async def _post_run(agent_id: str, trigger_type: str, status: str,
                    started_at: str, ended_at: str, output: dict) -> None:
    """Post a run record to the backend (best-effort, never fails the chat)."""
    if not BACKEND_URL or not agent_id:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{BACKEND_URL}/api/agents/{agent_id}/runs",
                json={
                    "trigger_type": trigger_type,
                    "status": status,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "output": output,
                },
            )
    except Exception as e:
        log.debug("Failed to post run record: %s", e)
    message: str,
    agent_name: str,
    model,
    system_prompt: str,
    toolsets: list,
    session_id: str,
    max_retries: int = MAX_RETRIES,
    base_backoff: int = BASE_BACKOFF,
) -> AsyncGenerator[str, None]:
    if not system_prompt.strip():
        yield f"data: {json.dumps({'type': 'error', 'content': 'system_prompt is required.'})}\n\n"
        return

    runner = Runner(
        agent=LlmAgent(
            model=model,
            name=agent_name.replace(" ", "_"),
            instruction=system_prompt,
            tools=toolsets if toolsets else [],
        ),
        app_name="greater-agents",
        session_service=session_service,
        auto_create_session=True,
    )
    user_message = types.Content(role="user", parts=[types.Part(text=message)])
    final_text = ""

    for attempt in range(max_retries + 1):
        run_error = None
        final_text = ""
        try:
            async for event in runner.run_async(
                user_id="portal_user",
                session_id=session_id,
                new_message=user_message,
            ):
                if event.get_function_calls():
                    for call in event.get_function_calls():
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': call.name, 'args': dict(call.args)})}\n\n"
                        await asyncio.sleep(0)

                if event.get_function_responses():
                    for resp in event.get_function_responses():
                        yield f"data: {json.dumps({'type': 'tool_result', 'tool': resp.name, 'result': str(resp.response)})}\n\n"
                        await asyncio.sleep(0)

                if not event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            yield f"data: {json.dumps({'type': 'text_delta', 'content': part.text})}\n\n"
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

        if is_rate and is_exhausted:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Daily quota exhausted. Switch model or API key.'})}\n\n"
            return

        if is_rate and attempt < max_retries:
            suggested = _parse_retry_delay(err_str)
            wait = max(suggested + 3, base_backoff * (2 ** attempt))
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': 'rate_limit_backoff', 'result': f'Rate limit — retrying in {wait:.0f}s ({attempt + 1}/{max_retries})...'})}\n\n"
            await asyncio.sleep(wait)
            continue

        if is_rate:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Rate limit exceeded after {max_retries} retries.'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'error', 'content': err_str})}\n\n"
        return

    yield f"data: {json.dumps({'type': 'final', 'content': final_text or 'Done.', 'session_id': session_id})}\n\n"


# ── Deploy endpoints ──────────────────────────────────────────────────────────

@app.post("/agent/deploy")
async def deploy_agent(req: DeployRequest):
    """Register an agent config so it's ready to serve chat requests."""
    _deployed_agents[req.agent_id] = DeployedAgent(req.agent_id, req)
    log.info("Deployed agent %s (%s) with LLM %s", req.agent_id, req.agent_name, req.llm)
    return {
        "agent_id": req.agent_id,
        "status": "running",
        "message": f"Agent '{req.agent_name}' deployed successfully",
    }


@app.get("/agent/status/{agent_id}")
async def agent_status(agent_id: str):
    da = _deployed_agents.get(agent_id)
    if not da:
        return {"agent_id": agent_id, "status": "stopped"}
    return {
        "agent_id": agent_id,
        "status": da.status,
        "agent_name": da.config.agent_name,
        "llm": da.config.llm,
        "deployed_at": da.deployed_at,
    }


@app.delete("/agent/undeploy/{agent_id}")
async def undeploy_agent(agent_id: str):
    da = _deployed_agents.pop(agent_id, None)
    if not da:
        raise HTTPException(404, f"Agent {agent_id} not deployed")
    log.info("Undeployed agent %s", agent_id)
    return {"agent_id": agent_id, "status": "stopped"}


@app.get("/agent/list")
async def list_deployed():
    return [
        {"agent_id": k, "status": v.status, "agent_name": v.config.agent_name, "llm": v.config.llm}
        for k, v in _deployed_agents.items()
    ]


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/agent/chat")
async def agent_chat(request: ChatRequest):
    # If agent_id given, use deployed agent config as defaults
    if request.agent_id and request.agent_id in _deployed_agents:
        da = _deployed_agents[request.agent_id]
        llm = request.llm or da.config.llm
        api_key = request.api_key or da.config.api_key
        system_prompt = request.system_prompt or da.config.system_prompt
        tools = request.tools or da.config.tools
        agent_name = request.agent_name or da.config.agent_name
    else:
        llm = request.llm
        api_key = request.api_key
        system_prompt = request.system_prompt
        tools = request.tools
        agent_name = request.agent_name

    model = _resolve_model(llm, api_key)
    toolsets = _build_toolsets(tools)
    session_id = request.session_id or str(uuid.uuid4())

    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc).isoformat()

    async def stream_with_run_record():
        final_text = ""
        run_status = "completed"
        try:
            async for chunk in _stream_adk(
                message=request.message,
                agent_name=agent_name,
                model=model,
                system_prompt=system_prompt,
                toolsets=toolsets,
                session_id=session_id,
                max_retries=request.max_retries,
                base_backoff=request.base_backoff,
            ):
                yield chunk
                # Track final response
                if b'"type": "final"' in chunk or b'"type":"final"' in chunk:
                    try:
                        import json as _json
                        data = _json.loads(chunk.decode().replace("data: ", "").strip())
                        if data.get("type") == "final":
                            final_text = data.get("content", "")
                    except Exception:
                        pass
                elif b'"type": "error"' in chunk or b'"type":"error"' in chunk:
                    run_status = "error"
        except Exception:
            run_status = "error"
            raise
        finally:
            ended_at = datetime.now(timezone.utc).isoformat()
            await _post_run(
                agent_id=request.agent_id,
                trigger_type="manual",
                status=run_status,
                started_at=started_at,
                ended_at=ended_at,
                output={"message": request.message, "response": final_text[:500]},
            )

    return StreamingResponse(
        stream_with_run_record(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/agent/session/{session_id}")
async def clear_session(session_id: str):
    try:
        await session_service.delete_session(
            app_name="greater-agents",
            user_id="portal_user",
            session_id=session_id,
        )
    except Exception:
        pass
    return {"cleared": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "groq_configured": bool(GROQ_API_KEY),
        "google_configured": bool(GOOGLE_API_KEY and GOOGLE_API_KEY != "placeholder-not-used"),
        "session_backend": "redis" if REDIS_URL else "in-memory",
        "deployed_agents": len(_deployed_agents),
        "deployed_agent_ids": list(_deployed_agents.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
