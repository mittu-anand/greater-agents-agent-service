"""
Greater Agents — Agent Service
Thin FastAPI entry point. All agent logic lives in app/agents/*.
"""
from __future__ import annotations
import os
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# ── MUST load env before ADK touches GOOGLE_API_KEY ──────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ADK checks GOOGLE_API_KEY at import time. Inject placeholder when only
# another provider key is present so the import succeeds.
from app import config as cfg
if not os.environ.get("GOOGLE_API_KEY") and (cfg.GROQ_API_KEY or cfg.OPENAI_API_KEY or cfg.ANTHROPIC_API_KEY):
    os.environ["GOOGLE_API_KEY"] = "placeholder-not-used"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx

from app.schemas import DeployRequest, ChatRequest, ToolConfig
from app.agents.factory import get_runner, list_frameworks

logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO))
log = logging.getLogger("greater-agent-service")


# ── Deployed agents registry ──────────────────────────────────────────────────

class DeployedAgent:
    def __init__(self, agent_id: str, req: DeployRequest):
        self.agent_id = agent_id
        self.config   = req
        self.status   = "running"
        self.deployed_at = datetime.now(timezone.utc).isoformat()


_deployed_agents: dict[str, DeployedAgent] = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-deploy when running as a single-agent container
    if cfg.AGENT_ID and cfg.AGENT_NAME and cfg.SYSTEM_PROMPT:
        tools = []
        for url in filter(None, cfg.MCP_SERVER_URLS_RAW.split(",")):
            tools.append(ToolConfig(type="mcp", url=url.strip()))

        req = DeployRequest(
            agent_id      = cfg.AGENT_ID,
            agent_name    = cfg.AGENT_NAME,
            system_prompt = cfg.SYSTEM_PROMPT,
            framework     = cfg.AGENT_FRAMEWORK,
            llm           = cfg.AGENT_LLM or cfg.DEFAULT_MODEL,
            api_key       = cfg.AGENT_API_KEY,
            temperature   = cfg.AGENT_TEMPERATURE,
            tools         = tools,
            config        = cfg.AGENT_CONFIG,
        )
        _deployed_agents[cfg.AGENT_ID] = DeployedAgent(cfg.AGENT_ID, req)
        log.info(
            "Auto-deployed agent %s (%s) | framework=%s | llm=%s | %d MCP server(s)",
            cfg.AGENT_ID, cfg.AGENT_NAME, cfg.AGENT_FRAMEWORK, req.llm, len(tools)
        )

    yield
    log.info("Agent service shutting down. %d agents were deployed.", len(_deployed_agents))


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Greater Agents — Agent Service", lifespan=lifespan)

cors_origins = [o.strip() for o in cfg.CORS_ORIGINS.split(",")] if cfg.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _post_run(agent_id: str, trigger_type: str, status: str,
                    started_at: str, ended_at: str, output: dict) -> None:
    """Post a run record to the backend (best-effort, never fails the chat)."""
    if not cfg.BACKEND_URL or not agent_id:
        return
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{cfg.BACKEND_URL}/api/agents/{agent_id}/runs",
                json={
                    "trigger_type": trigger_type,
                    "status":       status,
                    "started_at":   started_at,
                    "ended_at":     ended_at,
                    "output":       output,
                },
            )
    except Exception as e:
        log.debug("Failed to post run record: %s", e)


# ── Deploy endpoints ──────────────────────────────────────────────────────────

@app.post("/agent/deploy")
async def deploy_agent(req: DeployRequest):
    _deployed_agents[req.agent_id] = DeployedAgent(req.agent_id, req)
    log.info(
        "Deployed agent %s (%s) | framework=%s | llm=%s",
        req.agent_id, req.agent_name, req.framework, req.llm
    )
    return {
        "agent_id":  req.agent_id,
        "status":    "running",
        "framework": req.framework,
        "message":   f"Agent '{req.agent_name}' deployed successfully",
    }


@app.get("/agent/status/{agent_id}")
async def agent_status(agent_id: str):
    da = _deployed_agents.get(agent_id)
    if not da:
        return {"agent_id": agent_id, "status": "stopped"}
    return {
        "agent_id":    agent_id,
        "status":      da.status,
        "agent_name":  da.config.agent_name,
        "framework":   da.config.framework,
        "llm":         da.config.llm,
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
        {
            "agent_id":    k,
            "status":      v.status,
            "agent_name":  v.config.agent_name,
            "framework":   v.config.framework,
            "llm":         v.config.llm,
            "deployed_at": v.deployed_at,
        }
        for k, v in _deployed_agents.items()
    ]


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/agent/chat")
async def agent_chat(request: ChatRequest):
    # Merge deployed config as defaults, request fields override
    if request.agent_id and request.agent_id in _deployed_agents:
        da = _deployed_agents[request.agent_id]
        llm           = request.llm           or da.config.llm
        api_key       = request.api_key       or da.config.api_key
        system_prompt = request.system_prompt or da.config.system_prompt
        tools         = request.tools         or da.config.tools
        agent_name    = request.agent_name    or da.config.agent_name
        framework     = request.framework     or da.config.framework
        temperature   = request.temperature   if request.temperature != cfg.AGENT_TEMPERATURE else da.config.temperature
        extra_config  = {**da.config.config, **request.config}   # request overrides deployed
    else:
        llm           = request.llm
        api_key       = request.api_key
        system_prompt = request.system_prompt
        tools         = request.tools
        agent_name    = request.agent_name
        framework     = request.framework
        temperature   = request.temperature
        extra_config  = request.config

    session_id = request.session_id or str(uuid.uuid4())

    try:
        runner = get_runner(
            framework,
            agent_name    = agent_name,
            system_prompt = system_prompt,
            llm           = llm or cfg.DEFAULT_MODEL,
            api_key       = api_key,
            temperature   = temperature,
            tools         = tools,
            extra_config  = extra_config,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    started_at = datetime.now(timezone.utc).isoformat()

    async def _stream_with_record():
        final_text = ""
        run_status = "completed"
        try:
            async for chunk in runner.stream(
                message      = request.message,
                session_id   = session_id,
                max_retries  = request.max_retries,
                base_backoff = request.base_backoff,
            ):
                yield chunk
                if '"type": "final"' in chunk or '"type":"final"' in chunk:
                    try:
                        import json as _j
                        data = _j.loads(chunk.replace("data: ", "").strip())
                        if data.get("type") == "final":
                            final_text = data.get("content", "")
                    except Exception:
                        pass
                elif '"type": "error"' in chunk or '"type":"error"' in chunk:
                    run_status = "error"
        except Exception:
            run_status = "error"
            raise
        finally:
            ended_at = datetime.now(timezone.utc).isoformat()
            await _post_run(
                agent_id     = request.agent_id,
                trigger_type = "manual",
                status       = run_status,
                started_at   = started_at,
                ended_at     = ended_at,
                output       = {"message": request.message, "response": final_text[:500]},
            )

    return StreamingResponse(
        _stream_with_record(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Session management ────────────────────────────────────────────────────────

@app.delete("/agent/session/{session_id}")
async def clear_session(session_id: str):
    """Clear an ADK session (no-op for other frameworks)."""
    try:
        from app.agents.adk import get_session_service
        svc = get_session_service()
        await svc.delete_session(
            app_name   = cfg.APP_NAME,
            user_id    = "portal_user",
            session_id = session_id,
        )
    except Exception:
        pass
    return {"cleared": session_id}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":              "ok",
        "app_name":            cfg.APP_NAME,
        "default_model":       cfg.DEFAULT_MODEL,
        "default_framework":   cfg.DEFAULT_FRAMEWORK,
        "supported_frameworks": list_frameworks(),
        "google_configured":   bool(cfg.GOOGLE_API_KEY and cfg.GOOGLE_API_KEY != "placeholder-not-used"),
        "groq_configured":     bool(cfg.GROQ_API_KEY),
        "openai_configured":   bool(cfg.OPENAI_API_KEY),
        "anthropic_configured": bool(cfg.ANTHROPIC_API_KEY),
        "session_backend":     "redis" if cfg.REDIS_URL else "in-memory",
        "deployed_agents":     len(_deployed_agents),
        "deployed_agent_ids":  list(_deployed_agents.keys()),
    }


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=cfg.PORT, reload=True)
