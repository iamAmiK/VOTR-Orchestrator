from __future__ import annotations

"""
FastAPI service for the MCP Orchestrator.

Endpoints:
  GET  /health                      — health check (pings the router too)
  POST /chat                        — send a message, get an agent response
  POST /servers/register            — register an MCP server (stdio)
  POST /servers/register/sse        — register an MCP server (SSE/HTTP)
  GET  /servers                     — list registered servers
  DELETE /servers/{name}            — remove a server from the registry
  POST /session/clear               — clear session history
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orchestrator.agent import Orchestrator
from orchestrator.config import OrchestratorConfig, load_config, openai_api_key


# ── App lifecycle ─────────────────────────────────────────────────────────────

_orch: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    if _orch is None:
        raise HTTPException(503, detail="Orchestrator not initialised")
    return _orch


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orch
    cfg: OrchestratorConfig = app.state.cfg  # type: ignore[attr-defined]
    if not openai_api_key(cfg):
        raise RuntimeError(
            f"Environment variable '{cfg.openai_api_key_env}' is not set. "
            "The orchestrator requires an OpenAI API key."
        )
    _orch = Orchestrator(cfg)
    yield


def create_app(cfg: Optional[OrchestratorConfig] = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(
        title="MCP Orchestrator",
        version="0.1.0",
        description=(
            "LangChain agent orchestrator backed by MCP-Router vector retrieval. "
            "Decomposes user prompts into routing hops, retrieves relevant MCP tools, "
            "and executes them via the LangChain tool-calling agent."
        ),
        lifespan=lifespan,
    )
    app.state.cfg = cfg  # type: ignore[attr-defined]
    _register_routes(app)

    # Local dev: Swagger at /swagger/, external tools, and localhost vs 127.0.0.1 mixes.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Bundled OpenAPI + Swagger UI (also served by FastAPI at /openapi.json and /docs)
    swagger_dir = Path(__file__).resolve().parents[1] / "swagger"
    if swagger_dir.is_dir():
        app.mount(
            "/swagger",
            StaticFiles(directory=str(swagger_dir), html=True),
            name="swagger",
        )

    return app


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    hops: List[Dict[str, str]]
    tools_found: List[str]
    route_responses: List[Dict[str, Any]]


class RegisterStdioRequest(BaseModel):
    name: str
    command: str
    args: List[str] = []
    description: str = ""
    discovery_timeout: float = 20.0


class RegisterSSERequest(BaseModel):
    name: str
    url: str
    description: str = ""
    discovery_timeout: float = 20.0


# ── Route registration ────────────────────────────────────────────────────────

def _register_routes(app: FastAPI) -> None:

    @app.get("/health", tags=["admin"])
    def health() -> Dict[str, Any]:
        orch = get_orchestrator()
        router_status: Dict[str, Any]
        try:
            router_status = orch.router.health()
        except Exception as exc:
            router_status = {"error": str(exc)}
        return {
            "status": "ok",
            "router": router_status,
            "registered_servers": len(orch.list_servers()),
        }

    @app.post("/chat", response_model=ChatResponse, tags=["agent"])
    def chat(req: ChatRequest) -> ChatResponse:
        orch = get_orchestrator()
        try:
            result = orch.chat(req.message, req.session_id)
            return ChatResponse(**result)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/servers/register", tags=["servers"])
    def register_stdio(req: RegisterStdioRequest) -> Dict[str, Any]:
        orch = get_orchestrator()
        try:
            return orch.register_server_stdio(
                name=req.name,
                command=req.command,
                args=req.args,
                description=req.description,
                discovery_timeout=req.discovery_timeout,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/servers/register/sse", tags=["servers"])
    def register_sse(req: RegisterSSERequest) -> Dict[str, Any]:
        orch = get_orchestrator()
        try:
            return orch.register_server_sse(
                name=req.name,
                url=req.url,
                description=req.description,
                discovery_timeout=req.discovery_timeout,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/servers", tags=["servers"])
    def list_servers() -> Dict[str, Any]:
        return get_orchestrator().list_servers()

    @app.delete("/servers/{server_name}", tags=["servers"])
    def remove_server(server_name: str) -> Dict[str, str]:
        orch = get_orchestrator()
        removed = orch.remove_server(server_name)
        if not removed:
            raise HTTPException(
                status_code=404, detail=f"Server '{server_name}' not found"
            )
        return {"status": "ok", "removed": server_name}

    @app.post("/session/clear", tags=["session"])
    def clear_session(session_id: str) -> Dict[str, str]:
        get_orchestrator().clear_session(session_id)
        return {"status": "ok"}


# ── Standalone entry point ────────────────────────────────────────────────────

app = create_app()
