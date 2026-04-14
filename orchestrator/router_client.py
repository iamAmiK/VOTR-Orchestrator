from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field


# ── Mirror of the router's response models ──────────────────────────────────

class RoutedTool(BaseModel):
    tool_key: str
    server_name: str
    tool_name: str
    score: float
    compressed: str
    description: str
    parameter: Dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    tools: List[RoutedTool]
    adaptive_k: int
    top1_score: float = 0.0
    top2_score: float = 0.0
    score_gap: float = 0.0
    confidence: str = "low"
    recommended_handoff_k: int = 5
    null_route: bool = False
    overlap_ambiguous: bool = False
    overlap_tool_keys: List[str] = Field(default_factory=list)
    overlap_servers: List[str] = Field(default_factory=list)


# ── HTTP client for MCP-Router ────────────────────────────────────────────────

class RouterClient:
    """Thin synchronous wrapper around the MCP-Router REST API."""

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── Routing ──────────────────────────────────────────────────────────────

    def route(
        self,
        server_intent: str,
        tool_intent: str,
        session_id: Optional[str] = None,
        record_session: bool = True,
    ) -> RouteResponse:
        """
        POST /route — returns ranked tools matching the given intents.

        Per Policy.md:
          server_intent: 4-10 words naming app/domain
          tool_intent  : verb + object + key constraint, 6-16 words
        """
        payload = {
            "server_intent": server_intent,
            "tool_intent": tool_intent,
            "session_id": session_id,
            "record_session": record_session,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/route", json=payload)
            r.raise_for_status()
            return RouteResponse.model_validate(r.json())

    # ── Server registration ───────────────────────────────────────────────────

    def register_discover_stdio(
        self,
        server_name: str,
        server_description: str,
        command: str,
        args: List[str],
        timeout_seconds: float = 20.0,
    ) -> Dict[str, Any]:
        """POST /register/discover — register an MCP server over stdio."""
        payload = {
            "command": command,
            "args": args,
            "server_name": server_name,
            "server_description": server_description,
            "timeout_seconds": timeout_seconds,
        }
        with httpx.Client(timeout=timeout_seconds + 15) as client:
            r = client.post(f"{self.base_url}/register/discover", json=payload)
            r.raise_for_status()
            return r.json()

    def register_discover_sse(
        self,
        server_name: str,
        server_description: str,
        url: str,
        timeout_seconds: float = 20.0,
    ) -> Dict[str, Any]:
        """POST /register/discover/sse — register an MCP server over SSE/HTTP."""
        payload = {
            "url": url,
            "server_name": server_name,
            "server_description": server_description,
            "timeout_seconds": timeout_seconds,
        }
        with httpx.Client(timeout=timeout_seconds + 15) as client:
            r = client.post(f"{self.base_url}/register/discover/sse", json=payload)
            r.raise_for_status()
            return r.json()

    # ── Session ───────────────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> Dict[str, str]:
        with httpx.Client(timeout=5.0) as client:
            r = client.post(
                f"{self.base_url}/session/clear",
                params={"session_id": session_id},
            )
            r.raise_for_status()
            return r.json()

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()
