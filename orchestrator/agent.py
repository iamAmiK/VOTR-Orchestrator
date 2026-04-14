from __future__ import annotations

"""
Orchestrator — ties together intent decomposition, MCP-Router retrieval,
and LangChain agent execution.

Workflow for each user message:
  1. GPT-4o decomposes the prompt into routing hops (Policy.md format)
  2. Each hop is sent to /route on the MCP-Router → RouteResponse
  3. RoutedTools are wrapped as LangChain StructuredTools backed by MCPExecutor
  4. A LangChain tool-calling agent (GPT-4o) executes the task using those tools
  5. The final answer + metadata are returned; conversation history is updated
"""

import uuid
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from orchestrator.config import OrchestratorConfig
from orchestrator.intent import RouterHop, decompose_into_hops
from orchestrator.mcp_executor import MCPExecutor
from orchestrator.router_client import RouteResponse, RouterClient
from orchestrator.server_registry import ServerRegistry
from orchestrator.tool_builder import build_tools_from_responses


_AGENT_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to MCP (Model Context Protocol) tools \
that connect you to real external services.

Guidelines:
- Use the provided tools to fulfill the user's request accurately and completely.
- Pass tool arguments exactly as required by each tool's schema.
- Chain multiple tool calls when the task requires sequential steps.
- If a tool returns an error, explain clearly what went wrong and, where possible, \
  attempt an alternative approach.
- When you have all the information needed, give a concise, accurate final answer.
- Do not fabricate results; if a tool fails and no alternative exists, say so honestly.\
"""


class Orchestrator:
    """
    The main entry point for the MCP Orchestrator.

    Manages:
      - RouterClient  — communicates with the MCP-Router retrieval service
      - ServerRegistry — persists MCP server connection details
      - MCPExecutor   — executes tool calls against live MCP servers
      - Session history — per session_id in-memory conversation history
    """

    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        self.router = RouterClient(cfg.router_url, cfg.router_timeout_seconds)
        self.registry = ServerRegistry(cfg.registry_path)
        self.executor = MCPExecutor(self.registry, cfg.mcp_tool_timeout_seconds)
        self.llm = ChatOpenAI(
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
        )
        # session_id → list of (HumanMessage | AIMessage)
        self._history: Dict[str, List[BaseMessage]] = {}

    # ── Session helpers ───────────────────────────────────────────────────────

    def _ensure_session(self, session_id: Optional[str]) -> str:
        sid = session_id or str(uuid.uuid4())
        if sid not in self._history:
            self._history[sid] = []
        return sid

    def _trim_history(self, session_id: str) -> None:
        """Keep at most max_history_turns * 2 messages (each turn = 2 messages)."""
        cap = self.cfg.max_history_turns * 2
        if len(self._history[session_id]) > cap:
            self._history[session_id] = self._history[session_id][-cap:]

    # ── Core chat method ──────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message end-to-end and return the agent's response.

        Returns a dict with:
          session_id      - the session ID used (create new one if not provided)
          answer          - the agent's final text response
          hops            - the routing hops the LLM planned
          route_responses - summary of what the router returned per hop
          tools_found     - names of the LangChain tools built for this turn
        """
        session_id = self._ensure_session(session_id)
        history = self._history[session_id]

        # ── Step 1: Decompose prompt into routing hops ────────────────────────
        hops: List[RouterHop] = decompose_into_hops(user_message, self.llm)

        # ── Step 2: Query the router for each hop ─────────────────────────────
        route_responses: List[RouteResponse] = []
        for hop in hops:
            try:
                resp = self.router.route(
                    server_intent=hop.server_intent,
                    tool_intent=hop.tool_intent,
                    session_id=session_id,
                    record_session=True,
                )
                route_responses.append(resp)
            except Exception as exc:
                # Router unavailable for this hop — skip and continue
                if self.cfg.verbose:
                    print(f"[Orchestrator] Router hop failed ({hop.server_intent}): {exc}")

        # ── Step 3: Build LangChain tools ─────────────────────────────────────
        tools: List[StructuredTool] = build_tools_from_responses(route_responses, self.executor)

        # ── Step 4: Run the LangChain agent ───────────────────────────────────
        answer = self._run_agent(user_message, history, tools)

        # ── Step 5: Update conversation history ───────────────────────────────
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=answer))
        self._trim_history(session_id)

        return {
            "session_id": session_id,
            "answer": answer,
            "hops": [h.model_dump() for h in hops],
            "tools_found": [t.name for t in tools],
            "route_responses": [
                {
                    "server_intent": hops[i].server_intent if i < len(hops) else "",
                    "tool_intent": hops[i].tool_intent if i < len(hops) else "",
                    "confidence": r.confidence,
                    "recommended_k": r.recommended_handoff_k,
                    "null_route": r.null_route,
                    "overlap_ambiguous": r.overlap_ambiguous,
                    "tools_returned": [
                        {"key": t.tool_key, "name": t.tool_name, "score": round(t.score, 5)}
                        for t in r.tools[: r.recommended_handoff_k]
                    ],
                }
                for i, r in enumerate(route_responses)
            ],
        }

    @staticmethod
    def _final_text_from_agent_messages(messages: List[BaseMessage]) -> str:
        """Pick the last assistant text reply from a LangGraph agent message list."""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                text = m.content
                if isinstance(text, str) and text.strip():
                    return text
                if isinstance(text, list):
                    parts = []
                    for block in text:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    joined = "".join(parts).strip()
                    if joined:
                        return joined
        return ""

    def _run_agent(
        self,
        user_message: str,
        history: List[BaseMessage],
        tools: List[StructuredTool],
    ) -> str:
        """
        Run a tool-capable agent (LangChain 1.x ``create_agent`` + LangGraph).

        ``max_agent_iterations`` maps loosely to LangGraph ``recursion_limit``
        (each model/tool step consumes graph steps).
        """
        messages: List[BaseMessage] = list(history)
        messages.append(HumanMessage(content=user_message))

        if tools:
            graph = create_agent(
                self.llm,
                tools,
                system_prompt=_AGENT_SYSTEM_PROMPT,
                debug=self.cfg.verbose,
            )
            recursion_limit = max(25, self.cfg.max_agent_iterations * 6)
            result = graph.invoke(
                {"messages": messages},
                config={"recursion_limit": recursion_limit},
            )
            out_msgs = result.get("messages", [])
            text = self._final_text_from_agent_messages(out_msgs)
            return text if text else str(result)

        # No tools — single LLM turn with the same system policy
        direct: List[BaseMessage] = [
            SystemMessage(content=_AGENT_SYSTEM_PROMPT),
            *messages,
        ]
        response = self.llm.invoke(direct)
        return str(response.content)

    # ── Server registration ───────────────────────────────────────────────────

    def register_server_stdio(
        self,
        name: str,
        command: str,
        args: List[str],
        description: str,
        discovery_timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Register an MCP server (stdio transport):
          1. Calls /register/discover on the MCP-Router → embeds tools in the vector index
          2. Saves connection details in the local registry for execution
        """
        router_result = self.router.register_discover_stdio(
            server_name=name,
            server_description=description,
            command=command,
            args=args,
            timeout_seconds=discovery_timeout,
        )
        self.registry.register_stdio(name, command, args, description)
        return {
            "status": "ok",
            "server_name": name,
            "transport": "stdio",
            "router": router_result,
        }

    def register_server_sse(
        self,
        name: str,
        url: str,
        description: str,
        discovery_timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Register an MCP server (SSE/HTTP transport):
          1. Calls /register/discover/sse on the MCP-Router
          2. Saves the URL in the local registry for execution
        """
        router_result = self.router.register_discover_sse(
            server_name=name,
            server_description=description,
            url=url,
            timeout_seconds=discovery_timeout,
        )
        self.registry.register_sse(name, url, description)
        return {
            "status": "ok",
            "server_name": name,
            "transport": "sse",
            "router": router_result,
        }

    # ── Session management ────────────────────────────────────────────────────

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history locally and in the router."""
        self._history.pop(session_id, None)
        try:
            self.router.clear_session(session_id)
        except Exception:
            pass

    # ── Registry helpers ──────────────────────────────────────────────────────

    def list_servers(self) -> Dict[str, Any]:
        return self.registry.list_all()

    def remove_server(self, name: str) -> bool:
        return self.registry.remove(name)
