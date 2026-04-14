from __future__ import annotations

"""
Executes MCP tool calls against live MCP servers.

Implements the JSON-RPC framing protocol directly (matching what mcp_discovery.py
in the router uses) so no additional `mcp` SDK dependency is required.

Supports:
  - stdio transport  (spawns a subprocess, communicates over stdin/stdout)
  - SSE / HTTP transport  (plain JSON-RPC POST to an HTTP endpoint)
"""

import json
import subprocess
import time
from typing import Any, Dict, List

import httpx

from orchestrator.server_registry import ServerRegistry, SSEServerEntry, StdioServerEntry


class MCPExecutionError(RuntimeError):
    pass


# ── Low-level stdio helpers (identical framing to mcp_discovery.py) ───────────

def _encode(obj: Dict[str, Any]) -> bytes:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def _read_framed(stdout, deadline: float) -> Dict[str, Any]:
    header = b""
    while b"\r\n\r\n" not in header:
        if time.time() > deadline:
            raise MCPExecutionError("Timed out reading MCP response header")
        chunk = stdout.read(1)
        if not chunk:
            raise MCPExecutionError("MCP process closed pipe before responding")
        header += chunk
    head, _ = header.split(b"\r\n\r\n", 1)
    length: int | None = None
    for line in head.decode("utf-8", errors="replace").split("\r\n"):
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
            break
    if length is None:
        raise MCPExecutionError("Missing Content-Length in MCP framing")
    payload = stdout.read(length)
    if len(payload) != length:
        raise MCPExecutionError("Incomplete MCP payload read")
    return json.loads(payload.decode("utf-8"))


def _rpc_stdio(
    proc: subprocess.Popen,
    req_id: int,
    method: str,
    params: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(_encode({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}))
    proc.stdin.flush()
    deadline = time.time() + timeout
    while time.time() <= deadline:
        msg = _read_framed(proc.stdout, deadline)
        if msg.get("id") != req_id:
            continue  # skip notifications
        if "error" in msg:
            raise MCPExecutionError(f"RPC error [{method}]: {msg['error']}")
        return msg.get("result") or {}
    raise MCPExecutionError(f"Timed out waiting for RPC response: {method}")


# ── stdio tool execution ──────────────────────────────────────────────────────

def call_tool_stdio(
    command: str,
    args: List[str],
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
) -> Any:
    """Spawn an MCP stdio server, call one tool, return the result, terminate."""
    cmd = [command] + args
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MCPExecutionError(f"Could not start MCP server: {cmd}") from exc

    try:
        _rpc_stdio(
            proc, 1, "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-orchestrator", "version": "0.1.0"},
            },
            timeout_seconds,
        )
        # initialized notification (no response expected)
        assert proc.stdin is not None
        proc.stdin.write(
            _encode({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        )
        proc.stdin.flush()

        return _rpc_stdio(
            proc, 2, "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout_seconds,
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()


# ── SSE / HTTP tool execution ─────────────────────────────────────────────────

def call_tool_sse(
    url: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout_seconds: float = 30.0,
) -> Any:
    """Call an MCP tool over HTTP JSON-RPC (SSE endpoint)."""
    t = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=t) as client:
        try:
            r = client.post(
                url,
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "mcp-orchestrator", "version": "0.1.0"},
                    },
                },
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise MCPExecutionError(f"MCP initialize failed at {url}: {exc}") from exc

        # best-effort initialized notification
        try:
            client.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        except httpx.HTTPError:
            pass

        try:
            r2 = client.post(
                url,
                json={
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            )
            r2.raise_for_status()
        except httpx.HTTPError as exc:
            raise MCPExecutionError(f"MCP tools/call failed at {url}: {exc}") from exc

        res = r2.json()
        if "error" in res:
            raise MCPExecutionError(f"RPC error [tools/call]: {res['error']}")
        return res.get("result") or {}


# ── High-level executor ───────────────────────────────────────────────────────

class MCPExecutor:
    """
    Routes tool calls to the correct MCP server using the ServerRegistry.

    Usage:
        executor = MCPExecutor(registry, timeout_seconds=30.0)
        result = executor.call("GitHub", "list_pull_requests", {"repo": "owner/repo"})
    """

    def __init__(self, registry: ServerRegistry, timeout_seconds: float = 30.0):
        self.registry = registry
        self.timeout = timeout_seconds

    def call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        entry = self.registry.get(server_name)
        if entry is None:
            raise MCPExecutionError(
                f"Server '{server_name}' is not in the local registry. "
                "Register it with `register` or `register-sse` first."
            )
        if isinstance(entry, StdioServerEntry):
            return call_tool_stdio(entry.command, entry.args, tool_name, arguments, self.timeout)
        elif isinstance(entry, SSEServerEntry):
            return call_tool_sse(entry.url, tool_name, arguments, self.timeout)
        raise MCPExecutionError(f"Unknown transport type for server '{server_name}'")
