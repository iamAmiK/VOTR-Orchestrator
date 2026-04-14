from __future__ import annotations

"""
Converts RoutedTool objects (from the MCP-Router) into LangChain StructuredTools
backed by MCPExecutor for real execution.

The router's `parameter` dict contains entries like:
  "repo"  : "(str) The repository name in owner/repo format"
  "limit" : "(Optional, int) Maximum number of results to return"

We parse these to build typed Pydantic input schemas for each tool.
"""

import re
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from orchestrator.mcp_executor import MCPExecutionError, MCPExecutor
from orchestrator.router_client import RoutedTool, RouteResponse


# ── JSON Schema type → Python type mapping ────────────────────────────────────

_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "number": float,
    "float": float,
    "boolean": bool,
    "bool": bool,
    "object": dict,
    "array": list,
    "null": type(None),
    "any": Any,  # type: ignore[assignment]
}


def _parse_param(raw: str) -> tuple[type, bool, str]:
    """
    Parse a router parameter string into (python_type, is_optional, description).

    Examples:
      "(str) The repository slug"      -> (str, False, "The repository slug")
      "(Optional, int) Page number"    -> (int, True,  "Page number")
      "(optional, string) Filter text" -> (str, True,  "Filter text")
    """
    m = re.match(r"\(\s*(optional,\s*)?(\w+)\s*\)\s*(.*)", raw.strip(), re.IGNORECASE | re.DOTALL)
    if m:
        is_optional = m.group(1) is not None
        py_type = _TYPE_MAP.get(m.group(2).lower(), str)
        description = m.group(3).strip()
        return py_type, is_optional, description
    # Fallback: treat the whole string as a description, type=str, required
    return str, False, raw.strip()


def _build_input_model(tool_name: str, parameters: Dict[str, Any]) -> Type[BaseModel]:
    """Dynamically build a Pydantic model from the router's parameter dict."""
    fields: Dict[str, Any] = {}
    for pname, pspec in parameters.items():
        py_type, is_optional, description = _parse_param(str(pspec))
        if is_optional:
            fields[pname] = (Optional[py_type], Field(default=None, description=description))
        else:
            fields[pname] = (py_type, Field(description=description))

    # Pydantic requires at least one field; add a sentinel if the tool has no params
    if not fields:
        fields["no_args"] = (
            Optional[str],
            Field(default=None, description="This tool takes no arguments"),
        )

    # Build a unique model name from the tool name
    model_name = (
        "".join(w.capitalize() for w in re.split(r"[_\-\s]+", tool_name))
        + "Args"
    )
    return create_model(model_name, **fields)


# ── Tool builder ──────────────────────────────────────────────────────────────

def build_langchain_tool(routed: RoutedTool, executor: MCPExecutor) -> StructuredTool:
    """
    Wrap a RoutedTool as a LangChain StructuredTool.

    When the LangChain agent calls this tool, it:
      1. Receives validated kwargs from the Pydantic input schema
      2. Forwards the call to MCPExecutor → actual MCP server JSON-RPC
      3. Returns the result as a formatted string
    """
    input_model = _build_input_model(routed.tool_name, routed.parameter)
    _server = routed.server_name
    _tool = routed.tool_name
    _desc = routed.description or routed.compressed or f"{_server}: {_tool}"

    def _run(**kwargs: Any) -> str:
        # Drop the sentinel no-args field and None optionals
        clean = {k: v for k, v in kwargs.items() if k != "no_args" and v is not None}
        try:
            result = executor.call(_server, _tool, clean)
            return _format_result(result)
        except MCPExecutionError as exc:
            return f"[Tool error — {_server}/{_tool}]: {exc}"

    # LangChain tool names must be valid identifiers (no :: or spaces)
    lc_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"{_server}__{_tool}")

    return StructuredTool.from_function(
        func=_run,
        name=lc_name,
        description=f"[{_server}] {_desc}",
        args_schema=input_model,
    )


def _format_result(result: Any) -> str:
    """Render an MCP tool result as a readable string for the agent."""
    if not isinstance(result, dict):
        return str(result)

    content = result.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    import json
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    if isinstance(content, str):
        return content

    import json
    return json.dumps(result, indent=2, ensure_ascii=False)


# ── Batch builder ─────────────────────────────────────────────────────────────

def build_tools_from_responses(
    route_responses: List[RouteResponse],
    executor: MCPExecutor,
) -> List[StructuredTool]:
    """
    Build a deduplicated list of LangChain tools from multiple RouteResponse objects.

    Respects `recommended_handoff_k` from the confidence-based handoff policy:
      - high confidence  → top-1 tool
      - medium confidence → top-3 tools
      - low confidence   → top-5 tools

    Skips null_route responses (router found no suitable tools).
    """
    seen_keys: set[str] = set()
    tools: List[StructuredTool] = []

    for response in route_responses:
        if response.null_route:
            continue
        k = max(1, response.recommended_handoff_k)
        for routed in response.tools[:k]:
            if routed.tool_key in seen_keys:
                continue
            seen_keys.add(routed.tool_key)
            tools.append(build_langchain_tool(routed, executor))

    return tools
