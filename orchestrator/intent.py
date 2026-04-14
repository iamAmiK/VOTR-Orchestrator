from __future__ import annotations

"""
Intent decomposer — converts a natural-language user prompt into one or more
MCP-Router routing hops following the format specified in Policy.md:

  server_intent : 4-10 words naming the app/domain
  tool_intent   : verb + object + key constraint, 6-16 words

One hop per distinct action/verb+object the user wants performed.
"""

import json
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


# ── Data model ────────────────────────────────────────────────────────────────

class RouterHop(BaseModel):
    server_intent: str
    tool_intent: str


# ── System prompt follows Policy.md rules exactly ─────────────────────────────

_SYSTEM_PROMPT = """\
You are a tool-routing intent planner for MCP (Model Context Protocol) servers.

Given a user request, decompose it into one or more routing hops — one per \
distinct action the user wants to perform. Each hop will be sent as a separate \
query to a semantic vector search engine that finds the right MCP tool.

Output rules (MUST follow):
- "server_intent": 4-10 words naming the app or domain \
  (e.g. "GitHub repository operations", "Telegram messaging")
- "tool_intent"  : verb + object + key constraint, 6-16 words \
  (e.g. "list open pull requests for repository", "send message to channel with text")
- One hop per distinct verb+object action
- Fix obvious typos: githb→github, gh→github, prs→pull requests
- Do NOT invent missing facts (IDs, repo names, dates, usernames)
- If only one action is needed, return a single-element list

Output ONLY a valid JSON array, no explanation, no markdown fences:
[{"server_intent": "...", "tool_intent": "..."}, ...]
"""


# ── Main function ─────────────────────────────────────────────────────────────

def decompose_into_hops(user_prompt: str, llm: ChatOpenAI) -> List[RouterHop]:
    """
    Use GPT-4o to decompose a user prompt into routing hops.

    Returns a list of RouterHop objects; at minimum one hop.
    Falls back to a single generic hop if the LLM returns unparseable output.
    """
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    content = response.content.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(content)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Expected non-empty JSON array")
        return [RouterHop(**hop) for hop in data]
    except (json.JSONDecodeError, ValueError, TypeError):
        # Graceful fallback: treat the whole prompt as a single generic hop
        return [
            RouterHop(
                server_intent="general capability",
                tool_intent=user_prompt[:80],
            )
        ]
