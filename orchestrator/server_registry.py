from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# ── Server entry models ───────────────────────────────────────────────────────

class StdioServerEntry(BaseModel):
    transport: Literal["stdio"] = "stdio"
    name: str
    description: str = ""
    command: str
    args: List[str] = []


class SSEServerEntry(BaseModel):
    transport: Literal["sse"] = "sse"
    name: str
    description: str = ""
    url: str


ServerEntry = Union[StdioServerEntry, SSEServerEntry]


# ── Registry ─────────────────────────────────────────────────────────────────

class ServerRegistry:
    """
    Persists MCP server connection details (command/URL) keyed by server name.

    The MCP-Router stores server *embeddings* for retrieval; this registry stores
    the *connection info* so the orchestrator can actually execute tool calls.
    Both registrations happen together via Orchestrator.register_server_*.
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._entries: Dict[str, dict] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.registry_path.is_file():
            with open(self.registry_path, encoding="utf-8") as f:
                self._entries = json.load(f)

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register_stdio(
        self,
        name: str,
        command: str,
        args: List[str],
        description: str = "",
    ) -> None:
        self._entries[name] = {
            "transport": "stdio",
            "name": name,
            "description": description,
            "command": command,
            "args": args,
        }
        self._save()

    def register_sse(self, name: str, url: str, description: str = "") -> None:
        self._entries[name] = {
            "transport": "sse",
            "name": name,
            "description": description,
            "url": url,
        }
        self._save()

    def get(self, name: str) -> Optional[ServerEntry]:
        data = self._entries.get(name)
        if data is None:
            return None
        if data["transport"] == "stdio":
            return StdioServerEntry(**data)
        return SSEServerEntry(**data)

    def remove(self, name: str) -> bool:
        if name in self._entries:
            del self._entries[name]
            self._save()
            return True
        return False

    def list_all(self) -> Dict[str, dict]:
        return dict(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries
