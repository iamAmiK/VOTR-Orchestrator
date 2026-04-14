from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class OrchestratorConfig(BaseModel):
    # MCP-Router connection
    router_url: str = "http://localhost:8765"
    router_timeout_seconds: float = 30.0

    # LLM (GPT-4o powers both intent decomposition and the agent)
    openai_api_key_env: str = "OPENAI_API_KEY"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # Agent execution
    max_agent_iterations: int = 10
    verbose: bool = True

    # MCP tool execution timeout
    mcp_tool_timeout_seconds: float = 30.0

    # Session history kept in memory per Orchestrator instance
    max_history_turns: int = 20

    # Path to the persisted server registry JSON
    registry_path: Path = Field(default=Path("./data/server_registry.json"))

    # Orchestrator API server
    api_host: str = "0.0.0.0"
    api_port: int = 8766


def load_config(path: Optional[Path] = None) -> OrchestratorConfig:
    root = Path(__file__).resolve().parents[1]
    cfg_path = path or (root / "config.yaml")
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    # Resolve relative registry_path relative to the project root
    if "registry_path" in data and data["registry_path"] is not None:
        p = Path(data["registry_path"])
        data["registry_path"] = p if p.is_absolute() else (root / p)
    return OrchestratorConfig.model_validate(data)


def openai_api_key(cfg: OrchestratorConfig) -> Optional[str]:
    return os.environ.get(cfg.openai_api_key_env)
