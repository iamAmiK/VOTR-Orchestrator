<div align="center">

# VOTR-Orchestrator

**Agent execution layer for VOTR: intent decomposition, tool execution, and multi-turn MCP workflows**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-service-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-agent_stack-1C3C3C?style=for-the-badge)](https://www.langchain.com/)
[![VOTR Router](https://img.shields.io/badge/Depends_on-VOTR-blue?style=for-the-badge)](https://github.com/iamAmiK/VOTR)
[![MCP](https://img.shields.io/badge/MCP-compatible-6f42c1?style=for-the-badge)](https://modelcontextprotocol.io/)

</div>

VOTR-Orchestrator is the companion execution harness for VOTR.  
It takes user prompts, decomposes them into routing hops, calls the VOTR router to retrieve tools, wraps those tools for agent use, and executes them in a production-style multi-step tool-calling loop.

---

## What It Does

- Decomposes a user request into one or more `server_intent` / `tool_intent` hops.
- Calls VOTR `/route` for each hop.
- Builds LangChain `StructuredTool`s from routed MCP tools.
- Executes tools through registered MCP servers (stdio or SSE/HTTP).
- Maintains per-session memory and can clear both local and router session state.

---

## Architecture

For each user message:

1. Intent planner generates routing hops.
2. Router client calls VOTR for candidate tools.
3. Tool builder converts route responses into executable agent tools.
4. LangChain agent performs tool-calling execution.
5. Final answer + metadata are returned and session history is updated.

Core modules:

- `orchestrator/agent.py` - end-to-end orchestration pipeline
- `orchestrator/intent.py` - hop decomposition policy
- `orchestrator/router_client.py` - VOTR API client
- `orchestrator/mcp_executor.py` - executes MCP tool calls
- `orchestrator/server_registry.py` - persisted transport registry
- `api/app.py` - FastAPI service surface
- `cli.py` - interactive and automation-friendly CLI

---

## Requirements

- Python 3.10+
- Running VOTR router (default: `http://localhost:8765`)
- OpenAI API key for planner + agent model

Install:

```bash
python -m pip install -r requirements.txt
```

Set API key (PowerShell):

```powershell
$env:OPENAI_API_KEY="sk-..."
```

---

## Configuration

Main config file: `config.yaml`

Important settings:

- `router_url` - VOTR router base URL (default `http://localhost:8765`)
- `llm_model` - planner/agent model (default `gpt-4o`)
- `max_agent_iterations` - cap on tool-calling loop iterations
- `registry_path` - persisted MCP server connection JSON
- `api_host` / `api_port` - Orchestrator API bind settings (default `8766`)

---

## CLI Usage

Interactive chat:

```bash
python cli.py chat
```

Single-turn request:

```bash
python cli.py ask "List my open GitHub pull requests"
```

Register MCP server over stdio:

```bash
python cli.py register --name GitHub --command npx --args -y @modelcontextprotocol/server-github --description "GitHub API"
```

Register MCP server over SSE/HTTP:

```bash
python cli.py register-sse --name MyServer --url http://localhost:9000 --description "Custom MCP server"
```

List/remove servers:

```bash
python cli.py servers
python cli.py remove GitHub
```

Start orchestrator API:

```bash
python cli.py serve
```

---

## API Endpoints

From `api/app.py`:

- `GET /health` - health status (includes router ping)
- `POST /chat` - execute one user message end-to-end
- `POST /servers/register` - register stdio server
- `POST /servers/register/sse` - register SSE/HTTP server
- `GET /servers` - list registry
- `DELETE /servers/{name}` - remove server from registry
- `POST /session/clear` - clear a session

Swagger/static docs are also mounted under `/swagger` when available.

---

## Relationship to VOTR

- `VOTR` (router repo) handles retrieval, ranking, confidence gating, and registry indexing.
- `VOTR-Orchestrator` handles agent execution and tool-calling workflows using those routed tools.

Use VOTR alone for retrieval benchmarks and routing service deployment.  
Use VOTR-Orchestrator when you need full conversation/task execution over live MCP tools.
