#!/usr/bin/env python3
"""
MCP Orchestrator CLI

Commands:
  chat              Interactive multi-turn chat with MCP tools
  ask <prompt>      Single-turn prompt (non-interactive)
  register          Register an MCP server over stdio
  register-sse      Register an MCP server over SSE/HTTP
  servers           List all registered servers
  remove <name>     Remove a server from the local registry
  serve             Start the FastAPI API server

Examples:
  python cli.py chat
  python cli.py ask "List my open GitHub pull requests"
  python cli.py register --name GitHub --command npx --args -y @modelcontextprotocol/server-github --description "GitHub API"
  python cli.py register-sse --name MyServer --url http://localhost:9000 --description "Custom MCP server"
  python cli.py servers
  python cli.py serve
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from orchestrator.config import load_config, openai_api_key
from orchestrator.agent import Orchestrator


def _check_api_key(cfg) -> None:
    if not openai_api_key(cfg):
        print(
            f"ERROR: Environment variable '{cfg.openai_api_key_env}' is not set.\n"
            "Export your OpenAI API key before running the orchestrator:\n"
            f"  $env:{cfg.openai_api_key_env} = 'sk-...'    (PowerShell)\n"
            f"  export {cfg.openai_api_key_env}=sk-...       (bash/zsh)",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_chat(args, orch: Orchestrator) -> None:
    session_id = args.session_id or str(uuid.uuid4())
    print(f"MCP Orchestrator — interactive chat")
    print(f"Session ID : {session_id}")
    print(f"Router     : {orch.cfg.router_url}")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            result = orch.chat(user_input, session_id)
        except Exception as exc:
            print(f"[Error] {exc}")
            continue

        print(f"\nAssistant: {result['answer']}\n")

        # Show routing metadata if verbose
        if orch.cfg.verbose and result["tools_found"]:
            print(f"  Hops    : {len(result['hops'])}")
            for i, rr in enumerate(result["route_responses"]):
                tools_str = ", ".join(t["name"] for t in rr.get("tools_returned", []))
                print(
                    f"  Hop {i+1}  : [{rr['confidence']}] "
                    f"{rr['server_intent']} / {rr['tool_intent']}"
                    + (f" → {tools_str}" if tools_str else " → (null route)")
                )
            print(f"  Used    : {', '.join(result['tools_found'])}\n")


def cmd_ask(args, orch: Orchestrator) -> None:
    prompt = " ".join(args.prompt)
    try:
        result = orch.chat(prompt, args.session_id)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["answer"])


def cmd_register(args, orch: Orchestrator) -> None:
    try:
        result = orch.register_server_stdio(
            name=args.name,
            command=args.command,
            args=args.args or [],
            description=args.description,
        )
        print(f"Registered '{args.name}' (stdio)")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_register_sse(args, orch: Orchestrator) -> None:
    try:
        result = orch.register_server_sse(
            name=args.name,
            url=args.url,
            description=args.description,
        )
        print(f"Registered '{args.name}' (SSE)")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_servers(args, orch: Orchestrator) -> None:
    servers = orch.list_servers()
    if not servers:
        print("No servers registered.")
        return
    print(json.dumps(servers, indent=2, ensure_ascii=False))


def cmd_remove(args, orch: Orchestrator) -> None:
    removed = orch.remove_server(args.name)
    if removed:
        print(f"Removed '{args.name}' from local registry.")
        print("Note: the server remains in the MCP-Router's vector index until it is restarted.")
    else:
        print(f"Server '{args.name}' not found in local registry.", file=sys.stderr)
        sys.exit(1)


def cmd_serve(args) -> None:
    import uvicorn
    from api.app import create_app
    cfg = load_config()
    _check_api_key(cfg)
    app = create_app(cfg)
    print(f"Starting MCP Orchestrator API on http://{cfg.api_host}:{cfg.api_port}")
    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port, reload=False)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestrator",
        description="MCP Orchestrator — LangChain agent backed by MCP-Router",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive multi-turn chat")
    p_chat.add_argument("--session-id", default=None, help="Resume an existing session")

    # ask
    p_ask = sub.add_parser("ask", help="Single-turn prompt")
    p_ask.add_argument("prompt", nargs="+", help="The prompt to send")
    p_ask.add_argument("--session-id", default=None)
    p_ask.add_argument("--json", action="store_true", help="Print full JSON response")

    # register (stdio)
    p_reg = sub.add_parser("register", help="Register an MCP server (stdio)")
    p_reg.add_argument("--name", required=True, help="Unique server name")
    p_reg.add_argument("--command", required=True, help="Executable to start the server")
    p_reg.add_argument("--args", nargs="*", default=[], help="Arguments for the command")
    p_reg.add_argument("--description", default="", help="Human-readable description")

    # register-sse
    p_sse = sub.add_parser("register-sse", help="Register an MCP server (SSE/HTTP)")
    p_sse.add_argument("--name", required=True)
    p_sse.add_argument("--url", required=True, help="HTTP endpoint URL")
    p_sse.add_argument("--description", default="")

    # servers
    sub.add_parser("servers", help="List all registered servers")

    # remove
    p_rm = sub.add_parser("remove", help="Remove a server from the local registry")
    p_rm.add_argument("name", help="Server name to remove")

    # serve
    sub.add_parser("serve", help="Start the FastAPI API server")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "serve":
        cmd_serve(args)
        return

    cfg = load_config()
    _check_api_key(cfg)
    orch = Orchestrator(cfg)

    dispatch = {
        "chat": cmd_chat,
        "ask": cmd_ask,
        "register": cmd_register,
        "register-sse": cmd_register_sse,
        "servers": cmd_servers,
        "remove": cmd_remove,
    }
    handler = dispatch.get(args.command)
    if handler:
        handler(args, orch)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
