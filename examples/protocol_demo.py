"""Chapter 10 protocol exercises: MCP, A2A, and local ANP-style discovery."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from my_agents import (
    A2AClient, ANPDiscovery, AgentService, CalculatorTool, MCPClient, MCPTool,
    MyAgentsLLM, SimpleAgent, create_a2a_app, create_mcp_server,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EchoLLM:
    provider = "echo"

    def invoke(self, messages, **kwargs):
        return "Echo: " + messages[-1].content

    def stream(self, messages, **kwargs):
        yield self.invoke(messages, **kwargs)


def mcp_roundtrip() -> None:
    command = [sys.executable, "-m", "examples.protocol_demo", "mcp-server"]
    tools = MCPTool.discover_sync(MCPClient(command), prefix="demo")
    calculator = next(tool for tool in tools if tool.remote_name == "calculator")
    print(calculator.run_arguments({"input_text": "(10 + 20) / 2"}))


def a2a_server(
    host: str, port: int, use_model: bool, database: Path,
    base_url: str | None, tokens_file: Path | None,
) -> None:
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Demo A2A server must bind loopback; use a TLS reverse proxy")

    load_dotenv(PROJECT_ROOT / ".env")
    llm = MyAgentsLLM.from_env() if use_model else EchoLLM()
    tokens = (
        json.loads(tokens_file.read_text(encoding="utf-8"))
        if tokens_file is not None else None
    )
    app = create_a2a_app(
        lambda: SimpleAgent("a2a-assistant", llm),
        name="MyAgents A2A Assistant",
        description="Answers text questions through MyAgents",
        base_url=base_url or f"http://{host}:{port}",
        task_db_path=database,
        session_db_path=database,
        auth_tokens=tokens,
    )
    uvicorn.run(app, host=host, port=port)


def anp_demo(path: Path) -> None:
    discovery = ANPDiscovery()
    for item in json.loads(path.read_text(encoding="utf-8")):
        discovery.register(AgentService(
            service_id=item["service_id"], name=item["name"],
            service_type=item["service_type"],
            capabilities=tuple(item["capabilities"]), endpoint=item["endpoint"],
            load=float(item.get("load", 0)), cost=float(item.get("cost", 0)),
        ))
    for service in discovery.discover(capability="answer"):
        print(f"{service.service_id}: {service.endpoint} (load={service.load})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mcp-server", help="Serve CalculatorTool over MCP stdio")
    sub.add_parser("mcp-roundtrip", help="Discover and call the local MCP server")
    mcp_http = sub.add_parser("mcp-http-server", help="Serve CalculatorTool over MCP HTTP")
    mcp_http.add_argument("--host", default="127.0.0.1")
    mcp_http.add_argument("--port", type=int, default=8000)
    serve = sub.add_parser("a2a-server", help="Serve a SimpleAgent over A2A")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=6001)
    serve.add_argument("--model", action="store_true", help="Use .env LLM instead of offline echo")
    serve.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "data" / "a2a.sqlite3")
    serve.add_argument("--base-url", help="Public HTTPS URL when behind a reverse proxy")
    serve.add_argument("--tokens-file", type=Path, help="JSON object mapping principal to bearer token")
    ask = sub.add_parser("a2a-ask", help="Send a text question to an A2A agent")
    ask.add_argument("text")
    ask.add_argument("--url", default="http://127.0.0.1:6001")
    discover = sub.add_parser("anp-demo", help="Discover registered A2A endpoints")
    discover.add_argument(
        "--file", type=Path, default=PROJECT_ROOT / "examples" / "agent_services.example.json"
    )
    args = parser.parse_args()
    if args.command == "mcp-server":
        create_mcp_server([CalculatorTool()]).run("stdio")
    elif args.command == "mcp-roundtrip":
        mcp_roundtrip()
    elif args.command == "mcp-http-server":
        try:
            create_mcp_server([CalculatorTool()]).run(
                "streamable-http", host=args.host, port=args.port
            )
        except KeyboardInterrupt:
            pass
    elif args.command == "a2a-server":
        a2a_server(
            args.host, args.port, args.model, args.db_path,
            args.base_url, args.tokens_file,
        )
    elif args.command == "a2a-ask":
        load_dotenv(PROJECT_ROOT / ".env")
        print(asyncio.run(A2AClient(
            args.url, bearer_token=os.getenv("A2A_BEARER_TOKEN") or None
        ).send(args.text)))
    else:
        anp_demo(args.file)


if __name__ == "__main__":
    main()
