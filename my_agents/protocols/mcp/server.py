"""Expose existing MyAgents tools through a real MCP server."""

from collections.abc import Iterable

from ...tools.base import Tool


def create_mcp_server(tools: Iterable[Tool], name: str = "my-agents"):
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError("MCP requires: pip install 'my-agents-csy[protocol-mcp]'") from exc

    server = MCPServer(name)
    for tool in tools:
        def make_handler(current: Tool):
            def invoke(input_text: str) -> str:
                """Run an existing MyAgents tool with its text input."""
                return current.run(input_text)
            return invoke

        server.tool(name=tool.name, description=tool.description)(make_handler(tool))
    return server
