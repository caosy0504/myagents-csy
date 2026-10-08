"""Tool-layer adapters for MCP, A2A, and local ANP-style discovery."""

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

from ..protocols.a2a import A2AClient
from ..protocols.anp import ANPDiscovery, ANPDocumentDiscovery
from ..protocols.mcp import MCPClient
from .base import Tool


def _run_sync(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise RuntimeError("Synchronous protocol tools cannot run inside an event loop")


class MCPTool(Tool):
    """One remote MCP tool exposed under a stable, prefixed local name."""

    def __init__(self, client: MCPClient, remote_name: str, schema: dict, *,
                 prefix: str = "mcp", description: str = "",
                 max_result_chars: int = 12000) -> None:
        local_name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{prefix}_{remote_name}")
        super().__init__(local_name, description or f"MCP tool {remote_name}")
        self.client = client
        self.remote_name = remote_name
        self.schema = schema
        self.prefix = prefix
        self.max_result_chars = max_result_chars

    @classmethod
    async def discover(
        cls, client: MCPClient, *, prefix: str = "mcp", refresh: bool = False
    ) -> list["MCPTool"]:
        discovered = await client.list_tools(refresh=refresh)
        tools = [
            cls(
                client, item["name"], item["input_schema"],
                prefix=prefix, description=item["description"],
            )
            for item in discovered
        ]
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("MCP tool names collide after prefix normalization")
        return tools

    @classmethod
    def discover_sync(
        cls, client: MCPClient, *, prefix: str = "mcp", refresh: bool = False
    ) -> list["MCPTool"]:
        if getattr(client, "_portal", None) is not None:
            discovered = client.list_tools_sync(refresh=refresh)
            tools = [
                cls(client, item["name"], item["input_schema"], prefix=prefix,
                    description=item["description"])
                for item in discovered
            ]
            if len({tool.name for tool in tools}) != len(tools):
                raise ValueError("MCP tool names collide after prefix normalization")
            return tools
        return _run_sync(cls.discover(client, prefix=prefix, refresh=refresh))

    def function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }

    def run_arguments(self, arguments: dict[str, Any]) -> str:
        result = self.client.call_tool_sync(self.remote_name, arguments)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        if len(text) > self.max_result_chars:
            raise ValueError(
                f"MCP result exceeds {self.max_result_chars} characters; request a smaller result"
            )
        return text

    def run(self, input_text: str) -> str:
        arguments = json.loads(input_text)
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be a JSON object")
        return self.run_arguments(arguments)


class A2ATool(Tool):
    """Delegate a request to an A2A agent."""

    def __init__(self, url: str, *, name: str = "a2a_agent",
                 description: str = "Ask a remote A2A agent", client: A2AClient | None = None) -> None:
        super().__init__(name, description)
        self.client = client or A2AClient(url)

    def run(self, input_text: str) -> str:
        return _run_sync(self.client.send(input_text))


class ANPTool(Tool):
    """Expose local discovery and route to an A2A endpoint when requested."""

    def __init__(
        self,
        discovery: ANPDiscovery,
        *,
        client_factory: Callable[[str], A2AClient] = A2AClient,
        public_discovery: ANPDocumentDiscovery | None = None,
    ) -> None:
        super().__init__(
            "anp", "Discover locally registered agents by capability or route a question to one"
        )
        self.discovery = discovery
        self.client_factory = client_factory
        self.public_discovery = public_discovery

    def function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["discover", "route", "discover_public"],
                        },
                        "capability": {"type": "string"},
                        "service_type": {"type": "string"},
                        "query": {"type": "string"},
                        "domain": {"type": "string"},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

    def run_arguments(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action")
        if action == "discover_public":
            domain = arguments.get("domain")
            if self.public_discovery is None or not isinstance(domain, str):
                raise ValueError("discover_public requires an allowed domain")
            references = _run_sync(self.public_discovery.discover(domain))
            return json.dumps([
                {"name": reference.name, "description_url": reference.url}
                for reference in references
            ], ensure_ascii=False)
        capability = arguments.get("capability")
        service_type = arguments.get("service_type")
        if action not in {"discover", "route"} or not isinstance(capability, str) or not capability:
            raise ValueError("action and capability are required")
        if service_type is not None and not isinstance(service_type, str):
            raise ValueError("service_type must be a string")
        if action == "discover":
            matches = self.discovery.discover(
                service_type=service_type, capability=capability
            )
            return json.dumps([
                {
                    "service_id": service.service_id, "name": service.name,
                    "endpoint": service.endpoint, "load": service.load,
                    "cost": service.cost,
                }
                for service in matches
            ], ensure_ascii=False)
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("route requires a nonempty query")
        service = self.discovery.choose(capability, service_type=service_type)
        return _run_sync(self.client_factory(service.endpoint).send(query))

    def run(self, input_text: str) -> str:
        arguments = json.loads(input_text)
        if not isinstance(arguments, dict):
            raise ValueError("ANP input must be a JSON object")
        return self.run_arguments(arguments)
