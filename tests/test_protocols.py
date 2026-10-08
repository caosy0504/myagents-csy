"""The three layers: SDK-backed protocols, Tool adapters, and Agent use."""

import asyncio
import json
import shlex
import sqlite3
import sys
from time import monotonic

import httpx
import pytest

from my_agents import (
    A2AClient, A2ATool, ANPDiscovery, ANPTool, AgentService,
    CalculatorTool, MCPClient, MCPTool, ReActAgent, SimpleAgent, ToolRegistry,
    create_a2a_app, create_mcp_server,
)
from my_agents.core import ToolCall, ToolCompletion
from my_agents.protocols.anp import ANPNetwork
from my_agents.protocols.anp import ANPDocumentDiscovery


class FakeLLM:
    provider = "fake"

    def invoke(self, messages, **kwargs):
        return "回答：" + messages[-1].content

    def stream(self, messages, **kwargs):
        yield ""


def test_mcp_discovers_and_calls_existing_tool():
    pytest.importorskip("mcp")
    server = create_mcp_server([CalculatorTool()], name="test-tools")
    client = MCPClient(server)
    tools = MCPTool.discover_sync(client, prefix="remote")
    assert [tool.name for tool in tools] == ["remote_calculator"]
    assert tools[0].function_schema()["function"]["parameters"]["required"] == ["input_text"]
    assert json.loads(tools[0].run_arguments({"input_text": "2 + 3"})) == {"result": "5"}
    small = MCPTool(client, "calculator", tools[0].schema, max_result_chars=5)
    with pytest.raises(ValueError, match="exceeds"):
        small.run_arguments({"input_text": "2 + 3"})

    async def persistent_connection():
        async with MCPClient(server) as connected:
            assert len(await connected.list_tools()) == 1
            assert await connected.call_tool("calculator", {"input_text": "6 * 7"}) == {
                "result": "42"
            }

    asyncio.run(persistent_connection())


def test_mcp_cache_refresh_and_persistent_stdio():
    pytest.importorskip("mcp")
    from mcp.server import CacheHint, MCPServer

    server = MCPServer(
        "catalog", cache_hints={"tools/list": CacheHint(ttl_ms=60_000)}
    )

    @server.tool()
    def first(text: str) -> str:
        return text

    client = MCPClient(server)
    assert [item["name"] for item in client.list_tools_sync()] == ["first"]

    @server.tool()
    def second(text: str) -> str:
        return text

    assert [item["name"] for item in client.list_tools_sync()] == ["first"]
    assert [item["name"] for item in client.list_tools_sync(refresh=True)] == [
        "first", "second"
    ]

    command = [sys.executable, "-m", "examples.protocol_demo", "mcp-server"]
    subprocess_client = MCPClient(command)
    subprocess_client.open_sync()
    try:
        calculator = MCPTool.discover_sync(subprocess_client, prefix="live")[0]
        assert json.loads(calculator.run_arguments({"input_text": "6*7"})) == {
            "result": "42"
        }
        assert json.loads(calculator.run_arguments({"input_text": "2+3"})) == {
            "result": "5"
        }
    finally:
        subprocess_client.close_sync()


def test_react_uses_dynamically_discovered_mcp_tool():
    pytest.importorskip("mcp")
    class NativeLLM:
        provider = "fake"

        def __init__(self):
            self.calls = 0

        def complete_with_tools(self, messages, tools, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert tools[0]["function"]["name"] == "remote_calculator"
                return ToolCompletion(
                    "", [ToolCall("call-1", "remote_calculator", '{"input_text":"4*5"}')]
                )
            assert "20" in messages[-1]["content"]
            return ToolCompletion("答案是 20", [])

        def invoke(self, messages, **kwargs):
            return "摘要"

    server = create_mcp_server([CalculatorTool()])
    registry = ToolRegistry()
    for tool in MCPTool.discover_sync(MCPClient(server), prefix="remote"):
        registry.register(tool)
    agent = ReActAgent("tester", NativeLLM(), registry, tool_call_mode="native")
    assert agent.run("4 乘 5") == "答案是 20"
    assert agent.last_tool_errors == 0


def test_a2a_sdk_card_message_and_tool_adapter():
    pytest.importorskip("a2a")
    app = create_a2a_app(
        lambda: SimpleAgent("helper", FakeLLM()),
        name="helper", description="Answers short questions",
        base_url="http://127.0.0.1",
    )

    async def exchange():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as http_client:
            card = (await http_client.get("/.well-known/agent-card.json")).json()
            assert card["name"] == "helper"
            client = A2AClient("http://127.0.0.1", http_client=http_client)
            assert await client.send("hello") == "回答：hello"
            assert await client.send("again") == "回答：again"
    asyncio.run(exchange())
    # The Tool layer uses the same client contract.
    class LocalClient:
        async def send(self, text):
            return "peer:" + text

    tool = A2ATool("http://testserver", name="peer", client=LocalClient())
    assert tool.run("question") == "peer:question"


def test_a2a_context_persists_and_isolated_by_bearer(tmp_path):
    pytest.importorskip("a2a")
    pytest.importorskip("sqlalchemy")

    class MemoryLLM:
        provider = "fake"

        def invoke(self, messages, **kwargs):
            question = messages[-1].content
            if "记住蓝色" in question:
                return "已记住"
            return "蓝色" if any("记住蓝色" in item.content for item in messages[:-1]) else "不知道"

        def stream(self, messages, **kwargs):
            yield ""

    database = tmp_path / "a2a.sqlite3"

    def make_app():
        return create_a2a_app(
            lambda: SimpleAgent("memory", MemoryLLM()),
            name="memory", description="Keeps a conversation",
            base_url="http://127.0.0.1",
            task_db_path=database, session_db_path=database,
            auth_tokens={"alice": "token-a", "bob": "token-b"},
        )

    async def roundtrip():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()),
            base_url="http://127.0.0.1",
            headers={"Authorization": "Bearer token-a"},
        ) as http:
            card = (await http.get("/.well-known/agent-card.json")).json()
            assert "bearer" in card["securitySchemes"]
            alice = A2AClient("http://127.0.0.1", http_client=http)
            assert await alice.send("记住蓝色") == "已记住"
            context_id = alice.context_id
            assert context_id
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()),
            base_url="http://127.0.0.1",
            headers={"Authorization": "Bearer token-a"},
        ) as http:
            alice = A2AClient("http://127.0.0.1", http_client=http)
            alice.context_id = context_id
            assert await alice.send("我喜欢什么颜色？") == "蓝色"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=make_app()),
            base_url="http://127.0.0.1",
            headers={"Authorization": "Bearer token-b"},
        ) as http:
            bob = A2AClient("http://127.0.0.1", http_client=http)
            bob.context_id = context_id
            assert await bob.send("我喜欢什么颜色？") == "不知道"
            assert (await http.post("/a2a/jsonrpc", headers={
                "Authorization": "Bearer wrong"
            }, json={})).status_code == 401
    asyncio.run(roundtrip())

    with sqlite3.connect(database) as db:
        assert db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] >= 3
        assert db.execute("SELECT COUNT(*) FROM agent_sessions").fetchone()[0] == 2
    if sys.platform != "win32":
        assert database.stat().st_mode & 0o777 == 0o600


def test_remote_protocol_urls_require_https():
    with pytest.raises(ValueError, match="HTTPS"):
        MCPClient("http://remote.example/mcp")
    with pytest.raises(ValueError, match="HTTPS"):
        A2AClient("http://remote.example")
    with pytest.raises(ValueError, match="HTTPS"):
        create_a2a_app(
            lambda: SimpleAgent("x", FakeLLM()),
            name="x", description="x", base_url="http://remote.example",
        )
    assert MCPClient("https://remote.example/mcp", bearer_token="a")._cache.partition != (
        MCPClient("https://remote.example/mcp", bearer_token="b")._cache.partition
    )


def test_a2a_rejects_card_that_redirects_credentials():
    pytest.importorskip("a2a")
    from a2a.types import AgentCard, AgentInterface

    client = A2AClient("https://trusted.example", bearer_token="secret")
    card = AgentCard(supported_interfaces=[
        AgentInterface(
            url="https://other.example/a2a/jsonrpc",
            protocol_binding="JSONRPC", protocol_version="1.0",
        )
    ])
    with pytest.raises(ValueError, match="another origin"):
        client._validate_card(card)


def test_anp_discovery_routing_and_network():
    discovery = ANPDiscovery()
    discovery.register(AgentService(
        "busy", "Busy expert", "research", ("answer",),
        "http://127.0.0.1:6001", load=0.8, cost=0.1,
    ))
    discovery.register(AgentService(
        "free", "Free expert", "research", ("answer",),
        "http://127.0.0.1:6002", load=0.1, cost=0.2,
    ))
    discovery.register(AgentService(
        "expired", "Old expert", "research", ("answer",),
        "http://127.0.0.1:6003", registered_at=monotonic() - 100, ttl_seconds=1,
    ))
    assert [service.service_id for service in discovery.discover(capability="answer")] == [
        "free", "busy"
    ]

    class LocalClient:
        def __init__(self, url):
            self.url = url

        async def send(self, text):
            return self.url + ":" + text

    tool = ANPTool(discovery, client_factory=LocalClient)
    answer = tool.run_arguments({"action": "route", "capability": "answer", "query": "hello"})
    assert answer == "http://127.0.0.1:6002:hello"
    assert len(json.loads(tool.run('{"action":"discover","capability":"answer"}'))) == 2

    network = ANPNetwork("local")
    network.add_node("free", "http://127.0.0.1:6002")
    network.add_node("busy", "http://127.0.0.1:6001")
    network.connect_nodes("free", "busy")
    assert network.get_network_stats()["total_connections"] == 1


def test_anp_public_discovery_follows_pages_and_stays_on_origin():
    base = "http://127.0.0.1:8123"

    def handler(request):
        if request.url.path == "/.well-known/agent-descriptions":
            page = request.url.params.get("page")
            if page == "2":
                body = {"@context": {}, "@type": "CollectionPage", "items": [
                    {
                        "@type": "ad:AgentDescription", "name": "second",
                        "@id": base + "/agents/second.json",
                    }
                ]}
            else:
                body = {
                    "@context": {}, "@type": "CollectionPage",
                    "items": [{
                        "@type": "ad:AgentDescription", "name": "first",
                        "@id": base + "/agents/first.json",
                    }],
                    "next": base + "/.well-known/agent-descriptions?page=2",
                }
            return httpx.Response(200, json=body)
        return httpx.Response(200, json={
            "@type": "ad:AgentDescription", "@id": str(request.url), "name": "first"
        })

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http:
            discovery = ANPDocumentDiscovery(
                {"127.0.0.1:8123"}, http_client=http
            )
            references = await discovery.discover("127.0.0.1:8123")
            assert [item.name for item in references] == ["first", "second"]
            assert (await discovery.fetch_description(references[0]))["name"] == "first"
            with pytest.raises(ValueError, match="allowed"):
                await discovery.discover("evil.example")
    asyncio.run(check())


def test_cli_registers_configured_protocol_tools(monkeypatch, tmp_path):
    pytest.importorskip("mcp")
    from my_agents.cli import load_protocol_tools

    services = tmp_path / "services.json"
    services.write_text(json.dumps([{
        "service_id": "peer", "name": "Peer", "service_type": "assistant",
        "capabilities": ["answer"], "endpoint": "http://127.0.0.1:6001",
    }]), encoding="utf-8")
    command = shlex.join([
        sys.executable, "-m", "examples.protocol_demo", "mcp-server"
    ])
    monkeypatch.setenv("MCP_SERVER_COMMAND", command)
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.setenv("A2A_AGENT_URL", "http://127.0.0.1:6001")
    monkeypatch.setenv("ANP_SERVICES_FILE", str(services))
    assert {tool.name for tool in load_protocol_tools()} == {
        "mcp_calculator", "a2a_peer", "anp"
    }


def test_cli_refreshes_mcp_tools_in_both_registries():
    pytest.importorskip("mcp")
    from mcp.server import MCPServer
    from my_agents.cli import refresh_mcp_tools

    server = MCPServer("changing")

    @server.tool()
    def first(text: str) -> str:
        return text

    client = MCPClient(server)
    main = ToolRegistry()
    codebase = ToolRegistry()
    for tool in MCPTool.discover_sync(client, prefix="remote"):
        main.register(tool)
        codebase.register(tool)

    @server.tool()
    def second(text: str) -> str:
        return text

    assert refresh_mcp_tools([main, codebase]) == [
        "remote_first", "remote_second"
    ]
    assert codebase.get("remote_second") is not None


def test_cli_closes_persistent_mcp_client(monkeypatch):
    pytest.importorskip("mcp")
    from my_agents.cli import close_protocol_tools, load_protocol_tools

    command = shlex.join([
        sys.executable, "-m", "examples.protocol_demo", "mcp-server"
    ])
    monkeypatch.setenv("MCP_SERVER_COMMAND", command)
    for name in ("MCP_SERVER_URL", "A2A_AGENT_URL", "ANP_SERVICES_FILE", "ANP_DOMAINS"):
        monkeypatch.delenv(name, raising=False)
    tools = load_protocol_tools(persistent_mcp=True)
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    client = tools[0].client
    try:
        assert client._portal is not None
        assert json.loads(tools[0].run_arguments({"input_text": "3*4"})) == {
            "result": "12"
        }
    finally:
        close_protocol_tools(registry)
    assert client._portal is None


def test_cli_resolves_workspace_after_normal_install(monkeypatch, tmp_path):
    import my_agents.cli as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli, "__file__", str(tmp_path / "site-packages" / "my_agents" / "cli.py")
    )
    monkeypatch.delenv("MY_AGENTS_WORKSPACE", raising=False)
    assert cli._project_root() == tmp_path
    project = tmp_path / "chosen"
    project.mkdir()
    monkeypatch.setenv("MY_AGENTS_WORKSPACE", str(project))
    assert cli._project_root() == project
