"""Terminal entry point for the chapter 7 framework."""

import argparse
import json
import os
import shlex
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .agents import (
    DocumentQAAgent,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
    CodebaseAgent,
)
from .core import Agent, Config, MyAgentsError, MyAgentsLLM
from .context import ContextBuilder, ContextConfig
from .tools import CalculatorTool, RAGTool, TavilySearchTool, ToolRegistry
from .tools import A2ATool, ANPTool, MCPTool
from .tools.base import Tool
from .protocols import A2AClient, ANPDiscovery, ANPDocumentDiscovery, AgentService, MCPClient


def _project_root() -> Path:
    configured = os.getenv("MY_AGENTS_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    if (current / ".env").is_file() or (current / "pyproject.toml").is_file():
        return current
    source = Path(__file__).resolve().parents[1]
    return source if (source / "pyproject.toml").is_file() else current


PROJECT_ROOT = _project_root()


def load_protocol_tools(*, persistent_mcp: bool = False) -> list[Tool]:
    """Load only explicitly configured protocol connections."""
    result: list[Tool] = []
    a2a_url = os.getenv("A2A_AGENT_URL", "").strip()
    if a2a_url:
        result.append(A2ATool(
            a2a_url, name="a2a_peer",
            client=A2AClient(
                a2a_url, bearer_token=os.getenv("A2A_BEARER_TOKEN") or None
            ),
        ))
    registry_file = os.getenv("ANP_SERVICES_FILE", "").strip()
    domains = {
        value.strip() for value in os.getenv("ANP_DOMAINS", "").split(",")
        if value.strip()
    }
    if registry_file or domains:
        discovery = ANPDiscovery()
    if registry_file:
        path = Path(registry_file).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        services = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(services, list):
            raise ValueError("ANP_SERVICES_FILE must contain a JSON array")
        for item in services:
            if not isinstance(item, dict):
                raise ValueError("Every ANP service must be an object")
            capabilities = item["capabilities"]
            if not isinstance(capabilities, list) or not all(
                isinstance(value, str) for value in capabilities
            ):
                raise ValueError("ANP capabilities must be a string array")
            discovery.register(AgentService(
                service_id=item["service_id"],
                name=item["name"],
                service_type=item["service_type"],
                capabilities=tuple(capabilities),
                endpoint=item["endpoint"],
                load=float(item.get("load", 0)),
                cost=float(item.get("cost", 0)),
                ttl_seconds=float(item.get("ttl_seconds", 86400)),
            ))
    if registry_file or domains:
        public = ANPDocumentDiscovery(domains) if domains else None
        result.append(ANPTool(discovery, public_discovery=public))
    mcp_url = os.getenv("MCP_SERVER_URL", "").strip()
    mcp_command = os.getenv("MCP_SERVER_COMMAND", "").strip()
    if mcp_url and mcp_command:
        raise ValueError("Configure either MCP_SERVER_URL or MCP_SERVER_COMMAND")
    if mcp_url or mcp_command:
        if mcp_url and urlparse(mcp_url).scheme not in {"http", "https"}:
            raise ValueError("MCP_SERVER_URL must be HTTP(S)")
        source = mcp_url if mcp_url else shlex.split(mcp_command)
        client = MCPClient(
            source, bearer_token=os.getenv("MCP_BEARER_TOKEN") or None
        )
        if persistent_mcp:
            client.open_sync()
        try:
            result.extend(MCPTool.discover_sync(client, prefix="mcp"))
        except BaseException:
            client.close_sync()
            raise
    return result


def refresh_mcp_tools(registries: list[ToolRegistry]) -> list[str]:
    """Refresh MCP names and schemas in every Agent registry atomically."""
    sources = {}
    for registry in registries:
        for name in registry.list_tools():
            tool = registry.get(name)
            if isinstance(tool, MCPTool):
                sources[(id(tool.client), tool.prefix)] = (tool.client, tool.prefix)
    replacements = []
    for client, prefix in sources.values():
        replacements.extend(MCPTool.discover_sync(client, prefix=prefix, refresh=True))
    names = [tool.name for tool in replacements]
    if len(names) != len(set(names)):
        raise ValueError("MCP tools collide after refresh")
    for registry in registries:
        for name in names:
            current = registry.get(name)
            if current is not None and not isinstance(current, MCPTool):
                raise ValueError(f"MCP tool conflicts with local tool: {name}")
    for registry in registries:
        for name in registry.list_tools():
            if isinstance(registry.get(name), MCPTool):
                registry.unregister(name)
        for tool in replacements:
            registry.register(tool)
    return names


def close_protocol_tools(registry: ToolRegistry) -> None:
    seen = set()
    for name in registry.list_tools():
        tool = registry.get(name)
        if isinstance(tool, MCPTool) and id(tool.client) not in seen:
            seen.add(id(tool.client))
            tool.client.close_sync()


def build_application() -> tuple[dict[str, Agent], ToolRegistry, MyAgentsLLM]:
    load_dotenv(PROJECT_ROOT / ".env")
    config = Config.from_env()
    llm = MyAgentsLLM.from_env()

    tools = ToolRegistry()
    tools.register(CalculatorTool())
    search = TavilySearchTool()
    if search.available:
        tools.register(search)
    rag_tool = RAGTool(llm=llm)
    tools.register(rag_tool)
    protocol_tools = load_protocol_tools(persistent_mcp=True)
    for tool in protocol_tools:
        if tools.get(tool.name) is not None:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        tools.register(tool)
    context_builder = ContextBuilder(ContextConfig(max_tokens=config.context_max_tokens))

    agents: dict[str, Agent] = {
        "simple": SimpleAgent(
            name="基础助手",
            llm=llm,
            system_prompt="你是一个友好、准确、简洁的中文助手。",
            config=config,
            context_builder=context_builder,
        ),
        "react": ReActAgent(
            name="工具助手",
            llm=llm,
            tool_registry=tools,
            system_prompt="你负责通过必要的工具调用解决用户问题。",
            config=config,
            context_builder=context_builder,
            tool_call_mode=config.tool_call_mode,
        ),
        "reflection": ReflectionAgent(
            name="反思助手",
            llm=llm,
            system_prompt="你负责生成内容，并通过审查提升最终质量。",
            config=config,
        ),
        "plan": PlanAndSolveAgent(
            name="规划助手",
            llm=llm,
            system_prompt="你负责先规划，再严格按照计划解决复杂问题。",
            config=config,
        ),
        "rag": DocumentQAAgent(
            name="文档问答助手",
            llm=llm,
            rag_tool=rag_tool,
            system_prompt="你只根据已入库的文档回答，并明确标注参考来源。",
            config=config,
        ),
        "codebase": CodebaseAgent(
            name="代码库助手",
            llm=llm,
            workspace=PROJECT_ROOT,
            config=config,
            extra_tools=protocol_tools,
        ),
    }
    return agents, tools, llm


def print_help() -> None:
    print(
        """
命令：
  /agent <simple|react|reflection|plan|rag|codebase>  切换 Agent
  /add <文件路径>                       添加文档到 RAG 知识库
  /search <查询>                        只检索知识库，不生成答案
  /rag-stats                            查看 RAG 知识库状态
  /rag-clear                            清空当前 RAG namespace
  /history                              查看当前 Agent 历史
  /trace                                查看最近一次执行轨迹
  /tools                                查看工具
  /mcp-refresh                          刷新 MCP 工具目录
  /clear                                清空当前 Agent 历史
  /help                                 查看帮助
  /quit                                 退出
""".strip()
    )


def print_trace(agent: Agent) -> None:
    if isinstance(agent, ReActAgent):
        lines = agent.last_trace
    elif isinstance(agent, PlanAndSolveAgent):
        lines = [f"计划: {agent.last_plan}"] + [
            f"步骤 {index}: {result}" for index, result in enumerate(agent.step_results, 1)
        ]
    elif isinstance(agent, ReflectionAgent):
        lines = [
            f"第 {index} 轮反馈: {feedback}"
            for index, feedback in enumerate(agent.last_feedback, 1)
        ]
    elif isinstance(agent, DocumentQAAgent):
        lines = [f"来源: {source}" for source in agent.last_sources]
    else:
        lines = []

    print("\n".join(lines) if lines else "当前 Agent 暂无执行轨迹。")


def run_once(agent: Agent, prompt: str) -> str:
    print(f"\n[{agent.name}] 正在处理...")
    answer = agent.run(prompt)
    print(f"\n回答：\n{answer}")
    return answer


def interactive_loop(agents: dict[str, Agent], tools: ToolRegistry, initial_agent: str) -> None:
    current_name = initial_agent
    rag_tool = tools.get("rag")
    print("MyAgents CSY - Chapter 7/8/9/10 Terminal Edition")
    print_help()

    while True:
        agent = agents[current_name]
        try:
            user_input = input(f"\n[{current_name}] 你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return

        if not user_input:
            continue
        if user_input == "/quit":
            print("再见！")
            return
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/tools":
            print(agent.tool_registry.descriptions() if isinstance(agent, ReActAgent) else tools.descriptions())
            continue
        if user_input == "/mcp-refresh":
            registries = [tools]
            codebase = agents.get("codebase")
            if isinstance(codebase, ReActAgent):
                registries.append(codebase.tool_registry)
            try:
                print(f"已刷新 MCP 工具：{', '.join(refresh_mcp_tools(registries)) or '无'}")
            except Exception as exc:
                print(f"MCP 刷新失败：{exc}")
            continue
        if user_input == "/history":
            history = agent.get_history()
            print("\n".join(map(str, history)) if history else "历史为空。")
            continue
        if user_input == "/trace":
            print_trace(agent)
            continue
        if user_input == "/clear":
            agent.clear_history()
            print("历史已清空。")
            continue
        if user_input.startswith("/add "):
            if not isinstance(rag_tool, RAGTool):
                print("RAGTool 未配置。")
                continue
            try:
                print(rag_tool.add_document(user_input.split(maxsplit=1)[1]))
            except Exception as exc:
                print(f"文档入库失败：{exc}")
            continue
        if user_input.startswith("/search "):
            if not isinstance(rag_tool, RAGTool):
                print("RAGTool 未配置。")
                continue
            try:
                results = rag_tool.search(user_input.split(maxsplit=1)[1])
                print(rag_tool._context(results) if results else "没有检索到相关内容。")
            except Exception as exc:
                print(f"检索失败：{exc}")
            continue
        if user_input == "/rag-stats":
            try:
                print(rag_tool.stats() if isinstance(rag_tool, RAGTool) else "RAGTool 未配置。")
            except Exception as exc:
                print(f"读取 RAG 状态失败：{exc}")
            continue
        if user_input == "/rag-clear":
            try:
                print(rag_tool.clear() if isinstance(rag_tool, RAGTool) else "RAGTool 未配置。")
            except Exception as exc:
                print(f"清空 RAG 失败：{exc}")
            continue
        if user_input.startswith("/agent"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1] in agents:
                current_name = parts[1]
                print(f"已切换到 {current_name}。")
            else:
                print(f"可选 Agent：{', '.join(agents)}")
            continue

        try:
            run_once(agent, user_input)
        except (MyAgentsError, RuntimeError, ValueError) as exc:
            print(f"运行失败：{exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MyAgents CSY terminal application")
    parser.add_argument(
        "--agent",
        choices=("simple", "react", "reflection", "plan", "rag", "codebase"),
        default="simple",
        help="Agent pattern to use",
    )
    parser.add_argument("--prompt", help="Run one prompt and exit")
    args = parser.parse_args()

    try:
        agents, tools, llm = build_application()
    except (MyAgentsError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"配置失败：{exc}") from exc

    print(
        f"模型服务：{llm.provider} | 模型：{llm.model} | "
        f"工具：{', '.join(tools.list_tools()) or '无'}"
    )
    try:
        if args.prompt:
            run_once(agents[args.agent], args.prompt)
            return
        interactive_loop(agents, tools, args.agent)
    finally:
        close_protocol_tools(tools)


if __name__ == "__main__":
    main()
