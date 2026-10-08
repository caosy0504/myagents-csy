"""Fixed-question comparison of a text ReAct baseline and the context-aware agent.

Offline mode checks orchestration deterministically. Live mode uses the configured
LLM and records keyword-based answer checks for later human review.
"""

import argparse
import json
import os
import socket
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from dotenv import load_dotenv

from my_agents import CodebaseAgent, Config, ContextBuilder, ContextConfig, MyAgentsLLM, ReActAgent
from my_agents.core import ToolCall, ToolCompletion
from my_agents.tools import NoteTool, TerminalTool, ToolRegistry


CASES = [
    {
        "id": "login",
        "question": "登录超时关联哪个文件？",
        "expected": ["app.py"],
        "lookup": {"name": "note", "arguments": {"action": "search", "query": "登录超时"}},
        "evidence": "关联 app.py",
        "answer": "登录超时关联 app.py。",
    },
    {
        "id": "todo",
        "question": "限流 TODO 在哪个文件？",
        "expected": ["rate_limit.py"],
        "lookup": {"name": "terminal", "arguments": {"action": "search", "query": "TODO 限流"}},
        "evidence": "TODO 限流",
        "answer": "限流 TODO 在 rate_limit.py。",
    },
    {
        "id": "migration",
        "question": "数据库迁移的下一步是什么？",
        "expected": ["回滚测试"],
        "lookup": {"name": "note", "arguments": {"action": "search", "query": "数据库迁移"}},
        "evidence": "回滚测试",
        "answer": "数据库迁移的下一步是回滚测试。",
    },
    {
        "id": "history",
        "question": "之前确认的失败文件是哪个？",
        "expected": ["app.py"],
        "lookup": {"name": "terminal", "arguments": {"action": "search", "query": "失败文件"}},
        "evidence": "失败文件 app.py",
        "answer": "之前确认的失败文件是 app.py。",
        "seed_history": True,
    },
]


class OfflineLLM:
    """The same deterministic policy drives both modes; it is not an LLM judge."""

    provider = "offline"

    def __init__(self, case: dict) -> None:
        self.case = case
        self.calls = 0

    def _has_evidence(self, messages) -> bool:
        text = "\n".join(
            message.content if hasattr(message, "content") else str(message.get("content") or "")
            for message in messages
        )
        return self.case["evidence"] in text

    def invoke(self, messages, **kwargs) -> str:
        self.calls += 1
        if any("请把旧对话压缩为续聊摘要" in (
            message.content if hasattr(message, "content") else str(message.get("content") or "")
        ) for message in messages):
            return "旧结论：失败文件 app.py；当前仍需记住这个文件路径。"
        if self._has_evidence(messages):
            return f"Thought: 已找到依据\nAction: Finish[{self.case['answer']}]"
        lookup = self.case["lookup"]
        payload = json.dumps(lookup["arguments"], ensure_ascii=False)
        return f"Thought: 需要检索\nAction: {lookup['name']}[{payload}]"

    def complete_with_tools(self, messages, tools, **kwargs) -> ToolCompletion:
        self.calls += 1
        if self._has_evidence(messages):
            return ToolCompletion(self.case["answer"], [])
        lookup = self.case["lookup"]
        call = ToolCall(
            id=f"call-{self.calls}", name=lookup["name"],
            arguments=json.dumps(lookup["arguments"], ensure_ascii=False),
        )
        return ToolCompletion("", [call])


def make_fixture(root: Path) -> Path:
    (root / "app.py").write_text("# 登录超时处理入口\n", encoding="utf-8")
    (root / "rate_limit.py").write_text("# TODO 限流: 添加单元测试\n", encoding="utf-8")
    notes_dir = root / "notes"
    notes = NoteTool(notes_dir)
    notes.run(json.dumps({
        "action": "create", "title": "登录超时排查",
        "content": "登录超时关联 app.py，下一步检查异常处理。", "tags": ["登录"],
    }, ensure_ascii=False))
    notes.run(json.dumps({
        "action": "create", "title": "数据库迁移进度",
        "content": "迁移脚本已完成；下一步是回滚测试。", "tags": ["迁移"],
    }, ensure_ascii=False))
    return notes_dir


def evaluate(mode: str) -> dict:
    rows = []
    with TemporaryDirectory(prefix="context-eval-") as directory:
        root = Path(directory)
        notes_dir = make_fixture(root)
        for case in CASES:
            llm = OfflineLLM(case) if mode == "offline" else MyAgentsLLM.from_env()
            tools = ToolRegistry()
            tools.register(NoteTool(notes_dir))
            tools.register(TerminalTool(root))
            config = Config(max_steps=4, max_history_length=4, history_keep_recent_messages=2)
            profiles = (
                ("text_baseline", ReActAgent(
                    "baseline", llm, tools, config=config,
                    tool_call_mode="text",
                )),
                ("context_roles", ReActAgent(
                    "context", llm, tools, config=config,
                    context_builder=ContextBuilder(ContextConfig()),
                    tool_call_mode="text",
                )),
                ("retrieval_prefetch", CodebaseAgent(
                    "retrieval", llm, root, notes_dir=notes_dir,
                    config=config, tool_call_mode="text",
                )),
                ("native_tools", CodebaseAgent(
                    "native", llm, root, notes_dir=notes_dir,
                    config=config, tool_call_mode="auto",
                )),
            )
            for profile, agent in profiles:
                if case.get("seed_history"):
                    agent.save_turn("我们确认失败文件是 app.py", "是 app.py")
                    agent.save_turn("现在先处理其他问题", "好的")
                    agent.save_turn("继续", "完成")
                try:
                    answer = agent.run(case["question"])
                    error = None
                except Exception as exc:
                    answer = ""
                    error = str(exc)
                rows.append({
                    "case": case["id"], "profile": profile, "question": case["question"],
                    "expected": case["expected"], "answer": answer,
                    "quality_pass": all(term in answer for term in case["expected"]),
                    "input_tokens_estimate": agent.last_input_tokens,
                    "provider_prompt_tokens": agent.last_reported_prompt_tokens or None,
                    "tool_attempts": agent.last_tool_attempts,
                    "tool_errors": agent.last_tool_errors,
                    "tool_error_rate": round(agent.last_tool_errors / max(agent.last_tool_attempts, 1), 3),
                    "call_mode": agent.last_call_mode,
                    "error": error,
                })
    summary = {}
    for profile in ("text_baseline", "context_roles", "retrieval_prefetch", "native_tools"):
        group = [row for row in rows if row["profile"] == profile]
        attempts = sum(row["tool_attempts"] for row in group)
        summary[profile] = {
            "quality_pass_count": sum(row["quality_pass"] for row in group),
            "total_cases": len(group),
            "input_tokens_estimate": sum(row["input_tokens_estimate"] for row in group),
            "tool_errors": sum(row["tool_errors"] for row in group),
            "tool_attempts": attempts,
            "tool_error_rate": round(sum(row["tool_errors"] for row in group) / max(attempts, 1), 3),
        }
    return {
        "mode": mode,
        "quality_note": (
            "Offline mode checks deterministic orchestration, not actual model answer quality."
            if mode == "offline" else
            "Keyword checks are a proxy; review individual answers for correctness and support."
        ),
        "comparison_note": (
            "All profiles use the current NoteTool and TerminalTool; these rows isolate context "
            "assembly, prefetch, and native calls, not the older tool implementations."
        ),
        "profiles": summary,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "live":
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        endpoint = urlparse(os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
        if endpoint.hostname in {"localhost", "127.0.0.1", "::1"}:
            try:
                with socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=2):
                    pass
            except OSError as exc:
                raise SystemExit(
                    f"本地模型服务未启动：{endpoint.hostname}:{endpoint.port or 80}"
                ) from exc
    result = evaluate(args.mode)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
