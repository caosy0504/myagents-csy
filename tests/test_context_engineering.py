"""Behavioral checks for the chapter 9 context workflow."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from my_agents import CodebaseAgent, Config, ContextBuilder, ContextConfig, ContextPacket, Message, MyAgentsLLM, SimpleAgent
from my_agents.core import ToolCall, ToolCompletion
from my_agents.core.exceptions import LLMError
from my_agents.tools import NoteTool, TerminalTool


class RecordingLLM:
    provider = "fake"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return next(self.responses)

    def stream(self, messages, **kwargs):
        yield from next(self.responses)


def test_context_builder_keeps_task_and_role_within_budget():
    builder = ContextBuilder(ContextConfig(max_tokens=240, reserve_ratio=0.2, max_packet_tokens=60))
    old = datetime.now(timezone.utc) - timedelta(days=30)
    messages = builder.build_messages(
        "查找测试失败原因",
        "你是助手。",
        history=[Message(role="user", content="先前的问题", timestamp=old)],
        packets=[
            ContextPacket("测试失败，因为连接超时。" * 30, source="test_log"),
            ContextPacket("完全无关的资料" * 30, source="other", timestamp=old),
        ],
    )
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert "查找测试失败原因" in messages[-1].content
    assert any("test_log" in message.content for message in messages)
    assert builder.estimate_messages(messages) <= 192


def test_context_builder_rejects_oversized_current_task():
    builder = ContextBuilder(ContextConfig(max_tokens=30))
    with pytest.raises(ValueError, match="exceed"):
        builder.build_messages("任务" * 100, "系统指令")


def test_history_is_summarized_while_recent_roles_remain(tmp_path):
    llm = RecordingLLM(["目标是修复登录；已定位 app.py；下一步检查测试。"])
    agent = SimpleAgent(
        "history", llm,
        config=Config(max_history_length=4, history_keep_recent_messages=2),
        context_builder=ContextBuilder(ContextConfig(max_tokens=4096)),
    )
    for number in range(3):
        agent.save_turn(f"问题 {number}", f"回答 {number}")
    assert "app.py" in agent.get_history_summary()
    assert [message.role for message in agent.get_history()] == ["user", "assistant"]
    built = agent.build_messages("继续")
    assert [message.role for message in built] == ["system", "user", "user", "assistant", "user"]
    assert "Prior conversation summary" in built[1].content
    agent.clear_history()
    assert agent.get_history_summary() == ""


def test_note_tool_persists_and_searches(tmp_path):
    notes = NoteTool(tmp_path / "notes")
    created = json.loads(notes.run(json.dumps({
        "action": "create", "title": "测试进度", "content": "定位超时问题", "tags": ["任务"],
    }, ensure_ascii=False)))
    note_id = created["id"]
    assert len(json.loads(notes.run('{"action":"search","query":"测试为什么超时"}'))["items"]) == 1
    assert json.loads(notes.run(json.dumps({"action": "read", "id": note_id})))['content'] == "定位超时问题"
    notes.run(json.dumps({"action": "update", "id": note_id, "content": "已经修复"}, ensure_ascii=False))
    assert json.loads(NoteTool(tmp_path / "notes").run(json.dumps({"action": "read", "id": note_id})))['content'] == "已经修复"
    assert json.loads(notes.run('{"action":"summary"}'))["count"] == 1
    assert len(json.loads(notes.run('{"action":"list"}'))) == 1
    notes.run(json.dumps({"action": "delete", "id": note_id}))
    assert json.loads(notes.run('{"action":"summary"}'))["count"] == 0


def test_note_search_ranks_terms_and_filters_task(tmp_path):
    notes = NoteTool(tmp_path / "notes", project_id="repo-a")
    for title, content, task_id in [
        ("登录测试", "认证超时，需要检查 app.py", "login"),
        ("天气测试", "检查天气接口", "weather"),
        ("登录决定", "登录异常来自 session.py", "login"),
    ]:
        notes.run(json.dumps({
            "action": "create", "title": title, "content": content,
            "task_id": task_id, "tags": [],
        }, ensure_ascii=False))
    result = json.loads(notes.run(json.dumps({
        "action": "search", "query": "登录超时", "task_id": "login", "limit": 1,
    }, ensure_ascii=False)))
    assert result["total"] == 2
    assert result["items"][0]["title"] == "登录测试"
    assert result["items"][0]["score"] > 0
    assert result["next_offset"] == 1
    assert "source_path" in result["items"][0]


def test_terminal_is_bounded_and_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "sample.py").write_text("TODO: 检查更长的说明文字\n第二行", encoding="utf-8")
    outside = tmp_path / "private.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)
    (workspace / ".env").write_text("API_KEY=secret", encoding="utf-8")
    terminal = TerminalTool(workspace, max_output_chars=256)
    assert any(item["path"] == "sample.py" for item in json.loads(terminal.run('{"action":"list"}'))["items"])
    assert "sample.py" in json.loads(terminal.run('{"action":"glob","pattern":"*.py"}'))["items"]
    assert "TODO" in json.loads(terminal.run('{"action":"read","path":"sample.py","start_line":1,"end_line":1}'))["content"]
    assert json.loads(terminal.run('{"action":"search","query":"TODO"}'))["items"][0]["line"] == 1
    for path in ("../private.txt", "link.txt"):
        with pytest.raises(ValueError, match="之外"):
            terminal.run(json.dumps({"action": "read", "path": path}))
    with pytest.raises(ValueError, match="隐藏"):
        terminal.run('{"action":"read","path":".env"}')


def test_terminal_range_and_pagination(tmp_path):
    (tmp_path / "a.py").write_text("one\nTODO first\nthree\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("TODO second\n", encoding="utf-8")
    terminal = TerminalTool(tmp_path)
    line = json.loads(terminal.run('{"action":"read","path":"a.py","start_line":2,"end_line":2}'))
    assert line["content"] == "2: TODO first"
    first = json.loads(terminal.run('{"action":"search","query":"TODO","limit":1}'))
    assert first["next_offset"] == 1
    second = json.loads(terminal.run('{"action":"search","query":"TODO","limit":1,"offset":1}'))
    assert first["items"][0]["path"] != second["items"][0]["path"]


def test_terminal_long_line_can_be_resumed(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 500, encoding="utf-8")
    terminal = TerminalTool(tmp_path, max_output_chars=300)
    first = json.loads(terminal.run('{"action":"read","path":"long.txt","start_line":1,"end_line":1}'))
    assert first["truncated"] is True
    assert first["next_line"] == 1
    assert first["next_column"] > 0
    second = json.loads(terminal.run(json.dumps({
        "action": "read", "path": "long.txt", "start_line": first["next_line"],
        "end_line": 1, "start_column": first["next_column"],
    })))
    assert second["content"].startswith("1: xxx")


def test_codebase_agent_combines_notes_files_and_react(tmp_path):
    (tmp_path / "app.py").write_text("# login bug", encoding="utf-8")
    llm = RecordingLLM([
        'Thought: 先查看文件\nAction: terminal[{"action":"read","path":"app.py"}]',
        'Thought: 记录进度\nAction: note[{"action":"create","title":"登录修复","content":"检查 app.py","tags":[]}]',
        'Thought: 已定位\nAction: Finish[登录问题在 app.py]',
    ])
    agent = CodebaseAgent("维护", llm, tmp_path)
    agent.note_tool.run(json.dumps({
        "action": "create", "title": "登录问题", "content": "检查 app.py 的 login bug", "tags": [],
    }, ensure_ascii=False))
    assert "app.py" in agent.run("检查登录问题")
    first = llm.calls[0]
    assert first[0].role == "system"
    assert any("project_notes" in message.content for message in first)
    assert any("tool_trace" in message.content for message in llm.calls[1])
    assert any("login bug" in item for item in agent.last_trace)
    assert json.loads(agent.note_tool.run('{"action":"summary"}'))["count"] == 2


def test_native_tool_calls_preserve_assistant_tool_pair(tmp_path):
    (tmp_path / "app.py").write_text("login handler", encoding="utf-8")

    class NativeLLM:
        provider = "fake"

        def __init__(self):
            self.calls = []

        def complete_with_tools(self, messages, tools, **kwargs):
            self.calls.append((list(messages), tools))
            if len(self.calls) == 1:
                return ToolCompletion("", [ToolCall("call-1", "terminal", '{"action":"read","path":"app.py"}')], 42, 12)
            return ToolCompletion("登录逻辑在 app.py", [], 60, 9)

        def invoke(self, messages, **kwargs):
            return "摘要"

    llm = NativeLLM()
    agent = CodebaseAgent("代码", llm, tmp_path, tool_call_mode="native")
    assert agent.run("定位登录逻辑") == "登录逻辑在 app.py"
    assert agent.last_call_mode == "native"
    assert agent.last_tool_errors == 0
    assert agent.last_reported_prompt_tokens == 102
    assert agent.last_input_tokens > 0
    assert agent.last_tool_attempts == 1
    second_messages = llm.calls[1][0]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "call-1"
    assert llm.calls[0][1][0]["function"]["name"] == "note"


def test_openai_compatible_adapter_reads_native_tool_calls():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="id-1", function=SimpleNamespace(name="terminal", arguments='{"action":"list"}'),
                )],
            ))],
            usage=SimpleNamespace(prompt_tokens=28, completion_tokens=6),
        )

    client = object.__new__(MyAgentsLLM)
    client.model = "fake-model"
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = client.complete_with_tools(
        [Message(role="user", content="列出文件")],
        [{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
    )
    assert result.tool_calls[0].name == "terminal"
    assert result.prompt_tokens == 28
    assert seen["tool_choice"] == "auto"
    assert seen["messages"][0]["role"] == "user"


def test_native_tool_error_rate_counts_bad_arguments(tmp_path):
    class BadArgumentsLLM:
        provider = "fake"

        def __init__(self):
            self.count = 0

        def complete_with_tools(self, messages, tools, **kwargs):
            self.count += 1
            if self.count == 1:
                return ToolCompletion("", [ToolCall("bad-1", "terminal", "{invalid")])
            assert "参数无效" in messages[-1]["content"]
            return ToolCompletion("已发现参数错误", [])

        def invoke(self, messages, **kwargs):
            return "摘要"

    agent = CodebaseAgent("错误", BadArgumentsLLM(), tmp_path, tool_call_mode="native")
    assert agent.run("查看文件") == "已发现参数错误"
    assert agent.last_tool_attempts == 1
    assert agent.last_tool_errors == 1


def test_auto_mode_falls_back_when_provider_rejects_tools(tmp_path):
    class UnsupportedLLM:
        provider = "fake"

        def complete_with_tools(self, messages, tools, **kwargs):
            raise LLMError("model does not support tools")

        def invoke(self, messages, **kwargs):
            return "Thought: 无需工具\nAction: Finish[文本模式完成]"

    agent = CodebaseAgent("回退", UnsupportedLLM(), tmp_path, tool_call_mode="auto")
    assert agent.run("你好") == "文本模式完成"
    assert agent.last_call_mode == "text"


def test_offline_fixed_questions_detect_history_improvement():
    from examples.evaluate_context_engineering import evaluate

    report = evaluate("offline")
    assert report["profiles"]["text_baseline"]["quality_pass_count"] == 3
    assert report["profiles"]["context_roles"]["quality_pass_count"] == 4
    assert report["profiles"]["native_tools"]["tool_errors"] == 0
