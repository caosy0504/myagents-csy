from my_agents import (
    CalculatorTool,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
    ToolRegistry,
)


class FakeLLM:
    provider = "fake"

    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages, **kwargs):
        return next(self.responses)

    def stream(self, messages, **kwargs):
        yield from next(self.responses)


def test_simple_agent():
    agent = SimpleAgent("simple", FakeLLM(["你好"]))
    assert agent.run("hello") == "你好"
    assert len(agent.get_history()) == 2


def test_react_agent_calls_tool_then_finishes():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    llm = FakeLLM(
        [
            "**Thought:** 需要计算\n**Action:** calculator[2 + 3]",
            "Thought: 已得到结果\nAction: Finish[结果是5]",
        ]
    )
    agent = ReActAgent("react", llm, registry)
    assert agent.run("2+3是多少") == "结果是5"
    assert any("Observation: 5" in line for line in agent.last_trace)


def test_reflection_agent_stops_when_no_change_needed():
    agent = ReflectionAgent("reflection", FakeLLM(["初稿", "无需改进"]))
    assert agent.run("写一句话") == "初稿"
    assert agent.last_feedback == ["无需改进"]


def test_plan_and_solve_agent():
    llm = FakeLLM(['["计算", "汇总"]', "中间结果", "最终答案：5"])
    agent = PlanAndSolveAgent("plan", llm)
    assert agent.run("计算问题") == "最终答案：5"
    assert agent.last_plan == ["计算", "汇总"]


def test_plan_parser_accepts_markdown_code_block():
    raw = '```json\n["分析", "汇总最终答案"]\n```'
    assert PlanAndSolveAgent._parse_plan(raw) == ["分析", "汇总最终答案"]
