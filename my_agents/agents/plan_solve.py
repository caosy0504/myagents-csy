"""Plan first, then execute each step."""

import ast
import json
import re

from ..core import Agent, Message


PLANNER_PROMPT = """把下面的问题拆成按顺序执行的步骤。
只输出 JSON 字符串数组，不要输出说明或 Markdown，例如：
["分析问题", "完成计算", "汇总最终答案"]
最后一个步骤必须汇总前面结果并回答原始问题。

问题：{question}
"""

EXECUTOR_PROMPT = """你正在按计划解决问题。只完成当前步骤，并输出该步骤的结果。

原始问题：{question}
完整计划：{plan}
已有结果：{history}
当前步骤：{step}
"""


class PlanAndSolveAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_plan: list[str] = []
        self.step_results: list[str] = []

    def run(self, input_text: str, **kwargs: object) -> str:
        model_kwargs = self._model_kwargs(dict(kwargs))
        raw_plan = self.llm.invoke(
            [
                Message(content=self.system_prompt, role="system"),
                Message(content=PLANNER_PROMPT.format(question=input_text), role="user"),
            ],
            **model_kwargs,
        )
        self.last_plan = self._parse_plan(raw_plan)
        self.step_results = []

        if not self.last_plan:
            final = "无法解析模型生成的计划。"
            self.save_turn(input_text, final)
            return final

        for index, step in enumerate(self.last_plan, 1):
            history = "\n".join(
                f"步骤 {i}: {result}" for i, result in enumerate(self.step_results, 1)
            ) or "无"
            result = self.llm.invoke(
                [
                    Message(content=self.system_prompt, role="system"),
                    Message(
                        content=EXECUTOR_PROMPT.format(
                            question=input_text,
                            plan=self.last_plan,
                            history=history,
                            step=f"{index}. {step}",
                        ),
                        role="user",
                    ),
                ],
                **model_kwargs,
            )
            self.step_results.append(result)

        final = self.step_results[-1]
        self.save_turn(input_text, final)
        return final

    @staticmethod
    def _parse_plan(text: str) -> list[str]:
        cleaned = re.sub(r"```(?:json|python)?", "", text, flags=re.IGNORECASE).replace("```", "")
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        value = match.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
        return []
