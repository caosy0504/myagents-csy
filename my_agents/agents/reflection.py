"""Generate, reflect, and refine Agent."""

from ..core import Agent, Message


INITIAL_PROMPT = """请完成下面的任务，并给出完整、准确的初稿：
{task}
"""

REFLECT_PROMPT = """请审查下面的回答。
如果无需修改，只回答“无需改进”；否则列出最重要的改进意见。

任务：{task}
回答：{answer}
"""

REFINE_PROMPT = """请根据反馈给出改进后的最终回答，不要解释修改过程。

任务：{task}
上一版：{answer}
反馈：{feedback}
"""


class ReflectionAgent(Agent):
    def __init__(self, *args, max_iterations: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_iterations = (
            self.config.reflection_iterations if max_iterations is None else max_iterations
        )
        self.last_feedback: list[str] = []

    def _ask(self, prompt: str, model_kwargs: dict[str, object]) -> str:
        return self.llm.invoke(
            [
                Message(content=self.system_prompt, role="system"),
                Message(content=prompt, role="user"),
            ],
            **model_kwargs,
        )

    def run(self, input_text: str, **kwargs: object) -> str:
        model_kwargs = self._model_kwargs(dict(kwargs))
        answer = self._ask(INITIAL_PROMPT.format(task=input_text), model_kwargs)
        self.last_feedback = []

        for _ in range(self.max_iterations):
            feedback = self._ask(
                REFLECT_PROMPT.format(task=input_text, answer=answer),
                model_kwargs,
            )
            self.last_feedback.append(feedback)
            if "无需改进" in feedback:
                break
            answer = self._ask(
                REFINE_PROMPT.format(
                    task=input_text,
                    answer=answer,
                    feedback=feedback,
                ),
                model_kwargs,
            )

        self.save_turn(input_text, answer)
        return answer
