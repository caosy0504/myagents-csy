"""A small deterministic tool-use environment and trajectory dataset for Agentic RL."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL_SYSTEM_PROMPT = (
    "Solve the arithmetic task using calculate, then call submit_answer with the result. "
    "End with Final Answer: <number>."
)
OPERATIONS = {"add": "+", "subtract": "-", "multiply": "*"}


@dataclass(frozen=True)
class ToolTask:
    a: int
    b: int
    operation: str

    def __post_init__(self) -> None:
        if type(self.a) is not int or type(self.b) is not int:
            raise ValueError("Tool operands must be integers")
        if abs(self.a) > 100 or abs(self.b) > 100 or self.operation not in OPERATIONS:
            raise ValueError("Only bounded add/subtract/multiply tasks are supported")

    @property
    def answer(self) -> int:
        return {
            "add": self.a + self.b,
            "subtract": self.a - self.b,
            "multiply": self.a * self.b,
        }[self.operation]

    @property
    def question(self) -> str:
        return f"Calculate {self.a} {OPERATIONS[self.operation]} {self.b}."


class ArithmeticToolEnv:
    """One isolated rollout; the two public action methods are model tools."""

    def __init__(self) -> None:
        self._task: ToolTask | None = None
        self._used_correct_calculator = False
        self._submitted_answer: int | None = None
        self._tool_calls = 0
        self._tool_errors = 0

    def reset(self, a: int, b: int, operation: str, **_: Any) -> None:
        self._task = ToolTask(a, b, operation)
        self._used_correct_calculator = False
        self._submitted_answer = None
        self._tool_calls = 0
        self._tool_errors = 0

    def calculate(self, a: int, b: int, operation: str) -> int:
        """Calculate a bounded arithmetic expression.

        Args:
            a: First integer operand.
            b: Second integer operand.
            operation: One of add, subtract, or multiply.

        Returns:
            The integer result.
        """
        self._tool_calls += 1
        try:
            proposed = ToolTask(a, b, operation)
        except ValueError:
            self._tool_errors += 1
            raise
        self._used_correct_calculator |= proposed == self._task
        return proposed.answer

    def submit_answer(self, answer: int) -> str:
        """Submit the final integer answer for scoring.

        Args:
            answer: Integer answer to the arithmetic task.

        Returns:
            A receipt; correctness is revealed only by the reward.
        """
        self._tool_calls += 1
        if type(answer) is not int:
            self._tool_errors += 1
            raise ValueError("answer must be an integer")
        self._submitted_answer = answer
        return "Answer submitted."

    def get_reward(self) -> float:
        if self._task is None or self._submitted_answer != self._task.answer:
            return 0.0
        return 0.8 + 0.2 * float(self._used_correct_calculator)


def tool_schemas() -> list[dict]:
    from transformers.utils import get_json_schema

    return [
        get_json_schema(ArithmeticToolEnv.calculate),
        get_json_schema(ArithmeticToolEnv.submit_answer),
    ]


@dataclass(frozen=True)
class ToolTaskBundle:
    train: tuple[ToolTask, ...]
    validation: tuple[ToolTask, ...]
    test: tuple[ToolTask, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.validation or not self.test:
            raise ValueError("All tool task splits must be nonempty")
        seen = set()
        for split in (self.train, self.validation, self.test):
            for task in split:
                if task in seen:
                    raise ValueError("Duplicate tool task across splits")
                seen.add(task)

    def manifest(self) -> dict:
        return {
            name: {
                "count": len(getattr(self, name)),
                "sha256": hashlib.sha256(json.dumps(
                    [task.__dict__ for task in getattr(self, name)], sort_keys=True
                ).encode()).hexdigest(),
            }
            for name in ("train", "validation", "test")
        }

    def rows(self, split: str, format_type: str) -> list[dict]:
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown tool task split")
        tasks = getattr(self, split)
        if format_type == "grpo":
            return [{
                "prompt": [
                    {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                    {"role": "user", "content": task.question},
                ],
                "a": task.a,
                "b": task.b,
                "operation": task.operation,
            } for task in tasks]
        if format_type == "sft":
            schemas = tool_schemas()
            return [self._trajectory(task, schemas) for task in tasks]
        raise ValueError("format_type must be sft or grpo")

    @staticmethod
    def _trajectory(task: ToolTask, schemas: list[dict]) -> dict:
        def call(name: str, arguments: dict) -> dict:
            return {"role": "assistant", "tool_calls": [{
                "type": "function", "function": {"name": name, "arguments": arguments},
            }]}

        return {
            "messages": [
                {"role": "system", "content": TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": task.question},
                call("calculate", {"a": task.a, "b": task.b, "operation": task.operation}),
                {"role": "tool", "name": "calculate", "content": str(task.answer)},
                call("submit_answer", {"answer": task.answer}),
                {"role": "tool", "name": "submit_answer", "content": "Answer submitted."},
                {"role": "assistant", "content": f"Final Answer: {task.answer}"},
            ],
            "tools": schemas,
        }

    def hf_dataset(self, split: str, format_type: str):
        from datasets import Dataset

        return Dataset.from_list(self.rows(split, format_type), on_mixed_types="use_json")


def load_tool_task_bundle(
    train_file: str | Path, validation_file: str | Path, test_file: str | Path,
) -> ToolTaskBundle:
    def read(path: str | Path) -> tuple[ToolTask, ...]:
        path = Path(path)
        result = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                result.append(ToolTask(raw["a"], raw["b"], raw["operation"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid tool task: {path}:{index}") from exc
        return tuple(result)

    return ToolTaskBundle(read(train_file), read(validation_file), read(test_file))
