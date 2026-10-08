"""Side-effect-free multi-turn ReAct evaluation with scripted tool observations."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from ..agents.react import ReActAgent
from ..core.config import Config
from ..tools.base import Tool
from ..tools.registry import ToolRegistry
from .bfcl import _call_matches
from .gaia import gaia_match
from .stats import wilson_interval


@dataclass(frozen=True)
class TrajectoryCase:
    id: str
    turns: tuple[str, ...]
    tools: tuple[dict[str, Any], ...]
    expected_calls: tuple[dict[str, Any], ...]
    expected_final: str
    max_steps: int = 4
    recovery_required: bool = False

    def __post_init__(self) -> None:
        if not self.id or not self.turns or self.max_steps <= 0:
            raise ValueError("Trajectory case needs id, turns, and positive max_steps")
        if len({tool["schema"]["function"]["name"] for tool in self.tools}) != len(self.tools):
            raise ValueError("Trajectory tool names must be unique")


def load_trajectory_cases(path: str | Path) -> list[TrajectoryCase]:
    cases = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            case = TrajectoryCase(
                id=item["id"], turns=tuple(item["turns"]),
                tools=tuple(item["tools"]), expected_calls=tuple(item["expected_calls"]),
                expected_final=item["expected_final"],
                max_steps=int(item.get("max_steps", 4)),
                recovery_required=bool(item.get("recovery_required", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trajectory at {path}:{number}") from exc
        if case.id in seen:
            raise ValueError(f"Duplicate trajectory ID: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("Trajectory suite is empty")
    return cases


class _ScriptedTool(Tool):
    def __init__(self, definition: dict, call_log: list[dict]) -> None:
        schema = definition["schema"]
        function = schema["function"]
        super().__init__(function["name"], function.get("description", "Scripted sandbox tool"))
        self.schema = schema
        self.responses = definition.get("responses", ["ok"])
        self.fail_calls = set(definition.get("fail_calls", []))
        self.call_log = call_log
        self.count = 0

    def function_schema(self) -> dict:
        return self.schema

    def run(self, input_text: str) -> str:
        return self.run_arguments({"input_text": input_text})

    def run_arguments(self, arguments: dict[str, Any]) -> str:
        self.count += 1
        self.call_log.append({"name": self.name, "arguments": arguments})
        if self.count in self.fail_calls:
            raise RuntimeError("simulated transient tool failure")
        return str(self.responses[min(self.count - 1, len(self.responses) - 1)])


def _path_matches(predicted: Sequence[dict], expected: Sequence[dict]) -> bool:
    return len(predicted) == len(expected) and all(
        _call_matches(actual, target)[1] for actual, target in zip(predicted, expected)
    )


def run_trajectories(llm: Any, cases: Sequence[TrajectoryCase]) -> dict:
    """Exercise ReActAgent's native loop against scripted, isolated tools."""
    if not callable(getattr(llm, "complete_with_tools", None)) or not cases:
        raise ValueError("Native tool LLM and nonempty trajectory cases are required")
    results = []
    for case in cases:
        calls: list[dict] = []
        registry = ToolRegistry()
        for definition in case.tools:
            registry.register(_ScriptedTool(definition, calls))
        agent = ReActAgent(
            name=f"trajectory-{case.id}", llm=llm, tool_registry=registry,
            config=Config(max_steps=case.max_steps, temperature=0, tool_call_mode="native"),
            tool_call_mode="native",
        )
        answer = ""
        error = None
        tool_attempts = tool_errors = prompt_tokens = completion_tokens = 0
        traces = []
        started = monotonic()
        for turn in case.turns:
            try:
                answer = agent.run(turn)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
            tool_attempts += agent.last_tool_attempts
            tool_errors += agent.last_tool_errors
            prompt_tokens += agent.last_reported_prompt_tokens
            completion_tokens += agent.last_reported_completion_tokens
            traces.extend(agent.last_trace)
            if error:
                break
        final_correct = not error and gaia_match(answer, case.expected_final)
        path_correct = _path_matches(calls, case.expected_calls)
        results.append({
            "id": case.id, "turn_count": len(case.turns),
            "final_answer": answer, "final_correct": final_correct,
            "tool_path_correct": path_correct,
            "task_success": final_correct and path_correct,
            "recovery_required": case.recovery_required,
            "recovered": (final_correct and tool_errors > 0 and tool_attempts > tool_errors)
                         if case.recovery_required else None,
            "tool_attempts": tool_attempts, "tool_errors": tool_errors,
            "reported_prompt_tokens": prompt_tokens or None,
            "reported_completion_tokens": completion_tokens or None,
            "latency_seconds": round(monotonic() - started, 4),
            "calls": calls, "trace": traces, "error": error,
        })
    attempts = sum(row["tool_attempts"] for row in results)
    recovery = [row for row in results if row["recovery_required"]]
    prompts = [row["reported_prompt_tokens"] for row in results
               if row["reported_prompt_tokens"] is not None]
    completions = [row["reported_completion_tokens"] for row in results
                   if row["reported_completion_tokens"] is not None]
    return {
        "benchmark": "local_agent_trajectory",
        "official_score": False,
        "sample_count": len(results),
        "task_success_rate": sum(row["task_success"] for row in results) / len(results),
        "task_success_ci95": wilson_interval(
            sum(row["task_success"] for row in results), len(results)
        ),
        "tool_path_accuracy": sum(row["tool_path_correct"] for row in results) / len(results),
        "tool_error_rate": sum(row["tool_errors"] for row in results) / attempts
                           if attempts else None,
        "recovery_rate": sum(row["recovered"] for row in recovery) / len(recovery)
                         if recovery else None,
        "mean_latency_seconds": sum(row["latency_seconds"] for row in results)
                                / len(results),
        "reported_prompt_tokens": sum(prompts) if prompts else None,
        "reported_completion_tokens": sum(completions) if completions else None,
        "samples": results,
    }
