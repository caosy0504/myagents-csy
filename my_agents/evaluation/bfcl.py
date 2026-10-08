"""Safe native-call evaluation, with an optional official BFCL CLI handoff."""

import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from .stats import wilson_interval


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row must be an object")
                rows.append(value)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}") from exc
    return rows


@dataclass(frozen=True)
class BFCLCase:
    id: str
    category: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    expected: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.id or not self.category or not self.messages:
            raise ValueError("BFCL case needs id, category, and messages")
        if any(item.get("role") not in {"system", "user", "assistant", "tool"}
               for item in self.messages):
            raise ValueError("Invalid message role")
        if any(item.get("type") != "function" for item in self.tools):
            raise ValueError("Tools must use native function schemas")


def load_bfcl_cases(path: str | Path) -> list[BFCLCase]:
    """Read a small project-owned JSONL suite; this is not an official score."""
    cases = []
    seen = set()
    for row in _read_jsonl(path):
        case = BFCLCase(
            id=str(row["id"]), category=str(row["category"]),
            messages=tuple(row["messages"]), tools=tuple(row["tools"]),
            expected=tuple(row.get("expected", [])),
        )
        if case.id in seen:
            raise ValueError(f"Duplicate BFCL case id: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("BFCL suite is empty")
    return cases


def load_bfcl_v4_subset(
    questions_file: str | Path, answers_file: str | Path | None,
    *, category: str, max_samples: int | None = None,
) -> list[BFCLCase]:
    """Adapt official BFCL V4 single-turn JSONL; local matching remains a proxy."""
    if category not in {"simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"}:
        raise ValueError("Only BFCL V4 single-turn Python/irrelevance categories are supported")
    if category != "irrelevance" and answers_file is None:
        raise ValueError("This BFCL category requires its possible_answer file")
    answers = {row["id"]: row["ground_truth"] for row in _read_jsonl(answers_file)} if answers_file else {}
    rows = _read_jsonl(questions_file)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        rows = rows[:max_samples]
    cases = []
    for row in rows:
        turns = row.get("question")
        if not isinstance(turns, list) or len(turns) != 1 or not isinstance(turns[0], list):
            raise ValueError("Only single-turn BFCL entries can be adapted")
        schemas = []
        for item in row.get("function", []):
            params = dict(item.get("parameters", {}))
            if params.get("type") == "dict":
                params["type"] = "object"
            schemas.append({
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "parameters": params,
                },
            })
        if category != "irrelevance" and row["id"] not in answers:
            raise ValueError(f"Missing BFCL answer: {row['id']}")
        cases.append(BFCLCase(
            id=row["id"], category=category, messages=tuple(turns[0]),
            tools=tuple(schemas), expected=tuple(answers.get(row["id"], [])),
        ))
    if not cases:
        raise ValueError("BFCL subset is empty")
    return cases


def _value_matches(actual: Any, options: list[Any]) -> bool:
    return any(actual == option for option in options if option != "")


def _call_matches(predicted: dict, expected: dict) -> tuple[bool, bool]:
    name = predicted.get("name")
    expected_name, allowed = next(iter(expected.items()))
    if name != expected_name:
        return False, False
    arguments = predicted.get("arguments")
    if not isinstance(arguments, dict):
        return True, False
    for key, options in allowed.items():
        if key not in arguments:
            if "" in options:
                continue
            return True, False
        if not _value_matches(arguments[key], options):
            return True, False
    if set(arguments) - set(allowed):
        return True, False
    return True, True


def score_calls(predicted: Sequence[dict], expected: Sequence[dict]) -> dict:
    """Conservative argument matching; never label it the official BFCL AST metric."""
    if not expected:
        return {"selection_correct": len(predicted) == 0, "arguments_correct": len(predicted) == 0}
    selection = Counter(item.get("name") for item in predicted) == Counter(
        next(iter(item)) for item in expected
    )
    if not selection:
        return {"selection_correct": False, "arguments_correct": False}
    used = set()

    def match(index: int) -> bool:
        if index == len(predicted):
            return True
        for target_index, target in enumerate(expected):
            if target_index in used or not _call_matches(predicted[index], target)[1]:
                continue
            used.add(target_index)
            if match(index + 1):
                return True
            used.remove(target_index)
        return False

    return {"selection_correct": True, "arguments_correct": match(0)}


def run_bfcl(llm: Any, cases: Sequence[BFCLCase], *, max_samples: int | None = None) -> dict:
    """Ask the Agent's LLM client for native tool calls; no tools are executed."""
    if not callable(getattr(llm, "complete_with_tools", None)):
        raise ValueError("This BFCL runner requires native complete_with_tools")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    selected = list(cases[:max_samples])
    if not selected:
        raise ValueError("No BFCL cases selected")
    rows = []
    for case in selected:
        started = monotonic()
        predicted = []
        prompt_tokens = completion_tokens = None
        error = None
        try:
            response = llm.complete_with_tools(list(case.messages), list(case.tools), temperature=0)
            prompt_tokens = response.prompt_tokens
            completion_tokens = response.completion_tokens
            for call in response.tool_calls:
                predicted.append({"name": call.name, "arguments": json.loads(call.arguments)})
            result = score_calls(predicted, case.expected)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = {"selection_correct": False, "arguments_correct": False}
        rows.append({
            "id": case.id, "category": case.category, "predicted": predicted,
            **result, "error": error,
            "latency_seconds": round(monotonic() - started, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        group = [row for row in rows if row["category"] == category]
        categories[category] = {
            "count": len(group),
            "selection_accuracy": sum(row["selection_correct"] for row in group) / len(group),
            "argument_accuracy": sum(row["arguments_correct"] for row in group) / len(group),
            "argument_accuracy_ci95": wilson_interval(
                sum(row["arguments_correct"] for row in group), len(group)
            ),
        }
    return {
        "benchmark": "bfcl_style_local_proxy",
        "official_score": False,
        "sample_count": len(rows),
        "selection_accuracy": sum(row["selection_correct"] for row in rows) / len(rows),
        "argument_accuracy": sum(row["arguments_correct"] for row in rows) / len(rows),
        "argument_accuracy_ci95": wilson_interval(
            sum(row["arguments_correct"] for row in rows), len(rows)
        ),
        "error_rate": sum(row["error"] is not None for row in rows) / len(rows),
        "mean_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
        "reported_prompt_tokens": (
            sum(row["prompt_tokens"] or 0 for row in rows)
            if any(row["prompt_tokens"] is not None for row in rows) else None
        ),
        "reported_completion_tokens": (
            sum(row["completion_tokens"] or 0 for row in rows)
            if any(row["completion_tokens"] is not None for row in rows) else None
        ),
        "categories": categories,
        "samples": rows,
    }


def run_official_bfcl(
    model: str, category: str, *, project_root: str | Path,
    partial_eval: bool = False,
) -> dict:
    """Run the official evaluator on results generated with an official model handler."""
    executable = shutil.which("bfcl")
    if executable is None:
        raise RuntimeError("Install the official bfcl-eval package in a separate environment")
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir() or not (root / "result").is_dir():
        raise ValueError("BFCL_PROJECT_ROOT must contain official generated result files")
    command = [executable, "evaluate", "--model", model, "--test-category", category]
    if partial_eval:
        command.append("--partial-eval")
    completed = subprocess.run(
        command, cwd=root, env={**os.environ, "BFCL_PROJECT_ROOT": str(root)},
        capture_output=True, text=True, timeout=3600, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Official BFCL evaluator failed: {completed.stderr[-2000:]}")
    return {"official_evaluator": True, "command": command, "stdout": completed.stdout[-4000:]}
