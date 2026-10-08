"""Local Gaia2-inspired stateful scenarios, with no external app side effects."""

import json
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from ..agents.react import ReActAgent
from ..core.config import Config
from ..tools.base import Tool
from ..tools.registry import ToolRegistry
from .gaia import gaia_match
from .stats import wilson_interval


_SCHEMAS = {
    "get_state": {"key": "string"},
    "set_state": {"key": "string", "value": "string"},
    "wait": {"minutes": "integer"},
    "ask_user": {"question": "string"},
    "ask_peer": {"question": "string"},
}


@dataclass(frozen=True)
class DynamicCase:
    id: str
    question: str
    initial_state: dict[str, str]
    desired_state: dict[str, str]
    expected_answer: str | None = None
    events: tuple[dict[str, Any], ...] = ()
    fail_once: tuple[str, ...] = ()
    clarification: str | None = None
    peer_reply: str | None = None
    require_clarification: bool = False
    require_peer: bool = False
    max_steps: int = 6
    max_minutes: int = 30

    def __post_init__(self) -> None:
        if not self.id or not self.question or self.max_steps <= 0 or self.max_minutes <= 0:
            raise ValueError("Dynamic case needs id, question, and positive budgets")
        if any(name not in _SCHEMAS for name in self.fail_once):
            raise ValueError("Unknown simulated failure tool")


def load_dynamic_cases(path: str | Path) -> list[DynamicCase]:
    cases = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            case = DynamicCase(
                id=item["id"], question=item["question"],
                initial_state=dict(item.get("initial_state", {})),
                desired_state=dict(item.get("desired_state", {})),
                expected_answer=item.get("expected_answer"),
                events=tuple(item.get("events", [])),
                fail_once=tuple(item.get("fail_once", [])),
                clarification=item.get("clarification"), peer_reply=item.get("peer_reply"),
                require_clarification=bool(item.get("require_clarification", False)),
                require_peer=bool(item.get("require_peer", False)),
                max_steps=int(item.get("max_steps", 6)),
                max_minutes=int(item.get("max_minutes", 30)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dynamic scenario at {path}:{number}") from exc
        if case.id in seen:
            raise ValueError(f"Duplicate dynamic scenario: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("Dynamic suite is empty")
    return cases


class _World:
    def __init__(self, case: DynamicCase) -> None:
        self.case = case
        self.state = dict(case.initial_state)
        self.actions = []
        self.minutes = 0
        self.failed_tools = set()
        self.applied_events = set()
        self.clarified = False
        self.asked_peer = False

    def act(self, name: str, arguments: dict[str, Any]) -> str:
        self.actions.append({"name": name, "arguments": arguments})
        if name in self.case.fail_once and name not in self.failed_tools:
            self.failed_tools.add(name)
            raise RuntimeError("simulated transient API failure")
        if name == "get_state":
            key = arguments["key"]
            result = json.dumps({"key": key, "value": self.state.get(key)}, ensure_ascii=False)
        elif name == "set_state":
            if self.case.require_clarification and not self.clarified:
                raise ValueError("clarification required before changing state")
            if self.case.require_peer and not self.asked_peer:
                raise ValueError("peer consultation required before changing state")
            self.state[str(arguments["key"])] = str(arguments["value"])
            result = "state updated"
        elif name == "wait":
            minutes = arguments["minutes"]
            if type(minutes) is not int or minutes < 0 or self.minutes + minutes > self.case.max_minutes:
                raise ValueError("simulated time budget exceeded")
            self.minutes += minutes
            result = f"simulated minute {self.minutes}"
        elif name == "ask_user":
            self.clarified = True
            result = self.case.clarification or "No clarification available"
        elif name == "ask_peer":
            self.asked_peer = True
            result = self.case.peer_reply or "No peer response available"
        else:
            raise ValueError(f"Unknown world tool: {name}")
        for index, event in enumerate(self.case.events):
            if index in self.applied_events:
                continue
            action_ready = ("after_actions" not in event or
                            len(self.actions) >= int(event["after_actions"]))
            time_ready = ("at_minute" not in event or
                          self.minutes >= int(event["at_minute"]))
            if action_ready and time_ready:
                self.state.update({str(k): str(v) for k, v in event.get("updates", {}).items()})
                self.applied_events.add(index)
        return result

    def success(self, final_answer: str) -> bool:
        return (
            all(self.state.get(k) == v for k, v in self.case.desired_state.items())
            and (not self.case.require_clarification or self.clarified)
            and (not self.case.require_peer or self.asked_peer)
            and (self.case.expected_answer is None or
                 gaia_match(final_answer, self.case.expected_answer))
        )


class _WorldTool(Tool):
    def __init__(self, name: str, world: _World) -> None:
        super().__init__(name, f"Simulated local {name} action")
        self.world = world

    def function_schema(self) -> dict:
        properties = {key: {"type": kind} for key, kind in _SCHEMAS[self.name].items()}
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False},
        }}

    def run(self, input_text: str) -> str:
        return self.run_arguments(json.loads(input_text))

    def run_arguments(self, arguments: dict[str, Any]) -> str:
        expected = _SCHEMAS[self.name]
        if set(arguments) != set(expected):
            raise ValueError("incorrect simulated tool arguments")
        for key, kind in expected.items():
            if kind == "string" and not isinstance(arguments[key], str):
                raise ValueError("tool argument must be a string")
            if kind == "integer" and type(arguments[key]) is not int:
                raise ValueError("tool argument must be an integer")
        return self.world.act(self.name, arguments)


def run_dynamic(llm: Any, cases: Sequence[DynamicCase]) -> dict:
    if not cases or not callable(getattr(llm, "complete_with_tools", None)):
        raise ValueError("Dynamic evaluation requires cases and native tool calls")
    rows = []
    for case in cases:
        world = _World(case)
        registry = ToolRegistry()
        for name in _SCHEMAS:
            registry.register(_WorldTool(name, world))
        agent = ReActAgent(
            name=f"dynamic-{case.id}", llm=llm, tool_registry=registry,
            config=Config(max_steps=case.max_steps, temperature=0, tool_call_mode="native"),
            tool_call_mode="native",
        )
        started = monotonic()
        answer = ""
        error = None
        try:
            answer = agent.run(case.question)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
        success = not error and world.success(answer)
        rows.append({
            "id": case.id, "success": success, "final_answer": answer,
            "state": world.state, "actions": world.actions,
            "simulated_minutes": world.minutes,
            "tool_attempts": agent.last_tool_attempts,
            "tool_errors": agent.last_tool_errors,
            "recovered_after_failure": (success and bool(world.failed_tools))
                                       if case.fail_once else None,
            "latency_seconds": round(monotonic() - started, 4),
            "reported_prompt_tokens": agent.last_reported_prompt_tokens or None,
            "reported_completion_tokens": agent.last_reported_completion_tokens or None,
            "error": error,
        })
    attempts = sum(row["tool_attempts"] for row in rows)
    failure_cases = [row for row in rows if row["recovered_after_failure"] is not None]
    prompts = [row["reported_prompt_tokens"] for row in rows
               if row["reported_prompt_tokens"] is not None]
    completions = [row["reported_completion_tokens"] for row in rows
                   if row["reported_completion_tokens"] is not None]
    return {
        "benchmark": "gaia2_style_local_dynamic",
        "official_score": False,
        "scenario_count": len(rows),
        "pass_at_1": sum(row["success"] for row in rows) / len(rows),
        "pass_at_1_ci95": wilson_interval(sum(row["success"] for row in rows), len(rows)),
        "tool_error_rate": sum(row["tool_errors"] for row in rows) / attempts
                           if attempts else None,
        "failure_recovery_rate": (
            sum(row["recovered_after_failure"] for row in failure_cases) / len(failure_cases)
            if failure_cases else None
        ),
        "mean_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
        "reported_prompt_tokens": sum(prompts) if prompts else None,
        "reported_completion_tokens": sum(completions) if completions else None,
        "samples": rows,
    }
