"""ReAct Agent with native tool calls and a text-protocol fallback."""

import json
import re
from typing import Any

from ..core import Agent, Config, LLMClient, Message
from ..context import ContextBuilder, ContextPacket
from ..context.builder import TokenCounter
from ..core.exceptions import LLMError
from ..tools import ToolRegistry


REACT_PROMPT = """你是一个能够调用工具解决问题的助手。

可用工具：
{tools}

每轮必须严格输出：
Thought: 一句简短的判断
Action: 工具名[输入]

获得足够信息后输出：
Thought: 一句简短的判断
Action: Finish[最终答案]

用户问题：{question}

已有执行记录：
{trace}
"""


class ReActAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        system_prompt: str | None = None,
        config: Config | None = None,
        max_steps: int | None = None,
        context_builder: ContextBuilder | None = None,
        tool_call_mode: str | None = None,
    ) -> None:
        tool_call_mode = tool_call_mode or (config.tool_call_mode if config else "auto")
        if tool_call_mode not in {"auto", "native", "text"}:
            raise ValueError("tool_call_mode 必须是 auto/native/text")
        super().__init__(name, llm, system_prompt, config, context_builder)
        self.tool_registry = tool_registry
        self.max_steps = max_steps or self.config.max_steps
        self.last_trace: list[str] = []
        self.tool_call_mode = tool_call_mode
        self.last_call_mode = "text"
        self.last_input_tokens = 0
        self.last_reported_prompt_tokens = 0
        self.last_reported_completion_tokens = 0
        self.last_tool_attempts = 0
        self.last_tool_errors = 0
        self._counter = TokenCounter()

    def run(self, input_text: str, **kwargs: object) -> str:
        self.last_trace = []
        self.last_input_tokens = 0
        self.last_reported_prompt_tokens = 0
        self.last_reported_completion_tokens = 0
        self.last_tool_attempts = 0
        self.last_tool_errors = 0
        context_packets = kwargs.pop("context_packets", ())
        model_kwargs = self._model_kwargs(dict(kwargs))
        native_available = callable(getattr(self.llm, "complete_with_tools", None))
        if self.tool_call_mode == "native" and not native_available:
            raise LLMError("当前 LLM 不支持原生工具调用")
        if self.tool_call_mode != "text" and native_available:
            self.last_call_mode = "native"
            try:
                return self._run_native(input_text, context_packets, model_kwargs)
            except LLMError as exc:
                unsupported = any(word in str(exc).lower() for word in (
                    "unsupported", "not support", "does not support", "unknown tool", "不支持",
                ))
                if self.tool_call_mode != "auto" or not unsupported or self.last_trace:
                    raise
                self.last_trace = []
                self.last_input_tokens = 0
                self.last_reported_prompt_tokens = 0
                self.last_reported_completion_tokens = 0
                self.last_tool_attempts = 0
                self.last_tool_errors = 0
        self.last_call_mode = "text"
        return self._run_text(input_text, context_packets, model_kwargs)

    def _estimate(self, messages: list[Message] | list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            if isinstance(message, Message):
                total += self._counter.count(message.content) + 4
            else:
                total += self._counter.count(str(message.get("content") or "")) + 4
                if message.get("tool_calls"):
                    total += self._counter.count(json.dumps(message["tool_calls"], ensure_ascii=False))
        return total

    def _run_native(self, input_text: str, context_packets, model_kwargs: dict[str, object]) -> str:
        schemas = self.tool_registry.function_schemas()
        schema_tokens = self._counter.count(json.dumps(schemas, ensure_ascii=False))
        context_budget = (
            self.context_builder.config.input_budget - schema_tokens
            if self.context_builder is not None else None
        )
        initial = self.build_messages(
            input_text, context_packets=context_packets, context_budget=context_budget
        )
        messages: list[dict[str, Any]] = [message.to_api_dict() for message in initial]
        initial_count = len(messages)
        for step in range(1, self.max_steps + 1):
            initial_count = self._fit_native_context(messages, initial_count, schema_tokens)
            completion = self.llm.complete_with_tools(  # type: ignore[attr-defined]
                messages, schemas, **model_kwargs
            )
            self.last_input_tokens += self._estimate(messages) + schema_tokens
            self.last_reported_prompt_tokens += completion.prompt_tokens or 0
            self.last_reported_completion_tokens += completion.completion_tokens or 0
            if not completion.tool_calls:
                final = completion.content.strip() or "模型没有返回最终答案。"
                self.save_turn(input_text, final)
                return final
            messages.append({
                "role": "assistant", "content": completion.content or None,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name, "arguments": call.arguments}}
                    for call in completion.tool_calls
                ],
            })
            for call in completion.tool_calls:
                self.last_tool_attempts += 1
                try:
                    arguments = json.loads(call.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("工具参数必须是 JSON 对象")
                    observation, error = self.tool_registry.execute_arguments(call.name, arguments)
                except (ValueError, json.JSONDecodeError) as exc:
                    observation, error = f"错误：工具参数无效：{exc}", True
                self.last_tool_errors += int(error)
                if self.context_builder is not None:
                    observation = self.context_builder.counter.fit(
                        observation, self.context_builder.config.max_packet_tokens
                    )
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "content": observation,
                })
                self.last_trace.append(
                    f"步骤 {step} Action: {call.name}[{call.arguments}]\n"
                    f"步骤 {step} Observation: {observation}"
                )
        final = "未能在最大步骤数内完成任务。"
        self.save_turn(input_text, final)
        return final

    def _fit_native_context(
        self, messages: list[dict[str, Any]], initial_count: int, schema_tokens: int
    ) -> int:
        if self.context_builder is None:
            return initial_count
        budget = self.context_builder.config.input_budget - schema_tokens
        # Drop oldest complete assistant/tool groups while retaining the latest result.
        while self._estimate(messages) > budget:
            starts = [
                index for index in range(initial_count, len(messages))
                if messages[index].get("role") == "assistant" and messages[index].get("tool_calls")
            ]
            if len(starts) < 2:
                break
            del messages[starts[0]:starts[1]]
        # Then remove optional evidence from the initial prompt, never the current task.
        while self._estimate(messages) > budget:
            evidence_index = next((
                index for index in range(1, initial_count - 1)
                if str(messages[index].get("content", "")).startswith("[Evidence:")
            ), None)
            if evidence_index is None:
                break
            del messages[evidence_index]
            initial_count -= 1
        # If needed, discard the oldest retained conversation turn.
        while self._estimate(messages) > budget and initial_count > 2:
            first = 1
            if str(messages[first].get("content", "")).startswith("[Prior conversation summary"):
                del messages[first]
                initial_count -= 1
                continue
            next_user = next((
                index for index in range(first + 1, initial_count - 1)
                if messages[index].get("role") == "user"
            ), initial_count - 1)
            del messages[first:next_user]
            initial_count -= next_user - first
        # A single unusually large tool result can be shortened in place.
        if self._estimate(messages) > budget:
            for message in reversed(messages):
                if message.get("role") != "tool":
                    continue
                over = self._estimate(messages) - budget
                old = str(message["content"])
                target = max(0, self._counter.count(old) - over - 8)
                message["content"] = self._counter.fit(old, target)
                if self._estimate(messages) <= budget:
                    break
        return initial_count

    def _run_text(self, input_text: str, context_packets, model_kwargs: dict[str, object]) -> str:
        for step in range(1, self.max_steps + 1):
            prompt = REACT_PROMPT.format(
                tools=self.tool_registry.descriptions(),
                question=input_text,
                trace=("见上下文中的最近执行记录" if self.context_builder is not None else "\n".join(self.last_trace) or "无"),
            )
            if self.context_builder is not None:
                packets = list(context_packets)
                if self.last_trace:
                    packets.append(ContextPacket("\n".join(self.last_trace[-3:]), source="tool_trace"))
                messages = self.build_messages(input_text, context_packets=packets, task=prompt)
            else:
                messages = [Message(content=self.system_prompt, role="system"), Message(content=prompt, role="user")]
            self.last_input_tokens += self._estimate(messages)
            response = self.llm.invoke(messages, **model_kwargs)
            thought, action = self._parse_output(response)
            self._log(f"step={step}, thought={thought}, action={action}")

            if not action:
                self.last_tool_attempts += 1
                self.last_tool_errors += 1
                self.last_trace.append(f"步骤 {step}: 无法解析输出：{response}")
                continue

            name, value = self._parse_action(action)
            if name.lower() == "finish":
                self.save_turn(input_text, value)
                return value

            self.last_tool_attempts += 1
            observation, error = self.tool_registry.execute_with_status(name, value)
            self.last_tool_errors += int(error)
            self.last_trace.extend(
                [
                    f"步骤 {step} Thought: {thought or '无'}",
                    f"步骤 {step} Action: {name}[{value}]",
                    f"步骤 {step} Observation: {observation}",
                ]
            )

        final = "未能在最大步骤数内完成任务。"
        self.save_turn(input_text, final)
        return final

    @staticmethod
    def _parse_output(text: str) -> tuple[str | None, str | None]:
        normalized = text.replace("**", "").replace("：", ":")
        thought_match = re.search(
            r"^\s*Thought\s*:\s*(.+)$", normalized, re.IGNORECASE | re.MULTILINE
        )
        action_match = re.search(
            r"^\s*Action\s*:\s*(.+)$", normalized, re.IGNORECASE | re.MULTILINE
        )
        return (
            thought_match.group(1).strip() if thought_match else None,
            action_match.group(1).strip() if action_match else None,
        )

    @staticmethod
    def _parse_action(action: str) -> tuple[str, str]:
        normalized = action.strip().strip("`")
        match = re.match(r"([A-Za-z_]\w*)\s*[\[【](.*)[\]】]\s*$", normalized)
        if not match:
            return "invalid", normalized
        return match.group(1), match.group(2).strip()
