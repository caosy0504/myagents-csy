"""OpenAI-compatible LLM interface for cloud and local providers."""

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from .exceptions import ConfigurationError, LLMError
from .message import Message


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCompletion:
    content: str
    tool_calls: list[ToolCall]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    """The minimum model interface required by an Agent."""

    provider: str

    def invoke(self, messages: Sequence[Message | dict[str, Any]], **kwargs: object) -> str: ...

    def stream(self, messages: Sequence[Message | dict[str, Any]], **kwargs: object) -> Iterator[str]: ...


class MyAgentsLLM:
    """Small OpenAI-compatible client with provider auto-detection."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 120,
    ) -> None:
        if not model or not api_key or not base_url:
            raise ConfigurationError("model、api_key 和 base_url 都不能为空")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.provider = self._detect_provider(self.base_url)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def from_env(cls) -> "MyAgentsLLM":
        """Create a client from the project .env values."""
        return cls(
            model=os.getenv("LLM_MODEL_ID", "qwen2.5:7b"),
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        )

    @staticmethod
    def _detect_provider(base_url: str) -> str:
        lowered = base_url.lower()
        if ":11434" in lowered or "ollama" in lowered:
            return "ollama"
        if ":8000" in lowered or "vllm" in lowered:
            return "vllm"
        if "api.openai.com" in lowered:
            return "openai"
        return "openai-compatible"

    @staticmethod
    def _serialize(messages: Sequence[Message | dict[str, Any]]) -> list[dict[str, Any]]:
        return [message.to_api_dict() if isinstance(message, Message) else message for message in messages]

    def invoke(self, messages: Sequence[Message | dict[str, Any]], **kwargs: object) -> str:
        """Run one non-streaming completion."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._serialize(messages),  # type: ignore[arg-type]
                temperature=float(kwargs.get("temperature", 0.3)),
                max_tokens=kwargs.get("max_tokens"),  # type: ignore[arg-type]
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"模型调用失败: {exc}") from exc

    def complete_with_tools(
        self,
        messages: Sequence[Message | dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: object,
    ) -> ToolCompletion:
        """One native OpenAI-compatible tool-calling step."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._serialize(messages),  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                tool_choice="auto",
                temperature=float(kwargs.get("temperature", 0.3)),
                max_tokens=kwargs.get("max_tokens"),  # type: ignore[arg-type]
            )
            message = response.choices[0].message
            calls = [
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in (message.tool_calls or [])
            ]
            usage = response.usage
            return ToolCompletion(
                content=message.content or "",
                tool_calls=calls,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )
        except Exception as exc:
            raise LLMError(f"原生工具调用失败: {exc}") from exc

    def stream(self, messages: Sequence[Message | dict[str, Any]], **kwargs: object) -> Iterator[str]:
        """Yield text deltas from one streaming completion."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._serialize(messages),  # type: ignore[arg-type]
                temperature=float(kwargs.get("temperature", 0.3)),
                max_tokens=kwargs.get("max_tokens"),  # type: ignore[arg-type]
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as exc:
            raise LLMError(f"流式模型调用失败: {exc}") from exc
