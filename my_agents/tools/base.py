"""Tool contracts for Agent actions."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """One narrowly scoped capability exposed to an Agent."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Execute the tool with one text input."""

    def function_schema(self) -> dict[str, Any]:
        """OpenAI-compatible function definition for native tool calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"input_text": {"type": "string"}},
                    "required": ["input_text"],
                    "additionalProperties": False,
                },
            },
        }

    def run_arguments(self, arguments: dict[str, Any]) -> str:
        value = arguments.get("input_text")
        if not isinstance(value, str):
            raise ValueError("input_text 必须是字符串")
        return self.run(value)

    def __str__(self) -> str:
        return f"Tool(name={self.name})"


class FunctionTool(Tool):
    """Adapt a plain string-to-string function to the Tool interface."""

    def __init__(self, name: str, description: str, function) -> None:
        super().__init__(name, description)
        self.function = function

    def run(self, input_text: str) -> str:
        return str(self.function(input_text))
