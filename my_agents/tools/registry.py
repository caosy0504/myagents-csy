"""Registry that gives every Agent one tool-management interface."""

from collections.abc import Callable
from typing import Any

from .base import FunctionTool, Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        function: Callable[[str], str],
    ) -> None:
        self.register(FunctionTool(name, description, function))

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, input_text: str) -> str:
        return self.execute_with_status(name, input_text)[0]

    def execute_with_status(self, name: str, input_text: str) -> tuple[str, bool]:
        tool = self.get(name)
        if tool is None:
            return f"错误：未找到工具 '{name}'。", True
        try:
            return tool.run(input_text), False
        except Exception as exc:
            return f"错误：工具 '{name}' 执行失败：{exc}", True

    def execute_arguments(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        tool = self.get(name)
        if tool is None:
            return f"错误：未找到工具 '{name}'。", True
        try:
            return tool.run_arguments(arguments), False
        except Exception as exc:
            return f"错误：工具 '{name}' 执行失败：{exc}", True

    def function_schemas(self) -> list[dict[str, Any]]:
        return [tool.function_schema() for tool in self._tools.values()]

    def descriptions(self) -> str:
        if not self._tools:
            return "暂无可用工具"
        return "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self._tools.values()
        )

    def list_tools(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
