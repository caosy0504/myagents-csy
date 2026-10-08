"""Optional Tavily search tool."""

import os

from .base import Tool


class TavilySearchTool(Tool):
    def __init__(self, api_key: str | None = None, max_results: int = 3) -> None:
        super().__init__(
            name="search",
            description="搜索互联网信息并返回摘要与来源链接",
        )
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.max_results = max_results

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def run(self, input_text: str) -> str:
        query = input_text.strip()
        if not query:
            return "错误：搜索内容不能为空。"
        if not self.api_key:
            return "错误：请先在 .env 中配置 TAVILY_API_KEY。"

        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise RuntimeError("请安装搜索依赖：pip install -e '.[search]'") from exc

        response = TavilyClient(api_key=self.api_key).search(
            query=query,
            include_answer=True,
            max_results=self.max_results,
        )
        parts: list[str] = []
        if response.get("answer"):
            parts.append(f"摘要：{response['answer']}")
        for index, item in enumerate(response.get("results", []), 1):
            parts.append(
                f"[{index}] {item.get('title', '')}\n"
                f"{item.get('content', '')[:300]}\n"
                f"来源：{item.get('url', '')}"
            )
        return "\n\n".join(parts) or "没有找到相关结果。"
