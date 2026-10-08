"""Agent specialized in grounded document question answering."""

from pathlib import Path

from ..core import Agent
from ..tools.rag import RAGTool


class DocumentQAAgent(Agent):
    def __init__(self, *args, rag_tool: RAGTool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rag_tool = rag_tool
        self.last_sources: list[str] = []

    def add_document(self, file_path: str | Path) -> str:
        return self.rag_tool.add_document(file_path)

    def add_text(self, text: str, source: str = "manual") -> str:
        return self.rag_tool.add_text(text, source)

    def search(self, query: str, limit: int = 5) -> str:
        results = self.rag_tool.search(query, limit)
        self.last_sources = list(dict.fromkeys(result.source for result in results))
        return self.rag_tool._context(results) if results else "没有检索到相关内容。"

    def run(self, input_text: str, **kwargs: object) -> str:
        answer = self.rag_tool.ask(input_text, kwargs.get("limit"))
        self.last_sources = list(
            dict.fromkeys(result.source for result in self.rag_tool.last_results)
        )
        self.save_turn(input_text, answer)
        return answer
