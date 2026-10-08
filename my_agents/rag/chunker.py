"""Markdown-aware, token-bounded chunking with overlap."""

import re
from dataclasses import dataclass

from .models import DocumentChunk


@dataclass(slots=True)
class _Paragraph:
    content: str
    heading_path: str | None


class MarkdownChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding = None

    @property
    def encoding(self):
        if self._encoding is None:
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoding = False
        return self._encoding

    def count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        return max(1, len(text) // 2)

    def _token_slice(self, text: str, start: int, end: int) -> str:
        if self.encoding:
            return self.encoding.decode(self.encoding.encode(text)[start:end])
        return text[start * 2 : end * 2]

    @staticmethod
    def _paragraphs(markdown: str) -> list[_Paragraph]:
        heading_stack: list[str] = []
        paragraphs: list[_Paragraph] = []
        buffer: list[str] = []

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if content:
                path = " > ".join(heading_stack) if heading_stack else None
                paragraphs.append(_Paragraph(content, path))
            buffer.clear()

        in_code_block = False
        for line in markdown.splitlines():
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                buffer.append(line)
                continue

            heading = None if in_code_block else re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading:
                flush()
                level = len(heading.group(1))
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading.group(2).strip())
                continue

            if not line.strip() and not in_code_block:
                flush()
            else:
                buffer.append(line)
        flush()
        return paragraphs

    def split(
        self,
        markdown: str,
        source: str,
        document_id: str,
    ) -> list[DocumentChunk]:
        paragraphs = self._paragraphs(markdown)
        sections: list[tuple[str, str | None]] = []
        current: list[str] = []
        current_heading: str | None = None

        def flush_section() -> None:
            if current:
                sections.append(("\n\n".join(current).strip(), current_heading))
                current.clear()

        # 标题变化就是天然语义边界，不把两个章节硬拼进同一块。
        for paragraph in paragraphs:
            if current and paragraph.heading_path != current_heading:
                flush_section()
            if not current:
                current_heading = paragraph.heading_path
            current.append(paragraph.content)
        flush_section()

        base_chunks: list[tuple[str, str | None]] = []
        for section, heading_path in sections:
            prefix = f"标题路径：{heading_path}\n" if heading_path else ""
            prefix_tokens = self.count_tokens(prefix) if prefix else 0
            content_budget = max(1, self.chunk_size - prefix_tokens)
            section_tokens = self.count_tokens(section)

            if section_tokens <= content_budget:
                base_chunks.append((f"{prefix}{section}".strip(), heading_path))
                continue

            # 每个窗口都重复标题路径；重叠只发生在同一章节内部。
            # overlap 计入 chunk_size，因此最终块不会越切越大。
            content_overlap = min(self.chunk_overlap, content_budget - 1)
            content_step = max(1, content_budget - content_overlap)
            start = 0
            while start < section_tokens:
                piece = self._token_slice(
                    section,
                    start,
                    start + content_budget,
                ).strip()
                if piece:
                    base_chunks.append((f"{prefix}{piece}".strip(), heading_path))
                if start + content_budget >= section_tokens:
                    break
                start += content_step

        chunks: list[DocumentChunk] = []
        for index, (content, heading_path) in enumerate(base_chunks):
            chunks.append(
                DocumentChunk(
                    content=content,
                    source=source,
                    document_id=document_id,
                    chunk_index=index,
                    token_count=self.count_tokens(content),
                    heading_path=heading_path,
                )
            )
        return chunks
