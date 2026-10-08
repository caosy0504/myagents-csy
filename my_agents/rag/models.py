"""Data contracts for the RAG subsystem."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    content: str
    source: str
    document_id: str
    chunk_index: int
    token_count: int
    heading_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    content: str
    source: str
    score: float
    chunk_index: int
    heading_path: str | None = None
    document_id: str | None = None


@dataclass(slots=True)
class IngestReport:
    document_id: str
    source: str
    chunk_count: int
    character_count: int
    namespace: str
