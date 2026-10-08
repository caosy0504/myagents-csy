"""Document ingestion and semantic retrieval pipeline."""

import hashlib
from pathlib import Path

from .chunker import MarkdownChunker
from .converter import DocumentConverter
from .embedding import LocalEmbeddingModel
from .models import IngestReport, SearchResult
from .store import QdrantVectorStore


class RAGPipeline:
    def __init__(
        self,
        *,
        qdrant_url: str,
        collection_name: str,
        namespace: str,
        qdrant_api_key: str | None = None,
        embedding_model: LocalEmbeddingModel | None = None,
        converter: DocumentConverter | None = None,
        chunker: MarkdownChunker | None = None,
        store=None,
    ) -> None:
        self.namespace = namespace
        self.embedding_model = embedding_model or LocalEmbeddingModel()
        self.converter = converter or DocumentConverter()
        self.chunker = chunker or MarkdownChunker()
        self._store = store
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name

    @property
    def store(self):
        if self._store is None:
            self._store = QdrantVectorStore(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                vector_size=self.embedding_model.dimension,
            )
        return self._store

    @staticmethod
    def _document_id(source: str, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(source.encode("utf-8"))
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()[:24]

    def add_document(self, file_path: str | Path) -> IngestReport:
        path = Path(file_path).expanduser().resolve()
        markdown = self.converter.convert(path)
        return self.add_text(markdown, source=path.name)

    def add_text(self, text: str, source: str = "manual") -> IngestReport:
        normalized = text.strip()
        if not normalized:
            raise ValueError("待入库文本不能为空")
        document_id = self._document_id(source, normalized)
        chunks = self.chunker.split(normalized, source, document_id)
        if not chunks:
            raise ValueError("文档未产生有效分块")
        vectors = self.embedding_model.encode_documents(
            [chunk.content for chunk in chunks]
        )
        if hasattr(self.store, "delete_document"):
            self.store.delete_document(self.namespace, document_id)
        self.store.upsert(chunks, vectors, self.namespace)
        return IngestReport(
            document_id=document_id,
            source=source,
            chunk_count=len(chunks),
            character_count=len(normalized),
            namespace=self.namespace,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        question = query.strip()
        if not question:
            raise ValueError("检索问题不能为空")
        vector = self.embedding_model.encode_query(question)
        return self.store.search(
            vector,
            self.namespace,
            limit,
            score_threshold=score_threshold,
        )

    def stats(self) -> dict[str, object]:
        return {
            "namespace": self.namespace,
            "collection": self.collection_name,
            "chunks": self.store.count(self.namespace),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.dimension,
        }

    def clear(self) -> None:
        self.store.clear_namespace(self.namespace)
