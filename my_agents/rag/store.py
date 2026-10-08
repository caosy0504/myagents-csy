"""Qdrant-backed vector storage with namespace isolation."""

import uuid
from collections.abc import Sequence

from .models import DocumentChunk, SearchResult


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError(
                "缺少 qdrant-client，请安装：pip install -e '.[rag]'"
            ) from exc

        self.models = models
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key or None)
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def _namespace_filter(self, namespace: str, document_id: str | None = None):
        conditions = [
            self.models.FieldCondition(
                key="namespace",
                match=self.models.MatchValue(value=namespace),
            )
        ]
        if document_id:
            conditions.append(
                self.models.FieldCondition(
                    key="document_id",
                    match=self.models.MatchValue(value=document_id),
                )
            )
        return self.models.Filter(
            must=conditions
        )

    def delete_document(self, namespace: str, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self.models.FilterSelector(
                filter=self._namespace_filter(namespace, document_id)
            ),
            wait=True,
        )

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[Sequence[float]],
        namespace: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{namespace}:{chunk.document_id}:{chunk.chunk_index}",
                )
            )
            points.append(
                self.models.PointStruct(
                    id=point_id,
                    vector=list(vector),
                    payload={
                        "namespace": namespace,
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "heading_path": chunk.heading_path,
                        "content": chunk.content,
                    },
                )
            )
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        query_vector: Sequence[float],
        namespace: str,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(query_vector),
            query_filter=self._namespace_filter(namespace),
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
        )
        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    content=str(payload.get("content", "")),
                    source=str(payload.get("source", "unknown")),
                    score=float(hit.score),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    heading_path=payload.get("heading_path"),
                    document_id=payload.get("document_id"),
                )
            )
        return results

    def clear_namespace(self, namespace: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self.models.FilterSelector(
                filter=self._namespace_filter(namespace)
            ),
            wait=True,
        )

    def count(self, namespace: str) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._namespace_filter(namespace),
            exact=True,
        )
        return int(result.count)
