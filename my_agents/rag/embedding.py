"""Lazy local embedding provider."""

from collections.abc import Sequence


class LocalEmbeddingModel:
    BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str | None = None,
        query_instruction: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        if query_instruction is None and "bge" in model_name.lower() and "zh" in model_name.lower():
            query_instruction = self.BGE_ZH_QUERY_INSTRUCTION
        self.query_instruction = query_instruction
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 sentence-transformers，请安装：pip install -e '.[rag]'"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError("无法确定 Embedding 维度")
        return int(dimension)

    def _encode(
        self,
        texts: str | Sequence[str],
        *,
        is_query: bool,
    ) -> list[float] | list[list[float]]:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if is_query and self.query_instruction:
            values = [f"{self.query_instruction}{value}" for value in values]

        # BGE 中文模型通过显式 query instruction 区分问题与文档；其他
        # SentenceTransformer 模型优先使用新版的非对称检索接口。
        encoder = self.model.encode
        if not self.query_instruction:
            method_name = "encode_query" if is_query else "encode_document"
            encoder = getattr(self.model, method_name, encoder)

        vectors = encoder(
            values,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        converted = [vector.tolist() for vector in vectors]
        return converted[0] if single else converted

    def encode_documents(
        self,
        texts: str | Sequence[str],
    ) -> list[float] | list[list[float]]:
        return self._encode(texts, is_query=False)

    def encode_query(
        self,
        texts: str | Sequence[str],
    ) -> list[float] | list[list[float]]:
        return self._encode(texts, is_query=True)

    def encode(self, texts: str | Sequence[str]) -> list[float] | list[list[float]]:
        """Backward-compatible alias for document encoding."""
        return self.encode_documents(texts)
