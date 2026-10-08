import pytest

from my_agents import (
    DocumentConverter,
    DocumentQAAgent,
    LocalEmbeddingModel,
    MarkdownChunker,
    RAGPipeline,
    RAGTool,
)
from my_agents.rag.models import SearchResult


class FakeEmbedding:
    model_name = "fake-embedding"
    dimension = 3

    def encode_documents(self, texts):
        if isinstance(texts, str):
            return [1.0, 0.0, 0.0]
        return [[1.0, 0.0, 0.0] for _ in texts]

    def encode_query(self, texts):
        if isinstance(texts, str):
            return [1.0, 0.0, 0.0]
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    def __init__(self):
        self.chunks = []

    def upsert(self, chunks, vectors, namespace):
        self.chunks = list(chunks)
        self.namespace = namespace

    def delete_document(self, namespace, document_id):
        self.chunks = [
            chunk for chunk in self.chunks if chunk.document_id != document_id
        ]

    def search(self, query_vector, namespace, limit, score_threshold=None):
        return [
            SearchResult(
                content="MyAgents 通过统一 Agent.run() 接口执行任务。",
                source="guide.md",
                score=0.92,
                chunk_index=0,
                heading_path="核心接口 > Agent",
                document_id="doc-1",
            )
        ][:limit]

    def count(self, namespace):
        return len(self.chunks)

    def clear_namespace(self, namespace):
        self.chunks = []


class FakeLLM:
    provider = "fake"

    def invoke(self, messages, **kwargs):
        return "所有 Agent 使用统一的 run() 接口。[资料 1]"

    def stream(self, messages, **kwargs):
        yield "unused"


class FakeVector(list):
    def tolist(self):
        return list(self)


class RecordingSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, values, **kwargs):
        self.calls.append(list(values))
        return [FakeVector([1.0, 0.0, 0.0]) for _ in values]


def make_pipeline():
    return RAGPipeline(
        qdrant_url="http://unused",
        collection_name="test",
        namespace="tests",
        embedding_model=FakeEmbedding(),
        chunker=MarkdownChunker(chunk_size=60, chunk_overlap=10),
        store=FakeStore(),
    )


def test_markdown_chunker_preserves_heading_path_and_code_block():
    markdown = """# 第一章

这是第一段内容，用于测试标题路径。

## 代码示例

```python
def hello():
    return "hello"
```
"""
    chunks = MarkdownChunker(chunk_size=80, chunk_overlap=10).split(
        markdown, "demo.md", "doc-1"
    )
    assert chunks
    assert any(chunk.heading_path == "第一章 > 代码示例" for chunk in chunks)
    assert any("def hello" in chunk.content for chunk in chunks)


def test_markdown_chunker_counts_overlap_inside_size_limit():
    chunker = MarkdownChunker(chunk_size=40, chunk_overlap=8)
    chunks = chunker.split(
        "# 长章节\n\n" + "这是用于验证分块边界的内容。" * 80,
        "long.md",
        "doc-long",
    )
    assert len(chunks) > 1
    assert all(chunk.token_count <= 40 for chunk in chunks)


def test_document_converter_rejects_oversized_file(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * 1025)
    converter = DocumentConverter(max_file_size_mb=1)
    converter.max_file_size_bytes = 1024
    with pytest.raises(ValueError, match="大小限制"):
        converter.convert(path)


def test_bge_adds_instruction_only_to_queries():
    embedder = LocalEmbeddingModel("BAAI/bge-small-zh-v1.5")
    recorder = RecordingSentenceTransformer()
    embedder._model = recorder

    embedder.encode_documents("这是一段资料")
    embedder.encode_query("资料讲了什么？")

    assert recorder.calls[0] == ["这是一段资料"]
    assert recorder.calls[1][0].startswith(embedder.BGE_ZH_QUERY_INSTRUCTION)


def test_pipeline_ingests_and_searches_text():
    pipeline = make_pipeline()
    report = pipeline.add_text(
        "# MyAgents\n\n统一接口让不同 Agent 能被同一调度器调用。",
        source="guide.md",
    )
    assert report.chunk_count >= 1
    assert pipeline.stats()["chunks"] == report.chunk_count
    assert pipeline.search("Agent 如何调用")[0].source == "guide.md"


def test_rag_tool_returns_grounded_answer_and_sources():
    tool = RAGTool(FakeLLM(), namespace="tests", pipeline=make_pipeline())
    answer = tool.ask("Agent 使用什么接口？")
    assert "run()" in answer
    assert "参考来源" in answer
    assert "guide.md" in answer


def test_document_qa_agent_uses_rag_and_saves_history():
    tool = RAGTool(FakeLLM(), namespace="tests", pipeline=make_pipeline())
    agent = DocumentQAAgent("qa", FakeLLM(), rag_tool=tool)
    answer = agent.run("Agent 使用什么接口？")
    assert "资料 1" in answer
    assert len(agent.get_history()) == 2
    assert agent.last_sources == ["guide.md"]
