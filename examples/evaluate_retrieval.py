"""Compare embedding models on a tiny, reproducible retrieval set.

This is a learning example, not a production benchmark. Add questions from your
real documents before using its scores to choose a model or score threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from my_agents import LocalEmbeddingModel, MarkdownChunker


CASES = [
    ("消息的数据格式是由哪个接口规定的？", "Message"),
    ("温度和历史长度这些公共参数在哪里管理？", "Config"),
    ("不同智能体使用哪个统一执行入口？", "Agent"),
    ("文档问答的资料会保存到哪里？", "RAG 流程"),
]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def evaluate(model_name: str, markdown: str) -> tuple[float, float]:
    chunks = MarkdownChunker(chunk_size=120, chunk_overlap=20).split(
        markdown,
        source="sample_knowledge.md",
        document_id="evaluation",
    )
    embedder = LocalEmbeddingModel(model_name=model_name)
    document_vectors = embedder.encode_documents([chunk.content for chunk in chunks])

    reciprocal_ranks: list[float] = []
    top_one_hits = 0
    for query, expected_heading in CASES:
        query_vector = embedder.encode_query(query)
        ranking = sorted(
            range(len(chunks)),
            key=lambda index: dot(query_vector, document_vectors[index]),
            reverse=True,
        )
        expected_rank = next(
            (
                rank
                for rank, index in enumerate(ranking, 1)
                if expected_heading in (chunks[index].heading_path or "")
            ),
            0,
        )
        reciprocal_ranks.append(1 / expected_rank if expected_rank else 0.0)
        top_one_hits += int(expected_rank == 1)

    return top_one_hits / len(CASES), sum(reciprocal_ranks) / len(CASES)


def main() -> None:
    parser = argparse.ArgumentParser(description="对比本地 Embedding 检索效果")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-zh-v1.5",
        ],
    )
    args = parser.parse_args()

    path = Path(__file__).with_name("sample_knowledge.md")
    markdown = path.read_text(encoding="utf-8")
    print("model\tTop-1 accuracy\tMRR")
    for model_name in args.models:
        top_one, mrr = evaluate(model_name, markdown)
        print(f"{model_name}\t{top_one:.2%}\t{mrr:.3f}")


if __name__ == "__main__":
    main()
