"""Validated, non-overlapping math data for SFT and GRPO."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rewards import extract_final_answer


SYSTEM_PROMPT = "Solve the math problem. Explain briefly, then end with Final Answer: <number>."


@dataclass(frozen=True)
class MathExample:
    question: str
    answer: str
    source: str = "custom"

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("Math questions and answers must be nonempty")
        if extract_final_answer(self.answer) is None:
            raise ValueError("Answer must contain an explicit numeric final answer")

    @property
    def ground_truth(self) -> str:
        return extract_final_answer(self.answer) or ""


@dataclass(frozen=True)
class MathDatasetBundle:
    train: tuple[MathExample, ...]
    validation: tuple[MathExample, ...]
    test: tuple[MathExample, ...]
    source: str
    seed: int

    def __post_init__(self) -> None:
        if not self.train or not self.validation or not self.test:
            raise ValueError("Train, validation, and test splits must all be nonempty")
        seen: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            for row in getattr(self, split_name):
                key = re.sub(r"\s+", " ", row.question).strip().casefold()
                if key in seen:
                    raise ValueError(
                        f"Duplicate question across {seen[key]} and {split_name}: {row.question}"
                    )
                seen[key] = split_name

    def rows(self, split: str, format_type: str) -> list[dict[str, Any]]:
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        examples = getattr(self, split)
        if format_type == "sft":
            return [
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row.question},
                    ],
                    "completion": [{"role": "assistant", "content": row.answer}],
                }
                for row in examples
            ]
        if format_type == "grpo":
            return [
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row.question},
                    ],
                    "ground_truth": row.ground_truth,
                }
                for row in examples
            ]
        raise ValueError("format_type must be sft or grpo")

    def manifest(self) -> dict[str, Any]:
        def digest(rows: tuple[MathExample, ...]) -> str:
            payload = "\n".join(
                f"{row.question}\t{row.answer}" for row in rows
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        return {
            "source": self.source,
            "seed": self.seed,
            "splits": {
                name: {"count": len(getattr(self, name)), "sha256": digest(getattr(self, name))}
                for name in ("train", "validation", "test")
            },
        }

    def hf_dataset(self, split: str, format_type: str):
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("Training requires: pip install 'my-agents-csy[rl]'") from exc
        return Dataset.from_list(self.rows(split, format_type))


def _read_jsonl(path: str | Path) -> tuple[MathExample, ...]:
    path = Path(path)
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            rows.append(MathExample(
                question=data["question"], answer=data["answer"],
                source=str(data.get("source", path.name)),
            ))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: invalid math example") from exc
    return tuple(rows)


def load_jsonl_bundle(
    train_file: str | Path, validation_file: str | Path, test_file: str | Path,
    *, seed: int = 42,
) -> MathDatasetBundle:
    """Read three explicit splits, rejecting duplicate questions."""
    return MathDatasetBundle(
        train=_read_jsonl(train_file),
        validation=_read_jsonl(validation_file),
        test=_read_jsonl(test_file),
        source="jsonl",
        seed=seed,
    )


def load_gsm8k_bundle(
    *, max_train: int = 100, max_validation: int = 20,
    max_test: int = 40, seed: int = 42,
) -> MathDatasetBundle:
    """Use the official GSM8K test split only for final evaluation."""
    if min(max_train, max_validation, max_test) <= 0:
        raise ValueError("GSM8K sample counts must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("GSM8K loading requires: pip install 'my-agents-csy[rl]'") from exc
    training = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    testing = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    if max_train + max_validation > len(training) or max_test > len(testing):
        raise ValueError("Requested more GSM8K samples than available")

    def convert(rows) -> tuple[MathExample, ...]:
        return tuple(
            MathExample(question=row["question"], answer=row["answer"], source="gsm8k")
            for row in rows
        )

    return MathDatasetBundle(
        train=convert(training.select(range(max_train))),
        validation=convert(training.select(range(max_train, max_train + max_validation))),
        test=convert(testing.select(range(max_test))),
        source="openai/gsm8k:main",
        seed=seed,
    )
