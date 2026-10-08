"""Small, auditable rewards for exact-answer math training."""

import re
from decimal import Decimal, InvalidOperation
from typing import Any


_NUMBER = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_MARKER = re.compile(
    rf"(?:####|Final Answer\s*[:：]|最终答案\s*[:：]|答案\s*[:：])\s*({_NUMBER})",
    re.IGNORECASE,
)
_WHOLE_NUMBER = re.compile(rf"^\s*({_NUMBER})\s*[.!。]??\s*$")


def extract_final_answer(text: str) -> str | None:
    """Read an explicit final answer; never guess from intermediate numbers."""
    if not isinstance(text, str):
        return None
    markers = _MARKER.findall(text)
    if markers:
        return markers[-1].replace(",", "")
    match = _WHOLE_NUMBER.fullmatch(text)
    return match.group(1).replace(",", "") if match else None


def has_final_answer_marker(text: str) -> bool:
    """Whether the response explicitly marks its final numeric answer."""
    return isinstance(text, str) and _MARKER.search(text) is not None


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(
            part.get("content", "") for part in completion
            if isinstance(part, dict) and isinstance(part.get("content"), str)
        )
    return ""


def _as_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def accuracy_reward(
    completions: list[Any], ground_truth: list[str] | None = None, **_: Any
) -> list[float]:
    """TRL-compatible reward with one score per generated completion."""
    if ground_truth is None or len(completions) != len(ground_truth):
        raise ValueError("ground_truth must match the completions length")
    scores = []
    for completion, expected in zip(completions, ground_truth):
        actual = _as_decimal(extract_final_answer(_completion_text(completion)))
        target = _as_decimal(extract_final_answer(expected))
        scores.append(float(actual is not None and target is not None and actual == target))
    return scores


def math_reward(
    completions: list[Any], ground_truth: list[str] | None = None, **kwargs: Any
) -> list[float]:
    """Accuracy dominates; explicit final-answer format gets a small bonus."""
    accuracy = accuracy_reward(completions, ground_truth=ground_truth, **kwargs)
    formatting = format_reward(completions, ground_truth=ground_truth, **kwargs)
    return [score + 0.1 * fmt for score, fmt in zip(accuracy, formatting)]


def format_reward(
    completions: list[Any], ground_truth: list[str] | None = None, **_: Any
) -> list[float]:
    """A format bonus only for correct answers, avoiding formatted-wrong rewards."""
    correct = accuracy_reward(completions, ground_truth=ground_truth)
    return [
        score * float(has_final_answer_marker(_completion_text(item)))
        for score, item in zip(correct, completions)
    ]
