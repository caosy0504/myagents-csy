"""GAIA-style assistant evaluation with explicit attachment and answer boundaries."""

import json
import os
import re
import string
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Sequence

from .stats import wilson_interval


def _normalize_text(value: str, *, punctuation: bool = True) -> str:
    compact = re.sub(r"\s+", "", value).lower()
    return compact.translate(str.maketrans("", "", string.punctuation)) if punctuation else compact


def _number(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace("$", "").replace("%", "").replace(",", "").strip())
    except InvalidOperation:
        return None


def gaia_match(prediction: str, reference: str) -> bool:
    """Local quasi-exact proxy following GAIA's number/list/text cases."""
    prediction = str(prediction or "")
    reference = str(reference or "")
    if _number(reference) is not None:
        return _number(prediction) == _number(reference)
    if "," in reference or ";" in reference:
        gold = re.split(r"[,;]", reference)
        actual = re.split(r"[,;]", prediction)
        if len(actual) != len(gold):
            return False
        return all(
            _number(p) == _number(g) if _number(g) is not None
            else _normalize_text(p, punctuation=False) == _normalize_text(g, punctuation=False)
            for p, g in zip(actual, gold)
        )
    return _normalize_text(prediction) == _normalize_text(reference)


def extract_final_answer(output: str) -> str:
    match = re.search(r"(?:Final Answer|最终答案|答案)\s*[:：]\s*(.+)", output, re.I)
    return (match.group(1) if match else output).strip()


@dataclass(frozen=True)
class GAIACase:
    task_id: str
    question: str
    level: int
    answer: str | None = None
    attachment: str | None = None
    source_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.question or self.level not in {1, 2, 3}:
            raise ValueError("GAIA case requires task_id, question, and level 1/2/3")


def load_gaia_cases(path: str | Path, *, attachment_root: str | Path | None = None) -> list[GAIACase]:
    """Load local JSONL, including GAIA's original column names."""
    root = (Path(attachment_root).resolve() if attachment_root
            else Path(path).resolve().parent)
    cases = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            task_id = str(item["task_id"])
            attachment = item.get("file_path") or item.get("attachment") or None
            if attachment:
                resolved = (root / attachment).resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError("Attachment path escapes its dataset root")
                attachment = str(resolved)
            case = GAIACase(
                task_id=task_id,
                question=str(item.get("Question", item.get("question", ""))),
                level=int(item.get("Level", item.get("level", 0))),
                answer=item.get("Final answer", item.get("answer")),
                attachment=attachment,
            )
            if task_id in seen:
                raise ValueError("Duplicate GAIA task_id")
            seen.add(task_id)
            cases.append(case)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid GAIA row at {path}:{number}") from exc
    if not cases:
        raise ValueError("GAIA suite is empty")
    return cases


def load_gaia_official_validation(
    *, level: int | None = None, include_attachments: bool = False,
    revision: str | None = None,
) -> list[GAIACase]:
    """Opt-in gated GAIA validation load; never fetch private test answers."""
    if level not in {None, 1, 2, 3}:
        raise ValueError("level must be 1, 2, 3, or None")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN after accepting the GAIA dataset access terms")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    revision = revision or HfApi(token=token).dataset_info("gaia-benchmark/GAIA").sha
    config = f"2023_level{level}" if level else "2023_all"
    rows = load_dataset(
        "gaia-benchmark/GAIA", config, split="validation", token=token,
        revision=revision,
    )
    root = None
    if include_attachments:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            repo_id="gaia-benchmark/GAIA", repo_type="dataset", token=token,
            revision=revision,
        )).resolve()
    cases = []
    for item in rows:
        attachment = item.get("file_path") or None
        if attachment and root:
            resolved = (root / attachment).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("GAIA attachment path escapes dataset root")
            attachment = str(resolved)
        cases.append(GAIACase(
            task_id=item["task_id"], question=item["Question"],
            level=int(item["Level"]), answer=item.get("Final answer"),
            attachment=attachment, source_revision=revision,
        ))
    return cases


def run_gaia(
    agent_factory: Callable[[], Any], cases: Sequence[GAIACase], *,
    max_samples: int | None = None,
    attachment_handler: Callable[[str, str], str] | None = None,
) -> dict:
    """Fresh Agent per task; unsupported attachments are counted, not mis-scored."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    selected = list(cases[:max_samples])
    if not selected:
        raise ValueError("No GAIA cases selected")
    results = []
    eligible_count = sum(case.answer is not None and str(case.answer).strip() != ""
                         for case in selected)
    for case in selected:
        prompt = case.question
        if case.attachment:
            if attachment_handler is None or not Path(case.attachment).is_file():
                results.append({"task_id": case.task_id, "level": case.level,
                                "status": "unsupported_attachment", "correct": None})
                continue
            prompt = attachment_handler(case.attachment, prompt)
        started = monotonic()
        answer = ""
        error = None
        agent = None
        try:
            agent = agent_factory()
            answer = extract_final_answer(agent.run(prompt))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
        scored = case.answer is not None and str(case.answer).strip() != ""
        results.append({
            "task_id": case.task_id,
            "level": case.level,
            "status": "error" if error else "scored" if scored else "answer_private",
            "model_answer": answer,
            "correct": (gaia_match(answer, case.answer) if not error else False)
                       if scored else None,
            "error": error,
            "latency_seconds": round(monotonic() - started, 4),
            "tool_attempts": getattr(agent, "last_tool_attempts", 0),
            "tool_errors": getattr(agent, "last_tool_errors", 0),
            "input_tokens_estimate": getattr(agent, "last_input_tokens", None),
            "provider_prompt_tokens": getattr(agent, "last_reported_prompt_tokens", None),
            "provider_completion_tokens": getattr(
                agent, "last_reported_completion_tokens", None
            ),
        })
    scored_rows = [row for row in results if row.get("correct") is not None]
    levels = {}
    for level in (1, 2, 3):
        group = [row for row in scored_rows if row["level"] == level]
        if group:
            levels[str(level)] = {"count": len(group),
                                  "accuracy": sum(row["correct"] for row in group) / len(group),
                                  "accuracy_ci95": wilson_interval(
                                      sum(row["correct"] for row in group), len(group)
                                  )}
    attempts = sum(row.get("tool_attempts", 0) for row in results)
    attempted = [row for row in results if row["status"] != "unsupported_attachment"]
    prompt_usage = [row["provider_prompt_tokens"] for row in attempted
                    if row.get("provider_prompt_tokens")]
    completion_usage = [row["provider_completion_tokens"] for row in attempted
                        if row.get("provider_completion_tokens")]
    revisions = {case.source_revision for case in selected if case.source_revision}
    return {
        "benchmark": "gaia_validation_local_proxy",
        "official_score": False,
        "dataset_revision": next(iter(revisions)) if len(revisions) == 1 else None,
        "sample_count": len(results),
        "scored_count": len(scored_rows),
        "eligible_count": eligible_count,
        "coverage": len(scored_rows) / eligible_count if eligible_count else None,
        "execution_error_count": sum(row["status"] == "error" for row in results),
        "unsupported_attachment_count": sum(
            row["status"] == "unsupported_attachment" for row in results
        ),
        "accuracy": (sum(row["correct"] for row in scored_rows) / len(scored_rows)
                     if scored_rows else None),
        "accuracy_ci95": wilson_interval(
            sum(row["correct"] for row in scored_rows), len(scored_rows)
        ),
        "levels": levels,
        "tool_error_rate": sum(row.get("tool_errors", 0) for row in results) / attempts
                           if attempts else None,
        "mean_latency_seconds": (
            sum(row["latency_seconds"] for row in attempted) / len(attempted)
            if attempted else None
        ),
        "reported_prompt_tokens": sum(prompt_usage) if prompt_usage else None,
        "reported_completion_tokens": sum(completion_usage) if completion_usage else None,
        "samples": results,
    }


def export_gaia_submission(report: dict, output_path: str | Path) -> Path:
    """Write a local submission-shaped file; no leaderboard upload occurs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in report["samples"]:
            handle.write(json.dumps({
                "task_id": row["task_id"],
                "model_answer": row.get("model_answer", ""),
                "reasoning_trace": "",
            }, ensure_ascii=False) + "\n")
    return path
