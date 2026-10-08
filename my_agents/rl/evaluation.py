"""Held-out exact-answer evaluation before and after training."""

from pathlib import Path
from collections.abc import Mapping
from time import monotonic
from typing import Any

from .data import MathDatasetBundle, SYSTEM_PROMPT
from .rewards import accuracy_reward, extract_final_answer, has_final_answer_marker


def evaluate_predictions(predictions: list[str], ground_truth: list[str]) -> dict[str, Any]:
    scores = accuracy_reward(predictions, ground_truth=ground_truth)
    return {
        "num_samples": len(scores),
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "average_reward": sum(scores) / len(scores) if scores else 0.0,
    }


def evaluate_model(
    model_name_or_path: str,
    bundle: MathDatasetBundle,
    *,
    max_samples: int | None = None,
    max_new_tokens: int = 128,
    device: str = "auto",
    split: str = "test",
) -> dict[str, Any]:
    """Generate on an explicit validation or test split without updates."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if max_new_tokens <= 0 or split not in {"validation", "test"}:
        raise ValueError("Use a positive max_new_tokens and validation/test split")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Evaluation requires: pip install 'my-agents-csy[rl]'") from exc

    path = Path(model_name_or_path)
    adapter = path.is_dir() and (path / "adapter_config.json").is_file()
    base_name = model_name_or_path
    if adapter:
        from peft import PeftConfig, PeftModel

        base_name = PeftConfig.from_pretrained(model_name_or_path).base_model_name_or_path
        base_model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=torch.float32
        )
    tokenizer_source = model_name_or_path if (
        path.is_dir() and (path / "tokenizer_config.json").is_file()
    ) else base_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto/cpu/mps/cuda")
    model.to(device)
    model.eval()
    rows = getattr(bundle, split)[:max_samples]
    predictions = []
    output_lengths = []
    clipped = 0
    started = monotonic()
    with torch.inference_mode():
        for row in rows:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.question},
            ]
            if tokenizer.chat_template:
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                encoded = tokenizer(
                    f"{SYSTEM_PROMPT}\nQuestion: {row.question}\nAnswer:",
                    return_tensors="pt",
                )["input_ids"]
            if isinstance(encoded, Mapping):
                encoded = encoded["input_ids"]
            encoded = encoded.to(device)
            generated = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            output_ids = generated[0][encoded.shape[-1]:]
            output_lengths.append(len(output_ids))
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if len(output_ids) >= max_new_tokens and (
                eos_token_id is None or output_ids[-1] != eos_token_id
            ):
                clipped += 1
            predictions.append(tokenizer.decode(output_ids, skip_special_tokens=True))
    result = evaluate_predictions(
        predictions, [row.ground_truth for row in rows]
    )
    result.update({
        "model": model_name_or_path,
        "split": split,
        "device": device,
        "elapsed_seconds": round(monotonic() - started, 3),
        "mean_output_tokens": sum(output_lengths) / len(rows),
        "clipped_ratio": clipped / len(rows),
        "format_rate": sum(has_final_answer_marker(text) for text in predictions) / len(rows),
        "unparseable": sum(extract_final_answer(text) is None for text in predictions),
        "predictions": [
            {
                "question": row.question,
                "expected": row.ground_truth,
                "predicted": extract_final_answer(text),
                "completion": text[:1000],
            }
            for row, text in zip(rows, predictions)
        ],
    })
    return result
