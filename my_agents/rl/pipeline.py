"""Data → validation baselines → SFT → GRPO → final held-out test."""

import importlib.metadata
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .data import MathDatasetBundle, load_gsm8k_bundle, load_jsonl_bundle
from .evaluation import evaluate_model
from .rewards import math_reward
from .trainers import GRPOTrainerWrapper, SFTTrainerWrapper, TrainSettings


def load_pipeline_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("Pipeline config needs a data object")
    return config, path.parent


def prepare_bundle(data: dict, base_dir: Path, seed: int) -> MathDatasetBundle:
    source = data.get("source", "jsonl")
    if source == "jsonl":
        paths = []
        for field in ("train_file", "validation_file", "test_file"):
            value = data.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"data.{field} is required")
            path = Path(value).expanduser()
            paths.append(path if path.is_absolute() else base_dir / path)
        return load_jsonl_bundle(*paths, seed=seed)
    if source == "gsm8k":
        return load_gsm8k_bundle(
            max_train=int(data.get("max_train", 100)),
            max_validation=int(data.get("max_validation", 20)),
            max_test=int(data.get("max_test", 40)),
            seed=seed,
        )
    raise ValueError("data.source must be jsonl or gsm8k")


def stage_settings(
    config: dict, stage: str, model_name: str, output_root: Path, seed: int
) -> TrainSettings:
    options = config.get(stage, {})
    if not isinstance(options, dict):
        raise ValueError(f"{stage} settings must be an object")
    reserved = {"model_name", "output_dir", "seed"}
    allowed = {field.name for field in fields(TrainSettings)} - reserved
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown {stage} settings: {', '.join(sorted(unknown))}")
    return TrainSettings(
        model_name=model_name,
        output_dir=str(output_root / stage),
        seed=seed,
        **options,
    )


def validate_pipeline(config_path: str | Path) -> dict:
    """Check data, reward shape, stages, and optional-package availability."""
    config, base_dir = load_pipeline_config(config_path)
    seed = int(config.get("seed", 42))
    bundle = prepare_bundle(config["data"], base_dir, seed)
    model_name = config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name is required")
    output = Path(config.get("output_root", "../models/chapter11-demo"))
    output = output if output.is_absolute() else base_dir / output
    output = output.resolve()
    sft = stage_settings(config, "sft", model_name, output, seed)
    grpo = stage_settings(config, "grpo", model_name, output, seed)
    grpo.validate_grpo()
    if not sft.use_lora or not grpo.use_lora:
        raise ValueError("The end-to-end pipeline requires LoRA in both SFT and GRPO stages")
    reward_probe = math_reward(
        ["Final Answer: " + bundle.train[0].ground_truth],
        ground_truth=[bundle.train[0].ground_truth],
    )
    versions = {}
    for package in ("trl", "transformers", "torch", "datasets", "peft", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "model_name": model_name,
        "output_root": str(output.resolve()),
        "dataset": bundle.manifest(),
        "sft_settings": asdict(sft),
        "grpo_settings": asdict(grpo),
        "reward_probe": reward_probe,
        "package_versions": versions,
        "ready_to_train": all(value is not None for value in versions.values()),
        "note": "Validation does not download a model or prove that training will improve accuracy.",
    }


def run_pipeline(config_path: str | Path) -> dict:
    """Run all chapter 11 stages; never change the active Agent automatically."""
    plan = validate_pipeline(config_path)
    if not plan["ready_to_train"]:
        raise RuntimeError("Install optional RL dependencies: pip install 'my-agents-csy[rl]'")
    config, base_dir = load_pipeline_config(config_path)
    bundle = prepare_bundle(config["data"], base_dir, int(config.get("seed", 42)))
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    model_name = plan["model_name"]
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation settings must be an object")
    allowed_eval = {"max_samples", "max_new_tokens", "device"}
    if set(evaluation) - allowed_eval:
        raise ValueError("Unknown evaluation settings")

    report: dict[str, Any] = {"plan": plan, "stages": {}}

    def save_progress() -> None:
        (output / "pipeline_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    report["stages"]["baseline_validation"] = evaluate_model(
        model_name, bundle, split="validation", **evaluation
    )
    save_progress()
    sft = stage_settings(config, "sft", model_name, output, bundle.seed)
    report["stages"]["sft_train"] = SFTTrainerWrapper().train(bundle, sft)
    save_progress()
    report["stages"]["sft_validation"] = evaluate_model(
        sft.output_dir, bundle, split="validation", **evaluation
    )
    save_progress()
    grpo = stage_settings(config, "grpo", model_name, output, bundle.seed)
    report["stages"]["grpo_train"] = GRPOTrainerWrapper().train(
        bundle, grpo, sft_adapter_path=sft.output_dir if sft.use_lora else None
    )
    save_progress()
    report["stages"]["grpo_validation"] = evaluate_model(
        grpo.output_dir, bundle, split="validation", **evaluation
    )
    save_progress()
    candidates = {
        "base": model_name,
        "sft": sft.output_dir,
        "grpo": grpo.output_dir,
    }
    report["final_test"] = {
        name: evaluate_model(path, bundle, split="test", **evaluation)
        for name, path in candidates.items()
    }
    baseline = report["stages"]["baseline_validation"]["accuracy"]
    winner = max(
        candidates,
        key=lambda name: (
            report["stages"][f"{name if name != 'base' else 'baseline'}_validation"]["accuracy"],
            -report["stages"][f"{name if name != 'base' else 'baseline'}_validation"]["unparseable"],
        ),
    )
    improved = (
        report["stages"][f"{winner if winner != 'base' else 'baseline'}_validation"][
            "accuracy"
        ] > baseline
    )
    report["deployment"] = {
        "base_model": model_name,
        "candidate_model_path": candidates[winner] if improved else None,
        "validation_winner": winner,
        "improved_over_base": improved,
        "automatically_activated": False,
        "instruction": "Review validation and test results before manually loading "
                       "candidate_model_path with LocalTransformersLLM.",
    }
    save_progress()
    return report
