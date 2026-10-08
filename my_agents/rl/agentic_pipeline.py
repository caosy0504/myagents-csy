"""Explicit tool-trajectory SFT → environment GRPO experiment."""

import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

from .agentic import ArithmeticToolEnv, load_tool_task_bundle, tool_schemas
from .agentic_trainers import (
    ToolGRPOTrainerWrapper, ToolSFTTrainerWrapper, evaluate_tool_policy,
)
from .pipeline import load_pipeline_config, stage_settings


def _prepare(config: dict, base_dir: Path):
    data = config["data"]
    files = []
    for key in ("train_file", "validation_file", "test_file"):
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"data.{key} is required")
        path = Path(value).expanduser()
        files.append(path if path.is_absolute() else base_dir / path)
    return load_tool_task_bundle(*files)


def validate_agentic_pipeline(config_path: str | Path) -> dict:
    config, base_dir = load_pipeline_config(config_path)
    bundle = _prepare(config, base_dir)
    model_name = config.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name is required")
    seed = int(config.get("seed", 42))
    output = Path(config.get("output_root", "../models/chapter11-tool-demo"))
    output = (output if output.is_absolute() else base_dir / output).resolve()
    sft = stage_settings(config, "sft", model_name, output, seed)
    grpo = stage_settings(config, "grpo", model_name, output, seed)
    grpo.validate_grpo()
    if not sft.use_lora or not grpo.use_lora:
        raise ValueError("The tool pipeline requires LoRA in both SFT and GRPO stages")
    schemas = tool_schemas()
    first = bundle.train[0]
    env = ArithmeticToolEnv()
    env.reset(first.a, first.b, first.operation)
    env.calculate(first.a, first.b, first.operation)
    env.submit_answer(first.answer)
    versions = {}
    for package in ("trl", "transformers", "torch", "datasets", "peft", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "model_name": model_name,
        "output_root": str(output),
        "dataset": bundle.manifest(),
        "sft_settings": asdict(sft),
        "grpo_settings": asdict(grpo),
        "tool_names": [schema["function"]["name"] for schema in schemas],
        "reward_probe": env.get_reward(),
        "package_versions": versions,
        "ready_to_train": all(value is not None for value in versions.values()),
        "note": "Validation does not download a model or prove tool-call quality.",
    }


def run_agentic_pipeline(config_path: str | Path) -> dict:
    plan = validate_agentic_pipeline(config_path)
    if not plan["ready_to_train"]:
        raise RuntimeError("Install optional RL dependencies: pip install 'my-agents-csy[rl]'")
    config, base_dir = load_pipeline_config(config_path)
    bundle = _prepare(config, base_dir)
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    model_name = plan["model_name"]
    seed = int(config.get("seed", 42))
    sft = stage_settings(config, "sft", model_name, output, seed)
    grpo = stage_settings(config, "grpo", model_name, output, seed)
    report = {"plan": plan, "stages": {}, "final_test": {}}

    def save() -> None:
        (output / "pipeline_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    report["stages"]["baseline_validation"] = evaluate_tool_policy(
        bundle, grpo, model_name, split="validation"
    )
    save()
    report["stages"]["sft_train"] = ToolSFTTrainerWrapper().train(bundle, sft)
    report["stages"]["sft_validation"] = evaluate_tool_policy(
        bundle, grpo, sft.output_dir, split="validation"
    )
    save()
    report["stages"]["grpo_train"] = ToolGRPOTrainerWrapper().train(
        bundle, grpo, sft_adapter_path=sft.output_dir if sft.use_lora else None
    )
    report["stages"]["grpo_validation"] = evaluate_tool_policy(
        bundle, grpo, grpo.output_dir, split="validation"
    )
    save()
    policies = {"base": model_name, "sft": sft.output_dir, "grpo": grpo.output_dir}
    for name, path in policies.items():
        report["final_test"][name] = evaluate_tool_policy(bundle, grpo, path, split="test")
        save()
    scores = {
        name: report["stages"][f"{name if name != 'base' else 'baseline'}_validation"].get(
            "eval_reward", 0.0
        )
        for name in policies
    }
    winner = max(scores, key=scores.get)
    report["deployment"] = {
        "validation_winner": winner,
        "validation_rewards": scores,
        "candidate_model_path": policies[winner] if scores[winner] > scores["base"] else None,
        "automatically_activated": False,
    }
    save()
    return report
