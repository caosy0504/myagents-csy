"""Explicit training tool, intentionally not auto-registered on any Agent."""

import json
from pathlib import Path

from ..tools.base import Tool
from .evaluation import evaluate_model
from .pipeline import (
    load_pipeline_config, prepare_bundle, run_pipeline, stage_settings,
    validate_pipeline,
)
from .trainers import GRPOTrainerWrapper, SFTTrainerWrapper
from .agentic_pipeline import run_agentic_pipeline, validate_agentic_pipeline


class RLTrainingTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            "rl_training",
            "Explicit chapter 11 data validation, SFT/GRPO training, and held-out evaluation",
        )

    def run(self, input_text: str) -> str:
        request = json.loads(input_text)
        if not isinstance(request, dict):
            raise ValueError("RLTrainingTool input must be a JSON object")
        action = request.get("action")
        config_path = request.get("config_path")
        if not isinstance(config_path, str):
            raise ValueError("config_path is required")
        if action in {"load_dataset", "create_reward", "validate"}:
            result = validate_pipeline(config_path)
            if action == "load_dataset":
                result = result["dataset"]
            elif action == "create_reward":
                result = {
                    "name": "math_reward",
                    "policy": "exact numeric answer + small format bonus only if correct",
                    "probe": result["reward_probe"],
                }
        elif action == "pipeline":
            result = run_pipeline(config_path)
        elif action == "agentic_validate":
            result = validate_agentic_pipeline(config_path)
        elif action == "agentic_pipeline":
            result = run_agentic_pipeline(config_path)
        elif action in {"train", "evaluate"}:
            config, base_dir = load_pipeline_config(config_path)
            bundle = prepare_bundle(
                config["data"], base_dir, int(config.get("seed", 42))
            )
            model = config["model_name"]
            output = Path(config.get("output_root", "../models/chapter11-demo"))
            output = output if output.is_absolute() else base_dir / output
            if action == "train":
                algorithm = request.get("algorithm")
                if algorithm not in {"sft", "grpo"}:
                    raise ValueError("algorithm must be sft or grpo")
                settings = stage_settings(
                    config, algorithm, model, output, bundle.seed
                )
                if algorithm == "sft":
                    result = SFTTrainerWrapper().train(bundle, settings)
                else:
                    result = GRPOTrainerWrapper().train(
                        bundle, settings,
                        sft_adapter_path=request.get("sft_adapter_path"),
                    )
            else:
                model_path = request.get("model_path")
                if not isinstance(model_path, str):
                    raise ValueError("model_path is required")
                eval_config = config.get("evaluation", {})
                result = evaluate_model(
                    model_path, bundle, split=request.get("split", "test"), **eval_config
                )
        else:
            raise ValueError(
                "action must be load_dataset/create_reward/validate/train/evaluate/"
                "pipeline/agentic_validate/agentic_pipeline"
            )
        return json.dumps(result, ensure_ascii=False, default=str)
