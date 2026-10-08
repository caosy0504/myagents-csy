"""Chapter 11: data, rewards, SFT, GRPO, and held-out evaluation."""

from .data import MathDatasetBundle, MathExample, load_jsonl_bundle, load_gsm8k_bundle
from .rewards import (
    accuracy_reward, extract_final_answer, format_reward, has_final_answer_marker,
    math_reward,
)
from .trainers import GRPOTrainerWrapper, SFTTrainerWrapper, TrainSettings
from .evaluation import evaluate_model, evaluate_predictions
from .pipeline import run_pipeline, validate_pipeline
from .inference import LocalTransformersLLM
from .agentic import ArithmeticToolEnv, ToolTask, ToolTaskBundle, load_tool_task_bundle
from .agentic_trainers import ToolGRPOTrainerWrapper, ToolSFTTrainerWrapper, evaluate_tool_policy
from .agentic_pipeline import run_agentic_pipeline, validate_agentic_pipeline

__all__ = [
    "MathDatasetBundle", "MathExample", "load_jsonl_bundle",
    "load_gsm8k_bundle", "accuracy_reward", "extract_final_answer",
    "math_reward", "format_reward", "has_final_answer_marker",
    "GRPOTrainerWrapper", "SFTTrainerWrapper", "TrainSettings",
    "evaluate_model", "evaluate_predictions", "run_pipeline", "validate_pipeline",
    "LocalTransformersLLM",
    "ArithmeticToolEnv", "ToolTask", "ToolTaskBundle", "load_tool_task_bundle",
    "ToolSFTTrainerWrapper", "ToolGRPOTrainerWrapper", "evaluate_tool_policy",
    "run_agentic_pipeline", "validate_agentic_pipeline",
]
