"""Lazy TRL 1.14 wrappers for chapter 11 SFT and GRPO."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .data import MathDatasetBundle
from .rewards import accuracy_reward, format_reward


@dataclass(frozen=True)
class TrainSettings:
    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "models/chapter11"
    max_steps: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    max_length: int = 512
    max_completion_length: int = 128
    num_generations: int = 4
    use_lora: bool = True
    lora_rank: int = 8
    seed: int = 42
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    loss_type: Literal["grpo", "dapo", "dr_grpo"] = "dapo"
    scale_rewards: Literal["group", "batch", "none"] = "group"
    mask_truncated_completions: bool = True
    format_reward_weight: float = 0.1
    log_completions: bool = False
    packing: bool = False
    gradient_checkpointing: bool = False
    mixed_precision: Literal["none", "bf16", "fp16"] = "none"
    max_tool_calling_iterations: int = 3

    def __post_init__(self) -> None:
        if not self.model_name or not self.output_dir:
            raise ValueError("model_name and output_dir are required")
        if min(
            self.max_steps, self.batch_size, self.gradient_accumulation_steps,
            self.max_length, self.max_completion_length, self.num_generations,
            self.lora_rank, self.max_tool_calling_iterations,
        ) <= 0 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Training sizes, steps, and learning rate must be positive")
        if not math.isfinite(self.format_reward_weight) or not 0 <= self.format_reward_weight <= 0.25:
            raise ValueError("format_reward_weight must be between 0 and 0.25")

    def validate_grpo(self) -> None:
        effective = self.batch_size * self.gradient_accumulation_steps
        if effective % self.num_generations:
            raise ValueError(
                "batch_size * gradient_accumulation_steps must be divisible by num_generations"
            )


def _dependencies():
    try:
        import torch
        import trl
        import transformers
        from peft import LoraConfig, TaskType
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Training requires: pip install 'my-agents-csy[rl]'") from exc
    return torch, trl, transformers, LoraConfig, TaskType, AutoTokenizer


def _device_metadata(torch, choice: str) -> dict:
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    # This project's macOS training smoke crashed inside Trainer on MPS.
    # Keep MPS opt-in until this stack is verified on the target host.
    device = ("cuda" if available["cuda"] else "cpu") if choice == "auto" else choice
    if not available[device]:
        raise RuntimeError(f"Requested accelerator {device} is unavailable")
    return {"device": device, "torch_version": torch.__version__}


def _precision_flags(settings: TrainSettings, environment: dict) -> dict:
    if settings.mixed_precision != "none" and environment["device"] == "cpu":
        raise ValueError("Mixed precision is only enabled for MPS or CUDA")
    return {
        "bf16": settings.mixed_precision == "bf16",
        "fp16": settings.mixed_precision == "fp16",
    }


def _prepare_output(settings: TrainSettings, resume_from_checkpoint: str | None) -> Path:
    output = Path(settings.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume_from_checkpoint:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _lora(settings: TrainSettings, LoraConfig, TaskType):
    return LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    ) if settings.use_lora else None


def _manifest(
    output: Path, settings: TrainSettings, bundle: MathDatasetBundle,
    algorithm: str, trainer, versions: dict,
) -> dict:
    report = {
        "algorithm": algorithm,
        "settings": asdict(settings),
        "dataset": bundle.manifest(),
        "environment": versions,
        "global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history[-20:],
        "model_path": str(output),
        "adapter_only": settings.use_lora,
    }
    if "grpo" in algorithm:
        log = next(
            (item for item in reversed(trainer.state.log_history)
             if "frac_reward_zero_std" in item or "completions/clipped_ratio" in item),
            {},
        )
        report["diagnostics"] = {
            "reward_mean": log.get("reward"),
            "reward_std": log.get("reward_std"),
            "frac_reward_zero_std": log.get("frac_reward_zero_std"),
            "clipped_ratio": log.get("completions/clipped_ratio"),
            "tool_call_frequency": log.get("tools/call_frequency"),
            "tool_failure_frequency": log.get("tools/failure_frequency"),
            "issues": [
                issue for condition, issue in (
                    (log.get("frac_reward_zero_std", 0) >= 0.95,
                     "Nearly all prompt groups have identical rewards; inspect completions."),
                    (log.get("completions/clipped_ratio", 0) >= 0.5,
                     "Many completions are truncated; inspect completion length."),
                    ("tools/call_frequency" in log and log["tools/call_frequency"] == 0,
                     "No tool calls were observed."),
                    (log.get("tools/failure_frequency", 0) > 0,
                     "Some tool calls failed."),
                ) if condition
            ],
        }
    (output / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


class SFTTrainerWrapper:
    def train(
        self, bundle: MathDatasetBundle, settings: TrainSettings,
        *, resume_from_checkpoint: str | None = None,
    ) -> dict:
        torch, trl, transformers, LoraConfig, TaskType, AutoTokenizer = _dependencies()
        from trl import SFTConfig, SFTTrainer

        environment = _device_metadata(torch, settings.device)
        precision = _precision_flags(settings, environment)
        environment.update({
            "trl_version": trl.__version__,
            "transformers_version": transformers.__version__,
            "seed": settings.seed,
            "model_dtype": "float32",
            "mixed_precision": settings.mixed_precision,
        })
        output = _prepare_output(settings, resume_from_checkpoint)
        transformers.set_seed(settings.seed)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        args = SFTConfig(
            output_dir=str(output),
            max_steps=settings.max_steps,
            per_device_train_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            learning_rate=settings.learning_rate,
            max_length=settings.max_length,
            completion_only_loss=True,
            packing=settings.packing,
            model_init_kwargs={"dtype": "float32"},
            use_cpu=environment["device"] == "cpu",
            **precision,
            gradient_checkpointing=settings.gradient_checkpointing,
            optim="adamw_torch",
            logging_steps=1,
            save_steps=max(settings.max_steps, 1),
            save_total_limit=2,
            report_to="none",
            seed=settings.seed,
        )
        trainer = SFTTrainer(
            model=settings.model_name,
            args=args,
            train_dataset=bundle.hf_dataset("train", "sft"),
            eval_dataset=bundle.hf_dataset("validation", "sft"),
            processing_class=tokenizer,
            peft_config=_lora(settings, LoraConfig, TaskType),
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        trainer.save_state()
        return _manifest(output, settings, bundle, "sft", trainer, environment)


class GRPOTrainerWrapper:
    def train(
        self, bundle: MathDatasetBundle, settings: TrainSettings,
        *, sft_adapter_path: str | None = None,
        resume_from_checkpoint: str | None = None,
    ) -> dict:
        settings.validate_grpo()
        if sft_adapter_path and not settings.use_lora:
            raise ValueError("Continuing an SFT adapter requires use_lora=True")
        torch, trl, transformers, LoraConfig, TaskType, AutoTokenizer = _dependencies()
        from trl import GRPOConfig, GRPOTrainer

        environment = _device_metadata(torch, settings.device)
        precision = _precision_flags(settings, environment)
        environment.update({
            "trl_version": trl.__version__,
            "transformers_version": transformers.__version__,
            "seed": settings.seed,
            "model_dtype": "float32",
            "mixed_precision": settings.mixed_precision,
        })
        output = _prepare_output(settings, resume_from_checkpoint)
        transformers.set_seed(settings.seed)
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = settings.model_name
        peft_config = _lora(settings, LoraConfig, TaskType)
        if sft_adapter_path:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(
                    settings.model_name, dtype=torch.float32
                ),
                sft_adapter_path,
                is_trainable=True,
            )
            peft_config = None
        args = GRPOConfig(
            output_dir=str(output),
            max_steps=settings.max_steps,
            per_device_train_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            learning_rate=settings.learning_rate,
            num_generations=settings.num_generations,
            max_completion_length=settings.max_completion_length,
            model_init_kwargs=(
                None if sft_adapter_path else {"dtype": "float32"}
            ),
            loss_type=settings.loss_type,
            scale_rewards=settings.scale_rewards,
            mask_truncated_completions=settings.mask_truncated_completions,
            reward_weights=[1.0, settings.format_reward_weight],
            log_completions=settings.log_completions,
            beta=0.0,
            use_vllm=False,
            use_cpu=environment["device"] == "cpu",
            **precision,
            gradient_checkpointing=settings.gradient_checkpointing,
            optim="adamw_torch",
            logging_steps=1,
            save_steps=max(settings.max_steps, 1),
            save_total_limit=2,
            report_to="none",
            seed=settings.seed,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[accuracy_reward, format_reward],
            args=args,
            train_dataset=bundle.hf_dataset("train", "grpo"),
            eval_dataset=bundle.hf_dataset("validation", "grpo"),
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        trainer.save_state()
        return _manifest(output, settings, bundle, "grpo", trainer, environment)
