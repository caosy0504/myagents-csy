"""Explicit tool-trajectory SFT and environment GRPO for the arithmetic Agent."""

import json
from pathlib import Path

from .agentic import ArithmeticToolEnv, ToolTaskBundle
from .trainers import (
    TrainSettings, _dependencies, _device_metadata, _lora, _manifest,
    _precision_flags, _prepare_output,
)


def _add_validation(report: dict, output: Path, validation: dict) -> dict:
    report["validation"] = validation
    (output / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def evaluate_tool_policy(
    bundle: ToolTaskBundle,
    settings: TrainSettings,
    model_name_or_path: str,
    *,
    split: str,
) -> dict:
    """Score a policy in the same isolated tool environment on one split."""
    if split not in {"validation", "test"}:
        raise ValueError("Tool policy evaluation uses validation or test")
    settings.validate_grpo()
    torch, _, transformers, _, _, AutoTokenizer = _dependencies()
    from trl import GRPOConfig, GRPOTrainer

    environment = _device_metadata(torch, settings.device)
    precision = _precision_flags(settings, environment)
    transformers.set_seed(settings.seed)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    path = Path(model_name_or_path)
    adapter = path.is_dir() and (path / "adapter_config.json").is_file()
    if adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(settings.model_name, dtype=torch.float32),
            model_name_or_path,
        )
    else:
        model = model_name_or_path
    args = GRPOConfig(
        output_dir=str(Path(settings.output_dir).parent / "policy_evaluation"),
        max_steps=1,
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_generations=settings.num_generations,
        max_completion_length=settings.max_completion_length,
        max_tool_calling_iterations=settings.max_tool_calling_iterations,
        model_init_kwargs=None if adapter else {"dtype": "float32"},
        loss_type=settings.loss_type,
        mask_truncated_completions=settings.mask_truncated_completions,
        use_cpu=environment["device"] == "cpu",
        **precision,
        report_to="none",
        seed=settings.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=bundle.hf_dataset("train", "grpo"),
        eval_dataset=bundle.hf_dataset(split, "grpo"),
        processing_class=tokenizer,
        environment_factory=ArithmeticToolEnv,
    )
    metrics = trainer.evaluate(metric_key_prefix=split)
    return {"model": model_name_or_path, "split": split, **metrics}


class ToolSFTTrainerWrapper:
    def train(self, bundle: ToolTaskBundle, settings: TrainSettings) -> dict:
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
            "task_type": "tool_trajectory_sft",
        })
        output = _prepare_output(settings, None)
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
            assistant_only_loss=True,
            completion_only_loss=False,
            packing=settings.packing,
            model_init_kwargs={"dtype": "float32"},
            use_cpu=environment["device"] == "cpu",
            **precision,
            gradient_checkpointing=settings.gradient_checkpointing,
            optim="adamw_torch",
            logging_steps=1,
            save_steps=max(settings.max_steps, 1),
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
        trainer.train()
        validation = trainer.evaluate(metric_key_prefix="validation")
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        trainer.save_state()
        report = _manifest(output, settings, bundle, "tool_sft", trainer, environment)
        return _add_validation(report, output, validation)


class ToolGRPOTrainerWrapper:
    def train(
        self, bundle: ToolTaskBundle, settings: TrainSettings,
        *, sft_adapter_path: str | None = None,
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
            "task_type": "arithmetic_tool_environment",
        })
        output = _prepare_output(settings, None)
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
                AutoModelForCausalLM.from_pretrained(settings.model_name, dtype=torch.float32),
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
            max_tool_calling_iterations=settings.max_tool_calling_iterations,
            model_init_kwargs=None if sft_adapter_path else {"dtype": "float32"},
            loss_type=settings.loss_type,
            scale_rewards=settings.scale_rewards,
            mask_truncated_completions=settings.mask_truncated_completions,
            log_completions=settings.log_completions,
            beta=0.0,
            use_vllm=False,
            use_cpu=environment["device"] == "cpu",
            **precision,
            gradient_checkpointing=settings.gradient_checkpointing,
            optim="adamw_torch",
            logging_steps=1,
            save_steps=max(settings.max_steps, 1),
            report_to="none",
            seed=settings.seed,
        )
        trainer = GRPOTrainer(
            model=model,
            args=args,
            train_dataset=bundle.hf_dataset("train", "grpo"),
            eval_dataset=bundle.hf_dataset("validation", "grpo"),
            processing_class=tokenizer,
            peft_config=peft_config,
            environment_factory=ArithmeticToolEnv,
        )
        trainer.train()
        validation = trainer.evaluate(metric_key_prefix="validation")
        trainer.save_model(str(output))
        tokenizer.save_pretrained(output)
        trainer.save_state()
        report = _manifest(output, settings, bundle, "tool_grpo", trainer, environment)
        return _add_validation(report, output, validation)
