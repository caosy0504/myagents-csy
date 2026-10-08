"""Use a trained base model or LoRA adapter through the existing LLMClient API."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ..core.message import Message


class LocalTransformersLLM:
    """Local generation adapter for SimpleAgent and text-mode ReActAgent."""

    provider = "transformers"

    def __init__(
        self, model_name_or_path: str, *, device: str = "auto",
        max_input_tokens: int = 2048,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Local inference requires: pip install 'my-agents-csy[rl]'") from exc
        self.torch = torch
        self.model = model_name_or_path
        path = Path(model_name_or_path)
        adapter = path.is_dir() and (path / "adapter_config.json").is_file()
        base_name = model_name_or_path
        if adapter:
            from peft import PeftConfig, PeftModel

            base_name = PeftConfig.from_pretrained(model_name_or_path).base_model_name_or_path
            base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
            self._model = PeftModel.from_pretrained(base, model_name_or_path)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, dtype=torch.float32
            )
        tokenizer_source = model_name_or_path if (
            path.is_dir() and (path / "tokenizer_config.json").is_file()
        ) else base_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("device must be auto/cpu/mps/cuda")
        self.device = device
        self.max_input_tokens = max_input_tokens
        self._model.to(device)
        self._model.eval()

    def _encode(self, messages: Sequence[Message | dict[str, Any]]):
        conversation = []
        for message in messages:
            role = message.role if isinstance(message, Message) else message["role"]
            content = message.content if isinstance(message, Message) else message.get("content")
            if not isinstance(content, str):
                continue
            if role == "tool":
                role, content = "user", "[Tool result]\n" + content
            conversation.append({"role": role, "content": content})
        if self._tokenizer.chat_template:
            encoded = self._tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(encoded, "keys"):
                encoded = encoded["input_ids"]
        else:
            text = "\n".join(
                f"{item['role']}: {item['content']}" for item in conversation
            ) + "\nassistant:"
            encoded = self._tokenizer(text, return_tensors="pt")["input_ids"]
        if encoded.shape[-1] > self.max_input_tokens:
            raise ValueError("Input exceeds the local model token budget")
        return encoded.to(self.device)

    def invoke(self, messages: Sequence[Message | dict[str, Any]], **kwargs: object) -> str:
        tokens = self._encode(messages)
        temperature = float(kwargs.get("temperature", 0.0) or 0.0)
        max_new_tokens = int(kwargs.get("max_tokens") or 128)
        if max_new_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        with self.torch.inference_mode():
            generated = self._model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return self._tokenizer.decode(
            generated[0][tokens.shape[-1]:], skip_special_tokens=True
        )

    def stream(self, messages: Sequence[Message | dict[str, Any]], **kwargs: object) -> Iterator[str]:
        """Compatibility stream; emits the finished response as one chunk."""
        yield self.invoke(messages, **kwargs)
