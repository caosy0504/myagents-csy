import json

import pytest

from my_agents.rl import (
    MathExample, accuracy_reward, extract_final_answer, load_jsonl_bundle, math_reward,
)
from my_agents.rl.rewards import format_reward
from my_agents.rl.evaluation import evaluate_predictions
from my_agents.rl.trainers import TrainSettings
from my_agents.rl.pipeline import validate_pipeline
from my_agents.tools import RLTrainingTool


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_reward_uses_explicit_final_answer_and_decimal_equivalence():
    assert extract_final_answer("过程 2 + 3。Final Answer: 5") == "5"
    assert extract_final_answer("#### 1,000") == "1000"
    assert extract_final_answer("过程里有 5 和 7") is None
    scores = accuracy_reward(
        ["Final Answer: 5.0", "过程 5 + 2", [{"role": "assistant", "content": "#### 7"}]],
        ground_truth=["5", "7", "7.0"],
    )
    assert scores == [1.0, 0.0, 1.0]
    assert math_reward(["Final Answer: 5"], ground_truth=["5"]) == [1.1]
    assert accuracy_reward(["Final Answer: 7"], ground_truth=["5"]) == [0.0]
    assert format_reward(["Final Answer: 7"], ground_truth=["5"]) == [0.0]


def test_dataset_splits_and_training_formats(tmp_path):
    files = [tmp_path / f"{name}.jsonl" for name in ("train", "validation", "test")]
    for path, question, number in zip(files, ("2+3?", "3+4?", "4+5?"), (5, 7, 9)):
        _write(path, [{"question": question, "answer": f"{question} #### {number}"}])
    bundle = load_jsonl_bundle(*files)
    assert bundle.rows("train", "sft")[0]["completion"][0]["role"] == "assistant"
    assert bundle.rows("validation", "grpo")[0]["ground_truth"] == "7"
    assert bundle.manifest()["splits"]["test"]["count"] == 1
    assert len(bundle.manifest()["splits"]["train"]["sha256"]) == 64

    _write(files[2], [{"question": "2+3?", "answer": "#### 5"}])
    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl_bundle(*files)


def test_invalid_examples_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        MathExample("What is 2+3?", "Maybe five")
    train = tmp_path / "train.jsonl"
    _write(train, [{"question": "", "answer": "#### 1"}])
    with pytest.raises(ValueError, match="invalid math example"):
        load_jsonl_bundle(train, train, train)


def test_evaluation_and_grpo_batch_validation():
    assert evaluate_predictions(
        ["Final Answer: 2", "not a number"], ["2.0", "3"]
    ) == {"num_samples": 2, "accuracy": 0.5, "average_reward": 0.5}
    TrainSettings(batch_size=1, gradient_accumulation_steps=4, num_generations=4).validate_grpo()
    with pytest.raises(ValueError, match="divisible"):
        TrainSettings(batch_size=1, gradient_accumulation_steps=2, num_generations=4).validate_grpo()


def test_chapter11_dry_run_and_tool_interface():
    config = "examples/rl_config.json"
    plan = validate_pipeline(config)
    assert plan["dataset"]["splits"]["train"]["count"] == 8
    assert plan["reward_probe"] == [1.1]
    response = json.loads(RLTrainingTool().run(json.dumps({
        "action": "load_dataset", "config_path": config,
    })))
    assert response["splits"]["test"]["count"] == 2
    reward = json.loads(RLTrainingTool().run(json.dumps({
        "action": "create_reward", "config_path": config,
    })))
    assert reward["name"] == "math_reward"


def test_evaluation_handles_chat_template_batch_encoding(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.tokenization_utils_base import BatchEncoding
    from my_agents.rl.evaluation import evaluate_model
    from my_agents.rl.data import MathDatasetBundle

    class Tokenizer:
        chat_template = "template"
        pad_token = "<pad>"
        pad_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]])})

        def decode(self, *args, **kwargs):
            return "Final Answer: 6"

    class Model:
        def to(self, device):
            return self

        def eval(self):
            pass

        def generate(self, tokens, **kwargs):
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained",
        lambda *args, **kwargs: Model(),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    bundle = MathDatasetBundle(
        train=(MathExample("1+1?", "#### 2"),),
        validation=(MathExample("2+2?", "#### 4"),),
        test=(MathExample("15-9?", "#### 6"),),
        source="fake", seed=42,
    )
    assert evaluate_model("fake-model", bundle, device="cpu")["accuracy"] == 1.0
    validation = evaluate_model("fake-model", bundle, device="cpu", split="validation")
    assert validation["accuracy"] == 0.0
    assert validation["split"] == "validation"
    assert validation["mean_output_tokens"] == 1
    assert validation["predictions"][0]["completion"] == "Final Answer: 6"


def test_math_pipeline_uses_validation_before_final_test(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from my_agents.rl import pipeline

    source = Path("examples/rl_config.json").resolve()
    config = json.loads(source.read_text())
    config["data"] = {
        key: str(source.parent / value) if key.endswith("_file") else value
        for key, value in config["data"].items()
    }
    config["output_root"] = str(tmp_path / "runs")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    events = []

    def evaluate(model, bundle, *, split, **kwargs):
        events.append(("eval", split))
        return {"accuracy": 0.0, "unparseable": 1}

    def train_sft(self, bundle, settings):
        events.append(("train", "sft"))
        return {"global_step": 1}

    def train_grpo(self, bundle, settings, **kwargs):
        events.append(("train", "grpo"))
        return {"global_step": 1}

    monkeypatch.setattr(pipeline, "evaluate_model", evaluate)
    monkeypatch.setattr(pipeline.SFTTrainerWrapper, "train", train_sft)
    monkeypatch.setattr(pipeline.GRPOTrainerWrapper, "train", train_grpo)
    report = pipeline.run_pipeline(path)
    assert events == [
        ("eval", "validation"), ("train", "sft"),
        ("eval", "validation"), ("train", "grpo"),
        ("eval", "validation"),
        ("eval", "test"), ("eval", "test"), ("eval", "test"),
    ]
    assert report["deployment"]["candidate_model_path"] is None
    assert not report["deployment"]["automatically_activated"]
