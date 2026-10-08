import pytest

from my_agents.rl.agentic import (
    ArithmeticToolEnv, ToolTask, ToolTaskBundle, load_tool_task_bundle,
)
from my_agents.rl.agentic_pipeline import validate_agentic_pipeline
from my_agents.rl.trainers import TrainSettings


def test_isolated_tool_reward_requires_correct_submission_and_rewards_tool_use():
    env = ArithmeticToolEnv()
    env.reset(3, 4, "multiply")
    assert env.calculate(3, 4, "multiply") == 12
    assert env.submit_answer(12) == "Answer submitted."
    assert env.get_reward() == 1.0
    env.reset(3, 4, "multiply")
    assert env.get_reward() == 0.0
    env.submit_answer(12)
    assert env.get_reward() == 0.8
    env.reset(3, 4, "multiply")
    env.calculate(3, 4, "add")
    env.submit_answer(7)
    assert env.get_reward() == 0.0
    with pytest.raises(ValueError):
        env.calculate(3, 4, "divide")
    assert env._tool_errors == 1


def test_tool_dataset_has_schema_trajectory_and_no_split_leakage():
    bundle = load_tool_task_bundle(
        "examples/tool_rl_data/train.jsonl",
        "examples/tool_rl_data/validation.jsonl",
        "examples/tool_rl_data/test.jsonl",
    )
    assert bundle.manifest()["train"]["count"] == 8
    sft = bundle.rows("train", "sft")[0]
    assert sft["messages"][2]["tool_calls"][0]["function"]["name"] == "calculate"
    assert {item["function"]["name"] for item in sft["tools"]} == {
        "calculate", "submit_answer",
    }
    grpo = bundle.rows("train", "grpo")[0]
    assert grpo["operation"] == "add" and "prompt" in grpo
    with pytest.raises(ValueError, match="Duplicate"):
        ToolTaskBundle((ToolTask(2, 3, "add"),), (ToolTask(2, 3, "add"),),
                       (ToolTask(5, 7, "add"),))


def test_agentic_validation_and_training_settings():
    plan = validate_agentic_pipeline("examples/tool_rl_config.json")
    assert plan["reward_probe"] == 1.0
    assert plan["tool_names"] == ["calculate", "submit_answer"]
    assert plan["dataset"]["test"]["count"] == 2
    assert TrainSettings().loss_type == "dapo"
    assert TrainSettings().mask_truncated_completions
    with pytest.raises(ValueError, match="format_reward_weight"):
        TrainSettings(format_reward_weight=0.5)
