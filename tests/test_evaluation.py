import json
from pathlib import Path

import pytest

from my_agents.core import ToolCall, ToolCompletion
from my_agents.evaluation.bfcl import (
    load_bfcl_cases, load_bfcl_v4_subset, run_bfcl, score_calls,
)
from my_agents.evaluation.gaia import GAIACase, gaia_match, load_gaia_cases, run_gaia
from my_agents.evaluation.generation import (
    GeneratedProblem, compare_problems, create_review_template,
    generate_problems, judge_problems, review_interactively, summarize_reviews,
    verify_problem, calibrate_judge,
)
from my_agents.evaluation.report import save_report
from my_agents.evaluation.trajectory import load_trajectory_cases, run_trajectories
from my_agents.evaluation.dynamic import load_dynamic_cases, run_dynamic
from my_agents.evaluation.baseline import run_live_baseline
from my_agents.evaluation.stats import paired_bootstrap_delta, wilson_interval
from my_agents.evaluation.tools import BFCLEvaluationTool


FIXTURES = Path(__file__).resolve().parents[1] / "examples" / "evaluation"


def test_bfcl_selection_arguments_parallel_and_abstention():
    expected = [{"weather": {"city": ["Shanghai"]}},
                {"weather": {"city": ["Beijing"]}}]
    reversed_calls = [{"name": "weather", "arguments": {"city": "Beijing"}},
                      {"name": "weather", "arguments": {"city": "Shanghai"}}]
    assert score_calls(reversed_calls, expected) == {
        "selection_correct": True, "arguments_correct": True,
    }
    assert score_calls([{"name": "weather", "arguments": {"city": "Paris"}}],
                       [{"weather": {"city": ["Shanghai"]}}]) == {
        "selection_correct": True, "arguments_correct": False,
    }
    assert score_calls([], []) == {"selection_correct": True, "arguments_correct": True}
    assert not score_calls([{"name": "weather", "arguments": {}}], [])["selection_correct"]


def test_bfcl_runner_counts_errors_and_does_not_execute_tools():
    cases = load_bfcl_cases(FIXTURES / "bfcl_cases.jsonl")

    class FakeLLM:
        def complete_with_tools(self, messages, tools, **kwargs):
            question = messages[-1]["content"]
            if "Shanghai and Beijing" in question:
                calls = [ToolCall("1", "weather", '{"city":"Beijing"}'),
                         ToolCall("2", "weather", '{"city":"Shanghai"}')]
            elif "Shanghai" in question:
                calls = [ToolCall("1", "weather", '{"city":"Paris"}')]
            elif "6 plus 7" in question:
                calls = [ToolCall("1", "add", '{"a":6,"b":7}')]
            else:
                calls = []
            return ToolCompletion("", calls, prompt_tokens=10, completion_tokens=2)

    report = run_bfcl(FakeLLM(), cases)
    assert report["sample_count"] == 4
    assert report["selection_accuracy"] == 1.0
    assert report["argument_accuracy"] == 0.75
    assert report["reported_prompt_tokens"] == 40
    assert not report["official_score"]
    wrapped = json.loads(BFCLEvaluationTool(FakeLLM()).run(json.dumps({
        "cases_file": str(FIXTURES / "bfcl_cases.jsonl"),
    })))
    assert wrapped["argument_accuracy"] == 0.75


def test_bfcl_v4_subset_accepts_alternative_arguments(tmp_path):
    questions = tmp_path / "questions.jsonl"
    answers = tmp_path / "answers.jsonl"
    questions.write_text(json.dumps({
        "id": "simple_python_0",
        "question": [[{"role": "user", "content": "Area?"}]],
        "function": [{"name": "area", "parameters": {
            "type": "dict", "properties": {"unit": {"type": "string"}},
        }}],
    }) + "\n")
    answers.write_text(json.dumps({
        "id": "simple_python_0", "ground_truth": [{"area": {"unit": ["units", ""]}}],
    }) + "\n")
    cases = load_bfcl_v4_subset(questions, answers, category="simple_python")
    assert cases[0].tools[0]["function"]["parameters"]["type"] == "object"
    assert score_calls([{"name": "area", "arguments": {}}], cases[0].expected)[
        "arguments_correct"
    ]


def test_gaia_quasi_exact_and_attachment_coverage(tmp_path):
    assert gaia_match("1,000", "1000")
    assert gaia_match("$13", "13")
    assert gaia_match("2;3", "2, 3")
    assert gaia_match("New York!", "newyork")
    assert not gaia_match("2, 4", "2, 3")
    created = []

    class FakeAgent:
        def run(self, prompt):
            return "Final Answer: 13"

    def factory():
        created.append(1)
        return FakeAgent()

    cases = [GAIACase("ok", "What is 6+7?", 1, "13"),
             GAIACase("file", "Read the file", 2, "x", str(tmp_path / "missing.pdf"))]
    report = run_gaia(factory, cases)
    assert report["scored_count"] == 1
    assert report["eligible_count"] == 2
    assert report["coverage"] == 0.5
    assert report["unsupported_attachment_count"] == 1
    assert report["accuracy"] == 1.0
    assert len(created) == 1
    unsafe = tmp_path / "unsafe.jsonl"
    unsafe.write_text(json.dumps({"task_id": "unsafe", "Question": "Read", "Level": 1,
                                  "Final answer": "x", "file_path": "../secret.txt"}) + "\n")
    with pytest.raises(ValueError, match="Invalid GAIA row"):
        load_gaia_cases(unsafe)


def test_gaia_agent_failure_is_counted_as_wrong():
    class BrokenAgent:
        def run(self, prompt):
            raise RuntimeError("model unavailable")

    report = run_gaia(BrokenAgent, [GAIACase("failed", "Question", 1, "answer")])
    assert report["scored_count"] == 1
    assert report["coverage"] == 1.0
    assert report["execution_error_count"] == 1
    assert report["accuracy"] == 0.0


def test_generation_judge_pairwise_and_human_review(tmp_path):
    reference = [GeneratedProblem("ref", "What is 2+2?", "4", "2+2=4", "reference")]
    generated = [GeneratedProblem("gen", "What is 3+3?", "6", "3+3=6")]

    class FakeJudge:
        calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            if "比较两道" in messages[-1]["content"]:
                return json.dumps({"winner": "A" if self.calls % 2 == 0 else "B"})
            return json.dumps({"correctness": 5, "clarity": 4,
                               "difficulty_match": 3, "completeness": 4})

    judge = FakeJudge()
    result = judge_problems(judge, generated)
    assert result["mean_score"] == 4.0
    pair = compare_problems(judge, generated, reference)
    assert pair["valid_count"] == 1
    assert pair["wins"] + pair["losses"] + pair["ties"] == 1
    review = create_review_template(generated, tmp_path / "review.jsonl")
    assert summarize_reviews(review)["approval_rate"] is None
    row = json.loads(review.read_text())
    row.update(status="approved", correctness=5, clarity=4,
               difficulty_match=3, completeness=4)
    review.write_text(json.dumps(row) + "\n")
    assert summarize_reviews(review)["approval_rate"] == 1.0
    paths = save_report({"benchmark": "local", "official_score": False,
                         "judge": result, "pairwise": pair}, tmp_path / "report")
    assert Path(paths["json"]).is_file()
    with pytest.raises(FileExistsError):
        save_report({"benchmark": "local"}, tmp_path / "report")


def test_generation_rejects_duplicate_or_invalid_output():
    reference = [GeneratedProblem("ref", "What is 2+2?", "4", "2+2=4", "reference")]

    class FakeGenerator:
        calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"problem": "What is 2+2?", "answer": "4", "solution": "4"})
            return json.dumps({"problem": "What is 7+7?", "answer": "14", "solution": "7+7=14"})

    result = generate_problems(FakeGenerator(), reference, count=1, max_attempts=2)
    assert result["complete"]
    assert len(result["failures"]) == 1
    assert result["generated"][0].answer == "14"


def test_pairwise_order_conflict_is_not_a_tie():
    reference = [GeneratedProblem("ref", "What is 2+2?", "4", "2+2=4", "reference")]
    generated = [GeneratedProblem("gen", "What is 3+3?", "6", "3+3=6")]

    class FirstPositionJudge:
        def invoke(self, messages, **kwargs):
            return '{"winner":"A","reason":"first position"}'

    report = compare_problems(FirstPositionJudge(), generated, reference)
    assert report["disagreement_count"] == 1
    assert report["resolved_count"] == 0
    assert report["ties"] == 0
    assert report["win_rate"] is None


def test_human_review_terminal_saves_and_resumes(tmp_path):
    items = [GeneratedProblem("a", "What is 1+1?", "2", "1+1=2"),
             GeneratedProblem("b", "What is 2+2?", "4", "2+2=4")]
    path = create_review_template(items, tmp_path / "review.jsonl")
    replies = iter(["approved", "5", "4", "3", "5", "correct", "q"])
    result = review_interactively(path, input_fn=lambda _: next(replies), print_fn=lambda _: None)
    assert result["reviewed"] == 1
    assert result["counts"]["needs_review"] == 1
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_multi_turn_and_failure_recovery_use_isolated_sandbox_tools():
    class ScriptedLLM:
        def complete_with_tools(self, messages, tools, **kwargs):
            latest = messages[-1]
            content = latest.get("content") or ""
            if "Remember alpha" in content:
                calls = [ToolCall("1", "remember", '{"value":"alpha"}')]
            elif "What was the value" in content:
                calls = [ToolCall("2", "lookup", "{}")]
            elif "Fetch the value" in content or "simulated transient" in content:
                calls = [ToolCall("3", "fetch", '{"key":"answer"}')]
            else:
                calls = []
            answer = "alpha" if content == "alpha" else "42" if content == "42" else "stored"
            return ToolCompletion(answer, calls, prompt_tokens=5, completion_tokens=2)

    report = run_trajectories(
        ScriptedLLM(), load_trajectory_cases(FIXTURES / "trajectory_cases.jsonl")
    )
    assert report["sample_count"] == 2
    assert report["task_success_rate"] == 1.0
    assert report["tool_path_accuracy"] == 1.0
    assert report["recovery_rate"] == 1.0
    assert report["samples"][1]["tool_errors"] == 1
    assert report["reported_completion_tokens"] > 0


def test_deterministic_verifier_and_human_judge_calibration(tmp_path):
    valid = GeneratedProblem("ok", "What is 3+3?", "6", "3+3=6",
                             verification_expression="3 + 3")
    invalid = GeneratedProblem("bad", "What is 3+3?", "7", "3+3=7",
                               verification_expression="3 + 3")
    assert verify_problem(valid)["status"] == "passed"
    assert verify_problem(invalid)["status"] == "failed"

    class Judge:
        calls = 0

        def invoke(self, messages, **kwargs):
            self.calls += 1
            return '{"correctness":5,"clarity":4,"difficulty_match":4,"completeness":4}'

    llm = Judge()
    report = judge_problems(llm, [valid, invalid])
    assert llm.calls == 1
    assert report["deterministic_failed_count"] == 1
    review_path = create_review_template([valid], tmp_path / "review.jsonl")
    row = json.loads(review_path.read_text())
    row.update(status="approved", correctness=5, clarity=4,
               difficulty_match=3, completeness=4)
    review_path.write_text(json.dumps(row) + "\n")
    audit = calibrate_judge(report, review_path)
    assert audit["matched_count"] == 1
    assert audit["agreement"] == 1.0
    assert audit["dimension_mae"]["difficulty_match"] == 1.0


def test_dynamic_scenarios_cover_clarification_failure_time_and_peer():
    class ScriptedLLM:
        def complete_with_tools(self, messages, tools, **kwargs):
            question = next(message["content"] for message in messages
                            if message["role"] == "user")
            observed = [message["content"] for message in messages
                        if message["role"] == "tool"]
            if "which day" in question:
                steps = [("ask_user", {"question": "Which day?"}),
                         ("set_state", {"key": "meeting", "value": "Monday"})]
                final = "done"
            elif "API fails" in question:
                steps = [("get_state", {"key": "status"}),
                         ("get_state", {"key": "status"}),
                         ("set_state", {"key": "status", "value": "ready"})]
                final = "done"
            elif "five minutes" in question:
                steps = [("wait", {"minutes": 5}),
                         ("get_state", {"key": "alarm"})]
                final = "on"
            else:
                steps = [("ask_peer", {"question": "What is the decision?"}),
                         ("set_state", {"key": "decision", "value": "go"})]
                final = "done"
            if len(observed) >= len(steps):
                return ToolCompletion(final, [], prompt_tokens=5, completion_tokens=2)
            name, arguments = steps[len(observed)]
            return ToolCompletion("", [ToolCall(
                f"call-{len(observed)}", name, json.dumps(arguments),
            )], prompt_tokens=5, completion_tokens=2)

    report = run_dynamic(ScriptedLLM(), load_dynamic_cases(FIXTURES / "dynamic_cases.jsonl"))
    assert report["scenario_count"] == 4
    assert report["pass_at_1"] == 1.0
    assert report["failure_recovery_rate"] == 1.0
    assert report["samples"][2]["simulated_minutes"] == 5


def test_uncertainty_paired_comparison_and_baseline_manifest(tmp_path):
    interval = wilson_interval(4, 4)
    assert 0.50 < interval["lower"] < 0.52
    comparison = paired_bootstrap_delta(
        [{"id": "a", "correct": True}, {"id": "b", "correct": False}],
        [{"id": "a", "correct": False}, {"id": "b", "correct": False}],
        id_key="id", score_key="correct", resamples=200,
    )
    assert comparison["common_cases"] == 2
    assert comparison["delta"] == 0.5

    class SilentLLM:
        model = "offline-fixture"
        provider = "fake"

        def complete_with_tools(self, messages, tools, **kwargs):
            return ToolCompletion("", [])

    class SilentAgent:
        def run(self, prompt):
            return "Final Answer: unknown"

    report = run_live_baseline(
        SilentLLM(), SilentAgent, output_dir=tmp_path / "baseline",
        fixture_dir=FIXTURES, max_samples_per_suite=1,
    )
    assert set(report["suites"]) == {"bfcl", "gaia", "trajectory", "dynamic"}
    bfcl_report = json.loads((tmp_path / "baseline/bfcl/report.json").read_text())
    assert bfcl_report["provenance"]["dataset_files_sha256"]
    assert bfcl_report["argument_accuracy_ci95"] is not None
