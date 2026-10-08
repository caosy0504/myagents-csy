"""Chapter 12 Tool wrappers; callers explicitly provide an LLM or Agent factory."""

import json
from typing import Any, Callable

from ..tools.base import Tool
from .bfcl import load_bfcl_cases, load_bfcl_v4_subset, run_bfcl
from .gaia import load_gaia_cases, load_gaia_official_validation, run_gaia
from .generation import compare_problems, judge_problems, load_problems
from .report import save_report


def _request(input_text: str) -> dict:
    value = json.loads(input_text)
    if not isinstance(value, dict):
        raise ValueError("Evaluation tool input must be a JSON object")
    return value


def _finish(result: dict, options: dict) -> str:
    if options.get("output_dir"):
        result["report_files"] = save_report(result, options["output_dir"])
    return json.dumps(result, ensure_ascii=False, default=str)


class BFCLEvaluationTool(Tool):
    def __init__(self, llm: Any) -> None:
        super().__init__("bfcl_evaluation", "Explicit native tool-call evaluation")
        self.llm = llm

    def run(self, input_text: str) -> str:
        options = _request(input_text)
        if options.get("cases_file"):
            cases = load_bfcl_cases(options["cases_file"])
        else:
            cases = load_bfcl_v4_subset(
                options["questions_file"], options.get("answers_file"),
                category=options["category"], max_samples=options.get("max_samples"),
            )
        return _finish(run_bfcl(self.llm, cases, max_samples=options.get("max_samples")), options)


class GAIAEvaluationTool(Tool):
    def __init__(self, agent_factory: Callable[[], Any]) -> None:
        super().__init__("gaia_evaluation", "Explicit GAIA-style assistant evaluation")
        self.agent_factory = agent_factory

    def run(self, input_text: str) -> str:
        options = _request(input_text)
        cases = (
            load_gaia_official_validation(
                level=options.get("level"), revision=options.get("revision")
            )
            if options.get("official_validation") else
            load_gaia_cases(options["cases_file"], attachment_root=options.get("attachment_root"))
        )
        return _finish(run_gaia(
            self.agent_factory, cases, max_samples=options.get("max_samples")
        ), options)


class LLMJudgeTool(Tool):
    def __init__(self, llm: Any) -> None:
        super().__init__("llm_judge", "Explicit bounded 4-dimension generated-data review")
        self.llm = llm

    def run(self, input_text: str) -> str:
        options = _request(input_text)
        problems = load_problems(options["generated_file"], source="generated")
        result = {"benchmark": "aime_style_judge_local", "official_score": False,
                  "judge": judge_problems(self.llm, problems)}
        return _finish(result, options)


class WinRateTool(Tool):
    def __init__(self, llm: Any) -> None:
        super().__init__("win_rate", "Explicit order-balanced pairwise comparison")
        self.llm = llm

    def run(self, input_text: str) -> str:
        options = _request(input_text)
        generated = load_problems(options["generated_file"], source="generated")
        reference = load_problems(options["reference_file"])
        result = {"benchmark": "aime_style_pairwise_local", "official_score": False,
                  "pairwise": compare_problems(
                      self.llm, generated, reference, seed=int(options.get("seed", 42))
                  )}
        return _finish(result, options)
