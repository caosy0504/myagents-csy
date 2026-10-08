"""Fixed, versioned live baseline across the chapter 12 local suites."""

import json
from pathlib import Path
from typing import Any, Callable

from .bfcl import load_bfcl_cases, run_bfcl
from .dynamic import load_dynamic_cases, run_dynamic
from .gaia import load_gaia_cases, run_gaia
from .provenance import run_provenance
from .report import save_report
from .trajectory import load_trajectory_cases, run_trajectories


def run_live_baseline(
    llm: Any, agent_factory: Callable[[], Any], *,
    output_dir: str | Path, fixture_dir: str | Path,
    max_samples_per_suite: int | None = None,
    agent_description: str = "unspecified",
) -> dict:
    """Use the same fixed cases and model client across four distinct suites."""
    if max_samples_per_suite is not None and max_samples_per_suite <= 0:
        raise ValueError("max_samples_per_suite must be positive")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Baseline output already contains files: {output}")
    fixtures = Path(fixture_dir).expanduser().resolve()
    suites = [
        ("bfcl", "bfcl_cases.jsonl", load_bfcl_cases, run_bfcl),
        ("gaia", "gaia_cases.jsonl", load_gaia_cases, run_gaia),
        ("trajectory", "trajectory_cases.jsonl", load_trajectory_cases, run_trajectories),
        ("dynamic", "dynamic_cases.jsonl", load_dynamic_cases, run_dynamic),
    ]
    summary = {
        "benchmark": "chapter12_fixed_live_baseline",
        "official_score": False,
        "model": getattr(llm, "model", None),
        "provider": getattr(llm, "provider", None),
        "model_revision": getattr(llm, "model_revision", None),
        "agent_description": agent_description,
        "max_samples_per_suite": max_samples_per_suite,
        "suites": {},
    }
    for name, filename, loader, runner in suites:
        path = fixtures / filename
        cases = loader(path)
        if max_samples_per_suite is not None:
            cases = cases[:max_samples_per_suite]
        report = (runner(agent_factory, cases) if name == "gaia"
                  else runner(llm, cases))
        report["provenance"] = run_provenance(
            model=getattr(llm, "model", None), provider=getattr(llm, "provider", None),
            model_revision=getattr(llm, "model_revision", None),
            data_files=[path], scorer_files=[Path(__file__).with_name(f"{name}.py")],
            parameters={"suite": name, "max_samples": max_samples_per_suite,
                        "temperature": 0, "agent_factory": "fresh_per_gaia_case",
                        "agent_description": agent_description},
        )
        files = save_report(report, output / name)
        metrics = {
            key: report.get(key) for key in (
                "sample_count", "scenario_count", "accuracy", "coverage",
                "argument_accuracy", "task_success_rate", "pass_at_1",
                "tool_error_rate", "mean_latency_seconds",
                "reported_prompt_tokens", "reported_completion_tokens",
            ) if key in report
        }
        summary["suites"][name] = {"metrics": metrics, "files": files}
    summary["note"] = (
        "The four local suites measure different capabilities and are not averaged. "
        "They are not official BFCL, GAIA, or Gaia2 scores."
    )
    summary["report_files"] = save_report(summary, output)
    return summary
