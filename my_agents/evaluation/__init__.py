"""Chapter 12: reproducible tool, assistant, and generated-data evaluation."""

from .bfcl import BFCLCase, load_bfcl_cases, load_bfcl_v4_subset, run_bfcl, run_official_bfcl
from .gaia import (
    GAIACase, export_gaia_submission, gaia_match, load_gaia_cases,
    load_gaia_official_validation, run_gaia,
)
from .generation import (
    GeneratedProblem, generate_problems, judge_problems, load_problems,
    compare_problems, create_review_template, review_interactively, summarize_reviews,
    verify_problem, calibrate_judge,
)
from .report import save_report
from .trajectory import TrajectoryCase, load_trajectory_cases, run_trajectories
from .dynamic import DynamicCase, load_dynamic_cases, run_dynamic
from .stats import paired_bootstrap_delta, wilson_interval
from .baseline import run_live_baseline
from .provenance import run_provenance

__all__ = [
    "BFCLCase", "load_bfcl_cases", "load_bfcl_v4_subset", "run_bfcl",
    "run_official_bfcl", "GAIACase", "gaia_match", "load_gaia_cases",
    "run_gaia", "load_gaia_official_validation", "export_gaia_submission",
    "GeneratedProblem", "generate_problems", "judge_problems",
    "load_problems", "compare_problems", "create_review_template",
    "review_interactively", "summarize_reviews", "verify_problem",
    "calibrate_judge", "save_report",
    "TrajectoryCase", "load_trajectory_cases", "run_trajectories",
    "DynamicCase", "load_dynamic_cases", "run_dynamic",
    "paired_bootstrap_delta", "wilson_interval",
    "run_live_baseline", "run_provenance",
]
