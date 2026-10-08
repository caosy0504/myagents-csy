"""Run chapter 11 validation or the full training pipeline."""

import argparse
import json

from .pipeline import run_pipeline, validate_pipeline
from .agentic_pipeline import run_agentic_pipeline, validate_agentic_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "validate-agentic", "run-agentic"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = {
        "validate": validate_pipeline,
        "run": run_pipeline,
        "validate-agentic": validate_agentic_pipeline,
        "run-agentic": run_agentic_pipeline,
    }[args.action](args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
