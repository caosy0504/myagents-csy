"""Chapter 12 evaluation commands. Live runs use the project's configured LLM."""

import argparse
import json
import socket
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .bfcl import load_bfcl_cases, load_bfcl_v4_subset, run_bfcl, run_official_bfcl
from .gaia import load_gaia_cases, load_gaia_official_validation, run_gaia
from .generation import (
    compare_problems, create_review_template, generate_problems,
    judge_problems, load_problems, review_interactively, save_problems, summarize_reviews,
)
from .report import save_report
from .baseline import run_live_baseline
from .dynamic import load_dynamic_cases, run_dynamic
from .trajectory import load_trajectory_cases, run_trajectories
from .generation import calibrate_judge
from .provenance import run_provenance
from .stats import paired_bootstrap_delta


def _default_output(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return Path("evaluation_runs") / f"{name}-{stamp}"


def _live_llm():
    from my_agents import MyAgentsLLM

    load_dotenv(Path.cwd() / ".env")
    llm = MyAgentsLLM.from_env()
    endpoint = urlparse(llm.base_url)
    if endpoint.hostname in {"localhost", "127.0.0.1", "::1"}:
        try:
            with socket.create_connection(
                (endpoint.hostname, endpoint.port or 80), timeout=2
            ):
                pass
        except OSError as exc:
            raise RuntimeError(
                f"Local model service is unavailable at {endpoint.hostname}:{endpoint.port or 80}"
            ) from exc
    if llm.provider == "ollama":
        try:
            tags_url = f"{endpoint.scheme}://{endpoint.hostname}:{endpoint.port or 11434}/api/tags"
            with urllib.request.urlopen(tags_url, timeout=2) as response:
                tags = json.load(response)
            llm.model_revision = next(
                (item.get("digest") for item in tags.get("models", [])
                 if item.get("name") == llm.model), None
            )
        except (OSError, ValueError):
            llm.model_revision = None
    return llm


def _live_agent_factory(kind: str, llm=None):
    from my_agents import CalculatorTool, Config, ReActAgent, SimpleAgent, ToolRegistry

    llm = llm or _live_llm()

    def factory():
        if kind == "simple":
            return SimpleAgent(name="gaia-eval", llm=llm, config=Config(temperature=0))
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        return ReActAgent(
            name="gaia-eval", llm=llm, tool_registry=registry,
            config=Config(temperature=0, max_steps=4),
        )

    return factory


def _with_provenance(report: dict, llm, *, data_files=(), scorer_files=(), parameters=None):
    report["provenance"] = run_provenance(
        model=getattr(llm, "model", None), provider=getattr(llm, "provider", None),
        model_revision=getattr(llm, "model_revision", None),
        data_files=data_files, scorer_files=scorer_files, parameters=parameters,
        dataset_revision=report.get("dataset_revision"),
    )
    return report


def _demo(root: Path) -> dict:
    from my_agents.core import ToolCall, ToolCompletion

    fixtures = Path(__file__).resolve().parent / "fixtures"
    cases = load_bfcl_cases(fixtures / "bfcl_cases.jsonl")

    class FixtureToolLLM:
        def complete_with_tools(self, messages, tools, **kwargs):
            question = messages[-1]["content"]
            case = next(item for item in cases if item.messages[-1]["content"] == question)
            calls = []
            for index, expected in enumerate(case.expected):
                name, allowed = next(iter(expected.items()))
                args = {key: next(value for value in options if value != "")
                        for key, options in allowed.items() if any(v != "" for v in options)}
                calls.append(ToolCall(f"call-{index}", name, json.dumps(args)))
            return ToolCompletion("", calls, prompt_tokens=0, completion_tokens=0)

    bfcl = run_bfcl(FixtureToolLLM(), cases)
    bfcl.update({"mode": "offline_fixture", "quality_note": "Checks scoring code, not model ability."})
    bfcl_paths = save_report(bfcl, root / "bfcl")

    gaia_cases = load_gaia_cases(fixtures / "gaia_cases.jsonl")
    answers = {item.question: item.answer for item in gaia_cases}

    class FixtureAgent:
        def run(self, prompt):
            return f"Final Answer: {answers[prompt]}"

    gaia = run_gaia(FixtureAgent, gaia_cases)
    gaia.update({"mode": "offline_fixture", "quality_note": "Checks scoring code, not model ability."})
    gaia_paths = save_report(gaia, root / "gaia")

    class FixtureAgentLLM:
        def complete_with_tools(self, messages, tools, **kwargs):
            latest_user = next(message["content"] for message in reversed(messages)
                               if message["role"] == "user")
            observations = [message["content"] for message in messages
                            if message["role"] == "tool"]
            if "Remember alpha" in latest_user:
                steps, final = [("remember", {"value": "alpha"})], "stored"
            elif "What was the value" in latest_user:
                steps, final = [("lookup", {})], "alpha"
            elif "Fetch the value" in latest_user:
                steps = [("fetch", {"key": "answer"})] * 2
                final = "42"
            elif "which day" in latest_user:
                steps = [("ask_user", {"question": "Which day?"}),
                         ("set_state", {"key": "meeting", "value": "Monday"})]
                final = "done"
            elif "API fails" in latest_user:
                steps = [("get_state", {"key": "status"})] * 2 + [
                    ("set_state", {"key": "status", "value": "ready"})]
                final = "done"
            elif "five minutes" in latest_user:
                steps = [("wait", {"minutes": 5}),
                         ("get_state", {"key": "alarm"})]
                final = "on"
            else:
                steps = [("ask_peer", {"question": "What is the decision?"}),
                         ("set_state", {"key": "decision", "value": "go"})]
                final = "done"
            if len(observations) >= len(steps):
                return ToolCompletion(final, [], prompt_tokens=0, completion_tokens=0)
            name, arguments = steps[len(observations)]
            return ToolCompletion("", [ToolCall(
                f"call-{len(observations)}", name, json.dumps(arguments),
            )], prompt_tokens=0, completion_tokens=0)

    trajectory = run_trajectories(
        FixtureAgentLLM(), load_trajectory_cases(fixtures / "trajectory_cases.jsonl")
    )
    trajectory.update({"mode": "offline_fixture",
                       "quality_note": "Checks sandbox orchestration, not model ability."})
    trajectory_paths = save_report(trajectory, root / "trajectory")
    dynamic = run_dynamic(
        FixtureAgentLLM(), load_dynamic_cases(fixtures / "dynamic_cases.jsonl")
    )
    dynamic.update({"mode": "offline_fixture",
                    "quality_note": "Checks simulated world logic, not model ability."})
    dynamic_paths = save_report(dynamic, root / "dynamic")

    generated = load_problems(fixtures / "generated_problems.jsonl", source="generated")
    reference = load_problems(fixtures / "reference_problems.jsonl")

    class FixtureJudge:
        def invoke(self, messages, **kwargs):
            prompt = messages[-1]["content"]
            return ('{"winner":"Tie","reason":"fixture"}' if "比较两道" in prompt
                    else '{"correctness":4,"clarity":4,"difficulty_match":3,'
                         '"completeness":4,"comments":"fixture"}')

    judge = judge_problems(FixtureJudge(), generated)
    pairwise = compare_problems(FixtureJudge(), generated, reference)
    review_path = create_review_template(generated, root / "generation" / "review_queue.jsonl")
    generation = {
        "benchmark": "aime_style_generation_local_fixture", "official_score": False,
        "mode": "offline_fixture", "generated_count": len(generated),
        "judge": judge, "pairwise": pairwise,
        "human_review": "pending", "quality_note": "Checks orchestration, not math quality.",
    }
    generation_paths = save_report(generation, root / "generation")
    return {"bfcl": bfcl_paths, "gaia": gaia_paths,
            "trajectory": trajectory_paths, "dynamic": dynamic_paths,
            "generation": generation_paths, "review_queue": str(review_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    demo = actions.add_parser("demo", help="Offline deterministic checks of all three paths")
    demo.add_argument("--output", type=Path)

    bfcl = actions.add_parser("bfcl", help="Run the local native-call proxy on fixed cases")
    bfcl.add_argument("--cases", type=Path)
    bfcl.add_argument("--official-questions", type=Path)
    bfcl.add_argument("--official-answers", type=Path)
    bfcl.add_argument("--category", default="simple_python")
    bfcl.add_argument("--max-samples", type=int)
    bfcl.add_argument("--output", type=Path)

    gaia = actions.add_parser("gaia", help="Run the local GAIA-style assistant evaluator")
    gaia.add_argument("--cases", type=Path)
    gaia.add_argument("--official-validation", action="store_true")
    gaia.add_argument("--level", type=int, choices=(1, 2, 3))
    gaia.add_argument("--revision", help="Pin the official GAIA dataset commit")
    gaia.add_argument("--agent", choices=("simple", "react"), default="react")
    gaia.add_argument("--max-samples", type=int)
    gaia.add_argument("--output", type=Path)

    generation = actions.add_parser("generation", help="Generate and judge local math problems")
    generation.add_argument("--references", type=Path, required=True)
    generation.add_argument("--generated", type=Path, help="Existing generated JSONL; skips generation")
    generation.add_argument("--count", type=int, default=5)
    generation.add_argument("--judge-model", help="Separate judge model on the configured endpoint")
    generation.add_argument("--output", type=Path)

    baseline = actions.add_parser("baseline", help="Run fixed live suites without an aggregate score")
    baseline.add_argument("--agent", choices=("simple", "react"), default="react")
    baseline.add_argument("--max-samples", type=int, help="Per-suite limit for a short first run")
    baseline.add_argument("--output", type=Path)

    trajectory = actions.add_parser("trajectory", help="Run isolated multi-turn tool scenarios")
    trajectory.add_argument("--cases", type=Path, required=True)
    trajectory.add_argument("--max-samples", type=int)
    trajectory.add_argument("--output", type=Path)

    dynamic = actions.add_parser("dynamic", help="Run local Gaia2-style scenarios")
    dynamic.add_argument("--cases", type=Path, required=True)
    dynamic.add_argument("--max-samples", type=int)
    dynamic.add_argument("--output", type=Path)

    compare = actions.add_parser("compare-reports", help="Paired bootstrap on matching case IDs")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--score-key", required=True)
    compare.add_argument("--id-key", default="id")
    compare.add_argument("--seed", type=int, default=42)

    calibration = actions.add_parser("calibrate-judge", help="Compare Judge scores to human reviews")
    calibration.add_argument("--judge-report", type=Path, required=True)
    calibration.add_argument("--reviews", type=Path, required=True)
    calibration.add_argument("--output", type=Path)

    review = actions.add_parser("review-summary", help="Summarize a filled human review queue")
    review.add_argument("--file", type=Path, required=True)
    interactive = actions.add_parser("review", help="Review generated problems in the terminal")
    interactive.add_argument("--file", type=Path, required=True)

    official = actions.add_parser("bfcl-official", help="Run official CLI on official generated results")
    official.add_argument("--model", required=True)
    official.add_argument("--category", required=True)
    official.add_argument("--project-root", type=Path, required=True)
    official.add_argument("--partial-eval", action="store_true")

    args = parser.parse_args()
    if args.action == "review-summary":
        print(json.dumps(summarize_reviews(args.file), ensure_ascii=False, indent=2))
        return
    if args.action == "review":
        print(json.dumps(review_interactively(args.file), ensure_ascii=False, indent=2))
        return
    if args.action == "bfcl-official":
        print(json.dumps(run_official_bfcl(
            args.model, args.category, project_root=args.project_root,
            partial_eval=args.partial_eval,
        ), ensure_ascii=False, indent=2))
        return
    if args.action == "compare-reports":
        left = json.loads(args.left.read_text(encoding="utf-8"))
        right = json.loads(args.right.read_text(encoding="utf-8"))
        if left.get("benchmark") != right.get("benchmark"):
            raise ValueError("Paired reports must measure the same benchmark")
        result = paired_bootstrap_delta(
            left["samples"], right["samples"], id_key=args.id_key,
            score_key=args.score_key, seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    output = args.output or _default_output(args.action)
    if args.action == "calibrate-judge":
        judge = json.loads(args.judge_report.read_text(encoding="utf-8"))
        result = {"benchmark": "judge_human_calibration", "official_score": False,
                  "calibration": calibrate_judge(judge.get("judge", judge), args.reviews)}
        print(json.dumps(save_report(result, output), ensure_ascii=False, indent=2))
        return
    if args.action == "demo":
        result = _demo(output)
    elif args.action == "baseline":
        llm = _live_llm()
        result = run_live_baseline(
            llm, _live_agent_factory(args.agent, llm), output_dir=output,
            fixture_dir=Path(__file__).resolve().parent / "fixtures",
            max_samples_per_suite=args.max_samples,
            agent_description=f"{args.agent}_calculator_max_steps4",
        )
    elif args.action == "bfcl":
        if bool(args.cases) == bool(args.official_questions):
            parser.error("Choose exactly one of --cases or --official-questions")
        cases = (load_bfcl_cases(args.cases) if args.cases else load_bfcl_v4_subset(
            args.official_questions, args.official_answers, category=args.category,
            max_samples=args.max_samples,
        ))
        llm = _live_llm()
        report = run_bfcl(llm, cases, max_samples=args.max_samples)
        report["data_source"] = "official_bfcl_v4_subset" if args.official_questions else "local_jsonl"
        _with_provenance(
            report, llm,
            data_files=([args.official_questions, args.official_answers]
                        if args.official_questions and args.official_answers else
                        [args.cases] if args.cases else [args.official_questions]),
            scorer_files=[Path(__file__).with_name("bfcl.py")],
            parameters={"category": args.category, "max_samples": args.max_samples,
                        "temperature": 0},
        )
        result = save_report(report, output)
    elif args.action == "gaia":
        if bool(args.cases) == bool(args.official_validation):
            parser.error("Choose exactly one of --cases or --official-validation")
        if args.official_validation:
            load_dotenv(Path.cwd() / ".env")
        cases = (load_gaia_cases(args.cases) if args.cases else
                 load_gaia_official_validation(level=args.level, revision=args.revision))
        llm = _live_llm()
        report = run_gaia(
            _live_agent_factory(args.agent, llm), cases, max_samples=args.max_samples
        )
        report["data_source"] = "official_gaia_validation" if args.official_validation else "local_jsonl"
        _with_provenance(
            report, llm,
            data_files=[args.cases] if args.cases else [],
            scorer_files=[Path(__file__).with_name("gaia.py")],
            parameters={"agent": args.agent, "level": args.level,
                        "max_samples": args.max_samples, "temperature": 0},
        )
        result = save_report(report, output)
    elif args.action in {"trajectory", "dynamic"}:
        if args.max_samples is not None and args.max_samples <= 0:
            parser.error("--max-samples must be positive")
        llm = _live_llm()
        if args.action == "trajectory":
            cases = load_trajectory_cases(args.cases)
            runner = run_trajectories
        else:
            cases = load_dynamic_cases(args.cases)
            runner = run_dynamic
        report = runner(llm, cases[:args.max_samples])
        _with_provenance(
            report, llm, data_files=[args.cases],
            scorer_files=[Path(__file__).with_name(f"{args.action}.py")],
            parameters={"max_samples": args.max_samples, "temperature": 0},
        )
        result = save_report(report, output)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Generation output already contains files: {output}")
        references = load_problems(args.references)
        llm = _live_llm()
        if args.judge_model:
            from my_agents import MyAgentsLLM

            judge_llm = MyAgentsLLM(
                model=args.judge_model, api_key=llm.api_key,
                base_url=llm.base_url, timeout=llm.timeout,
            )
        else:
            judge_llm = llm
        if args.generated:
            problems = load_problems(args.generated, source="generated")
            generation_status = {"requested": len(problems), "complete": True, "attempts": 0,
                                 "failures": []}
        else:
            generation_status = generate_problems(llm, references, count=args.count)
            problems = generation_status.pop("generated")
        if not problems:
            raise RuntimeError("No valid generated problems; inspect generation failures")
        output.mkdir(parents=True, exist_ok=True)
        generated_path = save_problems(problems, output / "generated_problems.jsonl")
        judge = judge_problems(judge_llm, problems)
        pairwise = compare_problems(judge_llm, problems, references)
        review_path = create_review_template(problems, output / "review_queue.jsonl")
        report = {"benchmark": "aime_style_generation_local", "official_score": False,
                  "generated_count": len(problems), "generation": generation_status,
                  "judge": judge, "pairwise": pairwise, "human_review": "pending",
                  "judge_model": judge_llm.model,
                  "judge_note": (
                      "Generator and judge share a model; calibrate with independent human ratings."
                      if judge_llm.model == llm.model else
                      "Judge uses a different model ID on the same endpoint; human calibration is still required."
                  ),
                  "generated_file": str(generated_path), "review_queue": str(review_path)}
        _with_provenance(
            report, llm, data_files=[args.references, generated_path],
            scorer_files=[Path(__file__).with_name("generation.py")],
            parameters={"count": args.count, "judge_model": judge_llm.model},
        )
        result = save_report(report, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
