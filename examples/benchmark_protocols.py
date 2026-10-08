"""Compare repeated stdio MCP calls with and without a reused connection."""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from my_agents import MCPClient, MCPTool


def measure(calls: int, repeats: int) -> dict:
    command = [sys.executable, "-m", "examples.protocol_demo", "mcp-server"]
    samples = {"per_call_connection": [], "reused_connection": []}
    for _ in range(repeats):
        for mode in samples:
            client = MCPClient(command)
            start = perf_counter()
            if mode == "reused_connection":
                client.open_sync()
            try:
                tool = MCPTool.discover_sync(client)[0]
                answers = [
                    json.loads(tool.run_arguments({"input_text": "6*7"}))["result"]
                    for _ in range(calls)
                ]
                if answers != ["42"] * calls:
                    raise AssertionError("MCP results differ across connection modes")
            finally:
                client.close_sync()
            samples[mode].append(perf_counter() - start)
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "transport": "local stdio subprocess",
        "calls_per_repeat": calls,
        "repeats": repeats,
        "includes_discovery_and_cleanup": True,
        "median_total_seconds": {
            mode: round(statistics.median(values), 4)
            for mode, values in samples.items()
        },
        "samples_seconds": {
            mode: [round(value, 4) for value in values]
            for mode, values in samples.items()
        },
        "note": "Local process-startup measurement, not a model-quality or remote HTTP benchmark.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calls <= 0 or args.repeats <= 0:
        parser.error("calls and repeats must be positive")
    report = json.dumps(measure(args.calls, args.repeats), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
