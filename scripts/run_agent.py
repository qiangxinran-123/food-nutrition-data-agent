from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for relative in ("src", "src/data_agent/clustering/.m1_deps", "src/data_agent/clustering/.m2_deps", ".deps"):
    candidate = PROJECT_ROOT / relative
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from data_agent.agent import run_agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the food-nutrition LLM agent end to end.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default=None, help="run directory; must not exist")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--task-file", default="task_v1.md")
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    parser.add_argument("--dangerous-policy", choices=["deny", "allow"], default="deny")
    parser.add_argument("--dataset-version", default="1")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-bundle", action="store_true")
    args = parser.parse_args()

    summary = run_agent(
        args.project_root,
        output_dir=args.output,
        profile_name=args.profile,
        prompt_version=args.prompt_version,
        task_file=args.task_file,
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
        dangerous_policy=args.dangerous_policy,
        dataset_version=args.dataset_version,
        run_id=args.run_id,
        make_bundle=not args.no_bundle,
    )
    print(f"run_id      : {summary['run_id']}")
    print(f"status      : {summary['status']}")
    print(f"model calls : {summary['model_calls']}  tool calls: {summary['tool_calls']}")
    print(f"cost        : {summary['usage_and_cost']}")
    print(f"final data  : {summary['final_dataset']}")
    print(f"bundle      : {summary['bundle']}")
    print(f"output      : {summary['output_dir']}")
    if summary.get("error"):
        print(f"error       : {summary['error']}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
