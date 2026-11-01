from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if (PROJECT_ROOT / ".deps").exists():
    sys.path.insert(0, str(PROJECT_ROOT / ".deps"))

from data_agent.cleaning import execute_plan
from data_agent.contracts import CleaningPlan
from data_agent.io import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute an ordered cleaning plan. Use --persist per_step to write one "
            "version per decision, or --persist final to run every decision in "
            "memory and write only one final version. Dry-run is the default."
        )
    )
    parser.add_argument("--plan", required=True, help="JSON CleaningPlan file")
    parser.add_argument(
        "--profile",
        default=str(PROJECT_ROOT / "config" / "profiles" / "off_52k_v1.json"),
    )
    parser.add_argument(
        "--selection-dir",
        default=str(PROJECT_ROOT / "logs" / "checks" / "selections"),
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "logs" / "cleaning"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
    )
    parser.add_argument(
        "--persist",
        choices=("per_step", "final"),
        default="per_step",
        help=(
            "per_step writes v2, v3, ... for each action; final writes only the "
            "last dataset version with one aggregate change log."
        ),
    )
    parser.add_argument("--no-dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(
        args.profile,
        project_root=PROJECT_ROOT,
        verify_hash=not args.skip_hash_check,
    )
    payload = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    payload.setdefault("before_dataset", dataset.ref.model_dump(mode="json"))
    if args.no_dry_run:
        payload["dry_run"] = False
    else:
        payload.setdefault("dry_run", True)
    plan = CleaningPlan.model_validate(payload)

    result = execute_plan(
        dataset,
        plan,
        selection_artifact_dir=args.selection_dir,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        confirm=args.confirm,
        persist=args.persist,
    )
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / f"{plan.plan_id}.plan_result.json"
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset: {dataset.ref.dataset_id}")
    print(f"Plan status: {result.status.value}")
    print(f"Dry run: {result.dry_run}")
    print(f"Persist mode: {args.persist}")
    print(f"Steps: {len(result.results)}")
    print(
        f"Rows: {result.before_dataset.row_count} -> {result.after_dataset.row_count}"
    )
    print(f"Final dataset: {result.after_dataset.dataset_id}")
    if result.error_message:
        print(f"Error: {result.error_message}")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
