from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if (PROJECT_ROOT / ".deps").exists():
    sys.path.insert(0, str(PROJECT_ROOT / ".deps"))

from data_agent.cleaning import execute_cleaning, make_decision
from data_agent.contracts import IssueType
from data_agent.io import load_dataset


def _json_argument(value: str | None, default):
    if value is None:
        return default
    return json.loads(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one LLM-selected cleaning strategy. The default is a dry run; "
            "raw/base data is never overwritten."
        )
    )
    parser.add_argument(
        "--profile",
        default=str(PROJECT_ROOT / "config" / "profiles" / "off_52k_v1.json"),
    )
    parser.add_argument(
        "--issue-type",
        required=True,
        choices=[issue.value for issue in IssueType],
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--finding-id", required=True)
    parser.add_argument("--selection-artifact-id", default=None)
    parser.add_argument(
        "--selection-dir",
        default=str(PROJECT_ROOT / "logs" / "checks" / "selections"),
    )
    parser.add_argument("--target-columns", default=None, help="JSON array")
    parser.add_argument("--params", default=None, help="JSON object")
    parser.add_argument("--reason", default=None)
    parser.add_argument(
        "--artifact-dir",
        default=str(PROJECT_ROOT / "logs" / "cleaning"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Write a new processed dataset version. Dangerous strategies also need --confirm.",
    )
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(
        args.profile,
        project_root=PROJECT_ROOT,
        verify_hash=not args.skip_hash_check,
    )
    decision = make_decision(
        issue_type=args.issue_type,
        strategy_id=args.strategy,
        finding_id=args.finding_id,
        selection_artifact_id=args.selection_artifact_id,
        target_columns=_json_argument(args.target_columns, []),
        params=_json_argument(args.params, {}),
        reason=args.reason,
    )
    result = execute_cleaning(
        dataset,
        decision,
        selection_artifact_dir=args.selection_dir,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        dry_run=not args.no_dry_run,
        confirm=args.confirm,
    )
    output_path = Path(args.artifact_dir) / f"{decision.decision_id}.result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset: {dataset.ref.dataset_id}")
    print(f"Status: {result.status.value}")
    print(f"Dry run: {result.dry_run}")
    print(f"Rows: {result.before_dataset.row_count} -> {result.after_dataset.row_count}")
    if result.change_set is not None:
        print(f"Cells changed: {result.change_set.cells_changed}")
        print(f"Rows removed: {result.change_set.rows_removed}")
        print(f"Rows added: {result.change_set.rows_added}")
    print(f"Result JSON: {output_path}")


if __name__ == "__main__":
    main()
