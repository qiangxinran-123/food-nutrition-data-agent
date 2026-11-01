from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if (PROJECT_ROOT / ".deps").exists():
    sys.path.insert(0, str(PROJECT_ROOT / ".deps"))

from data_agent.checks.missing import MissingCheckOptions, check_missing
from data_agent.io import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing-value detection.")
    parser.add_argument(
        "--profile",
        default=str(PROJECT_ROOT / "config" / "profiles" / "off_52k_v1.json"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "logs" / "checks" / "off_52k_v1_missing.json"),
    )
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(
        args.profile,
        project_root=PROJECT_ROOT,
        verify_hash=not args.skip_hash_check,
    )
    result = check_missing(
        dataset,
        MissingCheckOptions(sample_limit=args.sample_limit),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Dataset: {dataset.ref.dataset_id}")
    print(f"Shape: {dataset.frame.shape[0]} rows x {dataset.frame.shape[1]} columns")
    print(f"Status: {result.status.value}")
    print(f"Missing cells: {result.summary.get('missing_cells')}")
    print(f"Rows with missing: {result.summary.get('rows_with_missing')}")
    print(f"Columns with missing: {result.summary.get('columns_with_missing')}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
