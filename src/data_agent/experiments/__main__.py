"""M3 reproducible demonstration using existing v1/v2 and one existing M1 run."""
import argparse
import json
from pathlib import Path

import numpy as np

from data_agent.clustering import load_input
from data_agent.io import sha256_file
from .clustering import compare_clustering, load_cluster_group
from .detection import detect_injected_errors
from .injection import evaluate_injected_errors, inject_errors, save_injection
from .quality import compare_quality


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True, help="new directory, never overwrite")
    parser.add_argument("--m1-run", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--count-per-type", type=int, default=5)
    args = parser.parse_args()
    root = args.project_root.resolve()
    before, after = load_input("1", project_root=root), load_input("2", project_root=root)
    paths = [root / "logs/checks/off_52k_v1_quality_summary.json", root / "logs/checks_v2/off_52k_v2_quality_summary.json"]
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    quality = compare_quality(before, after, *summaries, change_path=root / "data/processed/change.off_52k_v2_conservative.changes.parquet")
    quality["summary_sources"] = [{"path": str(p), "sha256": sha256_file(p)} for p in paths]
    if not 0 < args.sample_size <= len(after.frame):
        parser.error("sample-size must be positive and no greater than v2 rows")
    positions = np.sort(np.random.default_rng(args.seed).choice(len(after.frame), size=args.sample_size, replace=False))
    ids = after.frame.iloc[positions][after.row_id_column].tolist()
    injection = inject_errors(after, counts={x: args.count_per_type for x in ("missing", "duplicate", "range", "format", "conflict")}, seed=args.seed, sample_row_ids=ids)
    group = load_cluster_group(after, args.m1_run, args.k)
    comparison = compare_clustering([group])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "quality_comparison.json", quality)
    save_injection(injection, output)
    detections = detect_injected_errors(injection, after, output / "detector_selections")
    write_json(output / "detection_results.json", detections)
    evaluation = evaluate_injected_errors(injection, detections)
    write_json(output / "injection_evaluation.json", evaluation)
    write_json(output / "clustering_comparison.json", comparison)
    artifacts = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "manifest.json", {"source_versions": quality["datasets"], "seed": args.seed,
                                          "injection_id": injection.injection_id, "configuration": injection.config,
                                          "m1_run": str(Path(args.m1_run).resolve()), "output_dir": str(output), "artifacts": artifacts})
    print(f"M3 output={output}; injected={len(injection.truth)}; detection={evaluation['micro']['status']}; clustering={comparison['status']}")


if __name__ == "__main__":
    main()
