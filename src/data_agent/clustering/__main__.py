"""Run a reproducible K-Means sweep in a new output directory."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .core import DEFAULT_FEATURES, evaluate_clusters, run_kmeans, save_scaler, scale_features, select_features
from .dataset_adapter import load_input
from data_agent.io import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True, help="new run directory; must not exist")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 21])
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--m1-run", type=Path, help="reuse an existing M1 run instead of training")
    parser.add_argument("--analysis-k", type=int, choices=range(2, 9), help="explicitly choose k for M2 descriptive analysis")
    args = parser.parse_args()
    if args.m1_run and args.analysis_k is None:
        parser.error("--m1-run requires --analysis-k")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("provide at least two distinct seeds")
    root = args.project_root.resolve()
    data = load_input("5", project_root=root)
    if args.m1_run:
        from .analysis import load_m1_result
        _, scaled, result, metrics = load_m1_result(data, args.m1_run, args.analysis_k)
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        run_m2(data, scaled, result, metrics, output, args.m1_run.resolve())
        return
    selected = select_features(data)
    scaled = scale_features(selected)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_scaler(scaled, output / "scaler.json")
    selected.exclusions.to_csv(output / "excluded.csv", index=False)
    metrics = []
    analysis_result = None
    for k in range(2, 9):
        runs = [run_kmeans(scaled, k, random_state=seed, n_init=args.n_init, max_iter=args.max_iter) for seed in args.seeds]
        if k == args.analysis_k:
            analysis_result = runs[0]
        evaluation = evaluate_clusters(scaled, runs[0], stability_runs=runs[1:], sample_size=args.sample_size, sample_seed=42)
        labels = pd.DataFrame({"row_id": scaled.row_ids})
        for run in runs:
            labels[f"seed_{run.random_state}"] = run.labels
        labels.to_csv(output / f"labels_k{k}.csv", index=False)
        record = {"k": k, "status": runs[0].status, "algorithm": "KMeans", "algorithm_variant": "lloyd", "random_state": runs[0].random_state,
                  "n_init": runs[0].n_init, "max_iter": runs[0].max_iter, "tol": runs[0].tol, "iterations": runs[0].iterations,
                  "centers_standardized": runs[0].centers.tolist(), **evaluation}
        metrics.append(record)
        print(f"k={k} silhouette={record['silhouette']:.5f} CH={record['calinski_harabasz']:.3f} DB={record['davies_bouldin']:.5f} mean ARI={record['stability']['mean_ari']:.5f}", flush=True)
    files = sorted(p for p in output.iterdir() if p.is_file())
    payload = {"dataset_id": data.ref.dataset_id, "dataset_version": data.ref.version, "data_sha256": data.ref.sha256,
               "profile_path": str(data.profile_path), "manifest_path": str(data.manifest_path), "data_path": str(data.data_path),
               "row_id_column": data.row_id_column, "features": list(DEFAULT_FEATURES), "input_rows": selected.input_count,
               "valid_rows": len(selected.row_ids), "excluded_rows": selected.exclusion_counts,
               "matrix_sha256": scaled.matrix_sha256, "scaler_sha256": scaled.scaler_sha256,
               "scaler_file": "scaler.json", "python_version": platform.python_version(), "pandas_version": pd.__version__,
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__, "results": metrics,
               "artifact_sha256": {p.name: sha256_file(p) for p in files}}
    with (output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if analysis_result is not None:
        run_m2(data, scaled, analysis_result, payload, output, output)
    print(f"dataset={data.ref.dataset_id} version={data.ref.version} valid={len(selected.row_ids)}/{selected.input_count} seeds={args.seeds} output={output}")


def run_m2(data, scaled, result, metrics, output, m1_dir):
    from .analysis import analyze_clusters
    from .visualization import visualize_clusters
    analysis = analyze_clusters(data, scaled, result)
    analysis["m1_run"] = str(m1_dir)
    analysis["m1_metrics_sha256"] = sha256_file(m1_dir / "metrics.json")
    with (output / "cluster_analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2, allow_nan=False)
    visualize_clusters(scaled, result, analysis, metrics, output)
    print(f"M2 dataset={scaled.dataset_id} k={result.k} rows={len(scaled.row_ids)} output={output}", flush=True)


if __name__ == "__main__":
    main()
