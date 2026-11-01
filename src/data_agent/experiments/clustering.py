"""Compare M1 outputs on explicitly aligned common samples."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from data_agent.clustering.analysis import load_m1_result
from data_agent.clustering.core import evaluate_clusters, scale_features
from data_agent.io import sha256_file


def load_cluster_group(data, run_dir, k):
    selected, scaled, result, metrics = load_m1_result(data, run_dir, k)
    directory = Path(run_dir).resolve()
    record = next(r for r in metrics["results"] if r["k"] == k)
    seeds = list(record["stability"]["seeds"])
    if not seeds or len(seeds) != len(set(seeds)) or seeds[0] != record["random_state"]:
        raise ValueError("invalid recorded seed sequence")
    table = pd.read_csv(directory / f"labels_k{k}.csv", dtype={"row_id": str}).set_index("row_id").loc[list(scaled.row_ids)]
    label_map = {}
    for seed in seeds:
        column = f"seed_{seed}"
        if column not in table or table[column].isna().any() or not pd.api.types.is_integer_dtype(table[column]):
            raise ValueError(f"missing or invalid labels for seed {seed}")
        labels = table[column].to_numpy(dtype=int)
        if np.any(labels < 0) or np.any(labels >= k):
            raise ValueError("invalid cluster label")
        label_map[seed] = labels
    return {"selected": selected, "scaled": scaled, "result": result, "metrics": metrics, "record": record,
            "labels": label_map, "source_path": str(directory), "metrics_sha256": sha256_file(directory / "metrics.json"),
            "labels_sha256": metrics["artifact_sha256"][f"labels_k{k}.csv"], "input_rows": len(data.frame),
            "parent": data.manifest.get("parent")}


def compare_clustering(groups, *, max_evaluation_rows=2000, sample_seed=42):
    """Re-evaluate the same deterministic common subset using M1 metrics.

    Full-population metrics are descriptive only. No ranking is produced.
    """
    if not groups:
        raise ValueError("at least one cluster group is required")
    if not isinstance(max_evaluation_rows, int) or max_evaluation_rows < 2:
        raise ValueError("max_evaluation_rows must be at least 2")
    group_info = []
    for g in groups:
        s, r, rec = g["scaled"], g["result"], g["record"]
        group_info.append({"source_path": g["source_path"], "metrics_sha256": g["metrics_sha256"], "labels_sha256": g["labels_sha256"],
                           "dataset_id": s.dataset_id, "version": s.version, "data_sha256": s.data_sha256,
                           "input_rows": g["input_rows"], "clustered_rows": len(s.row_ids),
                           "retention": len(s.row_ids)/g["input_rows"] if g["input_rows"] else None,
                           "features": list(s.features), "scaler_sha256": s.scaler_sha256, "k": r.k,
                           "seeds": list(g["labels"]), "n_init": r.n_init, "max_iter": r.max_iter, "tol": r.tol,
                           "algorithm": rec.get("algorithm"), "algorithm_variant": rec.get("algorithm_variant"), "parent": g.get("parent"),
                           "full_population_metrics_descriptive_only": {key: rec[key] for key in ("silhouette", "silhouette_sampling", "calinski_harabasz", "davies_bouldin", "stability")}})
    result = {"groups": group_info, "status": "unavailable", "reason": None,
              "common_rows": None, "common_subset_metrics": None, "cross_group_ari": None,
              "interpretation": "Cluster metrics do not establish data truth or repair correctness; no ranking."}
    if len(groups) < 2:
        result["reason"] = "At least two independent existing clustering runs are required."
        return result
    if len({g["source_path"] for g in groups}) != len(groups) or len({g["metrics_sha256"] for g in groups}) != len(groups):
        result["reason"] = "Repeated or identical source artifacts are not independent comparison groups."
        return result
    base = group_info[0]
    for key in ("features", "scaler_sha256", "k", "seeds", "n_init", "max_iter", "tol", "algorithm", "algorithm_variant"):
        if any(g[key] != base[key] for g in group_info[1:]):
            result["reason"] = f"Incompatible {key}; no fair comparison calculated."
            return result
    def identity(g):
        return (g["dataset_id"], g["data_sha256"])
    def parent(g):
        p = g.get("parent") or {}
        return (p.get("dataset_id"), p.get("sha256")) if p.get("dataset_id") and p.get("sha256") else None
    for i in range(len(group_info)):
        for j in range(i + 1, len(group_info)):
            a, b = group_info[i], group_info[j]
            if identity(a) != identity(b) and identity(a) != parent(b) and identity(b) != parent(a) and not (parent(a) and parent(a) == parent(b)):
                result["reason"] = "Dataset lineage cannot establish shared row ID origin."
                return result
    common = sorted(set.intersection(*(set(g["scaled"].row_ids) for g in groups)))
    result["common_rows"] = len(common)
    for g in group_info:
        g["common_coverage_of_clustered"] = len(common)/g["clustered_rows"] if g["clustered_rows"] else None
    if len(common) <= base["k"]:
        result["reason"] = "Insufficient common row IDs for evaluation."
        return result
    full_aligned = []
    for g in groups:
        ids = g["scaled"].row_ids
        full_aligned.append({seed: pd.Series(values, index=ids).loc[common].to_numpy() for seed, values in g["labels"].items()})
    reference_seed = base["seeds"][0]
    result["cross_group_ari"] = [{"left": i, "right": j, "seed": reference_seed,
                                 "ari": float(adjusted_rand_score(full_aligned[i][reference_seed], full_aligned[j][reference_seed])),
                                 "sample_count": len(common)} for i in range(len(groups)) for j in range(i+1, len(groups))]
    # One shared deterministic subset covers every cluster in every recorded run.
    rng = np.random.default_rng(sample_seed)
    chosen = set()
    for aligned in full_aligned:
        for labels in aligned.values():
            if len(np.unique(labels)) != base["k"]:
                result["reason"] = "A cluster is absent from the common sample; common metrics unavailable."
                return result
            for label in range(base["k"]):
                chosen.add(int(rng.choice(np.flatnonzero(labels == label))))
    target = min(max_evaluation_rows, len(common))
    if len(chosen) > target or target <= base["k"]:
        result["reason"] = "Evaluation subset is too small to cover all clusters."
        return result
    remaining = np.array(sorted(set(range(len(common))) - chosen), dtype=int)
    chosen.update(map(int, rng.choice(remaining, target - len(chosen), replace=False)))
    indexes = sorted(chosen)
    ids = [common[i] for i in indexes]
    calculated = []
    for group, aligned in zip(groups, full_aligned):
        positions = {row_id: i for i, row_id in enumerate(group["scaled"].row_ids)}
        row_positions = [positions[row_id] for row_id in ids]
        selected = replace(group["selected"], frame=group["selected"].frame.iloc[row_positions].reset_index(drop=True), row_ids=tuple(ids))
        scaled = scale_features(selected, scaler=group["scaled"].scaler)
        runs = []
        for seed, labels in aligned.items():
            runs.append(replace(group["result"], row_ids=tuple(ids), labels=labels[indexes], random_state=seed,
                                matrix_sha256=scaled.matrix_sha256, scaler_sha256=scaled.scaler_sha256))
        # M1 reads inertia from the result object; on a subset it must not retain
        # the full training-set sum. Only the primary run's centers are persisted.
        residual = scaled.matrix.to_numpy() - runs[0].centers[runs[0].labels]
        runs[0] = replace(runs[0], inertia=float(np.sum(residual * residual)))
        calculated.append(evaluate_clusters(scaled, runs[0], stability_runs=runs[1:], sample_size=target+1, sample_seed=sample_seed))
    result.update(status="comparable", reason=None, common_subset_metrics=calculated,
                  evaluation_sample={"count": target, "common_population": len(common), "seed": sample_seed,
                                     "method": "one per cluster for each group and seed, then uniform without replacement",
                                     "row_ids": ids, "all_metrics_scope": "identical common evaluation subset"})
    return result
