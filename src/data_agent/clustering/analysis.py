"""Descriptive analysis of an existing M1 result; no model refitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_agent.io import sha256_file
from .core import ClusterResult, ScaledFeatures, _validate_scaled, load_scaler, scale_features, select_features
from .dataset_adapter import DatasetInput


def load_m1_result(data: DatasetInput, run_dir: str | Path, k: int):
    """Verify M1 files, reconstruct scaled input and its primary-seed result."""
    directory = Path(run_dir).resolve()
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    needed = ["scaler.json", f"labels_k{k}.csv"]
    for name in needed:
        expected = metrics.get("artifact_sha256", {}).get(name)
        if not expected or sha256_file(directory / name) != expected:
            raise ValueError(f"M1 artifact hash mismatch or missing hash: {name}")
    if (metrics["dataset_id"], metrics["dataset_version"], metrics["data_sha256"]) != (data.ref.dataset_id, data.ref.version, data.ref.sha256):
        raise ValueError("M1 input dataset identity mismatch")
    selected = select_features(data, metrics["features"])
    scaled = scale_features(selected, scaler=load_scaler(directory / "scaler.json"))
    if (scaled.matrix_sha256, scaled.scaler_sha256) != (metrics["matrix_sha256"], metrics["scaler_sha256"]):
        raise ValueError("reconstructed matrix or scaler differs from M1")
    matches = [r for r in metrics["results"] if r["k"] == k]
    if len(matches) != 1:
        raise ValueError(f"M1 results must contain exactly one result for k={k}")
    record = matches[0]
    table = pd.read_csv(directory / f"labels_k{k}.csv", dtype={"row_id": str})
    seed_col = f"seed_{record['random_state']}"
    if "row_id" not in table or seed_col not in table or table.row_id.isna().any() or table.row_id.duplicated().any() or set(table.row_id) != set(scaled.row_ids):
        raise ValueError("M1 label row IDs do not match selected input")
    ordered = table.set_index("row_id").loc[list(scaled.row_ids), seed_col]
    if ordered.isna().any() or not pd.api.types.is_integer_dtype(ordered.dtype):
        raise ValueError("invalid M1 labels")
    labels = ordered.to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=k) if (labels >= 0).all() else []
    if len(counts) != k or counts.tolist() != [c["count"] for c in record["clusters"]]:
        raise ValueError("M1 labels and cluster counts disagree")
    centers = np.asarray(record["centers_standardized"], dtype=float)
    if centers.shape != (k, len(scaled.features)) or not np.isfinite(centers).all():
        raise ValueError("invalid M1 cluster centers")
    result = ClusterResult(scaled.row_ids, labels, centers, record["inertia"], k, record["random_state"],
                           record["n_init"], record["max_iter"], record["tol"], record["iterations"],
                           scaled.matrix_sha256, scaled.scaler_sha256, scaled.dataset_id, scaled.version,
                           scaled.data_sha256, scaled.features, record["status"])
    return selected, scaled, result, metrics


def _category(row):
    for col in ("categories_en", "categories"):
        value = row.get(col)
        if pd.isna(value):
            continue
        # One first nonempty, meaningful category per food; no multi-counting.
        for part in str(value).split(","):
            part = " ".join(part.split())
            if part and part.casefold() not in {"unknown", "en:null", "null", "nan"}:
                return part
    return "(missing)"


def analyze_clusters(data: DatasetInput, scaled: ScaledFeatures, result: ClusterResult, *, representatives: int = 3) -> dict:
    _validate_scaled(scaled)
    if representatives < 1:
        raise ValueError("representatives must be positive")
    if (data.ref.dataset_id, data.ref.version, data.ref.sha256) != (scaled.dataset_id, scaled.version, scaled.data_sha256):
        raise ValueError("source data and scaled dataset differ")
    if (result.dataset_id, result.version, result.data_sha256, result.features, result.matrix_sha256, result.scaler_sha256) != (scaled.dataset_id, scaled.version, scaled.data_sha256, scaled.features, scaled.matrix_sha256, scaled.scaler_sha256):
        raise ValueError("result and scaled input differ")
    ids = data.frame[data.row_id_column]
    if ids.isna().any() or ids.duplicated().any() or len(result.labels) != len(result.row_ids) or len(set(result.row_ids)) != len(result.row_ids) or set(result.row_ids) != set(scaled.row_ids):
        raise ValueError("invalid row IDs or label count")
    source = data.frame.set_index(data.row_id_column, drop=False)
    if not set(scaled.row_ids).issubset(source.index):
        raise ValueError("model row IDs missing from source")
    raw = source.loc[list(scaled.row_ids)].copy()
    labels = pd.Series(result.labels, index=result.row_ids).loc[list(scaled.row_ids)].to_numpy()
    if (not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or np.any(labels >= result.k) or
        result.centers.shape != (result.k, len(scaled.features)) or not np.isfinite(result.centers).all() or
        result.status != "success" or len(np.unique(labels)) != result.k):
        raise ValueError("invalid or degenerate clustering result")
    original_values = raw.loc[:, list(scaled.features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.allclose(scaled.scaler.transform(pd.DataFrame(original_values, columns=scaled.features)), scaled.matrix.to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError("source nutrition values do not match scaled matrix")
    specs = {c.name: c for c in data.schema.columns}
    total_means = np.mean(original_values, axis=0)
    categories = raw.apply(_category, axis=1).to_numpy()
    clusters = []
    for label in range(result.k):
        positions = np.flatnonzero(labels == label)
        values = original_values[positions]
        means, medians = np.mean(values, axis=0), np.median(values, axis=0)
        distances = np.linalg.norm(scaled.matrix.to_numpy()[positions] - result.centers[label], axis=1)
        chosen = sorted(zip(distances, positions), key=lambda pair: (pair[0], scaled.row_ids[pair[1]]))[:representatives]
        foods = []
        for distance, pos in chosen:
            name = raw.iloc[pos].get("product_name")
            foods.append({"row_id": scaled.row_ids[pos], "product_name": None if pd.isna(name) or not str(name).strip() else str(name),
                          "category": categories[pos], "distance_standardized": float(distance),
                          "nutrition": {f: float(original_values[pos, i]) for i, f in enumerate(scaled.features)}})
        counts = pd.Series(categories[positions]).value_counts(sort=False)
        distribution = sorted(({"category": str(name), "count": int(count), "proportion": float(count / len(positions))} for name, count in counts.items()), key=lambda x: (-x["count"], x["category"]))
        stats = {f: {"unit": specs[f].unit, "mean": float(means[i]), "median": float(medians[i]),
                     "mean_minus_dataset_mean": float(means[i] - total_means[i])} for i, f in enumerate(scaled.features)}
        clusters.append({"label": label, "count": len(positions), "proportion": len(positions)/len(labels),
                         "nutrition": stats, "representatives": foods, "categories": distribution})
    return {"dataset_id": scaled.dataset_id, "dataset_version": scaled.version, "data_sha256": scaled.data_sha256,
            "matrix_sha256": scaled.matrix_sha256, "scaler_sha256": scaled.scaler_sha256,
            "features": list(scaled.features), "k": result.k, "random_state": result.random_state, "n_init": result.n_init,
            "max_iter": result.max_iter, "tol": result.tol, "sample_count": len(labels),
            "category_rule": "First nonempty comma-separated categories_en value; fallback categories; ignore unknown/en:null/null; otherwise (missing). One category per row.",
            "representative_rule": "Nearest Euclidean distance to cluster center in M1 standardized feature space; row ID breaks ties.",
            "clusters": clusters}
