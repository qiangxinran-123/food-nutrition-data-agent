"""Deterministic, traceable feature preparation and K-Means evaluation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .dataset_adapter import DatasetInput

DEFAULT_FEATURES = ("energy-kcal_100g", "proteins_100g", "fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g", "fiber_100g", "sodium_100g")


@dataclass
class SelectedFeatures:
    frame: pd.DataFrame
    row_ids: tuple[str, ...]
    features: tuple[str, ...]
    dataset_id: str
    version: str
    data_sha256: str
    input_count: int
    exclusion_counts: dict[str, int]
    exclusions: pd.DataFrame


@dataclass
class ScaledFeatures:
    matrix: pd.DataFrame
    row_ids: tuple[str, ...]
    features: tuple[str, ...]
    dataset_id: str
    version: str
    data_sha256: str
    scaler: StandardScaler
    matrix_sha256: str
    scaler_sha256: str


@dataclass
class ClusterResult:
    row_ids: tuple[str, ...]
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    k: int
    random_state: int
    n_init: int
    max_iter: int
    tol: float
    iterations: int
    matrix_sha256: str
    scaler_sha256: str
    dataset_id: str
    version: str
    data_sha256: str
    features: tuple[str, ...]
    status: str


def _matrix_hash(matrix: pd.DataFrame, row_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256(matrix.to_numpy(dtype='<f8').tobytes())
    digest.update(json.dumps(row_ids, ensure_ascii=False).encode('utf-8'))
    return digest.hexdigest()


def _validate_scaled(scaled: ScaledFeatures) -> None:
    if (scaled.matrix.shape != (len(scaled.row_ids), len(scaled.features)) or
        tuple(scaled.matrix.columns) != scaled.features or len(set(scaled.row_ids)) != len(scaled.row_ids) or
        not np.isfinite(scaled.matrix.to_numpy(dtype=float)).all() or
        _matrix_hash(scaled.matrix, scaled.row_ids) != scaled.matrix_sha256 or
        hashlib.sha256(json.dumps(_scaler_payload(scaled.scaler, scaled.features), sort_keys=True).encode()).hexdigest() != scaled.scaler_sha256):
        raise ValueError("scaled matrix, row mapping or scaler has changed")


def select_features(data: DatasetInput, features=DEFAULT_FEATURES) -> SelectedFeatures:
    features = tuple(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be nonempty and unique")
    for pair in (("energy_100g", "energy-kcal_100g"), ("salt_100g", "sodium_100g")):
        if set(pair).issubset(features):
            raise ValueError(f"redundant features: {pair}")
    schema = {item.name: item for item in data.schema.columns}
    missing = [x for x in features if x not in data.frame or x not in schema]
    if missing:
        raise ValueError(f"missing feature columns or metadata: {missing}")
    forbidden = {"code", "__row_id", "__source_row", "nutriscore_grade", data.row_id_column, data.schema.source_row_column}
    for name in features:
        spec = schema[name]
        if name in forbidden or spec.logical_type.value != "numeric" or spec.role in {"label", "row_id", "source_row", "entity_id"}:
            raise ValueError(f"illegal non-numeric or identifier feature: {name}")
        if spec.unit not in {"g/100g", "kcal/100g", "kJ/100g"}:
            raise ValueError(f"feature has unclear or incompatible unit: {name}")
    ids = data.frame[data.row_id_column]
    if ids.isna().any() or ids.duplicated().any() or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("row IDs must be unique nonempty strings")
    original = data.frame.loc[:, list(features)]
    numeric = original.apply(pd.to_numeric, errors="coerce")
    if any(pd.api.types.is_complex_dtype(numeric[c]) or pd.api.types.is_bool_dtype(original[c]) or
           pd.api.types.is_datetime64_any_dtype(original[c]) or pd.api.types.is_timedelta64_dtype(original[c]) for c in features):
        raise ValueError("boolean, complex and datetime values cannot safely represent nutrition measurements")
    numeric = numeric.astype(float)
    # Each excluded row gets one reason, in priority order, for additive counts.
    null = original.isna().any(axis=1).to_numpy()
    invalid = (numeric.isna() & original.notna()).any(axis=1).to_numpy() & ~null
    infinite = np.isinf(numeric.to_numpy()).any(axis=1) & ~null & ~invalid
    reasons = np.full(len(original), "", dtype=object)
    reasons[null], reasons[invalid], reasons[infinite] = "missing", "non_numeric", "infinite"
    valid = reasons == ""
    exclusions = pd.DataFrame({"row_id": ids.to_numpy()[~valid], "reason": reasons[~valid]})
    counts = {reason: int((reasons == reason).sum()) for reason in ("missing", "non_numeric", "infinite")}
    return SelectedFeatures(numeric.loc[valid].reset_index(drop=True), tuple(ids.to_numpy()[valid]), features,
                            data.ref.dataset_id, data.ref.version, data.ref.sha256, len(original), counts, exclusions)


def _scaler_payload(scaler: StandardScaler, features: tuple[str, ...]) -> dict:
    return {"type": "StandardScaler", "features": list(features), "mean": scaler.mean_.tolist(),
            "variance": scaler.var_.tolist(), "scale": scaler.scale_.tolist(),
            "n_samples_seen": np.asarray(scaler.n_samples_seen_).tolist()}


def save_scaler(scaled: ScaledFeatures, path: str | Path) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(_scaler_payload(scaled.scaler, scaled.features), handle, indent=2, allow_nan=False)


def load_scaler(path: str | Path) -> StandardScaler:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") != "StandardScaler":
        raise ValueError("unsupported scaler")
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["mean"], dtype=float)
    scaler.var_ = np.asarray(payload["variance"], dtype=float)
    scaler.scale_ = np.asarray(payload["scale"], dtype=float)
    scaler.n_samples_seen_ = np.asarray(payload["n_samples_seen"])
    scaler.feature_names_in_ = np.asarray(payload["features"], dtype=object)
    scaler.n_features_in_ = len(scaler.feature_names_in_)
    if any(len(x) != scaler.n_features_in_ or not np.isfinite(x).all() for x in (scaler.mean_, scaler.var_, scaler.scale_)) or (scaler.scale_ <= 0).any():
        raise ValueError("invalid scaler parameters")
    return scaler


def scale_features(selected: SelectedFeatures, *, scaler: StandardScaler | None = None) -> ScaledFeatures:
    frame = selected.frame
    if tuple(frame.columns) != selected.features or frame.shape != (len(selected.row_ids), len(selected.features)) or not len(frame):
        raise ValueError("feature columns, dimensions or row IDs do not match")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("features must be finite")
    if scaler is None:
        if (frame.nunique() <= 1).any():
            raise ValueError("constant feature columns cannot be fitted")
        scaler = StandardScaler().fit(frame)
    elif tuple(getattr(scaler, "feature_names_in_", ())) != selected.features:
        raise ValueError("scaler feature names or order differ")
    if (np.asarray(scaler.var_) <= 0).any():
        raise ValueError("scaler contains constant feature")
    matrix = pd.DataFrame(scaler.transform(frame), columns=selected.features)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("scaled features must be finite")
    scaler_hash = hashlib.sha256(json.dumps(_scaler_payload(scaler, selected.features), sort_keys=True).encode()).hexdigest()
    return ScaledFeatures(matrix, selected.row_ids, selected.features, selected.dataset_id, selected.version,
                          selected.data_sha256, scaler, _matrix_hash(matrix, selected.row_ids), scaler_hash)


def run_kmeans(scaled: ScaledFeatures, k: int | range = range(2, 9), *, random_state: int = 42,
               n_init: int = 10, max_iter: int = 300, tol: float = 1e-4) -> ClusterResult | list[ClusterResult]:
    if not isinstance(k, int):
        if not isinstance(k, (range, list, tuple)):
            raise ValueError("k must be an integer or a sequence of integers")
        return [run_kmeans(scaled, item, random_state=random_state, n_init=n_init, max_iter=max_iter, tol=tol) for item in k]
    n = len(scaled.matrix)
    if isinstance(k, bool) or not 2 <= k < n:
        raise ValueError("k must be an integer in [2, sample_count)")
    _validate_scaled(scaled)
    if not isinstance(random_state, int) or not 0 <= random_state <= 2**32 - 1 or not isinstance(n_init, int) or n_init < 1 or not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("invalid K-Means parameters")
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter, tol=tol, algorithm="lloyd")
    with threadpool_limits(limits=1):
        labels = model.fit_predict(scaled.matrix)
    status = "success" if len(np.unique(labels)) == k else "degenerate"
    return ClusterResult(scaled.row_ids, labels.copy(), model.cluster_centers_.copy(), float(model.inertia_), k, random_state,
                         n_init, max_iter, tol, int(model.n_iter_), scaled.matrix_sha256, scaled.scaler_sha256,
                         scaled.dataset_id, scaled.version, scaled.data_sha256, scaled.features, status)


def evaluate_clusters(scaled: ScaledFeatures, result: ClusterResult, *, stability_runs: list[ClusterResult] | None = None,
                      sample_size: int = 2000, sample_seed: int = 42) -> dict:
    _validate_scaled(scaled)
    def align(run):
        if (run.matrix_sha256 != scaled.matrix_sha256 or run.scaler_sha256 != scaled.scaler_sha256 or
            (run.dataset_id, run.version, run.data_sha256, run.features, run.k) !=
            (scaled.dataset_id, scaled.version, scaled.data_sha256, scaled.features, result.k)):
            raise ValueError("runs must use the same input, features, scaler and k")
        if (run.n_init, run.max_iter, run.tol) != (result.n_init, result.max_iter, result.tol):
            raise ValueError("stability runs must use identical K-Means parameters except random_state")
        if len(run.row_ids) != len(run.labels) or len(set(run.row_ids)) != len(run.row_ids) or set(run.row_ids) != set(scaled.row_ids):
            raise ValueError("run row IDs do not match input")
        if not np.issubdtype(run.labels.dtype, np.integer) or np.any(run.labels < 0) or np.any(run.labels >= result.k):
            raise ValueError("invalid cluster labels")
        lookup = dict(zip(run.row_ids, run.labels))
        return np.asarray([lookup[row_id] for row_id in scaled.row_ids])
    labels = align(result)
    if result.status != "success" or len(np.unique(labels)) != result.k or len(labels) <= result.k:
        raise ValueError("degenerate clustering: metrics unavailable")
    runs = [result] + list(stability_runs or [])
    aligned = [labels] + [align(run) for run in runs[1:]]
    if any(run.status != "success" or len(np.unique(a)) != result.k for run, a in zip(runs, aligned)):
        raise ValueError("degenerate stability run")
    n = len(labels)
    if not isinstance(sample_size, int) or sample_size <= result.k:
        raise ValueError("sample_size must exceed number of clusters")
    sample_count = min(n, sample_size)
    rng = np.random.default_rng(sample_seed)
    # Seed with one member of each cluster, then sample remaining rows uniformly.
    guaranteed = np.array([rng.choice(np.flatnonzero(labels == c)) for c in np.unique(labels)], dtype=int)
    remaining = np.setdiff1d(np.arange(n), guaranteed)
    chosen = np.sort(np.concatenate([guaranteed, rng.choice(remaining, sample_count - len(guaranteed), replace=False)]))
    sizes = np.bincount(labels, minlength=result.k)
    with threadpool_limits(limits=1):
        silhouette = silhouette_score(scaled.matrix.iloc[chosen], labels[chosen])
        ch = calinski_harabasz_score(scaled.matrix, labels)
        db = davies_bouldin_score(scaled.matrix, labels)
    pairs = [{"seed_a": runs[i].random_state, "seed_b": runs[j].random_state,
              "ari": float(adjusted_rand_score(aligned[i], aligned[j]))} for i, j in combinations(range(len(runs)), 2)]
    scores = [p["ari"] for p in pairs]
    return {"silhouette": float(silhouette), "silhouette_sampling": {"method": "full" if sample_count == n else "one per cluster then uniform without replacement",
            "sample_size": sample_count, "population_size": n, "seed": sample_seed, "sampled": sample_count < n},
            "calinski_harabasz": float(ch), "davies_bouldin": float(db), "inertia": result.inertia,
            "clusters": [{"label": int(i), "count": int(count), "proportion": float(count / n)} for i, count in enumerate(sizes)],
            "stability": {"seeds": [r.random_state for r in runs], "pairs": pairs,
                          "mean_ari": float(np.mean(scores)) if scores else None, "std_ari": float(np.std(scores)) if scores else None}}
