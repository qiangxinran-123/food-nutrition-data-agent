"""Reproducible static M1/M2 cluster visuals and their underlying plot data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .core import _validate_scaled


def visualize_clusters(scaled, result, analysis, metrics, output_dir, *, sample_size=5000, sample_seed=42):
    _validate_scaled(scaled)
    if (result.matrix_sha256 != scaled.matrix_sha256 or result.scaler_sha256 != scaled.scaler_sha256 or
        analysis["matrix_sha256"] != scaled.matrix_sha256 or analysis["scaler_sha256"] != scaled.scaler_sha256 or analysis["k"] != result.k or
        (metrics.get("dataset_id"), metrics.get("dataset_version"), metrics.get("data_sha256"), metrics.get("matrix_sha256"), metrics.get("scaler_sha256")) !=
        (scaled.dataset_id, scaled.version, scaled.data_sha256, scaled.matrix_sha256, scaled.scaler_sha256) or
        len(result.row_ids) != len(result.labels) or len(set(result.row_ids)) != len(result.row_ids) or
        set(result.row_ids) != set(scaled.row_ids)):
        raise ValueError("visualization inputs do not match")
    labels = pd.Series(result.labels, index=result.row_ids).loc[list(scaled.row_ids)].to_numpy()
    unique = np.unique(labels)
    if not isinstance(sample_size, int) or sample_size < len(unique) or len(unique) != result.k:
        raise ValueError("sample size or clusters invalid")
    if len(scaled.features) < 2 or len(labels) < 2:
        raise ValueError("PCA requires at least two samples and features")
    counts = [int(np.sum(labels == i)) for i in range(result.k)]
    if counts != [c["count"] for c in analysis["clusters"]]:
        raise ValueError("analysis counts do not match labels")
    rows = sorted(metrics["results"], key=lambda x: x["k"])
    if [r["k"] for r in rows] != list(range(2, 9)):
        raise ValueError("M1 metrics must contain k=2 through 8")
    target = Path(output_dir)
    if not target.is_dir():
        raise ValueError("output directory must already exist")
    names = ["pca_projection.csv", "plot_data.json", "pca_projection.png", "cluster_heatmap.png", "cluster_sizes.png", "k_metrics.png"]
    if any((target / name).exists() for name in names):
        raise FileExistsError("visualization output already exists")
    n = len(labels)
    sample_count = min(sample_size, n)
    rng = np.random.default_rng(sample_seed)
    guaranteed = np.array([rng.choice(np.flatnonzero(labels == c)) for c in unique], dtype=int)
    remaining = np.setdiff1d(np.arange(n), guaranteed)
    chosen = np.sort(np.concatenate((guaranteed, rng.choice(remaining, sample_count - len(guaranteed), replace=False))))
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(scaled.matrix.to_numpy())
    projected = pca.transform(scaled.matrix.to_numpy()[chosen])
    with (target / "pca_projection.csv").open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame({"row_id": np.asarray(scaled.row_ids)[chosen], "cluster": labels[chosen], "pc1": projected[:, 0], "pc2": projected[:, 1]}).to_csv(handle, index=False)
    means = np.array([[c["nutrition"][f]["mean"] for f in scaled.features] for c in analysis["clusters"]])
    heat = (means - scaled.scaler.mean_) / scaled.scaler.scale_
    curve = [{"k": r["k"], "silhouette": r["silhouette"], "silhouette_sampling": r["silhouette_sampling"],
              "calinski_harabasz": r["calinski_harabasz"], "davies_bouldin": r["davies_bouldin"]} for r in rows]
    plot_data = {"matrix_sha256": scaled.matrix_sha256, "k": result.k, "features": list(scaled.features),
                 "pca": {"fitted_rows": n, "sample_rows": sample_count, "sample_seed": sample_seed,
                         "sampling": "one per cluster, then uniform without replacement", "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                         "components": pca.components_.tolist(), "mean": pca.mean_.tolist(), "coordinates_file": "pca_projection.csv"},
                 "heatmap": {"meaning": "cluster mean in M1 scaler standardized units", "values": heat.tolist(), "color_clipped_to": [-3, 3]},
                 "cluster_counts": counts, "k_metrics": curve}
    with (target / "plot_data.json").open("x", encoding="utf-8") as handle:
        json.dump(plot_data, handle, ensure_ascii=False, indent=2, allow_nan=False)

    render_visualizations(target / "plot_data.json", target / "pca_projection.csv", target)
    return plot_data


def render_visualizations(plot_data_path, coordinates_path, output_dir):
    """Regenerate the four figures using only the saved JSON and CSV."""
    plot_data = json.loads(Path(plot_data_path).read_text(encoding="utf-8"))
    coordinates = pd.read_csv(coordinates_path, dtype={"row_id": str})
    target = Path(output_dir)
    if not target.is_dir():
        raise ValueError("output directory must already exist")
    names = ["pca_projection.png", "cluster_heatmap.png", "cluster_sizes.png", "k_metrics.png"]
    if any((target / name).exists() for name in names):
        raise FileExistsError("visualization output already exists")
    k = plot_data["k"]
    features = plot_data["features"]
    counts = plot_data["cluster_counts"]
    pca_info = plot_data["pca"]
    explained = pca_info["explained_variance_ratio"]
    projected = coordinates[["pc1", "pc2"]].to_numpy()
    sampled_labels = coordinates["cluster"].to_numpy()
    if len(coordinates) != pca_info["sample_rows"] or coordinates.row_id.duplicated().any():
        raise ValueError("invalid saved PCA coordinates")
    heat = np.asarray(plot_data["heatmap"]["values"])
    curve = plot_data["k_metrics"]
    n = pca_info["fitted_rows"]
    sample_count = pca_info["sample_rows"]

    def save(fig, name):
        with (target / name).open("xb") as handle:
            fig.savefig(handle, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.get_cmap("tab10")
    for label in range(k):
        mask = sampled_labels == label
        ax.scatter(projected[mask, 0], projected[mask, 1], s=12, alpha=.65, color=palette(label % 10), label=f"Cluster {label} (n={counts[label]:,})")
    ax.set(title=f"PCA projection, k={k}: {sample_count:,} of {n:,} rows", xlabel=f"PC1 ({100*explained[0]:.1f}% variance)", ylabel=f"PC2 ({100*explained[1]:.1f}% variance)")
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=.2)
    save(fig, "pca_projection.png")

    fig, ax = plt.subplots(figsize=(12, max(4, .65 * k + 2)))
    im = ax.imshow(heat, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(features)), features, rotation=35, ha="right")
    ax.set_yticks(range(k), [f"Cluster {i}" for i in range(k)])
    ax.set_title("Cluster means in M1 standardized units (color saturates at ±3)")
    for i in range(k):
        for j in range(len(features)):
            ax.text(j, i, f"{heat[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(heat[i,j]) > 1.7 else "black")
    fig.colorbar(im, ax=ax, label="Standard deviations from reference mean")
    save(fig, "cluster_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(k), counts, color=[palette(i % 10) for i in range(k)])
    ax.set_yscale("log"); ax.set_ylim(.7, max(counts) * 4)
    ax.set_xticks(range(k)); ax.set(xlabel="Cluster label", ylabel="Number of foods (log scale)", title=f"Cluster sizes, k={k} (all {n:,} rows)")
    for bar, count in zip(bars, counts):
        ax.annotate(f"{count:,}", (bar.get_x()+bar.get_width()/2, count), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    save(fig, "cluster_sizes.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ("silhouette", "calinski_harabasz", "davies_bouldin"), ("Silhouette (M1 sample)", "Calinski–Harabasz (all rows)", "Davies–Bouldin (all rows)")):
        ax.plot([r["k"] for r in curve], [r[key] for r in curve], marker="o"); ax.set(title=title, xlabel="k", ylabel=title.split(" (")[0]); ax.set_xticks(range(2, 9)); ax.grid(alpha=.25)
    sampling = curve[0]["silhouette_sampling"]
    sampling_text = (f"{sampling['sample_size']:,}/{sampling['population_size']:,} rows, sampled={sampling['sampled']}, seed={sampling['seed']}"
                     if all(r["silhouette_sampling"] == sampling for r in curve) else "sampling varies by k; see plot_data.json")
    fig.suptitle(f"M1 metrics; silhouette: {sampling_text}")
    fig.tight_layout()
    save(fig, "k_metrics.png")
