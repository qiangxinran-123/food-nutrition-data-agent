import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data_agent.io import sha256_file
from data_agent.clustering.analysis import analyze_clusters, load_m1_result
from data_agent.clustering.core import DEFAULT_FEATURES, run_kmeans, scale_features, select_features
from data_agent.clustering.dataset_adapter import load_input
from data_agent.clustering.visualization import render_visualizations, visualize_clusters

ROOT = Path(__file__).resolve().parents[3]


class M2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real = load_input("5", project_root=ROOT)
        cls.historical = load_input("2", project_root=ROOT)

    def fixture(self):
        frame = self.real.frame.iloc[:7].copy()
        frame["__row_id"] = [f"id{i}" for i in range(7)]
        for i, name in enumerate(DEFAULT_FEATURES):
            frame[name] = np.array([1., 2., 3., 20., 21., 22., 1000.]) * (i + 1)
        frame["product_name"] = ["A", None, "C", "D", "E", "F", "singleton"]
        frame["categories_en"] = ["Fruit, Snacks", "en:null", None, "Drinks, Fruit", "Drinks", "Drinks", "unknown"]
        frame["categories"] = [None, "Fallback, Other", None, None, None, None, "unknown"]
        data = replace(self.real, frame=frame)
        selected = select_features(data)
        scaled = scale_features(selected)
        result = run_kmeans(scaled, 3, n_init=3)
        return data, scaled, result

    def metrics(self, scaled):
        return {"dataset_id": scaled.dataset_id, "dataset_version": scaled.version, "data_sha256": scaled.data_sha256,
                "matrix_sha256": scaled.matrix_sha256, "scaler_sha256": scaled.scaler_sha256,
                "results": [{"k": k, "silhouette": .3, "silhouette_sampling": {"sample_size": 5, "population_size": 7, "sampled": True, "seed": 42}, "calinski_harabasz": 5., "davies_bouldin": 1.} for k in range(2, 9)]}

    def test_statistics_representatives_and_raw_units(self):
        data, scaled, result = self.fixture()
        before = data.frame.copy(deep=True)
        # Shuffled result and source verify explicit row ID alignment.
        shuffled = replace(result, row_ids=result.row_ids[::-1], labels=result.labels[::-1])
        data = replace(data, frame=data.frame.sample(frac=1, random_state=3))
        original = data.frame.copy(deep=True)
        analysis = analyze_clusters(data, scaled, shuffled, representatives=7)
        self.assertEqual(sum(c["count"] for c in analysis["clusters"]), 7)
        self.assertTrue(any(c["count"] == 1 for c in analysis["clusters"]))
        lookup = dict(zip(result.row_ids, result.labels))
        for cluster in analysis["clusters"]:
            rows = before[before["__row_id"].map(lookup) == cluster["label"]]
            for feature in DEFAULT_FEATURES:
                stat = cluster["nutrition"][feature]
                self.assertAlmostEqual(stat["mean"], rows[feature].mean())
                self.assertAlmostEqual(stat["median"], rows[feature].median())
                self.assertEqual(stat["unit"], "kcal/100g" if feature == "energy-kcal_100g" else "g/100g")
            categories = {x["category"]: x["count"] for x in cluster["categories"]}
            self.assertEqual(sum(categories.values()), cluster["count"])
            positions = np.flatnonzero(result.labels == cluster["label"])
            distances = np.linalg.norm(scaled.matrix.to_numpy()[positions] - result.centers[cluster["label"]], axis=1)
            expected = sorted(zip(distances, positions), key=lambda x: (x[0], scaled.row_ids[x[1]]))[:7]
            self.assertEqual([r["row_id"] for r in cluster["representatives"]], [scaled.row_ids[p] for _, p in expected])
            for rep in cluster["representatives"]:
                self.assertEqual(lookup[rep["row_id"]], cluster["label"])
        all_categories = {x["category"] for c in analysis["clusters"] for x in c["categories"]}
        self.assertTrue({"Fruit", "Fallback", "(missing)", "Drinks"}.issubset(all_categories))
        self.assertNotIn("Snacks", all_categories)
        foods = {rep["row_id"]: rep for cluster in analysis["clusters"] for rep in cluster["representatives"]}
        self.assertIsNone(foods["id1"]["product_name"])
        self.assertEqual(foods["id1"]["category"], "Fallback")
        self.assertEqual(foods["id2"]["category"], "(missing)")
        with self.assertRaises(ValueError):
            analyze_clusters(data, scaled, replace(result, row_ids=("different",) + result.row_ids[1:]))
        pd.testing.assert_frame_equal(original, data.frame)

    def test_visualization_and_regeneration(self):
        data, scaled, result = self.fixture()
        analysis = analyze_clusters(data, scaled, result)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "plots"
            target.mkdir()
            payload = visualize_clusters(scaled, result, analysis, self.metrics(scaled), target, sample_size=5)
            coordinates = pd.read_csv(target / "pca_projection.csv")
            self.assertEqual(len(coordinates), 5)
            self.assertEqual(set(coordinates.cluster), set(result.labels))
            labels = dict(zip(result.row_ids, result.labels))
            self.assertTrue(all(labels[row.row_id] == row.cluster for row in coordinates.itertuples()))
            self.assertEqual([x["k"] for x in payload["k_metrics"]], list(range(2, 9)))
            self.assertEqual(payload["k_metrics"][0]["silhouette_sampling"]["sample_size"], 5)
            self.assertEqual(len(payload["pca"]["components"]), 2)
            for row in coordinates.itertuples():
                pos = scaled.row_ids.index(row.row_id)
                expected = (scaled.matrix.to_numpy()[pos] - np.asarray(payload["pca"]["mean"])) @ np.asarray(payload["pca"]["components"]).T
                np.testing.assert_allclose([row.pc1, row.pc2], expected, atol=1e-12)
            for name in ("pca_projection.png", "cluster_heatmap.png", "cluster_sizes.png", "k_metrics.png"):
                self.assertGreater((target / name).stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                visualize_clusters(scaled, result, analysis, self.metrics(scaled), target)
            regenerated = Path(tmp) / "regenerated"
            regenerated.mkdir()
            render_visualizations(target / "plot_data.json", target / "pca_projection.csv", regenerated)
            self.assertTrue((regenerated / "k_metrics.png").is_file())

    def test_reuse_existing_m1_without_writes(self):
        path = ROOT / "outputs/clustering/m1_20261029_run03"
        names = ["metrics.json", "labels_k5.csv", "scaler.json"]
        before = {name: sha256_file(path / name) for name in names}
        selected, scaled, result, metrics = load_m1_result(self.historical, path, 5)
        self.assertEqual(len(selected.row_ids), 50000)
        self.assertEqual(len(result.labels), 50000)
        self.assertEqual(metrics["matrix_sha256"], scaled.matrix_sha256)
        self.assertEqual(before, {name: sha256_file(path / name) for name in names})


if __name__ == "__main__":
    unittest.main()
