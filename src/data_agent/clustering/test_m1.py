import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

from data_agent.clustering.dataset_adapter import load_input
from data_agent.clustering.core import (DEFAULT_FEATURES, evaluate_clusters, load_scaler, run_kmeans,
                                        save_scaler, scale_features, select_features)

ROOT = Path(__file__).resolve().parents[3]


class M1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_input("2", project_root=ROOT)

    def small_data(self):
        frame = self.data.frame.iloc[:8].copy()
        frame["__row_id"] = [f"r{i}" for i in range(8)]
        frame["fat_100g"] = [1., 2., 1., 2., 10., 11., 10., 11.]
        frame["proteins_100g"] = [1., 1., 2., 2., 10., 10., 11., 11.]
        return replace(self.data, frame=frame)

    def selected(self):
        return select_features(self.small_data(), ["fat_100g", "proteins_100g"])

    def test_default_and_read_only_real(self):
        before = self.data.frame.copy(deep=True)
        result = select_features(self.data)
        self.assertEqual(result.features, DEFAULT_FEATURES)
        self.assertEqual(result.input_count, len(before))
        self.assertEqual(len(result.row_ids) + sum(result.exclusion_counts.values()), len(before))
        self.assertTrue(np.isfinite(result.frame.to_numpy()).all())
        self.assertEqual(list(result.row_ids), before.loc[before["__row_id"].isin(result.row_ids), "__row_id"].tolist())
        pd.testing.assert_frame_equal(before, self.data.frame)

    def test_feature_constraints(self):
        for features in (["missing"], ["code"], ["__row_id"], ["__source_row"], ["nutriscore_grade"],
                         ["energy_100g", "energy-kcal_100g"], ["salt_100g", "sodium_100g"], ["serving_quantity"]):
            with self.subTest(features=features), self.assertRaises(ValueError):
                select_features(self.data, features)

    def test_exclusions_and_alignment(self):
        data = self.small_data()
        data.frame["fat_100g"] = data.frame["fat_100g"].astype(object)
        data.frame.loc[data.frame.index[0], "fat_100g"] = np.nan
        data.frame.loc[data.frame.index[1], "fat_100g"] = "not a number"
        data.frame.loc[data.frame.index[2], "fat_100g"] = np.inf
        selected = select_features(data, ["fat_100g", "proteins_100g"])
        self.assertEqual(selected.exclusion_counts, {"missing": 1, "non_numeric": 1, "infinite": 1})
        self.assertEqual(selected.row_ids, ("r3", "r4", "r5", "r6", "r7"))
        self.assertEqual(selected.exclusions.reason.tolist(), ["missing", "non_numeric", "infinite"])

    def test_scaler_roundtrip_and_validation(self):
        selected = self.selected()
        original = selected.frame.copy()
        scaled = scale_features(selected)
        np.testing.assert_allclose(scaled.matrix.mean(axis=0), [0, 0], atol=1e-14)
        np.testing.assert_allclose(scaled.matrix.std(axis=0, ddof=0), [1, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scaler.json"
            save_scaler(scaled, path)
            with self.assertRaises(FileExistsError):
                save_scaler(scaled, path)
            transformed = scale_features(selected, scaler=load_scaler(path))
            np.testing.assert_allclose(scaled.matrix, transformed.matrix)
            self.assertEqual(scaled.scaler_sha256, transformed.scaler_sha256)
            reversed_selected = replace(selected, frame=selected.frame.iloc[:, ::-1], features=selected.features[::-1])
            with self.assertRaisesRegex(ValueError, "order"):
                scale_features(reversed_selected, scaler=load_scaler(path))
        pd.testing.assert_frame_equal(original, selected.frame)
        with self.assertRaisesRegex(ValueError, "constant"):
            scale_features(replace(selected, frame=selected.frame.assign(fat_100g=1.0)))
        with self.assertRaisesRegex(ValueError, "finite"):
            scale_features(replace(selected, frame=selected.frame.assign(fat_100g=np.inf)))

    def test_kmeans_and_metrics(self):
        scaled = scale_features(self.selected())
        for k in (0, 1, 8, 9):
            with self.subTest(k=k), self.assertRaises(ValueError):
                run_kmeans(scaled, k)
        a = run_kmeans(scaled, 2, random_state=42, n_init=3)
        b = run_kmeans(scaled, 2, random_state=42, n_init=3)
        np.testing.assert_array_equal(a.labels, b.labels)
        self.assertEqual(len(a.labels), len(scaled.row_ids))
        other = run_kmeans(scaled, 2, random_state=7, n_init=3)
        result = evaluate_clusters(scaled, a, stability_runs=[other], sample_size=100)
        self.assertAlmostEqual(result["silhouette"], silhouette_score(scaled.matrix, a.labels))
        self.assertAlmostEqual(result["calinski_harabasz"], calinski_harabasz_score(scaled.matrix, a.labels))
        self.assertAlmostEqual(result["davies_bouldin"], davies_bouldin_score(scaled.matrix, a.labels))
        self.assertEqual(sum(c["count"] for c in result["clusters"]), 8)
        self.assertEqual(result["stability"]["mean_ari"], adjusted_rand_score(a.labels, other.labels))
        swapped = replace(a, labels=1-a.labels, random_state=11)
        self.assertAlmostEqual(evaluate_clusters(scaled, a, stability_runs=[swapped])["stability"]["mean_ari"], 1.0)
        shuffled = replace(swapped, row_ids=swapped.row_ids[::-1], labels=swapped.labels[::-1])
        self.assertAlmostEqual(evaluate_clusters(scaled, a, stability_runs=[shuffled])["stability"]["mean_ari"], 1.0)
        with self.assertRaises(ValueError):
            evaluate_clusters(scaled, replace(a, status="degenerate", labels=np.zeros(8, dtype=int)))
        with self.assertRaises(ValueError):
            evaluate_clusters(scaled, a, stability_runs=[replace(other, row_ids=("foreign",) + other.row_ids[1:])])
        with self.assertRaisesRegex(ValueError, "parameters"):
            evaluate_clusters(scaled, a, stability_runs=[replace(other, n_init=99)])
        changed = replace(scaled, matrix=scaled.matrix.copy())
        changed.matrix.iloc[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "changed"):
            run_kmeans(changed, 2)
        self.assertEqual([x.k for x in run_kmeans(scaled, range(2, 4), n_init=2)], [2, 3])

    def test_sample_is_deterministic_and_covers_clusters(self):
        scaled = scale_features(self.selected())
        result = run_kmeans(scaled, 2, n_init=2)
        a = evaluate_clusters(scaled, result, sample_size=5)
        b = evaluate_clusters(scaled, result, sample_size=5)
        self.assertEqual(a, b)
        self.assertTrue(a["silhouette_sampling"]["sampled"])
        self.assertEqual(a["silhouette_sampling"]["sample_size"], 5)

    def test_actual_degenerate_run_is_rejected(self):
        selected = self.selected()
        selected.frame.iloc[:4, :] = [1.0, 2.0]
        selected.frame.iloc[4:, :] = [10.0, 12.0]
        scaled = scale_features(selected)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_kmeans(scaled, 3, n_init=2)
        self.assertEqual(result.status, "degenerate")
        with self.assertRaisesRegex(ValueError, "degenerate"):
            evaluate_clusters(scaled, result)


if __name__ == "__main__":
    unittest.main()
