import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data_agent.clustering import load_input
from data_agent.clustering.core import run_kmeans, scale_features, select_features
from data_agent.experiments.clustering import compare_clustering
from data_agent.experiments.detection import detect_injected_errors
from data_agent.experiments.injection import ERROR_TYPES, evaluate_injected_errors, inject_errors, load_injection, save_injection
from data_agent.experiments.quality import compare_quality

ROOT = Path(__file__).resolve().parents[3]


class M3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = load_input("1", project_root=ROOT)
        cls.after = load_input("5", project_root=ROOT)
        cls.baseline = load_input("2", project_root=ROOT)
        cls.summaries = [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in
                         ("logs/checks/off_52k_v1_quality_summary.json", "logs/checks_v5/off_52k_v5_quality_summary.json")]
        cls.baseline_summaries = [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in
                                  ("logs/checks/off_52k_v1_quality_summary.json", "logs/checks_v2/off_52k_v2_quality_summary.json")]

    def test_quality_metrics_and_missing_or_failed_check(self):
        result = compare_quality(self.before, self.after, *self.summaries)
        self.assertEqual(result["completeness"]["semantic_missing_including_physical_cells"]["absolute_change"], 173016-297441)
        self.assertEqual(result["completeness"]["physical_null_cells"]["before"], int(self.before.frame.isna().sum().sum()))
        self.assertNotEqual(result["completeness"]["physical_null_cells"]["before"], result["completeness"]["semantic_missing_including_physical_cells"]["before"])
        self.assertEqual(result["uniqueness"]["redundant_rows"]["before"], 2000)
        self.assertEqual(result["range_validity"]["affected_rows"]["status"], "unavailable")
        self.assertEqual(result["range_validity"]["range_flagged_cells"]["after"], 0)
        self.assertEqual(result["field_consistency"]["affected_rows_union"]["after"], 0)
        self.assertEqual(result["retention"]["common_row_ids"], 50000)
        self.assertAlmostEqual(result["retention"]["retained_before_fraction"], 50000/52000)
        self.assertEqual(result["modifications"]["status"], "unavailable")
        baseline = compare_quality(
            self.before, self.baseline, *self.baseline_summaries,
            change_path=ROOT / "data/processed/change.off_52k_v2_conservative.changes.parquet",
        )
        self.assertEqual(baseline["modifications"]["deleted_row_ids"], 2000)
        changed = copy.deepcopy(self.summaries[1])
        changed["checks"]["missing"]["status"] = "failed"
        changed["checks"]["missing"]["summary"]["missing_cells"] = 0
        failed = compare_quality(self.before, self.after, self.summaries[0], changed)
        self.assertEqual(failed["completeness"]["semantic_missing_including_physical_cells"]["status"], "unavailable")
        self.assertIsNone(failed["completeness"]["semantic_missing_including_physical_cells"]["after"])
        self.assertEqual(failed["modifications"]["status"], "unavailable")
        del changed["checks"]["outliers"]
        self.assertEqual(compare_quality(self.before, self.after, self.summaries[0], changed)["range_validity"]["range_flagged_cells"]["status"], "unavailable")

    def injection(self, count=1):
        return inject_errors(self.after, counts={x: count for x in ERROR_TYPES}, seed=19,
                             sample_row_ids=self.after.frame[self.after.row_id_column].iloc[:500].tolist())

    def test_injection_reproducible_and_ground_truth(self):
        original = self.after.frame.copy(deep=True)
        a, b = self.injection(2), self.injection(2)
        pd.testing.assert_frame_equal(a.frame, b.frame)
        self.assertEqual(a.truth, b.truth)
        self.assertEqual(a.universe, b.universe)
        pd.testing.assert_frame_equal(original, self.after.frame)
        self.assertEqual(len(a.truth), 10)
        self.assertEqual(len(a.universe), 20)
        self.assertTrue(a.frame[self.after.row_id_column].is_unique)
        indexed = a.frame.set_index(self.after.row_id_column)
        for item in a.truth:
            self.assertEqual(item["source_sha256"], self.after.ref.sha256)
            if item["error_type"] == "missing":
                self.assertTrue(pd.isna(indexed.loc[item["row_id"], item["column"]]))
            elif item["error_type"] == "duplicate":
                self.assertNotEqual(item["row_id"], item["original_row_id"])
                self.assertIsInstance(item["original_value"], dict)
                self.assertEqual(indexed.loc[item["row_id"], "code"], indexed.loc[item["original_row_id"], "code"])
            else:
                self.assertEqual(indexed.loc[item["row_id"], item["column"]], item["injected_value"])
            if item["error_type"] == "conflict":
                self.assertGreater(indexed.loc[item["row_id"], item["column"]], indexed.loc[item["row_id"], "fat_100g"])
        with self.assertRaises(ValueError):
            self.injection(1000)
        with tempfile.TemporaryDirectory() as tmp:
            _, truth_path = save_injection(a, tmp)
            reloaded = load_injection(truth_path)
            self.assertEqual(reloaded.truth, a.truth)
            self.assertEqual(reloaded.universe, a.universe)
            self.assertEqual(reloaded.frame[self.after.row_id_column].tolist(), a.frame[self.after.row_id_column].tolist())
            format_item = next(x for x in a.truth if x["error_type"] == "format")
            reloaded_index = reloaded.frame.set_index(self.after.row_id_column)
            self.assertEqual(reloaded_index.loc[format_item["row_id"], format_item["column"]], format_item["injected_value"])
            with self.assertRaises(FileExistsError):
                save_injection(a, tmp)

    def test_detector_and_repair_metrics(self):
        injection = self.injection(1)
        detections = {}
        for kind in ERROR_TYPES:
            positive = next(x for x in injection.universe if x["error_type"] == kind and x["is_error"])
            detections[kind] = {"status": "success", "positions": [positive, positive]}
        negative = next(x for x in injection.universe if x["error_type"] == "missing" and not x["is_error"])
        detections["missing"]["positions"].append(negative)
        truth = [x for x in injection.truth if x["error_type"] in ("range", "format")]
        repairs = [{"error_type": x["error_type"], "row_id": x["row_id"], "column": x["column"], "new_value": x["expected_repair"] + n} for x,n in zip(truth, (1, 2))]
        repairs.append({"error_type": negative["error_type"], "row_id": negative["row_id"], "column": negative["column"], "new_value": 9})
        result = evaluate_injected_errors(injection, detections, repairs=repairs)
        self.assertEqual((result["micro"]["tp"], result["micro"]["fp"], result["micro"]["fn"]), (5, 1, 0))
        self.assertAlmostEqual(result["micro"]["precision"]["value"], 5/6)
        self.assertAlmostEqual(result["micro"]["f1"]["value"], 10/11)
        self.assertEqual(result["repair"]["mae"], 1.5)
        self.assertAlmostEqual(result["repair"]["rmse"], np.sqrt(2.5))
        self.assertEqual(result["repair"]["untreated_rate"]["value"], 3/5)
        self.assertEqual(result["repair"]["false_modification_rate"]["value"], 1/5)
        self.assertEqual(evaluate_injected_errors(injection, detections)["repair"]["status"], "unavailable")
        detections["range"]["status"] = "failed"
        self.assertEqual(evaluate_injected_errors(injection, detections)["micro"]["status"], "unavailable")

    def test_upstream_detectors_execute(self):
        injection = self.injection(1)
        with tempfile.TemporaryDirectory() as tmp:
            results = detect_injected_errors(injection, self.after, Path(tmp) / "selections")
            evaluated = evaluate_injected_errors(injection, results)
            self.assertEqual(evaluated["micro"]["status"], "available")
            self.assertEqual(evaluated["micro"]["tp"], 5)
            _, truth_path = save_injection(injection, tmp)
            loaded = load_injection(truth_path)
            rechecked = detect_injected_errors(loaded, self.after, Path(tmp) / "reloaded_selections")
            self.assertEqual(evaluate_injected_errors(loaded, rechecked)["micro"], evaluated["micro"])

    def groups(self):
        frame = self.after.frame.iloc[:30].copy()
        frame["fat_100g"] = [1+i*.01 for i in range(15)] + [15+i*.01 for i in range(15)]
        frame["proteins_100g"] = [2+i*.01 for i in range(15)] + [20+i*.01 for i in range(15)]
        data = replace(self.after, frame=frame)
        selected = select_features(data, ["fat_100g", "proteins_100g"])
        scaled = scale_features(selected)
        original = run_kmeans(scaled, 2, n_init=2)
        record = {"silhouette": .5, "silhouette_sampling": {"sample_size": 30}, "calinski_harabasz": 3., "davies_bouldin": .4, "stability": {"mean_ari": 1.}}
        group1 = {"selected": selected, "scaled": scaled, "result": original, "record": record,
                  "labels": {42: original.labels, 7: original.labels.copy()}, "metrics": {}, "source_path": "one", "metrics_sha256": "a", "labels_sha256": "c", "input_rows": 30}
        indexes = list(range(24, -1, -1))
        second = replace(selected, frame=selected.frame.iloc[indexes].reset_index(drop=True), row_ids=tuple(selected.row_ids[i] for i in indexes))
        scaled2 = scale_features(second, scaler=scaled.scaler)
        flipped = 1-original.labels[indexes]
        other = replace(original, row_ids=second.row_ids, labels=flipped, matrix_sha256=scaled2.matrix_sha256, scaler_sha256=scaled2.scaler_sha256)
        group2 = {"selected": second, "scaled": scaled2, "result": other, "record": record,
                  "labels": {42: flipped, 7: flipped.copy()}, "metrics": {}, "source_path": "two", "metrics_sha256": "b", "labels_sha256": "d", "input_rows": 30}
        return group1, group2

    def test_clustering_common_ids_fairness_and_metrics(self):
        a, b = self.groups()
        one = compare_clustering([a])
        self.assertEqual(one["status"], "unavailable")
        result = compare_clustering([a, b], max_evaluation_rows=20)
        self.assertEqual(result["status"], "comparable")
        self.assertEqual(result["common_rows"], 25)
        self.assertEqual(result["evaluation_sample"]["count"], 20)
        self.assertAlmostEqual(result["cross_group_ari"][0]["ari"], 1)
        self.assertEqual(result["common_subset_metrics"][0]["silhouette_sampling"]["population_size"], 20)
        self.assertAlmostEqual(result["groups"][1]["retention"], 25/30)
        altered = dict(b, scaled=replace(b["scaled"], scaler_sha256="different"))
        self.assertEqual(compare_clustering([a, altered])["status"], "unavailable")
        altered = dict(b, scaled=replace(b["scaled"], features=("different",)))
        self.assertEqual(compare_clustering([a, altered])["status"], "unavailable")
        self.assertEqual(compare_clustering([a, a])["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
