import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_agent.cleaning import (
    CLEANERS,
    CleaningConfirmationError,
    CleaningError,
    build_plan,
    clean_duplicates,
    clean_missing,
    clean_outliers,
    execute_plan,
    make_decision,
    mitigate_imbalance,
    normalize_types_formats,
    repair_labels,
    resolve_conflicts,
)
from data_agent.cleaning.strategies import get_strategy, validate_strategy_params
from data_agent.config import ColumnConfig, DatasetConfig, LabelColumnPolicy, LabelPolicy
from data_agent.contracts import (
    CleaningPlan,
    DataFormat,
    DatasetRef,
    IssueType,
    LogicalType,
)
from data_agent.io import LoadedDataset, load_dataset
from data_agent.profiling import build_schema_profile


def _selection(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


class CleaningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.processed = self.root / "processed"
        self.artifacts = self.root / "artifacts"
        self.frame = pd.DataFrame(
            {
                "code": ["A", "A", "B", "B", "C"],
                "energy": [10.0, None, 20.0, 20.0, 50.0],
                "grade": ["a", "a", "b", "b", "x"],
                "quality": [1, 3, 2, 4, 5],
                "__row_id": [f"r{i}" for i in range(1, 6)],
                "__source_row": [2, 3, 4, 5, 6],
            }
        )
        self.config = DatasetConfig(
            dataset_id="demo_v1",
            manifest="demo.manifest.json",
            row_id_column="__row_id",
            source_row_column="__source_row",
            primary_key_candidates=[["code"]],
            columns={
                "code": ColumnConfig(
                    role="entity_id",
                    logical_type=LogicalType.IDENTIFIER,
                    nullable=False,
                    unique=True,
                ),
                "energy": ColumnConfig(logical_type=LogicalType.NUMERIC),
                "grade": ColumnConfig(
                    role="label",
                    logical_type=LogicalType.CATEGORICAL,
                    allowed_values=["a", "b", "c"],
                ),
                "quality": ColumnConfig(logical_type=LogicalType.NUMERIC),
                "__row_id": ColumnConfig(
                    role="row_id",
                    logical_type=LogicalType.IDENTIFIER,
                    nullable=False,
                    unique=True,
                ),
                "__source_row": ColumnConfig(
                    role="source_row",
                    logical_type=LogicalType.IDENTIFIER,
                    nullable=False,
                    unique=True,
                ),
            },
            labels=LabelPolicy(
                columns={"grade": LabelColumnPolicy(unavailable_tokens=["unknown"])}
            ),
        )
        self.ref = DatasetRef(
            dataset_id="demo_v1",
            version="1",
            uri="demo.parquet",
            format=DataFormat.PARQUET,
            row_count=len(self.frame),
            column_count=len(self.frame.columns),
        )
        self.schema = build_schema_profile(self.ref, self.frame, self.config)
        self.dataset = LoadedDataset(
            ref=self.ref,
            schema=self.schema,
            frame=self.frame,
            manifest={},
            manifest_path=self.root / "demo.manifest.json",
            data_path=self.root / "demo.parquet",
            config=self.config,
            config_path=self.root / "config" / "profiles" / "demo_v1.json",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, function, decision, selection, **kwargs):
        return function(
            self.dataset,
            decision,
            selection_table=selection,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=False,
            **kwargs,
        )

    def test_missing_statistic_imputation_writes_new_version(self) -> None:
        decision = make_decision(
            issue_type=IssueType.MISSING,
            strategy_id="impute_statistic",
            finding_id="missing.energy",
            params={"method": "median"},
            target_columns=["energy"],
        )
        selection = _selection(
            [
                {
                    "finding_id": "missing.energy",
                    "row_index": 1,
                    "row_id": "r2",
                    "column": "energy",
                    "reason_code": "null",
                }
            ]
        )

        result = self._run(clean_missing, decision, selection)

        self.assertEqual(result.change_set.rows_removed, 0)
        self.assertEqual(result.change_set.cells_changed, 1)
        output = pd.read_parquet(self.processed / "demo_v2.parquet", engine="fastparquet")
        self.assertEqual(output.loc[1, "energy"], 20.0)
        manifest = json.loads(
            (self.processed / "demo_v2.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["parent"]["dataset_id"], "demo_v1")
        self.assertTrue(manifest["processing"]["cleaning_applied"])

    def test_duplicate_keep_first_drops_redundant_rows(self) -> None:
        decision = make_decision(
            issue_type=IssueType.DUPLICATE,
            strategy_id="keep_first",
            finding_id="duplicate.code",
        )
        selection = _selection(
            [
                {"group_id": "g1", "row_index": 0, "row_id": "r1", "group_rank": 1},
                {"group_id": "g1", "row_index": 1, "row_id": "r2", "group_rank": 2},
                {"group_id": "g2", "row_index": 2, "row_id": "r3", "group_rank": 1},
                {"group_id": "g2", "row_index": 3, "row_id": "r4", "group_rank": 2},
            ]
        )

        result = self._run(clean_duplicates, decision, selection)

        self.assertEqual(result.change_set.rows_removed, 2)
        output = pd.read_parquet(self.processed / "demo_v2.parquet", engine="fastparquet")
        self.assertEqual(output["code"].tolist(), ["A", "B", "C"])
        self.assertEqual(output["__row_id"].tolist(), ["r1", "r3", "r5"])

    def test_type_normalization_and_outlier_clipping(self) -> None:
        frame = self.frame.copy()
        frame.loc[0, "energy"] = 1000.0
        dataset = LoadedDataset(
            ref=self.ref.model_copy(update={"column_count": len(frame.columns)}),
            schema=build_schema_profile(self.ref, frame, self.config),
            frame=frame,
            manifest={},
            manifest_path=self.root / "demo.manifest.json",
            data_path=self.root / "demo.parquet",
            config=self.config,
        )
        decision = make_decision(
            issue_type=IssueType.OUTLIER,
            strategy_id="clip_value",
            finding_id="outlier.energy",
            params={"minimum": 0, "maximum": 100},
            target_columns=["energy"],
        )
        selection = _selection(
            [{"row_index": 0, "row_id": "r1", "column": "energy", "reason_code": "above_maximum"}]
        )
        result = clean_outliers(
            dataset,
            decision,
            selection_table=selection,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=False,
        )
        self.assertEqual(result.change_set.cells_changed, 1)
        output = pd.read_parquet(self.processed / "demo_v2.parquet", engine="fastparquet")
        self.assertEqual(output.loc[0, "energy"], 100.0)

        type_decision = make_decision(
            issue_type=IssueType.FORMAT,
            strategy_id="cast_numeric",
            finding_id="format.energy",
            target_columns=["energy"],
        )
        type_dataset = LoadedDataset(
            ref=self.ref,
            schema=self.schema,
            frame=self.frame.assign(energy=["10", "20", "30", "40", "50"]),
            manifest={},
            manifest_path=self.root / "demo.manifest.json",
            data_path=self.root / "demo.parquet",
            config=self.config,
        )
        type_selection = _selection(
            [{"row_index": 0, "row_id": "r1", "column": "energy", "reason_code": "unparseable_numeric"}]
        )
        normalize_types_formats(
            type_dataset,
            type_decision,
            selection_table=type_selection,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=False,
        )
        typed = pd.read_parquet(self.processed / "demo_v3.parquet", engine="fastparquet")
        self.assertEqual(typed.loc[0, "energy"], 10.0)

    def test_conflict_and_label_strategies_are_deterministic(self) -> None:
        conflict = make_decision(
            issue_type=IssueType.CONSTRAINT,
            strategy_id="trust_primary_source",
            finding_id="constraint.energy",
            params={"target_column": "energy", "source_column": "quality"},
        )
        row_selection = _selection([{"row_index": 0, "row_id": "r1"}])
        result = self._run(resolve_conflicts, conflict, row_selection)
        self.assertEqual(result.change_set.cells_changed, 1)
        output = pd.read_parquet(self.processed / "demo_v2.parquet", engine="fastparquet")
        self.assertEqual(output.loc[0, "energy"], 1.0)

        label = make_decision(
            issue_type=IssueType.LABEL,
            strategy_id="map_deterministic_value",
            finding_id="label.grade",
            params={"mapping": {"x": "c"}},
            target_columns=["grade"],
        )
        cell_selection = _selection(
            [{"row_index": 4, "row_id": "r5", "column": "grade", "reason_code": "invalid_label"}]
        )
        repair_labels(
            self.dataset,
            label,
            selection_table=cell_selection,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=False,
        )
        labelled = pd.read_parquet(self.processed / "demo_v3.parquet", engine="fastparquet")
        self.assertEqual(labelled.loc[4, "grade"], "c")

    def test_class_weight_and_strategy_guardrails(self) -> None:
        decision = make_decision(
            issue_type=IssueType.IMBALANCE,
            strategy_id="class_weight",
            finding_id="imbalance.grade",
            params={"target_column": "grade"},
        )
        result = mitigate_imbalance(
            self.dataset,
            decision,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=True,
        )
        self.assertEqual(result.status.value, "success")
        self.assertTrue(result.artifacts)
        weights = pd.read_parquet(result.artifacts[0].uri, engine="fastparquet")
        self.assertEqual(set(weights["class_label"]), {"a", "b"})

        unimplemented = make_decision(
            issue_type=IssueType.MISSING,
            strategy_id="impute_model",
            finding_id="missing.energy",
            params={"method": "knn"},
        )
        with self.assertRaises(CleaningError):
            clean_missing(self.dataset, unimplemented, selection_table=_selection([]))

        dangerous = make_decision(
            issue_type=IssueType.DUPLICATE,
            strategy_id="quarantine_group",
            finding_id="duplicate.code",
            params={"keep_candidate": True},
        )
        with self.assertRaises(CleaningConfirmationError):
            clean_duplicates(
                self.dataset,
                dangerous,
                selection_table=_selection(
                    [{"group_id": "g", "row_index": 0}, {"group_id": "g", "row_index": 1}]
                ),
                artifact_dir=self.artifacts,
                output_dir=self.processed,
                dry_run=False,
                confirm=False,
            )

    def test_processed_profile_can_reload_cleaned_dataset(self) -> None:
        decision = make_decision(
            issue_type=IssueType.MISSING,
            strategy_id="impute_statistic",
            finding_id="missing.energy",
            params={"method": "median"},
            target_columns=["energy"],
        )
        selection = _selection(
            [{"row_index": 1, "row_id": "r2", "column": "energy"}]
        )

        result = self._run(clean_missing, decision, selection)

        profile_path = Path(result.after_dataset.metadata["profile_uri"])
        reloaded = load_dataset(profile_path, project_root=self.root)
        manifest = json.loads(
            Path(result.after_dataset.metadata["manifest_uri"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(reloaded.ref.dataset_id, "demo_v2")
        self.assertEqual(reloaded.frame.loc[1, "energy"], 20.0)
        self.assertEqual(reloaded.ref.sha256, result.after_dataset.sha256)
        self.assertEqual(manifest["profile_file"]["dataset_id"], "demo_v2")
        self.assertEqual(manifest["parent"]["dataset_id"], "demo_v1")

    def test_plan_remaps_stable_row_ids_after_row_deletion(self) -> None:
        selections = self.root / "selections"
        selections.mkdir()
        _selection(
            [
                {"group_id": "A", "row_index": 0, "row_id": "r1", "group_rank": 1},
                {"group_id": "A", "row_index": 1, "row_id": "r2", "group_rank": 2},
                {"group_id": "B", "row_index": 2, "row_id": "r3", "group_rank": 1},
                {"group_id": "B", "row_index": 3, "row_id": "r4", "group_rank": 2},
            ]
        ).to_parquet(selections / "duplicates.parquet", engine="fastparquet", index=False)
        _selection(
            [{"row_index": 1, "row_id": "r2", "column": "energy"}]
        ).to_parquet(selections / "missing.parquet", engine="fastparquet", index=False)

        duplicate = make_decision(
            issue_type=IssueType.DUPLICATE,
            strategy_id="keep_last",
            finding_id="duplicate.code",
            selection_artifact_id="duplicates",
            decision_id="step.duplicate",
        )
        fill = make_decision(
            issue_type=IssueType.MISSING,
            strategy_id="fill_constant",
            finding_id="missing.energy",
            selection_artifact_id="missing",
            target_columns=["energy"],
            params={"value": 0.0},
            decision_id="step.fill",
        )
        plan = build_plan(
            self.dataset,
            [duplicate, fill],
            plan_id="demo_plan",
            dry_run=False,
        )

        result = execute_plan(
            self.dataset,
            plan,
            selection_artifact_dir=selections,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
        )

        self.assertEqual(result.status.value, "success")
        self.assertEqual(result.after_dataset.row_count, 3)
        self.assertEqual([item.after_dataset.dataset_id for item in result.results], ["demo_v2", "demo_v3"])
        final = pd.read_parquet(self.processed / "demo_v3.parquet", engine="fastparquet")
        self.assertEqual(final.loc[final["__row_id"] == "r2", "energy"].iloc[0], 0.0)
        self.assertTrue((self.processed / "demo_v3.manifest.json").exists())
        self.assertFalse((self.processed / "demo_v4.parquet").exists())

    def test_grouped_imputation_formula_derivation_and_quarantine(self) -> None:
        grouped = make_decision(
            issue_type=IssueType.MISSING,
            strategy_id="impute_grouped",
            finding_id="missing.energy",
            target_columns=["energy"],
            params={"method": "median", "group_columns": ["code"]},
        )
        selection = _selection(
            [{"row_index": 1, "row_id": "r2", "column": "energy"}]
        )
        result = clean_missing(
            self.dataset,
            grouped,
            selection_table=selection,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=True,
        )
        self.assertEqual(result.output_frame.loc[1, "energy"], 10.0)

        derived = make_decision(
            issue_type=IssueType.MISSING,
            strategy_id="derive_from_formula",
            finding_id="missing.energy",
            target_columns=["energy"],
            params={
                "target_column": "energy",
                "expression": "col(\"quality\") * 2",
            },
        )
        result = clean_missing(
            self.dataset,
            derived,
            selection_table=selection,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=True,
        )
        self.assertEqual(result.output_frame.loc[1, "energy"], 6.0)

        quarantine = make_decision(
            issue_type=IssueType.DUPLICATE,
            strategy_id="quarantine_group",
            finding_id="duplicate.code",
            params={"keep_candidate": True},
        )
        result = clean_duplicates(
            self.dataset,
            quarantine,
            selection_table=_selection(
                [
                    {"group_id": "A", "row_index": 0, "row_id": "r1", "group_rank": 1},
                    {"group_id": "A", "row_index": 1, "row_id": "r2", "group_rank": 2},
                ]
            ),
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=True,
            confirm=True,
        )
        self.assertEqual(result.output_frame["__row_id"].tolist(), ["r1", "r3", "r4", "r5"])
        quarantine_artifact = next(
            artifact
            for artifact in result.artifacts
            if artifact.kind.value == "row_list"
        )
        quarantined = pd.read_parquet(
            quarantine_artifact.uri, engine="fastparquet"
        )
        self.assertEqual(quarantined["__row_id_for_quarantine"].tolist(), ["r2"])

    def test_missing_imputation_does_not_overwrite_an_earlier_derivation(self) -> None:
        selection_dir = self.root / "idempotent_selections"
        selection_dir.mkdir()
        _selection(
            [
                {
                    "finding_id": "missing.energy",
                    "row_index": 1,
                    "row_id": "r2",
                    "column": "energy",
                }
            ]
        ).to_parquet(
            selection_dir / "selection.demo_v1.missing.energy.parquet",
            index=False,
            engine="fastparquet",
        )
        decisions = [
            make_decision(
                issue_type=IssueType.MISSING,
                strategy_id="derive_from_formula",
                finding_id="missing.energy",
                selection_artifact_id="selection.demo_v1.missing.energy",
                target_columns=["energy"],
                params={"target_column": "energy", "expression": 'col("quality") * 2'},
                decision_id="demo.missing.energy.derive",
            ),
            make_decision(
                issue_type=IssueType.MISSING,
                strategy_id="impute_statistic",
                finding_id="missing.energy",
                selection_artifact_id="selection.demo_v1.missing.energy",
                target_columns=["energy"],
                params={"method": "median"},
                decision_id="demo.missing.energy.median",
            ),
        ]
        plan = CleaningPlan(
            plan_id="demo_idempotent_plan",
            before_dataset=self.dataset.ref,
            decisions=decisions,
            dry_run=False,
            output_dataset_id="demo_v2",
        )

        result = execute_plan(
            self.dataset,
            plan,
            selection_artifact_dir=selection_dir,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            persist="final",
        )

        self.assertEqual(result.change_set.cells_changed, 1)
        output = pd.read_parquet(
            self.processed / "demo_v2.parquet", engine="fastparquet"
        )
        self.assertEqual(output.loc[output["__row_id"] == "r2", "energy"].iloc[0], 6.0)

    def test_final_plan_writes_one_version_and_remaps_selections(self) -> None:
        selection_dir = self.root / "plan_selections"
        selection_dir.mkdir()
        pd.DataFrame.from_records(
            [
                {
                    "finding_id": "duplicate.code",
                    "group_id": "A",
                    "row_index": 0,
                    "row_id": "r1",
                    "group_rank": 1,
                },
                {
                    "finding_id": "duplicate.code",
                    "group_id": "A",
                    "row_index": 1,
                    "row_id": "r2",
                    "group_rank": 2,
                },
            ]
        ).to_parquet(
            selection_dir / "selection.demo_v1.duplicate.code.parquet",
            index=False,
            engine="fastparquet",
        )
        pd.DataFrame.from_records(
            [
                {
                    "finding_id": "missing.energy",
                    "row_index": 1,
                    "row_id": "r2",
                    "column": "energy",
                    "reason_code": "null",
                }
            ]
        ).to_parquet(
            selection_dir / "selection.demo_v1.missing.energy.parquet",
            index=False,
            engine="fastparquet",
        )
        decisions = [
            make_decision(
                issue_type=IssueType.DUPLICATE,
                strategy_id="keep_last",
                finding_id="duplicate.code",
                selection_artifact_id="selection.demo_v1.duplicate.code",
                decision_id="demo.duplicate.keep_last",
            ),
            make_decision(
                issue_type=IssueType.MISSING,
                strategy_id="impute_statistic",
                finding_id="missing.energy",
                selection_artifact_id="selection.demo_v1.missing.energy",
                target_columns=["energy"],
                params={"method": "median"},
                decision_id="demo.missing.energy.median",
            ),
        ]
        plan = CleaningPlan(
            plan_id="demo_final_plan",
            before_dataset=self.dataset.ref,
            decisions=decisions,
            dry_run=False,
            output_dataset_id="demo_v2",
        )

        result = execute_plan(
            self.dataset,
            plan,
            selection_artifact_dir=selection_dir,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            persist="final",
        )

        self.assertEqual(result.after_dataset.dataset_id, "demo_v2")
        self.assertEqual(result.after_dataset.row_count, 4)
        self.assertIsNotNone(result.change_set)
        self.assertEqual(result.change_set.rows_removed, 1)
        self.assertEqual(result.change_set.cells_changed, 1)
        self.assertTrue((self.processed / "demo_v2.parquet").exists())
        self.assertFalse((self.processed / "demo_v3.parquet").exists())
        output = pd.read_parquet(
            self.processed / "demo_v2.parquet", engine="fastparquet"
        )
        self.assertEqual(output["__row_id"].tolist(), ["r2", "r3", "r4", "r5"])
        self.assertEqual(output.loc[output["__row_id"] == "r2", "energy"].iloc[0], 20.0)
        changes = pd.read_parquet(
            result.change_set.changes_uri, engine="fastparquet"
        )
        self.assertIn("drop", set(changes["change_type"]))
        self.assertIn("impute", set(changes["change_type"]))
        pd.testing.assert_frame_equal(self.dataset.frame, self.frame)

        manifest = json.loads(
            (self.processed / "demo_v2.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["parent"]["dataset_id"], "demo_v1")
        self.assertEqual(manifest["base_file"]["row_count"], 4)
        reloaded = load_dataset(
            manifest["profile_file"]["path"],
            project_root=self.root,
            verify_hash=True,
        )
        self.assertEqual(reloaded.ref.dataset_id, "demo_v2")
        self.assertEqual(len(reloaded.frame), 4)
        self.assertTrue(set(self.frame.columns).issubset(reloaded.frame.columns))

    def test_resampling_and_label_exclusion_produce_auditable_artifacts(self) -> None:
        imbalanced_frame = self.frame.assign(grade=["a", "a", "a", "a", "b"])
        imbalanced = LoadedDataset(
            ref=self.ref.model_copy(
                update={"row_count": len(imbalanced_frame)}
            ),
            schema=build_schema_profile(self.ref, imbalanced_frame, self.config),
            frame=imbalanced_frame,
            manifest={},
            manifest_path=self.root / "demo.manifest.json",
            data_path=self.root / "demo.parquet",
            config=self.config,
            config_path=self.root / "config" / "profiles" / "demo_v1.json",
        )
        oversample = make_decision(
            issue_type=IssueType.IMBALANCE,
            strategy_id="oversample_minority",
            finding_id="imbalance.grade",
            params={"target_column": "grade", "random_state": 7},
        )
        result = mitigate_imbalance(
            imbalanced,
            oversample,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=True,
            confirm=True,
        )
        self.assertEqual(len(result.output_frame), 8)
        self.assertEqual(result.change_set.rows_added, 3)

        split = make_decision(
            issue_type=IssueType.IMBALANCE,
            strategy_id="stratified_split",
            finding_id="imbalance.grade",
            params={"target_column": "grade", "test_size": 0.5, "random_state": 7},
        )
        split_result = mitigate_imbalance(
            imbalanced,
            split,
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=True,
        )
        split_table = pd.read_parquet(
            split_result.artifacts[0].uri, engine="fastparquet"
        )
        self.assertEqual(set(split_table["split"]), {"train", "test"})

        label = make_decision(
            issue_type=IssueType.LABEL,
            strategy_id="exclude_from_training",
            finding_id="label.grade",
            target_columns=["grade"],
        )
        label_result = repair_labels(
            self.dataset,
            label,
            selection_table=_selection(
                [{"row_index": 4, "row_id": "r5", "column": "grade"}]
            ),
            artifact_dir=self.artifacts,
            output_dir=self.processed,
            dry_run=True,
        )
        exclusions = pd.read_parquet(
            label_result.artifacts[0].uri, engine="fastparquet"
        )
        self.assertTrue(exclusions["excluded_from_training"].all())


if __name__ == "__main__":
    unittest.main()
