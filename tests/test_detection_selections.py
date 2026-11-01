import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_agent.checks.conflicts import detect_conflicts
from data_agent.checks.duplicates import detect_duplicates
from data_agent.checks.labels import analyse_label_column, detect_label_errors
from data_agent.checks.missing import MissingCheckOptions, detect_missing
from data_agent.checks.types_formats import detect_type_format_issues
from data_agent.config import (
    DuplicateRuleConfig,
    LabelColumnPolicy,
    LabelPolicy,
    TypeFormatPolicy,
)
from data_agent.contracts import (
    ColumnSpec,
    ConstraintKind,
    ConstraintSpec,
    LogicalType,
    SchemaProfile,
    ToolStatus,
)


def generic_schema() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="generic_demo",
        columns=[
            ColumnSpec(
                name="sku",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="cost",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="grade",
                role="label",
                logical_type=LogicalType.CATEGORICAL,
                physical_dtype="str",
                allowed_values=["a", "b", "c"],
            ),
        ],
    )


class SelectionArtifactTests(unittest.TestCase):
    def test_missing_selection_matches_count_and_uses_row_index_fallback(self) -> None:
        frame = pd.DataFrame(
            {
                "sku": ["s1", "s2", "s3", "s4"],
                "cost": [1.0, None, 3.0, None],
                "grade": ["a", "b", None, "c"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            result = detect_missing(
                frame,
                generic_schema(),
                MissingCheckOptions(artifact_dir=directory),
            )
            self.assertEqual(result.status, ToolStatus.SUCCESS)
            self.assertEqual(len(result.artifacts), 2)
            total_rows = 0
            for finding in result.findings:
                self.assertIsNotNone(finding.selection_artifact_id)
                artifact = next(
                    artifact
                    for artifact in result.artifacts
                    if artifact.artifact_id == finding.selection_artifact_id
                )
                selected = pd.read_parquet(artifact.uri)
                self.assertEqual(len(selected), finding.affected_cells)
                self.assertEqual(
                    set(selected["row_id"]),
                    {f"row_index:{index}" for index in selected["row_index"]},
                )
                if finding.columns == ["cost"]:
                    self.assertTrue(selected["value_is_null"].all())
                total_rows += len(selected)
            self.assertEqual(total_rows, result.summary["missing_cells"])

    def test_duplicate_selection_contains_whole_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "sku": ["a", "a", "b", "b", "b"],
                "cost": [1.0, 1.0, 2.0, 2.0, 2.0],
                "grade": ["a", "a", "b", "b", "b"],
            }
        )
        rule = DuplicateRuleConfig(rule_id="sku_unique", columns=["sku"])
        with tempfile.TemporaryDirectory() as directory:
            result = detect_duplicates(
                frame,
                generic_schema(),
                [rule],
                {"artifact_dir": directory},
            )
            self.assertEqual(result.summary["rules"][0]["redundant_rows"], 3)
            self.assertEqual(len(result.artifacts), 1)
            selected = pd.read_parquet(result.artifacts[0].uri)
            self.assertEqual(len(selected), 5)
            self.assertEqual(selected["group_id"].nunique(), 2)
            self.assertEqual(int(selected["is_candidate_keep"].sum()), 2)
            self.assertEqual(int(selected["is_redundant"].sum()), 3)

    def test_label_aliases_are_mapped_before_validation(self) -> None:
        frame = pd.DataFrame({"sku": ["s1", "s2"], "cost": [1.0, 2.0], "grade": ["A+", "a"]})
        policy = LabelColumnPolicy(aliases={"A+": "a"}, case_sensitive=False)
        analysis = analyse_label_column(frame, generic_schema().columns[2], policy)
        self.assertEqual(int(analysis.valid_mask.sum()), 2)
        self.assertEqual(int(analysis.invalid_mask.sum()), 0)
        result = detect_label_errors(
            frame,
            generic_schema(),
            LabelPolicy(columns={"grade": policy}),
        )
        self.assertEqual(result.findings, [])

    def test_string_infinity_is_detected_as_non_finite(self) -> None:
        frame = pd.DataFrame(
            {
                "sku": ["s1", "s2", "s3"],
                "cost": ["1.0", "inf", "not-a-number"],
                "grade": ["a", "b", "c"],
            }
        )
        result = detect_type_format_issues(frame, generic_schema(), TypeFormatPolicy())
        findings = {finding.finding_id: finding for finding in result.findings}
        self.assertEqual(findings["format.numeric.cost"].affected_rows, 1)
        self.assertEqual(findings["format.non_finite.cost"].affected_rows, 1)


def constraint_schema() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="constraint_demo",
        columns=[
            ColumnSpec(
                name="row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="qty",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="int64",
            ),
            ColumnSpec(
                name="grade",
                logical_type=LogicalType.CATEGORICAL,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="price",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="int64",
            ),
        ],
        row_id_column="row_id",
    )


class ConstraintKindTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "row_id": [f"r{index}" for index in range(1, 7)],
                "qty": [5, 20, 6, 7, 7, 7],
                "grade": ["a", "b", "X", "c", "c", "c"],
                "price": [50, 50, 50, 200, 100, 90],
            }
        )

    def test_range_allowed_values_and_conditional_constraints(self) -> None:
        constraints = [
            ConstraintSpec(
                constraint_id="qty_range",
                kind=ConstraintKind.RANGE,
                columns=["qty"],
                parameters={"minimum": 0, "maximum": 10},
                message="qty out of range",
            ),
            ConstraintSpec(
                constraint_id="grade_allowed",
                kind=ConstraintKind.ALLOWED_VALUES,
                columns=["grade"],
                parameters={"allowed_values": ["a", "b", "c"]},
                message="grade not allowed",
            ),
            ConstraintSpec(
                constraint_id="premium_price",
                kind=ConstraintKind.CONDITIONAL,
                columns=["qty", "price"],
                parameters={"when": 'col("qty") > 5'},
                expression='col("price") <= 100',
                message="premium product price too high",
            ),
            ConstraintSpec(
                constraint_id="row_id_unique",
                kind=ConstraintKind.UNIQUE,
                columns=["row_id"],
                message="row id must be unique",
            ),
            ConstraintSpec(
                constraint_id="external_rule",
                kind=ConstraintKind.CUSTOM,
                columns=["grade"],
                message="external rule",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = detect_conflicts(
                self.frame,
                constraint_schema(),
                constraints,
                {"artifact_dir": directory},
            )
        findings = {finding.finding_id: finding for finding in result.findings}
        self.assertEqual(result.status, ToolStatus.PARTIAL)
        self.assertEqual(findings["conflict.qty_range"].affected_rows, 1)
        self.assertEqual(findings["conflict.grade_allowed"].affected_rows, 1)
        self.assertEqual(findings["conflict.premium_price"].affected_rows, 1)
        self.assertNotIn("conflict.row_id_unique", findings)
        self.assertEqual(result.summary["constraints_not_evaluated"], 1)
        self.assertTrue(result.warnings)

    def test_missing_constraint_parameters_are_not_reported_as_success(self) -> None:
        constraint = ConstraintSpec(
            constraint_id="bad_range",
            kind=ConstraintKind.RANGE,
            columns=["qty"],
            message="bad range",
        )
        result = detect_conflicts(
            self.frame,
            constraint_schema(),
            [constraint],
        )
        self.assertEqual(result.status, ToolStatus.PARTIAL)
        self.assertEqual(result.summary["constraints_checked"], 0)
        self.assertEqual(result.summary["constraints_not_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
