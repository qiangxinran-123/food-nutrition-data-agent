import unittest

import pandas as pd

from data_agent.checks.conflicts import detect_conflicts
from data_agent.contracts import (
    ColumnSpec,
    ConstraintKind,
    ConstraintSpec,
    LogicalType,
    SchemaProfile,
    Severity,
)
from data_agent.expressions import ExpressionError, evaluate_boolean_expression


def schema() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="conflict_demo",
        columns=[
            ColumnSpec(
                name="row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="kj",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="float64",
            ),
            ColumnSpec(
                name="kcal",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="float64",
            ),
        ],
        row_id_column="row_id",
    )


class ConflictCheckTests(unittest.TestCase):
    def test_formula_conflicts_are_detected_and_missing_rows_are_skipped(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2", "r3", "r4"],
                "kj": [418.4, 500.0, 1000.0, None],
                "kcal": [100.0, 50.0, 100.0, 100.0],
            }
        )
        constraint = ConstraintSpec(
            constraint_id="energy_consistency",
            kind=ConstraintKind.FORMULA,
            columns=["kj", "kcal"],
            expression='abs(col("kj") - col("kcal") * 4.184) <= 0.15 * col("kj")',
            severity=Severity.HIGH,
            message="energy mismatch",
        )
        result = detect_conflicts(frame, schema(), [constraint])
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.affected_rows, 2)
        self.assertEqual(finding.evidence["evaluated_rows"], 3)
        self.assertEqual(finding.evidence["skipped_missing_rows"], 1)

    def test_expression_evaluator_rejects_unsafe_calls(self) -> None:
        frame = pd.DataFrame({"kj": [1.0], "kcal": [1.0]})
        with self.assertRaises(ExpressionError):
            evaluate_boolean_expression('__import__("os")', frame)


if __name__ == "__main__":
    unittest.main()
