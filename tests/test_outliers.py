import unittest

import numpy as np
import pandas as pd

from data_agent.checks.outliers import detect_outliers
from data_agent.config import OutlierPolicy
from data_agent.contracts import (
    ColumnSpec,
    LogicalType,
    NumericRange,
    SchemaProfile,
)


def schema(with_range: bool) -> SchemaProfile:
    return SchemaProfile(
        dataset_id="outlier_demo",
        columns=[
            ColumnSpec(
                name="row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="amount",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="float64",
                range=NumericRange(minimum=0, maximum=100) if with_range else None,
            ),
        ],
        row_id_column="row_id",
    )


class OutlierCheckTests(unittest.TestCase):
    def test_declared_range_and_non_finite_values_are_flagged(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
                "amount": [1.0, 50.0, -5.0, 150.0, np.inf, None],
            }
        )
        result = detect_outliers(frame, schema(True), OutlierPolicy())
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.affected_rows, 3)
        self.assertEqual(finding.evidence["below_minimum_count"], 1)
        self.assertEqual(finding.evidence["above_maximum_count"], 2)
        self.assertEqual(finding.evidence["non_finite_count"], 1)

    def test_iqr_tail_outlier_is_detected(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": [f"r{index}" for index in range(21)],
                "amount": [float(index) for index in range(1, 21)] + [1000.0],
            }
        )
        policy = OutlierPolicy(
            statistical_columns=["amount"],
            iqr_multiplier=1.5,
            minimum_non_null=5,
        )
        result = detect_outliers(frame, schema(False), policy)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].affected_rows, 1)
        self.assertIn("upper_bound", result.findings[0].evidence)


if __name__ == "__main__":
    unittest.main()
