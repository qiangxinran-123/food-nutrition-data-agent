import unittest

import pandas as pd

from data_agent.checks.imbalance import detect_imbalance
from data_agent.checks.labels import analyse_label_column, detect_label_errors
from data_agent.config import (
    ImbalancePolicy,
    LabelColumnPolicy,
    LabelPolicy,
)
from data_agent.contracts import ColumnSpec, LogicalType, SchemaProfile


def schema() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="label_demo",
        columns=[
            ColumnSpec(
                name="row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
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
        row_id_column="row_id",
    )


def label_policy() -> LabelPolicy:
    return LabelPolicy(
        columns={
            "grade": LabelColumnPolicy(
                unavailable_tokens=["unknown", "not-applicable"],
                case_sensitive=False,
                trim_whitespace=True,
            )
        }
    )


class LabelAndImbalanceTests(unittest.TestCase):
    def test_unavailable_and_invalid_labels_are_separate(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
                "grade": ["a", "b", "unknown", "not-applicable", "X", None],
            }
        )
        analysis = analyse_label_column(
            frame,
            schema().columns[1],
            label_policy().columns["grade"],
        )
        self.assertEqual(int(analysis.valid_mask.sum()), 2)
        self.assertEqual(int(analysis.unavailable_mask.sum()), 3)
        self.assertEqual(int(analysis.invalid_mask.sum()), 1)

        result = detect_label_errors(frame, schema(), label_policy())
        findings = {finding.finding_id: finding for finding in result.findings}
        self.assertEqual(
            findings["label.unavailable.grade"].evidence["unavailable_counts"]["unknown"],
            1,
        )
        self.assertEqual(findings["label.invalid.grade"].affected_rows, 1)

    def test_imbalance_uses_only_usable_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": [f"r{index}" for index in range(100)],
                "grade": ["a"] * 90 + ["b"] * 10,
            }
        )
        policy = ImbalancePolicy(
            columns=["grade"],
            minimum_minority_share=0.15,
            maximum_majority_minority_ratio=5.0,
        )
        result = detect_imbalance(
            frame,
            schema(),
            label_policy(),
            policy,
        )
        self.assertTrue(result.summary["has_imbalance"])
        self.assertEqual(result.findings[0].affected_rows, 10)
        self.assertAlmostEqual(result.findings[0].affected_rate, 0.1)

    def test_balanced_labels_do_not_trigger_imbalance(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": [f"r{index}" for index in range(100)],
                "grade": ["a"] * 50 + ["b"] * 50,
            }
        )
        result = detect_imbalance(
            frame,
            schema(),
            label_policy(),
            ImbalancePolicy(columns=["grade"]),
        )
        self.assertFalse(result.summary["has_imbalance"])


if __name__ == "__main__":
    unittest.main()
