import unittest

import pandas as pd

from data_agent.checks.missing import MissingCheckOptions, detect_missing
from data_agent.contracts import ColumnSpec, LogicalType, SchemaProfile, ToolStatus


def synthetic_schema() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="synthetic_v1",
        columns=[
            ColumnSpec(
                name="row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
                nullable=False,
            ),
            ColumnSpec(
                name="amount",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="float64",
            ),
            ColumnSpec(
                name="category",
                logical_type=LogicalType.CATEGORICAL,
                physical_dtype="str",
                missing_tokens=["unknown"],
            ),
            ColumnSpec(
                name="note",
                logical_type=LogicalType.TEXT,
                physical_dtype="str",
            ),
        ],
        row_id_column="row_id",
    )


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3", "r4"],
            "amount": [1.0, None, 3.0, 4.0],
            "category": ["ok", " ", "UNKNOWN", "x"],
            "note": ["a", "b", None, "unknown"],
        }
    )


class MissingCheckTests(unittest.TestCase):
    def test_detects_nulls_empty_strings_and_declared_tokens(self) -> None:
        result = detect_missing(synthetic_frame(), synthetic_schema())
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        findings = {finding.columns[0]: finding for finding in result.findings}

        self.assertEqual(findings["amount"].affected_rows, 1)
        self.assertEqual(findings["category"].affected_rows, 2)
        self.assertEqual(
            findings["category"].evidence["declared_token_count"], 1
        )
        self.assertEqual(findings["category"].evidence["empty_string_count"], 1)
        self.assertEqual(findings["note"].affected_rows, 1)
        self.assertEqual(findings["category"].sample_row_ids, ["r2", "r3"])

        self.assertEqual(result.summary["missing_cells"], 4)
        self.assertEqual(result.summary["rows_with_missing"], 2)
        self.assertEqual(result.summary["complete_rows"], 2)
        self.assertEqual(
            result.summary["declared_missing_tokens"],
            {"category": ["unknown"]},
        )

    def test_non_nullable_column_is_critical(self) -> None:
        schema = synthetic_schema()
        schema.columns[0].nullable = False
        frame = synthetic_frame()
        frame.loc[1, "row_id"] = None

        result = detect_missing(frame, schema)
        finding = next(
            finding for finding in result.findings if finding.columns == ["row_id"]
        )
        self.assertEqual(finding.severity.value, "critical")
        self.assertIn("investigate_required_field", finding.suggested_actions)

    def test_clean_data_produces_no_findings(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2"],
                "amount": [1.0, 2.0],
                "category": ["a", "b"],
                "note": ["x", "y"],
            }
        )
        result = detect_missing(frame, synthetic_schema())
        self.assertEqual(result.findings, [])
        self.assertFalse(result.summary["has_missing"])
        self.assertEqual(result.summary["complete_rows"], 2)

    def test_schema_mismatch_returns_failed_tool_result(self) -> None:
        frame = synthetic_frame().drop(columns=["amount"])
        result = detect_missing(frame, synthetic_schema())
        self.assertEqual(result.status, ToolStatus.FAILED)
        self.assertEqual(result.error_code, "schema_mismatch")
        self.assertIn("amount", result.error_message)

    def test_extra_columns_are_ignored_with_warning(self) -> None:
        frame = synthetic_frame()
        frame["extra"] = "ignored"
        result = detect_missing(frame, synthetic_schema())
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertTrue(any("extra" in warning for warning in result.warnings))

    def test_sample_limit_is_enforced(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": [f"r{index}" for index in range(10)],
                "amount": [None] * 10,
                "category": ["ok"] * 10,
                "note": ["ok"] * 10,
            }
        )
        result = detect_missing(
            frame,
            synthetic_schema(),
            MissingCheckOptions(sample_limit=3),
        )
        self.assertEqual(len(result.findings[0].sample_row_ids), 3)


if __name__ == "__main__":
    unittest.main()
