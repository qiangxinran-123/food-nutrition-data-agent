import unittest

import numpy as np
import pandas as pd

from data_agent.checks.types_formats import detect_type_format_issues
from data_agent.config import TypeFormatPolicy
from data_agent.contracts import ColumnSpec, LogicalType, SchemaProfile


def schema() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="format_demo",
        columns=[
            ColumnSpec(
                name="row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="code",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
                missing_tokens=["unknown"],
                metadata={"format_regex": r"^(?:\d{8}|\d{13})$"},
            ),
            ColumnSpec(
                name="amount",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="str",
            ),
        ],
        row_id_column="row_id",
    )


class TypeFormatCheckTests(unittest.TestCase):
    def test_regex_and_numeric_parsing_issues_are_reported(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2", "r3", "r4", "r5"],
                "code": ["12345678", "abc", "1234567890123", "unknown", None],
                "amount": ["1.2", "oops", "3.4", None, "5"],
            }
        )
        result = detect_type_format_issues(frame, schema(), TypeFormatPolicy())
        findings = {finding.finding_id: finding for finding in result.findings}
        self.assertEqual(findings["format.regex.code"].affected_rows, 1)
        self.assertEqual(findings["format.numeric.amount"].affected_rows, 1)
        self.assertTrue(result.summary["has_type_or_format_issues"])

    def test_non_finite_numeric_values_are_reported(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2"],
                "code": ["12345678", "12345678"],
                "amount": [1.0, np.inf],
            }
        )
        result = detect_type_format_issues(frame, schema(), TypeFormatPolicy())
        finding = next(
            finding
            for finding in result.findings
            if finding.finding_id == "format.non_finite.amount"
        )
        self.assertEqual(finding.affected_rows, 1)


if __name__ == "__main__":
    unittest.main()
