import unittest

import pandas as pd

from data_agent.checks.duplicates import detect_duplicates
from data_agent.config import DuplicateRuleConfig
from data_agent.contracts import ColumnSpec, LogicalType, SchemaProfile


def schema() -> SchemaProfile:
    return SchemaProfile(
        dataset_id="duplicate_demo",
        columns=[
            ColumnSpec(
                name="row_id",
                role="row_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
                nullable=False,
            ),
            ColumnSpec(
                name="code",
                role="entity_id",
                logical_type=LogicalType.IDENTIFIER,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="name",
                logical_type=LogicalType.TEXT,
                physical_dtype="str",
            ),
            ColumnSpec(
                name="value",
                logical_type=LogicalType.NUMERIC,
                physical_dtype="float64",
            ),
        ],
        row_id_column="row_id",
    )


class DuplicateCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2", "r3", "r4"],
                "code": ["001", "001", "002", "003"],
                "name": ["A", "A", "A", "B"],
                "value": [1.0, 1.0, 1.0, 2.0],
            }
        )
        self.rules = [
            DuplicateRuleConfig(
                rule_id="primary_key",
                columns=["code"],
                severity="critical",
            ),
            DuplicateRuleConfig(
                rule_id="business_exact",
                use_all_columns=True,
                exclude_columns=["row_id"],
            ),
            DuplicateRuleConfig(
                rule_id="content_signature",
                use_all_columns=True,
                exclude_columns=["row_id", "code"],
            ),
        ]

    def test_reports_all_three_duplicate_scopes(self) -> None:
        result = detect_duplicates(self.frame, schema(), self.rules)
        self.assertTrue(result.summary["has_duplicates"])
        summaries = {item["rule_id"]: item for item in result.summary["rules"]}
        self.assertEqual(summaries["primary_key"]["duplicate_group_count"], 1)
        self.assertEqual(summaries["primary_key"]["redundant_rows"], 1)
        self.assertEqual(summaries["business_exact"]["redundant_rows"], 1)
        self.assertEqual(summaries["content_signature"]["duplicate_group_count"], 1)
        self.assertEqual(summaries["content_signature"]["rows_in_duplicate_groups"], 3)
        self.assertEqual(summaries["content_signature"]["redundant_rows"], 2)
        self.assertEqual(result.summary["rows_flagged_union"], 2)

    def test_normalisation_can_merge_case_and_whitespace(self) -> None:
        frame = pd.DataFrame(
            {
                "row_id": ["r1", "r2", "r3"],
                "name": [" Apple ", "apple", "pear"],
            }
        )
        schema = SchemaProfile(
            dataset_id="normalise_demo",
            columns=[
                ColumnSpec(
                    name="row_id",
                    role="row_id",
                    logical_type=LogicalType.IDENTIFIER,
                    physical_dtype="str",
                ),
                ColumnSpec(
                    name="name",
                    logical_type=LogicalType.TEXT,
                    physical_dtype="str",
                ),
            ],
            row_id_column="row_id",
        )
        rule = DuplicateRuleConfig(
            rule_id="normalised_name",
            columns=["name"],
            normalize_text=True,
            case_sensitive=False,
        )
        result = detect_duplicates(frame, schema, [rule])
        summary = result.summary["rules"][0]
        self.assertEqual(summary["duplicate_group_count"], 1)
        self.assertEqual(summary["redundant_rows"], 1)


if __name__ == "__main__":
    unittest.main()
