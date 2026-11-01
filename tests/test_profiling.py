import unittest

import pandas as pd
from pydantic import ValidationError

from data_agent.config import ColumnConfig, DatasetConfig, MissingPolicy
from data_agent.contracts import DataFormat, DatasetRef, LogicalType
from data_agent.profiling import build_schema_profile


class ProfilingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "code": ["000123", "000456"],
                "grade": ["a", "unknown"],
                "energy": [100.0, None],
                "note": ["first", "second"],
            }
        )
        self.config = DatasetConfig(
            dataset_id="demo_v1",
            domain_pack="demo",
            manifest="data/base/demo.parquet",
            primary_key_candidates=[["code"]],
            missing=MissingPolicy(tokens_by_column={"grade": ["unknown"]}),
            columns={
                "code": ColumnConfig(
                    role="entity_id",
                    logical_type=LogicalType.IDENTIFIER,
                    nullable=False,
                    unique=True,
                ),
                "grade": ColumnConfig(logical_type=LogicalType.CATEGORICAL),
                "energy": ColumnConfig(
                    role="energy",
                    logical_type=LogicalType.NUMERIC,
                    unit="kcal/100g",
                ),
                "note": ColumnConfig(logical_type=LogicalType.TEXT),
            },
        )
        self.reference = DatasetRef(
            dataset_id="demo_v1",
            uri="data/base/demo.parquet",
            format=DataFormat.PARQUET,
            row_count=2,
            column_count=4,
        )

    def test_profile_merges_config_and_infers_statistics(self) -> None:
        profile = build_schema_profile(self.reference, self.frame, self.config)
        columns = {column.name: column for column in profile.columns}

        self.assertEqual(columns["code"].physical_dtype, "str")
        self.assertEqual(columns["code"].logical_type, LogicalType.IDENTIFIER)
        self.assertFalse(columns["code"].nullable)
        self.assertTrue(columns["code"].unique)
        self.assertEqual(columns["grade"].missing_tokens, ["unknown"])
        self.assertEqual(columns["energy"].unit, "kcal/100g")
        self.assertEqual(columns["energy"].metadata["null_count"], 1)
        self.assertEqual(profile.primary_key_candidates, [["code"]])
        self.assertEqual(profile.metadata["row_count"], 2)

    def test_profile_rejects_config_for_missing_dataframe_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent from data"):
            build_schema_profile(
                self.reference,
                self.frame.drop(columns=["note"]),
                self.config,
            )

    def test_profile_rejects_dataset_id_mismatch(self) -> None:
        reference = self.reference.model_copy(update={"dataset_id": "other"})
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_schema_profile(reference, self.frame, self.config)

    def test_config_rejects_unknown_token_column_and_blank_token(self) -> None:
        with self.assertRaises(ValidationError):
            DatasetConfig(
                dataset_id="demo",
                manifest="demo.json",
                missing=MissingPolicy(tokens_by_column={"missing": ["unknown"]}),
            )
        with self.assertRaises(ValidationError):
            MissingPolicy(tokens=["  "])


if __name__ == "__main__":
    unittest.main()
