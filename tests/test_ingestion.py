import unittest

import pandas as pd
from pydantic import ValidationError

from data_agent.config import ColumnConfig, DatasetConfig, IngestionPolicy
from data_agent.contracts import DataFormat, DatasetRef, LogicalType
from data_agent.ingestion import IngestionError, apply_ingestion_policy
from data_agent.profiling import build_schema_profile


class IngestionTests(unittest.TestCase):
    def test_alias_mapping_and_leading_zeros_are_preserved(self) -> None:
        frame = pd.DataFrame({"Barcode": [123, 456], "value": [1.0, 2.0]})
        policy = IngestionPolicy(
            column_aliases={"Barcode": "code"},
            identifier_columns=["code"],
        )

        result = apply_ingestion_policy(frame, policy)

        self.assertEqual(result.frame.columns.tolist(), ["code", "value"])
        self.assertEqual(result.frame["code"].tolist(), ["123", "456"])
        self.assertEqual(result.source_column_map, {"Barcode": "code"})
        self.assertEqual(result.source_column_count, 2)

    def test_bom_and_case_insensitive_aliases_are_supported(self) -> None:
        frame = pd.DataFrame({"\ufeffCode": ["000123"], "VALUE": [1.0]})
        policy = IngestionPolicy(
            column_aliases={"code": "code", "value": "value"},
            identifier_columns=["code"],
            case_sensitive_columns=False,
        )

        result = apply_ingestion_policy(frame, policy)

        self.assertEqual(result.frame.columns.tolist(), ["code", "value"])
        self.assertEqual(result.frame.loc[0, "code"], "000123")

    def test_raw_copy_and_generated_row_id_are_added_without_touching_source(self) -> None:
        frame = pd.DataFrame({"code": ["000123", "000456"], "value": [1, 2]})
        policy = IngestionPolicy(
            identifier_columns=["code"],
            preserve_raw_columns=["value"],
            ensure_row_id=True,
            row_id_prefix="row",
            row_id_width=3,
        )

        result = apply_ingestion_policy(
            frame,
            policy,
            row_id_column="__row_id",
        )

        self.assertEqual(result.frame["code"].tolist(), ["000123", "000456"])
        self.assertEqual(result.frame["__raw__value"].tolist(), [1, 2])
        self.assertEqual(result.frame["__row_id"].tolist(), ["row_001", "row_002"])
        self.assertEqual(result.row_id_generated, True)
        self.assertNotIn("__row_id", frame.columns)

    def test_raw_copy_columns_are_not_exposed_as_business_schema_columns(self) -> None:
        frame = pd.DataFrame({"code": ["000123"], "value": [1.0]})
        policy = IngestionPolicy(
            identifier_columns=["code"],
            preserve_raw_columns=["value"],
        )
        ingested = apply_ingestion_policy(frame, policy)
        reference = DatasetRef(
            dataset_id="demo",
            uri="demo.parquet",
            format=DataFormat.PARQUET,
            row_count=1,
            column_count=2,
        )
        config = DatasetConfig(
            dataset_id="demo",
            manifest="demo.json",
            columns={
                "code": ColumnConfig(logical_type=LogicalType.IDENTIFIER),
                "value": ColumnConfig(logical_type=LogicalType.NUMERIC),
            },
        )

        profile = build_schema_profile(
            reference,
            ingested.frame,
            config,
            ingestion_metadata=ingested.metadata(),
        )

        self.assertEqual([column.name for column in profile.columns], ["code", "value"])
        self.assertEqual(profile.metadata["raw_columns"], ["__raw__value"])

    def test_alias_targets_and_identifier_references_are_validated(self) -> None:
        with self.assertRaises(ValidationError):
            DatasetConfig(
                dataset_id="demo",
                manifest="demo.json",
                columns={"code": ColumnConfig()},
                ingestion=IngestionPolicy(
                    column_aliases={"a": "code", "b": "code"},
                    identifier_columns=["code"],
                ),
            )
        with self.assertRaises(ValidationError):
            DatasetConfig(
                dataset_id="demo",
                manifest="demo.json",
                columns={"code": ColumnConfig()},
                ingestion=IngestionPolicy(identifier_columns=["missing"]),
            )

    def test_identifier_must_exist_in_source(self) -> None:
        frame = pd.DataFrame({"value": [1.0]})
        with self.assertRaises(IngestionError):
            apply_ingestion_policy(
                frame,
                IngestionPolicy(identifier_columns=["code"]),
            )


if __name__ == "__main__":
    unittest.main()
