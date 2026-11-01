import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_agent.io import DatasetIntegrityError, load_dataset, sha256_file


class DatasetLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_path = self.root / "data" / "base" / "demo.parquet"
        self.data_path.parent.mkdir(parents=True)
        self.frame = pd.DataFrame(
            {
                "code": ["000123", "000456"],
                "value": [1.0, 2.0],
            }
        )
        self.frame.to_parquet(
            self.data_path,
            engine="fastparquet",
            compression="snappy",
            index=False,
        )
        self.manifest_path = self.root / "data" / "base" / "demo.manifest.json"
        self.config_path = self.root / "config" / "profiles" / "demo.json"
        self.config_path.parent.mkdir(parents=True)
        self._write_metadata()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest(self) -> dict:
        return {
            "contract_version": "0.1",
            "dataset_id": "demo_v1",
            "status": "frozen",
            "base_file": {
                "path": "data/base/demo.parquet",
                "format": "parquet",
                "engine": "fastparquet",
                "sha256": sha256_file(self.data_path),
                "row_count": 2,
                "column_count": 2,
            },
            "schema": {},
            "processing": {},
        }

    def _config(self) -> dict:
        return {
            "contract_version": "0.1",
            "dataset_id": "demo_v1",
            "manifest": "data/base/demo.manifest.json",
            "read": {
                "engine": "fastparquet",
                "dtype": {"code": "string"},
                "options": {},
            },
            "columns": {
                "code": {
                    "role": "entity_id",
                    "logical_type": "identifier",
                    "nullable": False,
                    "unique": True,
                },
                "value": {"logical_type": "numeric"},
            },
        }

    def _write_metadata(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self._manifest(), indent=2),
            encoding="utf-8",
        )
        self.config_path.write_text(
            json.dumps(self._config(), indent=2),
            encoding="utf-8",
        )

    def test_loads_manifest_data_and_preserves_identifier_text(self) -> None:
        dataset = load_dataset(self.config_path, project_root=self.root)
        self.assertEqual(dataset.ref.dataset_id, "demo_v1")
        self.assertEqual(dataset.frame.shape, (2, 2))
        self.assertEqual(dataset.frame.loc[0, "code"], "000123")
        self.assertEqual(dataset.schema.columns[0].physical_dtype, "str")

    def test_rejects_hash_mismatch(self) -> None:
        self.data_path.write_bytes(b"corrupted")
        with self.assertRaises(DatasetIntegrityError):
            load_dataset(self.config_path, project_root=self.root)

    def test_rejects_shape_mismatch(self) -> None:
        manifest = self._manifest()
        manifest["base_file"]["row_count"] = 3
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "row count mismatch"):
            load_dataset(self.config_path, project_root=self.root)

    def test_ingestion_additions_do_not_break_manifest_shape_check(self) -> None:
        config = self._config()
        config["row_id_column"] = "__row_id"
        config["columns"]["__row_id"] = {
            "role": "row_id",
            "logical_type": "identifier",
            "nullable": False,
            "unique": True,
        }
        config["ingestion"] = {
            "identifier_columns": ["code"],
            "preserve_raw_columns": ["value"],
            "ensure_row_id": True,
            "row_id_prefix": "row",
            "row_id_width": 3,
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        dataset = load_dataset(self.config_path, project_root=self.root)

        self.assertEqual(dataset.frame.shape, (2, 4))
        self.assertEqual(dataset.frame["__row_id"].tolist(), ["row_001", "row_002"])
        self.assertEqual(
            [column.name for column in dataset.schema.columns],
            ["code", "value", "__row_id"],
        )
        self.assertEqual(dataset.schema.metadata["raw_columns"], ["__raw__value"])


if __name__ == "__main__":
    unittest.main()
