import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from data_agent.clustering import ClusterConfig, load_input
from data_agent.io import DatasetIntegrityError, sha256_file

ROOT = Path(__file__).resolve().parents[3]


class M0Tests(unittest.TestCase):
    def test_real_versions_and_identity(self):
        for version, rows in (("1", 52000), ("2", 50000), ("5", 50000)):
            with self.subTest(version=version):
                data = load_input(version, project_root=ROOT)
                self.assertEqual((len(data.frame), len(data.columns)), (rows, 20))
                self.assertTrue(data.frame[data.row_id_column].is_unique)
                self.assertFalse(data.frame[data.row_id_column].isna().any())
                self.assertEqual(data.ref.sha256, sha256_file(data.data_path))
                self.assertEqual(tuple(data.frame.columns), data.columns)

    def test_unknown_version(self):
        with self.assertRaisesRegex(ValueError, "unsupported dataset version"):
            load_input("6", project_root=ROOT)

    def test_config_validation(self):
        args = dict(dataset_id="off_52k_v1", dataset_version="1", data_sha256="a" * 64,
                    profile_path="p", manifest_path="m", row_id_column="__row_id",
                    features=["fat_100g"], algorithm="future", artifact_path="outputs/clustering/run")
        self.assertEqual(ClusterConfig(**args).random_state, 42)
        for changes in ({"data_sha256": "bad"}, {"features": ["fat_100g", "fat_100g"]},
                        {"features": ["__row_id"]}, {"random_state": -1}, {"dataset_version": "6"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                ClusterConfig(**(args | changes))

    def test_portable_v2_hash_and_read_only(self):
        # Minimal independent snapshot under two different project roots.
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("first root", "second root"):
                root = Path(tmp) / name
                (root / "config/profiles").mkdir(parents=True)
                (root / "data/processed").mkdir(parents=True)
                data_path = root / "data/processed/off_52k_v2.parquet"
                pd.DataFrame({"code": ["01", "02"], "__row_id": ["r1", "r2"], "fat_100g": [1., 2.]}).to_parquet(data_path, engine="fastparquet", index=False)
                profile = {"dataset_id": "off_52k_v2", "manifest": "D:/data_mining/data/processed/off_52k_v2.manifest.json",
                           "row_id_column": "__row_id", "read": {"engine": "fastparquet"},
                           "columns": {"code": {"logical_type": "identifier"},
                                       "__row_id": {"logical_type": "identifier"},
                                       "fat_100g": {"logical_type": "numeric"}}}
                profile_path = root / "config/profiles/off_52k_v2.json"
                profile_path.write_text(json.dumps(profile), encoding="utf-8")
                manifest_path = root / "data/processed/off_52k_v2.manifest.json"
                manifest = {"dataset_id": "off_52k_v2", "version": "2", "base_file": {"path": "data/processed/off_52k_v2.parquet", "format": "parquet", "sha256": sha256_file(data_path), "row_count": 2, "column_count": 3}}
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                sentinel = root / "existing-output.txt"
                sentinel.write_text("keep", encoding="utf-8")
                before = {p: sha256_file(p) for p in (data_path, profile_path, manifest_path, sentinel)}
                loaded = load_input("2", project_root=root)
                loaded.frame.loc[0, "fat_100g"] = 99
                self.assertEqual(load_input("2", project_root=root).frame.loc[0, "fat_100g"], 1.)
                self.assertEqual(before, {p: sha256_file(p) for p in before})
                data_path.write_bytes(b"corrupted")
                with self.assertRaises(DatasetIntegrityError):
                    load_input("2", project_root=root)

    def test_duplicate_row_id_rejected(self):
        from unittest.mock import patch
        from data_agent.io import DatasetLoadError
        loaded = load_input("1", project_root=ROOT)
        loaded.frame.loc[1, loaded.row_id_column] = loaded.frame.loc[0, loaded.row_id_column]
        # Patch the existing loader return, without touching real data.
        from types import SimpleNamespace
        original = SimpleNamespace(ref=loaded.ref, schema=loaded.schema, frame=loaded.frame, manifest=loaded.manifest,
                                   manifest_path=loaded.manifest_path, data_path=loaded.data_path)
        with patch("data_agent.clustering.dataset_adapter.load_dataset", return_value=original):
            with self.assertRaisesRegex(DatasetLoadError, "row identifier"):
                load_input("1", project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
