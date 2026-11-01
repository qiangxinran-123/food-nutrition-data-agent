"""Load known snapshots using the existing integrity-checking loader."""
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_agent.contracts import DatasetRef, SchemaProfile
from data_agent.io import DatasetLoadError, load_dataset


@dataclass(slots=True)
class DatasetInput:
    frame: pd.DataFrame
    ref: DatasetRef
    schema: SchemaProfile
    manifest: dict[str, Any]
    profile_path: Path
    manifest_path: Path
    data_path: Path
    row_id_column: str
    columns: tuple[str, ...]


def load_input(version: str, *, project_root: str | Path) -> DatasetInput:
    if version not in ("1", "2"):
        raise ValueError(f"unsupported dataset version: {version!r}; expected '1' or '2'")
    root = Path(project_root).resolve()
    profile = root / "config" / "profiles" / f"off_52k_v{version}.json"
    if not profile.is_file():
        raise DatasetLoadError(f"profile not found: {profile}")
    payload = json.loads(profile.read_text(encoding="utf-8"))
    old_path = "D:/data_mining/data/processed/off_52k_v2.manifest.json"
    # Only this known non-portable upstream path is remapped. No search or hash bypass.
    if version == "2" and payload.get("manifest") == old_path:
        local_manifest = root / "data" / "processed" / "off_52k_v2.manifest.json"
        if not local_manifest.is_file():
            raise DatasetLoadError(f"v2 manifest not found at expected project path: {local_manifest}")
        payload["manifest"] = str(local_manifest)
        with tempfile.TemporaryDirectory(prefix="m0_profile_") as directory:
            temp_profile = Path(directory) / "profile.json"
            temp_profile.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_dataset(temp_profile, project_root=root, verify_hash=True)
    else:
        loaded = load_dataset(profile, project_root=root, verify_hash=True)
    if loaded.ref.dataset_id != f"off_52k_v{version}" or loaded.ref.version != version:
        raise DatasetLoadError("profile/manifest dataset identity or version mismatch")
    row_id = loaded.schema.row_id_column
    if not row_id or row_id not in loaded.frame or loaded.frame[row_id].isna().any() or loaded.frame[row_id].duplicated().any():
        raise DatasetLoadError("row identifier is missing, null, or not unique")
    return DatasetInput(loaded.frame.copy(deep=True), loaded.ref, loaded.schema, loaded.manifest,
                        profile, loaded.manifest_path, loaded.data_path, row_id,
                        tuple(loaded.frame.columns))
