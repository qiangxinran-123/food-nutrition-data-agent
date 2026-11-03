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


def _local_manifest_payload(payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Return a manifest payload whose external reference paths use this checkout."""
    prefix = "D:/data_mining/"
    localised = json.loads(json.dumps(payload))

    def local_path(value: Any) -> str:
        text = str(value)
        return str(project_root / text[len(prefix):]) if text.startswith(prefix) else text

    for key in ("base_file", "change_file", "profile_file"):
        item = localised.get(key)
        if isinstance(item, dict) and item.get("path"):
            item["path"] = local_path(item["path"])
    parent = localised.get("parent")
    if isinstance(parent, dict) and parent.get("uri"):
        parent["uri"] = local_path(parent["uri"])
    source = localised.get("source")
    if isinstance(source, dict) and source.get("path"):
        source["path"] = local_path(source["path"])
    return localised


def load_input(version: str, *, project_root: str | Path) -> DatasetInput:
    if version not in ("1", "2", "3", "4", "5"):
        raise ValueError(
            f"unsupported dataset version: {version!r}; expected a known version from '1' to '5'"
        )
    root = Path(project_root).resolve()
    profile = root / "config" / "profiles" / f"off_52k_v{version}.json"
    if not profile.is_file():
        raise DatasetLoadError(f"profile not found: {profile}")
    payload = json.loads(profile.read_text(encoding="utf-8"))
    old_path = f"D:/data_mining/data/processed/off_52k_v{version}.manifest.json"
    # Only this known non-portable upstream path is remapped. No search or hash bypass.
    if version in ("2", "3", "4", "5") and payload.get("manifest") == old_path:
        local_manifest = root / "data" / "processed" / f"off_52k_v{version}.manifest.json"
        if not local_manifest.is_file():
            raise DatasetLoadError(
                f"v{version} manifest not found at expected project path: {local_manifest}"
            )
        local_payload = _local_manifest_payload(
            json.loads(local_manifest.read_text(encoding="utf-8")), root
        )
        with tempfile.TemporaryDirectory(prefix="m0_profile_") as directory:
            temp_manifest = Path(directory) / "manifest.json"
            temp_manifest.write_text(json.dumps(local_payload), encoding="utf-8")
            payload["manifest"] = str(temp_manifest)
            temp_profile = Path(directory) / "profile.json"
            temp_profile.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_dataset(temp_profile, project_root=root, verify_hash=True)
            manifest_path = local_manifest
    else:
        loaded = load_dataset(profile, project_root=root, verify_hash=True)
        manifest_path = loaded.manifest_path
    if loaded.ref.dataset_id != f"off_52k_v{version}" or loaded.ref.version != version:
        raise DatasetLoadError("profile/manifest dataset identity or version mismatch")
    row_id = loaded.schema.row_id_column
    if not row_id or row_id not in loaded.frame or loaded.frame[row_id].isna().any() or loaded.frame[row_id].duplicated().any():
        raise DatasetLoadError("row identifier is missing, null, or not unique")
    return DatasetInput(loaded.frame.copy(deep=True), loaded.ref, loaded.schema, loaded.manifest,
                        profile, manifest_path, loaded.data_path, row_id,
                        tuple(loaded.frame.columns))
