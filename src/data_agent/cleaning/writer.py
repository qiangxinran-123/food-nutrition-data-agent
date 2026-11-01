from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import DatasetConfig
from ..contracts import Artifact, ArtifactKind, DataFormat, DatasetRef, utc_now


class CleaningWriteError(RuntimeError):
    pass


_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("._")
    return cleaned[:180] or "artifact"


def validate_dataset_id(value: str) -> str:
    if not _DATASET_ID_PATTERN.fullmatch(value):
        raise CleaningWriteError(
            "dataset_id must contain only letters, digits, dots, underscores and hyphens: "
            f"{value!r}"
        )
    return value


def next_dataset_id(dataset_id: str, requested: str | None = None) -> str:
    if requested:
        return validate_dataset_id(requested)
    validate_dataset_id(dataset_id)
    match = re.match(r"^(.*)_v(\d+)$", dataset_id)
    if match:
        prefix, number = match.groups()
        return f"{prefix}_v{int(number) + 1}"
    return f"{dataset_id}_cleaned"


def available_output_id(
    output_dir: str | Path,
    dataset_id: str,
    requested: str | None = None,
) -> str:
    candidate = next_dataset_id(dataset_id, requested)
    directory = Path(output_dir)
    candidate_exists = (
        (directory / f"{candidate}.parquet").exists()
        or (directory / f"{candidate}.manifest.json").exists()
    )
    if not candidate_exists:
        return candidate
    if requested:
        raise CleaningWriteError(f"processed dataset already exists: {candidate}")
    match = re.match(r"^(.*)_v(\d+)$", candidate)
    if match:
        prefix, number = match.groups()
        next_number = int(number) + 1
        while (
            (directory / f"{prefix}_v{next_number}.parquet").exists()
            or (directory / f"{prefix}_v{next_number}.manifest.json").exists()
        ):
            next_number += 1
        return f"{prefix}_v{next_number}"
    base = candidate
    index = 2
    while (
        (directory / f"{candidate}.parquet").exists()
        or (directory / f"{candidate}.manifest.json").exists()
    ):
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def write_table_artifact(
    table: pd.DataFrame,
    *,
    output_dir: str | Path,
    artifact_id: str,
    kind: ArtifactKind,
    dataset_id: str,
    description: str,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_filename(artifact_id)}.parquet"
    table.to_parquet(path, index=False, engine="fastparquet")
    return Artifact(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.resolve().as_posix(),
        media_type="application/vnd.apache.parquet",
        sha256=sha256_file(path),
        row_count=int(len(table)),
        dataset_id=dataset_id,
        description=description,
        metadata=metadata or {},
    )


def write_processed_dataset(
    frame: pd.DataFrame,
    *,
    before: DatasetRef,
    output_dir: str | Path,
    dataset_id: str | None = None,
    change_id: str | None = None,
    changes_table: pd.DataFrame | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[DatasetRef, Path, Path | None]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_dataset_id = available_output_id(
        directory,
        before.dataset_id,
        requested=dataset_id,
    )
    path = directory / f"{output_dataset_id}.parquet"
    frame.to_parquet(path, index=False, engine="fastparquet", compression="snappy")
    digest = sha256_file(path)

    changes_path: Path | None = None
    changes_artifact_id: str | None = None
    if changes_table is not None and not changes_table.empty:
        changes_artifact_id = f"{change_id or output_dataset_id}.changes"
        changes_path = directory / f"{_safe_filename(changes_artifact_id)}.parquet"
        changes_table.to_parquet(
            changes_path,
            index=False,
            engine="fastparquet",
        )

    after = DatasetRef(
        dataset_id=output_dataset_id,
        version=str(int(before.version) + 1) if before.version.isdigit() else before.version,
        uri=path.as_posix(),
        format=DataFormat.PARQUET,
        sha256=digest,
        row_count=int(len(frame)),
        column_count=int(len(frame.columns)),
        source=before.uri,
        parent_id=before.dataset_id,
        metadata={
            **(metadata or {}),
            "change_id": change_id,
            "changes_uri": changes_path.as_posix() if changes_path else None,
            "changes_artifact_id": changes_artifact_id,
            "manifest_uri": (directory / f"{output_dataset_id}.manifest.json").as_posix(),
        },
    )

    manifest = {
        "contract_version": "0.1",
        "dataset_id": output_dataset_id,
        "version": after.version,
        "status": "processed",
        "created_at": utc_now().isoformat(),
        "parent": {
            "dataset_id": before.dataset_id,
            "version": before.version,
            "sha256": before.sha256,
            "uri": before.uri,
        },
        "base_file": {
            "path": path.as_posix(),
            "format": "parquet",
            "engine": "fastparquet",
            "compression": "snappy",
            "sha256": digest,
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
        },
        "change_file": (
            {
                "path": changes_path.as_posix(),
                "sha256": sha256_file(changes_path),
            }
            if changes_path is not None
            else None
        ),
        "processing": {
            "cleaning_applied": True,
            "description": description,
            "change_id": change_id,
        },
        "metadata": metadata or {},
    }
    manifest_path = directory / f"{output_dataset_id}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return after, path, changes_path


def write_processed_profile(
    config: DatasetConfig,
    *,
    after: DatasetRef,
    manifest_path: str | Path,
    profile_dir: str | Path,
) -> Path:
    """Write a profile that can load the processed dataset with the normal loader."""

    directory = Path(profile_dir)
    directory.mkdir(parents=True, exist_ok=True)
    dataset_id = validate_dataset_id(after.dataset_id)
    profile = config.model_copy(
        deep=True,
        update={
            "dataset_id": dataset_id,
            "manifest": Path(manifest_path).resolve().as_posix(),
        },
    )
    profile_path = directory / f"{dataset_id}.json"
    profile_path.write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["profile_file"] = {
        "path": profile_path.resolve().as_posix(),
        "dataset_id": dataset_id,
        "sha256": sha256_file(profile_path),
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile_path


__all__ = [
    "CleaningWriteError",
    "available_output_id",
    "next_dataset_id",
    "sha256_file",
    "validate_dataset_id",
    "write_processed_profile",
    "write_processed_dataset",
    "write_table_artifact",
]
