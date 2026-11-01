from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetConfig, ReadOptions, load_dataset_config
from .contracts import DataFormat, DatasetRef, SchemaProfile
from .ingestion import apply_ingestion_policy
from .profiling import build_schema_profile


class DatasetLoadError(RuntimeError):
    pass


class DatasetIntegrityError(DatasetLoadError):
    pass


class DatasetShapeError(DatasetLoadError):
    pass


@dataclass(slots=True)
class LoadedDataset:
    ref: DatasetRef
    schema: SchemaProfile
    frame: pd.DataFrame
    manifest: dict[str, Any]
    manifest_path: Path
    data_path: Path
    config: DatasetConfig
    config_path: Path | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DatasetLoadError(f"manifest not found: {path}") from error
    if not isinstance(payload, dict):
        raise DatasetLoadError("manifest root must be a JSON object")
    return payload


def _canonical_dtype(value: str) -> str:
    normalised = value.strip().lower()
    if normalised in {"str", "string", "object"}:
        return "string"
    if normalised in {"bool", "boolean"}:
        return "boolean"
    return value


def _source_dtype(read: ReadOptions, config: DatasetConfig) -> dict[str, str]:
    reverse_aliases = {
        target: source for source, target in config.ingestion.column_aliases.items()
    }
    result: dict[str, str] = {}
    for column, dtype in read.dtype.items():
        result[reverse_aliases.get(column, column)] = dtype
    for column in config.ingestion.identifier_columns:
        result[reverse_aliases.get(column, column)] = "string"
    return result


def _read_frame(path: Path, data_format: DataFormat, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    read = config.read
    options = dict(read.options)
    source_dtype = _source_dtype(read, config)
    if config.ingestion.encoding and data_format in {DataFormat.CSV, DataFormat.JSONL}:
        options.setdefault("encoding", config.ingestion.encoding)

    if data_format is DataFormat.PARQUET:
        engine = read.engine or options.pop("engine", None) or "auto"
        frame = pd.read_parquet(path, engine=engine, **options)
    elif data_format is DataFormat.CSV:
        if source_dtype:
            options.setdefault("dtype", source_dtype)
        frame = pd.read_csv(path, **options)
    elif data_format is DataFormat.XLSX:
        if source_dtype:
            options.setdefault("dtype", source_dtype)
        frame = pd.read_excel(path, **options)
    elif data_format is DataFormat.JSONL:
        options.setdefault("lines", True)
        frame = pd.read_json(path, **options)
    else:
        raise DatasetLoadError(f"unsupported dataset format: {data_format.value}")

    ingested = apply_ingestion_policy(
        frame,
        config.ingestion,
        row_id_column=config.row_id_column,
    )
    frame = ingested.frame
    for column, dtype in read.dtype.items():
        if column in frame.columns:
            frame[column] = frame[column].astype(_canonical_dtype(dtype))
    for column in config.ingestion.identifier_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    return frame, ingested.metadata()


def _dataset_ref(manifest: dict[str, Any]) -> DatasetRef:
    try:
        base_file = manifest["base_file"]
        dataset_id = str(manifest["dataset_id"])
        data_format = DataFormat(str(base_file["format"]).lower())
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetLoadError("manifest is missing required dataset metadata") from error

    source = manifest.get("source")
    source_value = source.get("path") if isinstance(source, dict) else None
    return DatasetRef(
        dataset_id=dataset_id,
        version=str(manifest.get("version", "1")),
        uri=str(base_file["path"]),
        format=data_format,
        sha256=base_file.get("sha256"),
        row_count=int(base_file["row_count"]),
        column_count=int(base_file["column_count"]),
        source=source_value,
        metadata={
            "manifest_status": manifest.get("status"),
            "processing": manifest.get("processing", {}),
            "schema": manifest.get("schema", {}),
        },
    )


def load_dataset(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_hash: bool = True,
) -> LoadedDataset:
    config_path = Path(config_path).resolve()
    config = load_dataset_config(config_path)
    root = Path(project_root).resolve() if project_root is not None else Path.cwd()
    manifest_path = _resolve_path(root, config.manifest)
    manifest = _read_manifest(manifest_path)
    reference = _dataset_ref(manifest)

    declared_format = reference.format
    if config.read.format is not None and config.read.format is not declared_format:
        raise DatasetLoadError(
            "config read format does not match manifest format: "
            f"{config.read.format.value!r} != {declared_format.value!r}"
        )

    base_file = manifest["base_file"]
    data_path = _resolve_path(root, str(base_file["path"]))
    if not data_path.exists():
        raise DatasetLoadError(f"dataset file not found: {data_path}")

    if verify_hash:
        if not reference.sha256:
            raise DatasetIntegrityError("manifest does not provide a dataset SHA256")
        actual_sha256 = sha256_file(data_path)
        if actual_sha256.lower() != reference.sha256.lower():
            raise DatasetIntegrityError(
                "dataset SHA256 mismatch: "
                f"expected {reference.sha256}, got {actual_sha256}"
            )

    frame, ingestion_metadata = _read_frame(data_path, declared_format, config)
    source_row_count = int(
        ingestion_metadata.get("source_row_count", len(frame))
    )
    source_column_count = int(
        ingestion_metadata.get("source_column_count", len(frame.columns))
    )
    if source_row_count != reference.row_count:
        raise DatasetShapeError(
            "row count mismatch: "
            f"manifest={reference.row_count}, data={source_row_count}"
        )
    if source_column_count != reference.column_count:
        raise DatasetShapeError(
            "column count mismatch: "
            f"manifest={reference.column_count}, data={source_column_count}"
        )

    schema = build_schema_profile(
        reference,
        frame,
        config,
        ingestion_metadata=ingestion_metadata,
    )
    return LoadedDataset(
        ref=reference,
        schema=schema,
        frame=frame,
        manifest=manifest,
        manifest_path=manifest_path,
        data_path=data_path,
        config=config,
        config_path=config_path,
    )


__all__ = [
    "DatasetIntegrityError",
    "DatasetLoadError",
    "DatasetShapeError",
    "LoadedDataset",
    "load_dataset",
    "sha256_file",
]
