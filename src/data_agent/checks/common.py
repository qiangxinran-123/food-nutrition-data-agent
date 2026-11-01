from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import (
    Artifact,
    ArtifactKind,
    Finding,
    SchemaProfile,
    Severity,
    ToolResult,
    ToolStatus,
)


SELECTION_SCHEMA_VERSION = "1"


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    return str(value)


def _value_text(value: Any) -> str | None:
    converted = _json_value(value)
    return None if converted is None else str(converted)


def _row_id_values(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    positions: np.ndarray,
) -> list[str]:
    row_id_column = schema.row_id_column
    if row_id_column is not None and row_id_column in frame.columns:
        values = frame[row_id_column].iloc[positions].tolist()
    else:
        values = [None] * len(positions)
    result: list[str] = []
    for position, value in zip(positions, values, strict=True):
        text = _value_text(value)
        result.append(text if text not in {None, ""} else f"row_index:{int(position)}")
    return result


def _mask_positions(mask: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    aligned = mask.reindex(frame.index, fill_value=False).fillna(False).astype(bool)
    return np.flatnonzero(aligned.to_numpy(dtype=bool))


def _reason_values(
    reason: str | pd.Series | None,
    frame: pd.DataFrame,
    positions: np.ndarray,
    default: str,
) -> list[str]:
    if reason is None:
        return [default] * len(positions)
    if isinstance(reason, str):
        return [reason] * len(positions)
    aligned = reason.reindex(frame.index, fill_value=default).fillna(default)
    return [str(value) for value in aligned.iloc[positions].tolist()]


def cell_selection_table(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    *,
    finding_id: str,
    rule_id: str,
    mask: pd.Series,
    column: str,
    reason: str | pd.Series | None = None,
) -> pd.DataFrame:
    positions = _mask_positions(mask, frame)
    if positions.size == 0:
        return pd.DataFrame(
            columns=[
                "finding_id",
                "rule_id",
                "selection_type",
                "row_index",
                "row_id",
                "column_index",
                "column",
                "reason_code",
                "value_text",
                "value_is_null",
            ]
        )
    column_index = int(frame.columns.get_loc(column))
    raw_values = frame[column].iloc[positions].tolist()
    return pd.DataFrame(
        {
            "finding_id": finding_id,
            "rule_id": rule_id,
            "selection_type": "cell",
            "row_index": positions.astype("int64"),
            "row_id": _row_id_values(frame, schema, positions),
            "column_index": column_index,
            "column": column,
            "reason_code": _reason_values(reason, frame, positions, rule_id),
            "value_text": [_value_text(value) for value in raw_values],
            "value_is_null": [_json_value(value) is None for value in raw_values],
        }
    )


def row_selection_table(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    *,
    finding_id: str,
    rule_id: str,
    mask: pd.Series,
    reason: str | pd.Series | None = None,
    value_columns: list[str] | None = None,
) -> pd.DataFrame:
    positions = _mask_positions(mask, frame)
    columns = [
        "finding_id",
        "rule_id",
        "selection_type",
        "row_index",
        "row_id",
        "reason_code",
        "row_values_json",
    ]
    if positions.size == 0:
        return pd.DataFrame(columns=columns)
    if value_columns:
        row_values = []
        for position in positions:
            payload = {
                column: _json_value(frame[column].iloc[int(position)])
                for column in value_columns
            }
            row_values.append(
                json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )
    else:
        row_values = [None] * len(positions)
    return pd.DataFrame(
        {
            "finding_id": finding_id,
            "rule_id": rule_id,
            "selection_type": "row",
            "row_index": positions.astype("int64"),
            "row_id": _row_id_values(frame, schema, positions),
            "reason_code": _reason_values(reason, frame, positions, rule_id),
            "row_values_json": row_values,
        }
    )


def duplicate_group_table(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    normalised: pd.DataFrame,
    *,
    finding_id: str,
    rule_id: str,
) -> pd.DataFrame:
    columns = list(normalised.columns)
    grouped = normalised.groupby(columns, dropna=False, sort=False)
    records: list[dict[str, Any]] = []
    for positions in grouped.indices.values():
        if len(positions) < 2:
            continue
        ordered = np.sort(np.asarray(positions, dtype="int64"))
        key_payload = {
            column: _json_value(normalised[column].iloc[int(ordered[0])])
            for column in columns
        }
        key_json = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
        group_id = (
            f"{rule_id}:"
            f"{hashlib.sha1(key_json.encode('utf-8')).hexdigest()[:16]}"
        )
        row_ids = _row_id_values(frame, schema, ordered)
        for rank, (position, row_id) in enumerate(
            zip(ordered, row_ids, strict=True), start=1
        ):
            records.append(
                {
                    "finding_id": finding_id,
                    "rule_id": rule_id,
                    "selection_type": "duplicate_group",
                    "group_id": group_id,
                    "duplicate_key": key_json,
                    "row_index": int(position),
                    "row_id": row_id,
                    "group_rank": rank,
                    "group_size": len(ordered),
                    "is_candidate_keep": rank == 1,
                    "is_redundant": rank > 1,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "finding_id",
            "rule_id",
            "selection_type",
            "group_id",
            "duplicate_key",
            "row_index",
            "row_id",
            "group_rank",
            "group_size",
            "is_candidate_keep",
            "is_redundant",
        ],
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("._")
    return cleaned[:180] or "artifact"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table_artifact(
    table: pd.DataFrame,
    *,
    artifact_dir: str | Path | None,
    dataset_id: str,
    artifact_id: str,
    kind: ArtifactKind,
    finding_id: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact | None:
    if artifact_dir is None:
        return None
    directory = Path(artifact_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_filename(artifact_id)}.parquet"
    table.to_parquet(path, index=False, engine="fastparquet")
    artifact_metadata = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        **(metadata or {}),
    }
    return Artifact(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.as_posix(),
        media_type="application/vnd.apache.parquet",
        sha256=_sha256_file(path),
        row_count=int(len(table)),
        dataset_id=dataset_id,
        finding_ids=[finding_id] if finding_id else [],
        description=description,
        metadata=artifact_metadata,
    )


def attach_selection_artifact(
    finding: Finding,
    table: pd.DataFrame,
    *,
    artifact_dir: str | Path | None,
    dataset_id: str,
    kind: ArtifactKind,
    description: str,
    metadata: dict[str, Any] | None = None,
) -> Artifact | None:
    artifact_id = f"selection.{dataset_id}.{finding.finding_id}"
    artifact = write_table_artifact(
        table,
        artifact_dir=artifact_dir,
        dataset_id=dataset_id,
        artifact_id=artifact_id,
        kind=kind,
        finding_id=finding.finding_id,
        description=description,
        metadata={
            "finding_id": finding.finding_id,
            "rule_id": finding.rule_id,
            "issue_type": finding.issue_type.value,
            **(metadata or {}),
        },
    )
    if artifact is not None:
        finding.selection_artifact_id = artifact.artifact_id
    return artifact


def sample_row_ids(
    frame: pd.DataFrame,
    mask: pd.Series,
    schema: SchemaProfile,
    limit: int,
) -> list[str]:
    if limit == 0 or not bool(mask.any()):
        return []
    positions = _mask_positions(mask, frame)[:limit]
    return _row_id_values(frame, schema, positions)


def severity_for_rate(
    rate: float,
    *,
    medium: float = 0.2,
    high: float = 0.5,
    critical: float = 0.9,
) -> Severity:
    if rate >= critical:
        return Severity.CRITICAL
    if rate >= high:
        return Severity.HIGH
    if rate >= medium:
        return Severity.MEDIUM
    return Severity.LOW


def severity_order() -> dict[Severity, int]:
    return {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
        Severity.INFO: 4,
    }


def failed_tool_result(
    *,
    tool_name: str,
    tool_version: str,
    dataset_id: str,
    started_at: float,
    code: str,
    message: str,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        tool_version=tool_version,
        dataset_id=dataset_id,
        status=ToolStatus.FAILED,
        warnings=warnings or [],
        error_code=code,
        error_message=message,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata=metadata or {},
    )


def missing_schema_columns(frame: pd.DataFrame, schema: SchemaProfile) -> list[str]:
    return [column.name for column in schema.columns if column.name not in frame.columns]


__all__ = [
    "SELECTION_SCHEMA_VERSION",
    "attach_selection_artifact",
    "cell_selection_table",
    "duplicate_group_table",
    "failed_tool_result",
    "missing_schema_columns",
    "row_selection_table",
    "sample_row_ids",
    "severity_for_rate",
    "severity_order",
    "write_table_artifact",
]
