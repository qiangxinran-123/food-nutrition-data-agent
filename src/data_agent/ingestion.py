from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import IngestionPolicy


class IngestionError(ValueError):
    pass


@dataclass(frozen=True)
class IngestionResult:
    frame: pd.DataFrame
    source_column_map: dict[str, str] = field(default_factory=dict)
    raw_columns_added: list[str] = field(default_factory=list)
    row_id_generated: bool = False
    source_row_count: int = 0
    source_column_count: int = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "source_column_map": self.source_column_map,
            "raw_columns_added": self.raw_columns_added,
            "row_id_generated": self.row_id_generated,
            "source_row_count": self.source_row_count,
            "source_column_count": self.source_column_count,
        }


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _identifier_text(value: Any) -> str | pd.NA:
    if _is_missing_scalar(value):
        return pd.NA
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
    return str(value)


def _stringify_identifier(series: pd.Series) -> pd.Series:
    return series.map(_identifier_text).astype("string")


def apply_ingestion_policy(
    frame: pd.DataFrame,
    policy: IngestionPolicy,
    *,
    row_id_column: str | None = None,
) -> IngestionResult:
    source_row_count = int(len(frame))
    source_column_count = int(len(frame.columns))
    result = frame.copy()

    if policy.strip_bom:
        result.columns = [
            str(column).lstrip("\ufeff") if isinstance(column, str) else column
            for column in result.columns
        ]

    if policy.drop_unnamed_columns:
        unnamed = [
            column
            for column in result.columns
            if isinstance(column, str)
            and (not column.strip() or re.fullmatch(r"Unnamed:\s*\d+", column))
        ]
        if unnamed:
            result = result.drop(columns=unnamed)

    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise IngestionError(f"source columns are not unique: {duplicates}")

    rename_map: dict[str, str] = {}
    source_column_map: dict[str, str] = {}
    present = list(result.columns)
    if policy.case_sensitive_columns:
        lookup = {column: column for column in present}
    else:
        lookup = {str(column).casefold(): column for column in present}

    for source, target in policy.column_aliases.items():
        lookup_key = source if policy.case_sensitive_columns else source.casefold()
        actual = lookup.get(lookup_key)
        if actual is None:
            continue
        rename_map[actual] = target
        source_column_map[str(actual)] = target

    if rename_map:
        result = result.rename(columns=rename_map)
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise IngestionError(f"column aliases create duplicate names: {duplicates}")

    missing_identifiers = [
        column for column in policy.identifier_columns if column not in result.columns
    ]
    if missing_identifiers:
        raise IngestionError(
            f"identifier columns are absent from the source: {missing_identifiers}"
        )

    raw_columns_added: list[str] = []
    raw_targets = (
        [column for column in result.columns if not str(column).startswith("__")]
        if "*" in policy.preserve_raw_columns
        else list(policy.preserve_raw_columns)
    )
    for column in raw_targets:
        if column not in result.columns:
            raise IngestionError(f"raw-copy column is absent: {column!r}")
        raw_name = f"{policy.raw_column_prefix}{column}"
        if raw_name in result.columns:
            raise IngestionError(f"raw-copy column already exists: {raw_name!r}")
        result[raw_name] = result[column].copy()
        raw_columns_added.append(raw_name)

    for column in policy.identifier_columns:
        result[column] = _stringify_identifier(result[column])

    row_id_generated = False
    if policy.ensure_row_id:
        if row_id_column is None:
            raise IngestionError(
                "row_id_column must be configured when ingestion.ensure_row_id is true"
            )
        if row_id_column not in result.columns:
            result[row_id_column] = [
                f"{policy.row_id_prefix}_{index:0{policy.row_id_width}d}"
                for index in range(1, len(result) + 1)
            ]
            row_id_generated = True
        row_ids = result[row_id_column]
        if row_ids.isna().any():
            raise IngestionError("row_id_column contains missing values")
        if row_ids.duplicated().any():
            raise IngestionError("row_id_column is not unique")

    return IngestionResult(
        frame=result,
        source_column_map=source_column_map,
        raw_columns_added=raw_columns_added,
        row_id_generated=row_id_generated,
        source_row_count=source_row_count,
        source_column_count=source_column_count,
    )


__all__ = [
    "IngestionError",
    "IngestionResult",
    "apply_ingestion_policy",
]
