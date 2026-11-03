"""Build runtime field rules from dataset configuration."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandas.api.types as ptypes

from .config import ColumnConfig, DatasetConfig
from .contracts import ColumnSpec, DatasetRef, LogicalType, SchemaProfile


def _physical_dtype(series: pd.Series) -> str:
    dtype = str(series.dtype)
    if dtype in {"object", "string", "str"}:
        return "str"
    if dtype.startswith("datetime"):
        return "datetime"
    if dtype == "boolean":
        return "bool"
    return dtype


def _infer_logical_type(series: pd.Series, name: str) -> LogicalType:
    """Infer a logical type without scanning the column values."""
    if ptypes.is_bool_dtype(series):
        return LogicalType.BOOLEAN
    if ptypes.is_datetime64_any_dtype(series):
        return LogicalType.DATETIME
    if ptypes.is_numeric_dtype(series):
        return LogicalType.NUMERIC

    normalised_name = name.strip().lower()
    identifier_like = (
        normalised_name in {"id", "code", "key", "uuid"}
        or normalised_name.endswith("_id")
    )
    if identifier_like:
        return LogicalType.IDENTIFIER
    return LogicalType.TEXT


def _merge_missing_tokens(
    global_tokens: list[str],
    policy_tokens: list[str],
    column_config: ColumnConfig | None,
) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    candidates = list(global_tokens)
    candidates.extend(policy_tokens)
    if column_config is not None:
        candidates.extend(column_config.missing_tokens)
    for token in candidates:
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tokens


def _rule_metadata(column_config: ColumnConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if column_config.format_regex is not None:
        metadata["format_regex"] = column_config.format_regex
    if column_config.datetime_format is not None:
        metadata["datetime_format"] = column_config.datetime_format
    if column_config.dayfirst is not None:
        metadata["dayfirst"] = column_config.dayfirst
    if column_config.yearfirst is not None:
        metadata["yearfirst"] = column_config.yearfirst
    if column_config.boolean_values is not None:
        metadata["boolean_values"] = column_config.boolean_values
    metadata.update(column_config.metadata)
    return metadata


def build_schema_profile(
    dataset: DatasetRef,
    frame: pd.DataFrame,
    config: DatasetConfig | None = None,
    *,
    ingestion_metadata: dict[str, Any] | None = None,
) -> SchemaProfile:
    if config is not None and config.dataset_id != dataset.dataset_id:
        raise ValueError(
            "dataset config id does not match dataset reference: "
            f"{config.dataset_id!r} != {dataset.dataset_id!r}"
        )
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"dataframe contains duplicate column names: {duplicates}")

    configured_columns = config.columns if config is not None else {}
    missing_configured = set(configured_columns) - set(frame.columns)
    if missing_configured:
        raise ValueError(
            f"dataset config references columns absent from data: {sorted(missing_configured)}"
        )

    ingestion = dict(ingestion_metadata or {})
    raw_columns = set(ingestion.get("raw_columns_added", []))

    global_tokens = config.missing.tokens if config is not None else []
    policy = config.missing if config is not None else None
    columns: list[ColumnSpec] = []
    for name in frame.columns:
        if name in raw_columns:
            continue
        series = frame[name]
        column_config = configured_columns.get(name)

        logical_type = (
            column_config.logical_type
            if column_config is not None and column_config.logical_type is not None
            else _infer_logical_type(series, name)
        )
        nullable = (
            column_config.nullable
            if column_config is not None and column_config.nullable is not None
            else True
        )
        unique = (
            column_config.unique
            if column_config is not None and column_config.unique is not None
            else False
        )

        columns.append(
            ColumnSpec(
                name=name,
                role=column_config.role if column_config is not None else None,
                logical_type=logical_type,
                physical_dtype=_physical_dtype(series),
                unit=column_config.unit if column_config is not None else None,
                nullable=nullable,
                unique=unique,
                missing_tokens=_merge_missing_tokens(
                    global_tokens,
                    (
                        policy.tokens_by_column.get(name, [])
                        if policy is not None
                        else []
                    ),
                    column_config,
                ),
                allowed_values=(
                    list(column_config.allowed_values)
                    if column_config is not None
                    else []
                ),
                range=column_config.range if column_config is not None else None,
                description=(
                    column_config.description if column_config is not None else None
                ),
                metadata=_rule_metadata(column_config) if column_config is not None else {},
            )
        )

    return SchemaProfile(
        dataset_id=dataset.dataset_id,
        domain_pack=config.domain_pack if config is not None else None,
        columns=columns,
        primary_key_candidates=(
            [list(candidate) for candidate in config.primary_key_candidates]
            if config is not None
            else []
        ),
        row_id_column=config.row_id_column if config is not None else None,
        source_row_column=config.source_row_column if config is not None else None,
        metadata={
            "row_count": int(len(frame)),
            "source_uri": dataset.uri,
            "missing_policy": (
                {
                    "treat_empty_string_as_missing": policy.treat_empty_string_as_missing,
                    "trim_whitespace": policy.trim_whitespace,
                    "case_sensitive": policy.case_sensitive,
                    "severity_medium_threshold": policy.severity_medium_threshold,
                    "severity_high_threshold": policy.severity_high_threshold,
                    "severity_critical_threshold": policy.severity_critical_threshold,
                }
                if policy is not None
                else {}
            ),
            "config_dataset_id": config.dataset_id if config is not None else None,
            "ingestion": ingestion,
            "raw_columns": sorted(raw_columns),
        },
    )


__all__ = ["build_schema_profile"]
