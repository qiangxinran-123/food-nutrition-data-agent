from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts import (
    Artifact,
    ArtifactKind,
    Finding,
    FindingScope,
    IssueType,
    SchemaProfile,
    Severity,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from ..io import LoadedDataset
from ..masking import build_missing_mask
from .common import (
    attach_selection_artifact,
    cell_selection_table,
    sample_row_ids,
    severity_for_rate,
    severity_order,
)


class MissingCheckOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_limit: int = Field(default=10, ge=0, le=50)
    treat_empty_string_as_missing: bool = True
    trim_whitespace: bool = True
    case_sensitive: bool = False
    medium_threshold: float = Field(default=0.2, ge=0, le=1)
    high_threshold: float = Field(default=0.5, ge=0, le=1)
    critical_threshold: float = Field(default=0.9, ge=0, le=1)
    artifact_dir: str | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> MissingCheckOptions:
        if not (
            self.medium_threshold
            <= self.high_threshold
            <= self.critical_threshold
        ):
            raise ValueError(
                "missing thresholds must satisfy medium <= high <= critical"
            )
        return self

    @classmethod
    def from_schema(cls, schema: SchemaProfile) -> MissingCheckOptions:
        policy = schema.metadata.get("missing_policy", {})
        return cls(
            treat_empty_string_as_missing=bool(
                policy.get("treat_empty_string_as_missing", True)
            ),
            trim_whitespace=bool(policy.get("trim_whitespace", True)),
            case_sensitive=bool(policy.get("case_sensitive", False)),
            medium_threshold=float(policy.get("severity_medium_threshold", 0.2)),
            high_threshold=float(policy.get("severity_high_threshold", 0.5)),
            critical_threshold=float(policy.get("severity_critical_threshold", 0.9)),
        )


MISSING_CHECK_TOOL_SPEC = ToolSpec(
    name="check_missing",
    version="0.2.0",
    category=ToolCategory.DETECTION,
    description=(
        "Detect nulls, empty strings, and schema-declared missing tokens without "
        "modifying the dataset."
    ),
    mutates_data=False,
    input_schema=MissingCheckOptions.model_json_schema(),
    output_schema=ToolResult.model_json_schema(),
)


def _suggested_actions(logical_type: str, nullable: bool) -> list[str]:
    if not nullable:
        return ["investigate_required_field", "repair_missing_values"]
    if logical_type == "numeric":
        return ["keep_missing", "impute_with_domain_strategy", "review_missing_pattern"]
    if logical_type in {"categorical", "boolean", "datetime"}:
        return ["keep_missing", "impute_mode_or_explicit_unknown", "review_source"]
    if logical_type == "identifier":
        return ["investigate_required_field", "review_missing_pattern"]
    return ["keep_missing", "review_source", "review_missing_pattern"]


def _failed_result(
    *,
    dataset_id: str,
    started_at: float,
    code: str,
    message: str,
    warnings: list[str] | None = None,
) -> ToolResult:
    return ToolResult(
        tool_name="check_missing",
        tool_version="0.2.0",
        dataset_id=dataset_id,
        status=ToolStatus.FAILED,
        warnings=warnings or [],
        error_code=code,
        error_message=message,
        duration_ms=(perf_counter() - started_at) * 1000,
    )


def detect_missing(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    options: MissingCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    started_at = perf_counter()
    if options is None:
        options = MissingCheckOptions.from_schema(schema)
    elif isinstance(options, dict):
        options = MissingCheckOptions.model_validate(options)

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        return _failed_result(
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="duplicate_columns",
            message=f"dataframe contains duplicate column names: {duplicates}",
        )

    schema_columns = [column.name for column in schema.columns]
    missing_columns = [name for name in schema_columns if name not in frame.columns]
    if missing_columns:
        return _failed_result(
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="schema_mismatch",
            message=f"dataframe is missing schema columns: {missing_columns}",
        )

    extra_columns = [name for name in frame.columns if name not in set(schema_columns)]
    warnings = (
        [f"columns not described by schema were ignored: {extra_columns}"]
        if extra_columns
        else []
    )
    if frame.empty:
        warnings.append("dataset has zero rows")

    masks: dict[str, pd.Series] = {}
    findings: list[Finding] = []
    artifacts: list[Artifact] = []
    column_summaries: list[dict[str, Any]] = []
    token_policy: dict[str, list[str]] = {}

    for spec in schema.columns:
        series = frame[spec.name]
        mask_info = build_missing_mask(
            series,
            spec,
            treat_empty_string_as_missing=options.treat_empty_string_as_missing,
            trim_whitespace=options.trim_whitespace,
            case_sensitive=options.case_sensitive,
        )
        null_mask = mask_info.null
        empty_mask = mask_info.empty_string
        token_mask = mask_info.declared_token
        token_counts = mask_info.token_counts
        if spec.missing_tokens:
            token_policy[spec.name] = list(spec.missing_tokens)

        missing_mask = mask_info.total
        masks[spec.name] = missing_mask
        affected_rows = int(missing_mask.sum())
        if affected_rows == 0:
            continue

        row_count = int(len(frame))
        missing_rate = affected_rows / row_count if row_count else 0.0
        severity = severity_for_rate(
            missing_rate,
            medium=options.medium_threshold,
            high=options.high_threshold,
            critical=options.critical_threshold,
        )
        if spec.nullable is False:
            severity = Severity.CRITICAL

        reason = pd.Series("declared_token", index=frame.index, dtype="object")
        reason.loc[empty_mask] = "empty_string"
        reason.loc[null_mask] = "null"
        finding = Finding(
            finding_id=f"missing.{spec.name}",
            issue_type=IssueType.MISSING,
            severity=severity,
            scope=FindingScope.COLUMN,
            columns=[spec.name],
            affected_rows=affected_rows,
            affected_cells=affected_rows,
            affected_rate=round(missing_rate, 6),
            evidence={
                "missing_count": affected_rows,
                "missing_rate": round(missing_rate, 6),
                "null_count": int(null_mask.sum()),
                "empty_string_count": int(empty_mask.sum()),
                "declared_token_count": int(token_mask.sum()),
                "missing_token_counts": {
                    token: count for token, count in token_counts.items() if count > 0
                },
                "non_null_count": int((~null_mask).sum()),
                "nullable": spec.nullable,
                "logical_type": spec.logical_type.value,
            },
            sample_row_ids=sample_row_ids(
                frame, missing_mask, schema, options.sample_limit
            ),
            suggested_actions=_suggested_actions(
                spec.logical_type.value, spec.nullable
            ),
            confidence=1.0,
            rule_id="missing.null_empty_or_declared_token",
            metadata={"physical_dtype": spec.physical_dtype},
        )
        artifact = attach_selection_artifact(
            finding,
            cell_selection_table(
                frame,
                schema,
                finding_id=finding.finding_id,
                rule_id=finding.rule_id or finding.finding_id,
                mask=missing_mask,
                column=spec.name,
                reason=reason,
            ),
            artifact_dir=options.artifact_dir,
            dataset_id=schema.dataset_id,
            kind=ArtifactKind.CELL_LIST,
            description=(
                f"Complete cell selection for {finding.finding_id}; one row per "
                "affected cell with stable row and column locations."
            ),
            metadata={
                "column": spec.name,
                "flagged_cells": affected_rows,
                "row_id_column": schema.row_id_column,
            },
        )
        if artifact is not None:
            artifacts.append(artifact)
        findings.append(finding)
        column_summaries.append(
            {
                "column": spec.name,
                "missing_count": affected_rows,
                "missing_rate": round(missing_rate, 6),
                "severity": severity.value,
            }
        )

    order = severity_order()
    findings.sort(
        key=lambda finding: (
            order[finding.severity],
            -finding.affected_rate,
            finding.columns[0],
        )
    )
    column_summaries.sort(
        key=lambda item: (-item["missing_count"], item["column"])
    )

    row_count = int(len(frame))
    incomplete_rows = pd.Series(False, index=frame.index, dtype=bool)
    for mask in masks.values():
        incomplete_rows = incomplete_rows | mask
    rows_with_missing = int(incomplete_rows.sum())
    cells_checked = row_count * len(schema.columns)
    missing_cells = sum(int(mask.sum()) for mask in masks.values())
    severity_counts = {severity.value: 0 for severity in Severity}
    for finding in findings:
        severity_counts[finding.severity.value] += 1
    if findings and options.artifact_dir is None:
        warnings.append(
            "artifact_dir was not provided; complete cell selections were not persisted"
        )

    return ToolResult(
        tool_name="check_missing",
        tool_version="0.2.0",
        dataset_id=schema.dataset_id,
        status=ToolStatus.SUCCESS,
        findings=findings,
        artifacts=artifacts,
        summary={
            "has_missing": missing_cells > 0,
            "row_count": row_count,
            "column_count": len(schema.columns),
            "cells_checked": cells_checked,
            "missing_cells": missing_cells,
            "missing_cell_rate": (
                round(missing_cells / cells_checked, 8) if cells_checked else 0.0
            ),
            "rows_with_missing": rows_with_missing,
            "complete_rows": row_count - rows_with_missing,
            "complete_row_rate": (
                round((row_count - rows_with_missing) / row_count, 8)
                if row_count
                else 0.0
            ),
            "columns_with_missing": len(findings),
            "severity_counts": severity_counts,
            "affected_columns": column_summaries,
            "declared_missing_tokens": token_policy,
        },
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={"options": options.model_dump(mode="json")},
    )


def check_missing(
    dataset: LoadedDataset,
    options: MissingCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    return detect_missing(dataset.frame, dataset.schema, options)


__all__ = [
    "MISSING_CHECK_TOOL_SPEC",
    "MissingCheckOptions",
    "check_missing",
    "detect_missing",
]
