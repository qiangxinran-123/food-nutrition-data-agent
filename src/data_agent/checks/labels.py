from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..config import LabelColumnPolicy, LabelPolicy
from ..contracts import (
    Artifact,
    ArtifactKind,
    ColumnSpec,
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
from ..masking import as_bool_mask
from .common import (
    attach_selection_artifact,
    cell_selection_table,
    failed_tool_result,
    missing_schema_columns,
    sample_row_ids,
    severity_for_rate,
    severity_order,
)


class LabelCheckOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_limit: int = Field(default=10, ge=0, le=50)
    artifact_dir: str | None = None


LABEL_CHECK_TOOL_SPEC = ToolSpec(
    name="check_labels",
    version="0.2.0",
    category=ToolCategory.DETECTION,
    description=(
        "Detect invalid, null, or explicitly unavailable values in label columns."
    ),
    mutates_data=False,
    input_schema=LabelCheckOptions.model_json_schema(),
    output_schema=ToolResult.model_json_schema(),
)


@dataclass(frozen=True)
class LabelValueAnalysis:
    normalised: pd.Series
    allowed_values: list[str]
    valid_mask: pd.Series
    unavailable_mask: pd.Series
    invalid_mask: pd.Series
    null_count: int
    empty_count: int
    unavailable_counts: dict[str, int]
    invalid_counts: dict[str, int]


def _normalise(series: pd.Series, policy: LabelColumnPolicy) -> pd.Series:
    strings = series.astype("string")
    if policy.trim_whitespace:
        strings = strings.str.strip()
    if not policy.case_sensitive:
        strings = strings.str.casefold()
    return strings


def _normalise_token(value: str, policy: LabelColumnPolicy) -> str:
    token = value.strip() if policy.trim_whitespace else value
    return token if policy.case_sensitive else token.casefold()


def analyse_label_column(
    frame: pd.DataFrame,
    spec: ColumnSpec,
    policy: LabelColumnPolicy,
) -> LabelValueAnalysis:
    series = frame[spec.name]
    raw_normalised = _normalise(series, policy)
    aliases = {
        _normalise_token(source, policy): _normalise_token(target, policy)
        for source, target in policy.aliases.items()
    }
    normalised = raw_normalised.map(
        lambda value: aliases.get(value, value) if pd.notna(value) else value
    ).astype("string")
    index = frame.index

    null_mask = as_bool_mask(series.isna(), index)
    empty_mask = as_bool_mask(normalised.eq(""), index) & (~null_mask)
    unavailable_mask = null_mask | empty_mask
    unavailable_counts: dict[str, int] = {
        "null": int(null_mask.sum()),
        "empty_string": int(empty_mask.sum()),
    }
    for token in policy.unavailable_tokens:
        current = as_bool_mask(
            normalised.eq(_normalise_token(token, policy)), index
        )
        unavailable_counts[token] = int(current.sum())
        unavailable_mask = unavailable_mask | current

    allowed_values = [str(value) for value in spec.allowed_values]
    allowed_normalised = {_normalise_token(value, policy) for value in allowed_values}
    allowed_mask = as_bool_mask(normalised.isin(allowed_normalised), index)
    invalid_mask = (~unavailable_mask) & (~allowed_mask)
    invalid_counts = {
        str(value): int(count)
        for value, count in raw_normalised[invalid_mask].value_counts().items()
    }
    return LabelValueAnalysis(
        normalised=normalised,
        allowed_values=allowed_values,
        valid_mask=allowed_mask,
        unavailable_mask=unavailable_mask,
        invalid_mask=invalid_mask,
        null_count=int(null_mask.sum()),
        empty_count=int(empty_mask.sum()),
        unavailable_counts={
            key: value for key, value in unavailable_counts.items() if value > 0
        },
        invalid_counts=invalid_counts,
    )


def label_specs(schema: SchemaProfile, policy: LabelPolicy) -> list[ColumnSpec]:
    configured = set(policy.columns)
    return [
        spec
        for spec in schema.columns
        if spec.name in configured or spec.role == "label"
    ]


def detect_label_errors(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    policy: LabelPolicy,
    options: LabelCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    started_at = perf_counter()
    if options is None:
        options = LabelCheckOptions()
    elif isinstance(options, dict):
        options = LabelCheckOptions.model_validate(options)

    missing_columns = missing_schema_columns(frame, schema)
    if missing_columns:
        return failed_tool_result(
            tool_name="check_labels",
            tool_version="0.2.0",
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="schema_mismatch",
            message=f"dataframe is missing schema columns: {missing_columns}",
        )

    findings: list[Finding] = []
    artifacts: list[Artifact] = []
    summaries: list[dict[str, Any]] = []
    masks: list[pd.Series] = []
    warnings: list[str] = []

    for spec in label_specs(schema, policy):
        if not spec.allowed_values:
            warnings.append(
                f"skipped label column {spec.name!r}: no allowed_values configured"
            )
            continue
        column_policy = policy.columns.get(spec.name, LabelColumnPolicy())
        analysis = analyse_label_column(frame, spec, column_policy)
        total = len(frame)
        unavailable_count = int(analysis.unavailable_mask.sum())
        invalid_count = int(analysis.invalid_mask.sum())
        valid_count = int(analysis.valid_mask.sum())
        summaries.append(
            {
                "column": spec.name,
                "valid_rows": valid_count,
                "unavailable_rows": unavailable_count,
                "invalid_rows": invalid_count,
                "allowed_values": analysis.allowed_values,
                "unavailable_counts": analysis.unavailable_counts,
                "invalid_counts": analysis.invalid_counts,
            }
        )

        if unavailable_count > 0:
            masks.append(analysis.unavailable_mask)
            rate = unavailable_count / total if total else 0.0
            finding = Finding(
                finding_id=f"label.unavailable.{spec.name}",
                issue_type=IssueType.LABEL,
                severity=severity_for_rate(
                    rate,
                    medium=policy.severity_medium_threshold,
                    high=policy.severity_high_threshold,
                    critical=policy.severity_critical_threshold,
                ),
                scope=FindingScope.COLUMN,
                columns=[spec.name],
                affected_rows=unavailable_count,
                affected_cells=unavailable_count,
                affected_rate=round(rate, 8),
                evidence={
                    "method": "label_availability",
                    "allowed_values": analysis.allowed_values,
                    "unavailable_counts": analysis.unavailable_counts,
                    "valid_rows": valid_count,
                },
                sample_row_ids=sample_row_ids(
                    frame,
                    analysis.unavailable_mask,
                    schema,
                    options.sample_limit,
                ),
                suggested_actions=[
                    "exclude_from_supervised_training",
                    "collect_or_impute_labels",
                ],
                confidence=1.0,
                rule_id=f"label.unavailable.{spec.name}",
            )
            reason = pd.Series("unavailable_label", index=frame.index, dtype="object")
            reason.loc[frame[spec.name].isna()] = "label_null"
            artifact = attach_selection_artifact(
                finding,
                cell_selection_table(
                    frame,
                    schema,
                    finding_id=finding.finding_id,
                    rule_id=finding.rule_id or finding.finding_id,
                    mask=analysis.unavailable_mask,
                    column=spec.name,
                    reason=reason,
                ),
                artifact_dir=options.artifact_dir,
                dataset_id=schema.dataset_id,
                kind=ArtifactKind.CELL_LIST,
                description=(
                    f"Complete unavailable-label cells for {finding.finding_id}."
                ),
                metadata={
                    "column": spec.name,
                    "flagged_cells": unavailable_count,
                    "row_id_column": schema.row_id_column,
                },
            )
            if artifact is not None:
                artifacts.append(artifact)
            findings.append(finding)

        if invalid_count > 0:
            masks.append(analysis.invalid_mask)
            finding = Finding(
                finding_id=f"label.invalid.{spec.name}",
                issue_type=IssueType.LABEL,
                severity=Severity.HIGH,
                scope=FindingScope.COLUMN,
                columns=[spec.name],
                affected_rows=invalid_count,
                affected_cells=invalid_count,
                affected_rate=(
                    round(invalid_count / total, 8) if total else 0.0
                ),
                evidence={
                    "method": "allowed_values",
                    "allowed_values": analysis.allowed_values,
                    "invalid_counts": analysis.invalid_counts,
                },
                sample_row_ids=sample_row_ids(
                    frame,
                    analysis.invalid_mask,
                    schema,
                    options.sample_limit,
                ),
                suggested_actions=[
                    "map_to_allowed_labels",
                    "review_label_source",
                ],
                confidence=1.0,
                rule_id=f"label.invalid.{spec.name}",
            )
            artifact = attach_selection_artifact(
                finding,
                cell_selection_table(
                    frame,
                    schema,
                    finding_id=finding.finding_id,
                    rule_id=finding.rule_id or finding.finding_id,
                    mask=analysis.invalid_mask,
                    column=spec.name,
                    reason="invalid_label",
                ),
                artifact_dir=options.artifact_dir,
                dataset_id=schema.dataset_id,
                kind=ArtifactKind.CELL_LIST,
                description=f"Complete invalid-label cells for {finding.finding_id}.",
                metadata={
                    "column": spec.name,
                    "flagged_cells": invalid_count,
                    "row_id_column": schema.row_id_column,
                },
            )
            if artifact is not None:
                artifacts.append(artifact)
            findings.append(finding)

    order = severity_order()
    findings.sort(
        key=lambda finding: (order[finding.severity], -finding.affected_rows)
    )
    union_mask = pd.Series(False, index=frame.index, dtype=bool)
    for mask in masks:
        union_mask = union_mask | mask
    if findings and options.artifact_dir is None:
        warnings.append(
            "artifact_dir was not provided; complete cell selections were not persisted"
        )
    return ToolResult(
        tool_name="check_labels",
        tool_version="0.2.0",
        dataset_id=schema.dataset_id,
        status=ToolStatus.SUCCESS,
        findings=findings,
        artifacts=artifacts,
        summary={
            "has_label_issues": bool(union_mask.any()),
            "row_count": int(len(frame)),
            "label_columns_checked": len(summaries),
            "columns_with_unavailable_labels": sum(
                1 for summary in summaries if summary["unavailable_rows"] > 0
            ),
            "columns_with_invalid_labels": sum(
                1 for summary in summaries if summary["invalid_rows"] > 0
            ),
            "rows_flagged_union": int(union_mask.sum()),
            "labels": summaries,
        },
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={"options": options.model_dump(mode="json")},
    )


def check_labels(
    dataset: LoadedDataset,
    options: LabelCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    return detect_label_errors(
        dataset.frame,
        dataset.schema,
        dataset.config.labels,
        options,
    )


__all__ = [
    "LABEL_CHECK_TOOL_SPEC",
    "LabelCheckOptions",
    "LabelValueAnalysis",
    "analyse_label_column",
    "check_labels",
    "detect_label_errors",
    "label_specs",
]
