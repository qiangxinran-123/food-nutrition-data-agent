from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
import pandas.api.types as ptypes
from pydantic import BaseModel, ConfigDict, Field

from ..config import DuplicateRuleConfig
from ..contracts import (
    Artifact,
    ArtifactKind,
    Finding,
    FindingScope,
    IssueType,
    SchemaProfile,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from ..io import LoadedDataset
from .common import (
    attach_selection_artifact,
    duplicate_group_table,
    failed_tool_result,
    missing_schema_columns,
    sample_row_ids,
    severity_order,
)


class DuplicateCheckOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_limit: int = Field(default=10, ge=0, le=50)
    sample_group_limit: int = Field(default=5, ge=0, le=20)
    group_row_limit: int = Field(default=10, ge=1, le=50)
    artifact_dir: str | None = None


DUPLICATE_CHECK_TOOL_SPEC = ToolSpec(
    name="check_duplicates",
    version="0.2.0",
    category=ToolCategory.DETECTION,
    description=(
        "Detect exact, primary-key, and content-level duplicate groups without "
        "removing rows."
    ),
    mutates_data=False,
    input_schema=DuplicateCheckOptions.model_json_schema(),
    output_schema=ToolResult.model_json_schema(),
)


def _rule_columns(rule: DuplicateRuleConfig, frame: pd.DataFrame) -> list[str]:
    if rule.use_all_columns:
        candidates = list(frame.columns)
    else:
        candidates = list(rule.columns)
    excluded = set(rule.exclude_columns)
    return [column for column in candidates if column not in excluded]


def _normalise_for_rule(
    frame: pd.DataFrame,
    columns: list[str],
    rule: DuplicateRuleConfig,
) -> pd.DataFrame:
    result = frame[columns].copy()
    if not rule.normalize_text:
        return result
    for column in columns:
        series = result[column]
        if ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series):
            strings = series.astype("string").str.strip()
            if not rule.case_sensitive:
                strings = strings.str.casefold()
            result[column] = strings
    return result


def _sample_groups(
    frame: pd.DataFrame,
    normalised: pd.DataFrame,
    columns: list[str],
    schema: SchemaProfile,
    options: DuplicateCheckOptions,
) -> list[dict[str, Any]]:
    if options.sample_group_limit == 0:
        return []
    row_id_column = schema.row_id_column
    grouped = normalised.groupby(columns, dropna=False, sort=False)
    samples: list[dict[str, Any]] = []
    for group_number, positions in enumerate(grouped.indices.values(), start=1):
        if len(positions) < 2:
            continue
        row_labels = frame.index.take(positions)
        if row_id_column is not None and row_id_column in frame.columns:
            identifiers = frame.loc[row_labels, row_id_column].astype(str).tolist()
        else:
            identifiers = [f"row_index:{label}" for label in row_labels]
        samples.append(
            {
                "group_number": group_number,
                "size": len(positions),
                "row_ids": identifiers[: options.group_row_limit],
            }
        )
        if len(samples) >= options.sample_group_limit:
            break
    return samples


def detect_duplicates(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    rules: list[DuplicateRuleConfig],
    options: DuplicateCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    started_at = perf_counter()
    if options is None:
        options = DuplicateCheckOptions()
    elif isinstance(options, dict):
        options = DuplicateCheckOptions.model_validate(options)

    if frame.columns.duplicated().any():
        return failed_tool_result(
            tool_name="check_duplicates",
            tool_version="0.2.0",
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="duplicate_columns",
            message="dataframe contains duplicate column names",
        )
    missing_columns = missing_schema_columns(frame, schema)
    if missing_columns:
        return failed_tool_result(
            tool_name="check_duplicates",
            tool_version="0.2.0",
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="schema_mismatch",
            message=f"dataframe is missing schema columns: {missing_columns}",
        )

    findings: list[Finding] = []
    artifacts: list[Artifact] = []
    rule_summaries: list[dict[str, Any]] = []
    rule_masks: list[pd.Series] = []
    warnings: list[str] = []

    for rule in rules:
        columns = _rule_columns(rule, frame)
        if not columns:
            warnings.append(f"duplicate rule {rule.rule_id!r} selected no columns")
            continue

        normalised = _normalise_for_rule(frame, columns, rule)
        redundant_mask = normalised.duplicated(keep="first")
        group_sizes = normalised.groupby(columns, dropna=False, sort=False).size()
        duplicate_sizes = group_sizes[group_sizes > 1]
        group_count = int(len(duplicate_sizes))
        rows_in_groups = int(duplicate_sizes.sum())
        redundant_rows = rows_in_groups - group_count
        rule_masks.append(redundant_mask)

        summary = {
            "rule_id": rule.rule_id,
            "columns_compared": columns,
            "duplicate_group_count": group_count,
            "rows_in_duplicate_groups": rows_in_groups,
            "redundant_rows": redundant_rows,
        }
        rule_summaries.append(summary)
        if redundant_rows == 0:
            continue

        finding = Finding(
            finding_id=f"duplicate.{rule.rule_id}",
            issue_type=IssueType.DUPLICATE,
            severity=rule.severity,
            scope=FindingScope.DATASET,
            columns=columns,
            affected_rows=redundant_rows,
            affected_rate=round(redundant_rows / len(frame), 8) if len(frame) else 0.0,
            evidence={
                **summary,
                "duplicate_group_rate": (
                    round(group_count / len(frame), 8) if len(frame) else 0.0
                ),
                "sample_groups": _sample_groups(
                    frame, normalised, columns, schema, options
                ),
                "normalisation": {
                    "normalize_text": rule.normalize_text,
                    "case_sensitive": rule.case_sensitive,
                },
            },
            sample_row_ids=sample_row_ids(
                frame, redundant_mask, schema, options.sample_limit
            ),
            suggested_actions=(
                ["resolve_primary_key_conflict", "review_duplicate_groups"]
                if rule.rule_id in {"primary_key", "barcode_unique"}
                else ["review_duplicate_groups", "choose_retention_rule"]
            ),
            confidence=1.0 if not rule.normalize_text else 0.95,
            rule_id=f"duplicate.{rule.rule_id}",
            metadata={"description": rule.description},
        )
        artifact = attach_selection_artifact(
            finding,
            duplicate_group_table(
                frame,
                schema,
                normalised,
                finding_id=finding.finding_id,
                rule_id=finding.rule_id or finding.finding_id,
            ),
            artifact_dir=options.artifact_dir,
            dataset_id=schema.dataset_id,
            kind=ArtifactKind.GROUP_LIST,
            description=(
                f"Complete duplicate groups for {finding.finding_id}; every group "
                "member is listed with candidate-keep and redundant flags."
            ),
            metadata={
                "columns_compared": columns,
                "duplicate_group_count": group_count,
                "rows_in_duplicate_groups": rows_in_groups,
                "redundant_rows": redundant_rows,
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
    for mask in rule_masks:
        union_mask = union_mask | mask
    if findings and options.artifact_dir is None:
        warnings.append(
            "artifact_dir was not provided; complete duplicate groups were not persisted"
        )

    return ToolResult(
        tool_name="check_duplicates",
        tool_version="0.2.0",
        dataset_id=schema.dataset_id,
        status=ToolStatus.SUCCESS,
        findings=findings,
        artifacts=artifacts,
        summary={
            "has_duplicates": bool(union_mask.any()),
            "row_count": int(len(frame)),
            "rule_count": len(rules),
            "rules_with_duplicates": sum(
                1 for summary in rule_summaries if summary["redundant_rows"] > 0
            ),
            "rows_flagged_union": int(union_mask.sum()),
            "rules": rule_summaries,
        },
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={"options": options.model_dump(mode="json")},
    )


def check_duplicates(
    dataset: LoadedDataset,
    options: DuplicateCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    return detect_duplicates(
        dataset.frame,
        dataset.schema,
        dataset.config.duplicates,
        options,
    )


__all__ = [
    "DUPLICATE_CHECK_TOOL_SPEC",
    "DuplicateCheckOptions",
    "check_duplicates",
    "detect_duplicates",
]
