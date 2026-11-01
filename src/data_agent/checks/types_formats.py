from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from pydantic import BaseModel, ConfigDict, Field

from ..config import TypeFormatPolicy
from ..contracts import (
    Artifact,
    ArtifactKind,
    Finding,
    FindingScope,
    IssueType,
    LogicalType,
    SchemaProfile,
    Severity,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from ..io import LoadedDataset
from ..masking import build_missing_mask, missing_policy_from_schema
from .common import (
    attach_selection_artifact,
    cell_selection_table,
    failed_tool_result,
    missing_schema_columns,
    sample_row_ids,
    severity_order,
)


class TypeFormatCheckOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_limit: int = Field(default=10, ge=0, le=50)
    artifact_dir: str | None = None


TYPE_FORMAT_CHECK_TOOL_SPEC = ToolSpec(
    name="check_types_formats",
    version="0.2.0",
    category=ToolCategory.DETECTION,
    description=(
        "Detect regex, logical-type, numeric parsing, and non-finite format issues."
    ),
    mutates_data=False,
    input_schema=TypeFormatCheckOptions.model_json_schema(),
    output_schema=ToolResult.model_json_schema(),
)


def _finding(
    *,
    frame: pd.DataFrame,
    schema: SchemaProfile,
    column: str,
    mask: pd.Series,
    rule_id: str,
    severity: Severity,
    evidence: dict[str, Any],
    suggested_actions: list[str],
    sample_limit: int,
    reason: str | pd.Series | None,
    artifact_dir: str | None,
    artifacts: list[Artifact],
) -> Finding:
    count = int(mask.sum())
    finding = Finding(
        finding_id=rule_id,
        issue_type=IssueType.FORMAT,
        severity=severity,
        scope=FindingScope.COLUMN,
        columns=[column],
        affected_rows=count,
        affected_cells=count,
        affected_rate=round(count / len(frame), 8) if len(frame) else 0.0,
        evidence={**evidence, "flagged_count": count},
        sample_row_ids=sample_row_ids(frame, mask, schema, sample_limit),
        suggested_actions=suggested_actions,
        confidence=1.0,
        rule_id=rule_id,
    )
    artifact = attach_selection_artifact(
        finding,
        cell_selection_table(
            frame,
            schema,
            finding_id=finding.finding_id,
            rule_id=rule_id,
            mask=mask,
            column=column,
            reason=reason,
        ),
        artifact_dir=artifact_dir,
        dataset_id=schema.dataset_id,
        kind=ArtifactKind.CELL_LIST,
        description=(
            f"Complete cell selection for {finding.finding_id}; includes each "
            "flagged cell and its format reason."
        ),
        metadata={
            "column": column,
            "flagged_cells": count,
            "row_id_column": schema.row_id_column,
        },
    )
    if artifact is not None:
        artifacts.append(artifact)
    return finding


def detect_type_format_issues(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    policy: TypeFormatPolicy,
    options: TypeFormatCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    started_at = perf_counter()
    if options is None:
        options = TypeFormatCheckOptions()
    elif isinstance(options, dict):
        options = TypeFormatCheckOptions.model_validate(options)

    missing_columns = missing_schema_columns(frame, schema)
    if missing_columns:
        return failed_tool_result(
            tool_name="check_types_formats",
            tool_version="0.2.0",
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="schema_mismatch",
            message=f"dataframe is missing schema columns: {missing_columns}",
        )

    missing_policy = missing_policy_from_schema(schema)
    findings: list[Finding] = []
    artifacts: list[Artifact] = []
    masks: list[pd.Series] = []
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    policy_boolean_values = {
        str(value).strip().casefold() for value in policy.boolean_values
    }

    for spec in schema.columns:
        series = frame[spec.name]
        missing_mask = build_missing_mask(series, spec, **missing_policy).total
        present_mask = ~missing_mask

        format_regex = spec.metadata.get("format_regex")
        if format_regex:
            strings = series.astype("string")
            if missing_policy["trim_whitespace"]:
                strings = strings.str.strip()
            matched = strings.str.fullmatch(str(format_regex), na=False)
            invalid_mask = present_mask & (~matched)
            checks.append(
                {
                    "column": spec.name,
                    "check": "regex",
                    "flagged_count": int(invalid_mask.sum()),
                }
            )
            if bool(invalid_mask.any()):
                masks.append(invalid_mask)
                findings.append(
                    _finding(
                        frame=frame,
                        schema=schema,
                        column=spec.name,
                        mask=invalid_mask,
                        rule_id=f"format.regex.{spec.name}",
                        severity=policy.regex_severity,
                        evidence={
                            "expected_regex": format_regex,
                            "examples": series[present_mask].astype("string").head(3).tolist(),
                        },
                        suggested_actions=[
                            "review_format_examples",
                            "normalise_or_quarantine_values",
                        ],
                        sample_limit=options.sample_limit,
                        reason="regex_mismatch",
                        artifact_dir=options.artifact_dir,
                        artifacts=artifacts,
                    )
                )

        if policy.check_logical_types:
            if spec.logical_type is LogicalType.NUMERIC:
                parsed = pd.to_numeric(series, errors="coerce").astype("float64")
                invalid_mask = present_mask & parsed.isna()
                checks.append(
                    {
                        "column": spec.name,
                        "check": "numeric_parse",
                        "flagged_count": int(invalid_mask.sum()),
                    }
                )
                if bool(invalid_mask.any()):
                    masks.append(invalid_mask)
                    findings.append(
                        _finding(
                            frame=frame,
                            schema=schema,
                            column=spec.name,
                            mask=invalid_mask,
                            rule_id=f"format.numeric.{spec.name}",
                            severity=policy.numeric_parse_severity,
                            evidence={"expected": "finite or parseable numeric"},
                            suggested_actions=[
                                "repair_numeric_values",
                                "review_source_encoding",
                            ],
                            sample_limit=options.sample_limit,
                            reason="unparseable_numeric",
                            artifact_dir=options.artifact_dir,
                            artifacts=artifacts,
                        )
                    )

            if spec.logical_type is LogicalType.DATETIME and not ptypes.is_datetime64_any_dtype(series):
                datetime_format = spec.metadata.get(
                    "datetime_format", policy.datetime_format
                )
                dayfirst = bool(spec.metadata.get("dayfirst", policy.dayfirst))
                yearfirst = bool(spec.metadata.get("yearfirst", policy.yearfirst))
                parsed_datetime = pd.to_datetime(
                    series,
                    errors="coerce",
                    format=datetime_format,
                    dayfirst=dayfirst,
                    yearfirst=yearfirst,
                )
                invalid_mask = present_mask & parsed_datetime.isna()
                checks.append(
                    {
                        "column": spec.name,
                        "check": "datetime_parse",
                        "flagged_count": int(invalid_mask.sum()),
                    }
                )
                if bool(invalid_mask.any()):
                    masks.append(invalid_mask)
                    findings.append(
                        _finding(
                            frame=frame,
                            schema=schema,
                            column=spec.name,
                            mask=invalid_mask,
                            rule_id=f"format.datetime.{spec.name}",
                            severity=policy.datetime_severity,
                            evidence={
                                "expected": "parseable datetime",
                                "format": datetime_format,
                                "dayfirst": dayfirst,
                                "yearfirst": yearfirst,
                            },
                            suggested_actions=["standardise_datetime_format"],
                            sample_limit=options.sample_limit,
                            reason="invalid_datetime",
                            artifact_dir=options.artifact_dir,
                            artifacts=artifacts,
                        )
                    )

            if spec.logical_type is LogicalType.BOOLEAN:
                values = series.astype("string").str.strip().str.casefold()
                configured_boolean_values = spec.metadata.get("boolean_values")
                boolean_values = (
                    {
                        str(value).strip().casefold()
                        for value in configured_boolean_values
                    }
                    if configured_boolean_values
                    else policy_boolean_values
                )
                invalid_mask = present_mask & (~values.isin(boolean_values))
                checks.append(
                    {
                        "column": spec.name,
                        "check": "boolean_values",
                        "flagged_count": int(invalid_mask.sum()),
                    }
                )
                if bool(invalid_mask.any()):
                    masks.append(invalid_mask)
                    findings.append(
                        _finding(
                            frame=frame,
                            schema=schema,
                            column=spec.name,
                            mask=invalid_mask,
                            rule_id=f"format.boolean.{spec.name}",
                            severity=policy.boolean_severity,
                            evidence={"allowed_values": sorted(boolean_values)},
                            suggested_actions=["standardise_boolean_values"],
                            sample_limit=options.sample_limit,
                            reason="invalid_boolean",
                            artifact_dir=options.artifact_dir,
                            artifacts=artifacts,
                        )
                    )

        numeric_checked = spec.logical_type is LogicalType.NUMERIC
        if policy.check_numeric_finite and numeric_checked:
            numeric = pd.to_numeric(series, errors="coerce").astype("float64")
            finite = pd.Series(
                np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan)),
                index=frame.index,
            )
            invalid_mask = present_mask & numeric.notna() & (~finite)
            checks.append(
                {
                    "column": spec.name,
                    "check": "numeric_finite",
                    "flagged_count": int(invalid_mask.sum()),
                }
            )
            if bool(invalid_mask.any()):
                masks.append(invalid_mask)
                findings.append(
                    _finding(
                        frame=frame,
                        schema=schema,
                        column=spec.name,
                        mask=invalid_mask,
                        rule_id=f"format.non_finite.{spec.name}",
                        severity=policy.non_finite_severity,
                        evidence={"expected": "finite numeric values"},
                        suggested_actions=["replace_or_quarantine_non_finite_values"],
                        sample_limit=options.sample_limit,
                        reason="non_finite_numeric",
                        artifact_dir=options.artifact_dir,
                        artifacts=artifacts,
                    )
                )
        elif policy.check_numeric_finite and ptypes.is_numeric_dtype(series):
            numeric = series.astype("float64")
            finite = pd.Series(
                np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan)),
                index=frame.index,
            )
            invalid_mask = present_mask & numeric.notna() & (~finite)
            checks.append(
                {
                    "column": spec.name,
                    "check": "numeric_finite",
                    "flagged_count": int(invalid_mask.sum()),
                }
            )
            if bool(invalid_mask.any()):
                masks.append(invalid_mask)
                findings.append(
                    _finding(
                        frame=frame,
                        schema=schema,
                        column=spec.name,
                        mask=invalid_mask,
                        rule_id=f"format.non_finite.{spec.name}",
                        severity=policy.non_finite_severity,
                        evidence={"expected": "finite numeric values"},
                        suggested_actions=["replace_or_quarantine_non_finite_values"],
                        sample_limit=options.sample_limit,
                        reason="non_finite_numeric",
                        artifact_dir=options.artifact_dir,
                        artifacts=artifacts,
                    )
                )

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

    issue_checks = [check for check in checks if check["flagged_count"] > 0]
    return ToolResult(
        tool_name="check_types_formats",
        tool_version="0.2.0",
        dataset_id=schema.dataset_id,
        status=ToolStatus.SUCCESS,
        findings=findings,
        artifacts=artifacts,
        summary={
            "has_type_or_format_issues": bool(issue_checks),
            "row_count": int(len(frame)),
            "checks_performed": len(checks),
            "checks_with_issues": len(issue_checks),
            "columns_with_issues": len({finding.columns[0] for finding in findings}),
            "rows_flagged_union": int(union_mask.sum()),
            "check_results": checks,
        },
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={"policy": policy.model_dump(mode="json")},
    )


def check_types_formats(
    dataset: LoadedDataset,
    options: TypeFormatCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    return detect_type_format_issues(
        dataset.frame,
        dataset.schema,
        dataset.config.types_formats,
        options,
    )


__all__ = [
    "TYPE_FORMAT_CHECK_TOOL_SPEC",
    "TypeFormatCheckOptions",
    "check_types_formats",
    "detect_type_format_issues",
]
