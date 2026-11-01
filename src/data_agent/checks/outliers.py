from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from pydantic import BaseModel, ConfigDict, Field

from ..config import OutlierPolicy
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
from ..masking import build_missing_mask, missing_policy_from_schema
from .common import (
    attach_selection_artifact,
    cell_selection_table,
    failed_tool_result,
    missing_schema_columns,
    sample_row_ids,
    severity_order,
)


class OutlierCheckOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_limit: int = Field(default=10, ge=0, le=50)
    artifact_dir: str | None = None


OUTLIER_CHECK_TOOL_SPEC = ToolSpec(
    name="check_outliers",
    version="0.2.0",
    category=ToolCategory.DETECTION,
    description=(
        "Detect hard numeric range violations and conservative IQR tail outliers."
    ),
    mutates_data=False,
    input_schema=OutlierCheckOptions.model_json_schema(),
    output_schema=ToolResult.model_json_schema(),
)


def _numeric_values(series: pd.Series) -> pd.Series:
    if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce")


def _finding_for_mask(
    *,
    frame: pd.DataFrame,
    schema: SchemaProfile,
    column: str,
    mask: pd.Series,
    rule_id: str,
    severity: Severity,
    evidence: dict[str, Any],
    suggested_actions: list[str],
    confidence: float,
    sample_limit: int,
    reason: str | pd.Series | None,
    artifact_dir: str | None,
    artifacts: list[Artifact],
) -> Finding:
    affected_rows = int(mask.sum())
    finding = Finding(
        finding_id=rule_id,
        issue_type=IssueType.OUTLIER,
        severity=severity,
        scope=FindingScope.COLUMN,
        columns=[column],
        affected_rows=affected_rows,
        affected_cells=affected_rows,
        affected_rate=round(affected_rows / len(frame), 8) if len(frame) else 0.0,
        evidence={**evidence, "flagged_count": affected_rows},
        sample_row_ids=sample_row_ids(frame, mask, schema, sample_limit),
        suggested_actions=suggested_actions,
        confidence=confidence,
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
            "flagged cell and its outlier reason."
        ),
        metadata={
            "column": column,
            "flagged_cells": affected_rows,
            "row_id_column": schema.row_id_column,
        },
    )
    if artifact is not None:
        artifacts.append(artifact)
    return finding


def detect_outliers(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    policy: OutlierPolicy,
    options: OutlierCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    started_at = perf_counter()
    if options is None:
        options = OutlierCheckOptions()
    elif isinstance(options, dict):
        options = OutlierCheckOptions.model_validate(options)

    missing_columns = missing_schema_columns(frame, schema)
    if missing_columns:
        return failed_tool_result(
            tool_name="check_outliers",
            tool_version="0.2.0",
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="schema_mismatch",
            message=f"dataframe is missing schema columns: {missing_columns}",
        )

    missing_policy = missing_policy_from_schema(schema)
    findings: list[Finding] = []
    artifacts: list[Artifact] = []
    warnings: list[str] = []
    range_masks: list[pd.Series] = []
    statistical_masks: list[pd.Series] = []
    range_summaries: list[dict[str, Any]] = []
    statistical_summaries: list[dict[str, Any]] = []

    for spec in schema.columns:
        if spec.range is None:
            continue
        series = frame[spec.name]
        parsed = _numeric_values(series)
        missing_mask = build_missing_mask(series, spec, **missing_policy).total
        valid_mask = (~missing_mask) & parsed.notna()
        finite_mask = valid_mask & pd.Series(
            np.isfinite(parsed.to_numpy(dtype="float64", na_value=np.nan)),
            index=frame.index,
        )

        lower = spec.range.minimum
        upper = spec.range.maximum
        low_mask = pd.Series(False, index=frame.index, dtype=bool)
        high_mask = pd.Series(False, index=frame.index, dtype=bool)
        if lower is not None:
            low_mask = valid_mask & (
                parsed.lt(lower) if spec.range.inclusive_minimum else parsed.le(lower)
            )
        if upper is not None:
            high_mask = valid_mask & (
                parsed.gt(upper) if spec.range.inclusive_maximum else parsed.ge(upper)
            )
        nonfinite_mask = valid_mask & (~finite_mask)
        range_mask = low_mask | high_mask | nonfinite_mask
        range_masks.append(range_mask)
        range_summaries.append(
            {
                "column": spec.name,
                "minimum": lower,
                "maximum": upper,
                "flagged_count": int(range_mask.sum()),
            }
        )
        if not bool(range_mask.any()):
            continue

        finite_values = parsed[finite_mask]
        range_reason = pd.Series("range_violation", index=frame.index, dtype="object")
        range_reason.loc[low_mask] = "below_minimum"
        range_reason.loc[high_mask] = "above_maximum"
        range_reason.loc[nonfinite_mask] = "non_finite"
        findings.append(
            _finding_for_mask(
                frame=frame,
                schema=schema,
                column=spec.name,
                mask=range_mask,
                rule_id=f"outlier.range.{spec.name}",
                severity=policy.range_severity,
                evidence={
                    "method": "declared_range",
                    "minimum": lower,
                    "maximum": upper,
                    "inclusive_minimum": spec.range.inclusive_minimum,
                    "inclusive_maximum": spec.range.inclusive_maximum,
                    "below_minimum_count": int(low_mask.sum()),
                    "above_maximum_count": int(high_mask.sum()),
                    "non_finite_count": int(nonfinite_mask.sum()),
                    "observed_minimum": (
                        float(finite_values.min()) if not finite_values.empty else None
                    ),
                    "observed_maximum": (
                        float(finite_values.max()) if not finite_values.empty else None
                    ),
                },
                suggested_actions=[
                    "quarantine_out_of_range_rows",
                    "verify_source_unit_and_scale",
                ],
                confidence=1.0,
                sample_limit=options.sample_limit,
                reason=range_reason,
                artifact_dir=options.artifact_dir,
                artifacts=artifacts,
            )
        )

    schema_by_name = {column.name: column for column in schema.columns}
    for column in policy.statistical_columns:
        spec = schema_by_name.get(column)
        if spec is None:
            continue
        series = frame[column]
        parsed = _numeric_values(series)
        missing_mask = build_missing_mask(series, spec, **missing_policy).total
        finite = (
            ~missing_mask
            & parsed.notna()
            & pd.Series(
                np.isfinite(parsed.to_numpy(dtype="float64", na_value=np.nan)),
                index=frame.index,
            )
        )
        values = parsed[finite]
        if len(values) < policy.minimum_non_null:
            warnings.append(
                f"skipped IQR check for {column!r}: only {len(values)} finite values"
            )
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            warnings.append(f"skipped IQR check for {column!r}: IQR is zero")
            continue
        lower = q1 - policy.iqr_multiplier * iqr
        upper = q3 + policy.iqr_multiplier * iqr
        low_mask = finite & parsed.lt(lower)
        high_mask = finite & parsed.gt(upper)
        statistical_mask = low_mask | high_mask
        statistical_masks.append(statistical_mask)
        statistical_summaries.append(
            {
                "column": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "flagged_count": int(statistical_mask.sum()),
            }
        )
        if not bool(statistical_mask.any()):
            continue

        statistical_reason = pd.Series(
            "iqr_tail", index=frame.index, dtype="object"
        )
        statistical_reason.loc[low_mask] = "below_iqr_lower_bound"
        statistical_reason.loc[high_mask] = "above_iqr_upper_bound"
        findings.append(
            _finding_for_mask(
                frame=frame,
                schema=schema,
                column=column,
                mask=statistical_mask,
                rule_id=f"outlier.iqr.{column}",
                severity=policy.statistical_severity,
                evidence={
                    "method": "iqr",
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "multiplier": policy.iqr_multiplier,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "below_lower_count": int(low_mask.sum()),
                    "above_upper_count": int(high_mask.sum()),
                    "finite_value_count": int(len(values)),
                },
                suggested_actions=[
                    "review_distribution_tail",
                    "do_not_remove_without_domain_check",
                ],
                confidence=0.8,
                sample_limit=options.sample_limit,
                reason=statistical_reason,
                artifact_dir=options.artifact_dir,
                artifacts=artifacts,
            )
        )

    order = severity_order()
    findings.sort(
        key=lambda finding: (order[finding.severity], -finding.affected_rows)
    )
    union_mask = pd.Series(False, index=frame.index, dtype=bool)
    for mask in range_masks + statistical_masks:
        union_mask = union_mask | mask
    if findings and options.artifact_dir is None:
        warnings.append(
            "artifact_dir was not provided; complete cell selections were not persisted"
        )

    return ToolResult(
        tool_name="check_outliers",
        tool_version="0.2.0",
        dataset_id=schema.dataset_id,
        status=ToolStatus.SUCCESS,
        findings=findings,
        artifacts=artifacts,
        summary={
            "has_outliers": bool(union_mask.any()),
            "row_count": int(len(frame)),
            "range_columns_checked": len(range_summaries),
            "statistical_columns_checked": len(statistical_summaries),
            "range_flagged_cells": sum(
                summary["flagged_count"] for summary in range_summaries
            ),
            "statistical_flagged_cells": sum(
                summary["flagged_count"] for summary in statistical_summaries
            ),
            "rows_flagged_union": int(union_mask.sum()),
            "range_results": range_summaries,
            "statistical_results": statistical_summaries,
        },
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={"policy": policy.model_dump(mode="json")},
    )


def check_outliers(
    dataset: LoadedDataset,
    options: OutlierCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    return detect_outliers(
        dataset.frame,
        dataset.schema,
        dataset.config.outliers,
        options,
    )


__all__ = [
    "OUTLIER_CHECK_TOOL_SPEC",
    "OutlierCheckOptions",
    "check_outliers",
    "detect_outliers",
]
