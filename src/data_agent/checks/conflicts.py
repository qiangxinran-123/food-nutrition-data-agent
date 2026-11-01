from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..contracts import (
    Artifact,
    ArtifactKind,
    ConstraintKind,
    ConstraintSpec,
    Finding,
    FindingScope,
    IssueType,
    SchemaProfile,
    ToolCategory,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from ..expressions import ExpressionError, evaluate_boolean_expression
from ..io import LoadedDataset
from .common import (
    attach_selection_artifact,
    failed_tool_result,
    missing_schema_columns,
    row_selection_table,
    sample_row_ids,
    severity_order,
)


class ConflictCheckOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sample_limit: int = Field(default=10, ge=0, le=50)
    artifact_dir: str | None = None


CONFLICT_CHECK_TOOL_SPEC = ToolSpec(
    name="check_conflicts",
    version="0.2.0",
    category=ToolCategory.DETECTION,
    description=(
        "Evaluate configured cross-field formulas and relations to find conflicting values."
    ),
    mutates_data=False,
    input_schema=ConflictCheckOptions.model_json_schema(),
    output_schema=ToolResult.model_json_schema(),
)


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sample_values(
    frame: pd.DataFrame,
    mask: pd.Series,
    columns: list[str],
    schema: SchemaProfile,
    limit: int,
) -> list[dict[str, Any]]:
    if limit == 0 or not bool(mask.any()):
        return []
    row_id_column = schema.row_id_column
    samples: list[dict[str, Any]] = []
    for index, row in frame.loc[mask, columns].head(limit).iterrows():
        record: dict[str, Any] = {
            "row_id": (
                str(frame.loc[index, row_id_column])
                if row_id_column is not None and row_id_column in frame.columns
                else f"row_index:{index}"
            )
        }
        record.update({column: _json_value(row[column]) for column in columns})
        samples.append(record)
    return samples


class ConstraintNotEvaluated(ValueError):
    pass


@dataclass(frozen=True)
class ConstraintEvaluation:
    evaluated_mask: pd.Series
    violated_mask: pd.Series
    evidence: dict[str, Any]
    suggested_actions: list[str]
    reason: pd.Series | str


def _present_all(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].notna().all(axis=1)


def _present_any(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].notna().any(axis=1)


def _evaluate_constraint(
    frame: pd.DataFrame,
    constraint: ConstraintSpec,
) -> ConstraintEvaluation:
    kind = constraint.kind
    columns = constraint.columns

    if kind in {ConstraintKind.FORMULA, ConstraintKind.RELATION}:
        if not constraint.expression:
            raise ConstraintNotEvaluated("expression is required")
        try:
            passed = evaluate_boolean_expression(constraint.expression, frame)
        except ExpressionError as error:
            raise ConstraintNotEvaluated(f"expression error: {error}") from error
        valid_rows = _present_all(frame, columns)
        violated = valid_rows & (~passed)
        return ConstraintEvaluation(
            evaluated_mask=valid_rows,
            violated_mask=violated,
            evidence={
                "expression": constraint.expression,
                "evaluated_rows": int(valid_rows.sum()),
                "skipped_missing_rows": int(len(frame) - valid_rows.sum()),
            },
            suggested_actions=(
                ["recompute_derived_field", "verify_unit_conversion"]
                if kind is ConstraintKind.FORMULA
                else ["review_field_pair", "verify_source_consistency"]
            ),
            reason=constraint.constraint_id,
        )

    if kind is ConstraintKind.RANGE:
        minimum = constraint.parameters.get("minimum")
        maximum = constraint.parameters.get("maximum")
        if minimum is None and maximum is None:
            raise ConstraintNotEvaluated(
                "range constraint requires parameters.minimum and/or parameters.maximum"
            )
        inclusive_minimum = bool(
            constraint.parameters.get("inclusive_minimum", True)
        )
        inclusive_maximum = bool(
            constraint.parameters.get("inclusive_maximum", True)
        )
        violated = pd.Series(False, index=frame.index, dtype=bool)
        details: dict[str, Any] = {}
        for column in columns:
            series = frame[column]
            parsed = pd.to_numeric(series, errors="coerce").astype("float64")
            present = series.notna()
            non_numeric = present & parsed.isna()
            low = pd.Series(False, index=frame.index, dtype=bool)
            high = pd.Series(False, index=frame.index, dtype=bool)
            if minimum is not None:
                low = present & parsed.notna() & (
                    parsed.lt(minimum)
                    if inclusive_minimum
                    else parsed.le(minimum)
                )
            if maximum is not None:
                high = present & parsed.notna() & (
                    parsed.gt(maximum)
                    if inclusive_maximum
                    else parsed.ge(maximum)
                )
            violated = violated | low | high | non_numeric
            details[column] = {
                "below_minimum": int(low.sum()),
                "above_maximum": int(high.sum()),
                "non_numeric": int(non_numeric.sum()),
            }
        evaluated = _present_any(frame, columns)
        return ConstraintEvaluation(
            evaluated_mask=evaluated,
            violated_mask=evaluated & violated,
            evidence={
                "method": "range",
                "minimum": minimum,
                "maximum": maximum,
                "inclusive_minimum": inclusive_minimum,
                "inclusive_maximum": inclusive_maximum,
                "columns_detail": details,
            },
            suggested_actions=[
                "review_field_range",
                "verify_unit_and_scale",
            ],
            reason="range_violation",
        )

    if kind is ConstraintKind.UNIQUE:
        ignore_nulls = bool(constraint.parameters.get("ignore_nulls", True))
        evaluated = (
            _present_all(frame, columns)
            if ignore_nulls
            else pd.Series(True, index=frame.index, dtype=bool)
        )
        key_frame = frame.loc[evaluated, columns]
        redundant = key_frame.duplicated(keep="first")
        violated = pd.Series(False, index=frame.index, dtype=bool)
        violated.loc[redundant.index[redundant.to_numpy(dtype=bool)]] = True
        group_sizes = key_frame.groupby(columns, dropna=False, sort=False).size()
        duplicate_groups = int((group_sizes > 1).sum())
        return ConstraintEvaluation(
            evaluated_mask=evaluated,
            violated_mask=violated,
            evidence={
                "method": "unique",
                "ignore_nulls": ignore_nulls,
                "duplicate_group_count": duplicate_groups,
                "redundant_rows": int(redundant.sum()),
                "skipped_missing_rows": int(len(frame) - evaluated.sum()),
            },
            suggested_actions=[
                "resolve_key_conflict",
                "choose_retention_rule",
            ],
            reason="duplicate_key_value",
        )

    if kind is ConstraintKind.ALLOWED_VALUES:
        configured = constraint.parameters.get("allowed_values")
        if configured is None:
            raise ConstraintNotEvaluated(
                "allowed_values constraint requires parameters.allowed_values"
            )
        case_sensitive = bool(constraint.parameters.get("case_sensitive", True))
        trim_whitespace = bool(
            constraint.parameters.get("trim_whitespace", False)
        )
        violated = pd.Series(False, index=frame.index, dtype=bool)
        details: dict[str, Any] = {}
        for column in columns:
            allowed = configured.get(column) if isinstance(configured, dict) else configured
            if not isinstance(allowed, list):
                raise ConstraintNotEvaluated(
                    f"allowed values for {column!r} must be a list"
                )
            series = frame[column]
            values = series.astype("string")
            if trim_whitespace:
                values = values.str.strip()
            if not case_sensitive:
                values = values.str.casefold()
            allowed_values = {
                str(value).strip() if trim_whitespace else str(value)
                for value in allowed
            }
            if not case_sensitive:
                allowed_values = {value.casefold() for value in allowed_values}
            present = series.notna()
            invalid = present & (~values.isin(allowed_values))
            violated = violated | invalid
            details[column] = {
                "invalid_count": int(invalid.sum()),
                "observed_values": [
                    str(value)
                    for value in values[invalid].value_counts().head(5).index
                ],
            }
        evaluated = _present_any(frame, columns)
        return ConstraintEvaluation(
            evaluated_mask=evaluated,
            violated_mask=evaluated & violated,
            evidence={
                "method": "allowed_values",
                "case_sensitive": case_sensitive,
                "trim_whitespace": trim_whitespace,
                "columns_detail": details,
            },
            suggested_actions=[
                "map_to_allowed_values",
                "review_source_encoding",
            ],
            reason="value_not_allowed",
        )

    if kind is ConstraintKind.CONDITIONAL:
        when_expression = constraint.parameters.get(
            "when_expression", constraint.parameters.get("when")
        )
        if not when_expression or not constraint.expression:
            raise ConstraintNotEvaluated(
                "conditional constraint requires parameters.when and expression"
            )
        try:
            when_mask = evaluate_boolean_expression(str(when_expression), frame)
            then_mask = evaluate_boolean_expression(constraint.expression, frame)
        except ExpressionError as error:
            raise ConstraintNotEvaluated(f"expression error: {error}") from error
        evaluated = _present_all(frame, columns) & when_mask
        violated = evaluated & (~then_mask)
        return ConstraintEvaluation(
            evaluated_mask=evaluated,
            violated_mask=violated,
            evidence={
                "method": "conditional",
                "when_expression": str(when_expression),
                "then_expression": constraint.expression,
                "condition_true_rows": int(when_mask.sum()),
            },
            suggested_actions=[
                "review_conditional_rule",
                "verify_source_consistency",
            ],
            reason="conditional_failure",
        )

    if kind is ConstraintKind.CUSTOM:
        raise ConstraintNotEvaluated(
            "custom constraints require an external evaluator and were not run"
        )

    raise ConstraintNotEvaluated(f"unsupported constraint kind: {kind.value}")


def detect_conflicts(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    constraints: list[ConstraintSpec],
    options: ConflictCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    started_at = perf_counter()
    if options is None:
        options = ConflictCheckOptions()
    elif isinstance(options, dict):
        options = ConflictCheckOptions.model_validate(options)

    missing_columns = missing_schema_columns(frame, schema)
    if missing_columns:
        return failed_tool_result(
            tool_name="check_conflicts",
            tool_version="0.2.0",
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="schema_mismatch",
            message=f"dataframe is missing schema columns: {missing_columns}",
        )

    findings: list[Finding] = []
    artifacts: list[Artifact] = []
    masks: list[pd.Series] = []
    summaries: list[dict[str, Any]] = []
    not_evaluated: list[dict[str, Any]] = []
    warnings: list[str] = []

    for constraint in constraints:
        if not constraint.enabled:
            continue
        unknown = [column for column in constraint.columns if column not in frame.columns]
        if unknown:
            return failed_tool_result(
                tool_name="check_conflicts",
                tool_version="0.2.0",
                dataset_id=schema.dataset_id,
                started_at=started_at,
                code="constraint_columns_missing",
                message=(
                    f"constraint {constraint.constraint_id!r} references missing "
                    f"columns: {unknown}"
                ),
            )

        try:
            evaluation = _evaluate_constraint(frame, constraint)
        except ConstraintNotEvaluated as error:
            record = {
                "constraint_id": constraint.constraint_id,
                "kind": constraint.kind.value,
                "columns": constraint.columns,
                "status": "not_evaluated",
                "reason": str(error),
            }
            not_evaluated.append(record)
            warnings.append(
                f"constraint {constraint.constraint_id!r} was not evaluated: {error}"
            )
            continue

        evaluated_rows = int(evaluation.evaluated_mask.sum())
        violated_mask = evaluation.violated_mask
        violation_count = int(violated_mask.sum())
        masks.append(violated_mask)
        summary = {
            "constraint_id": constraint.constraint_id,
            "kind": constraint.kind.value,
            "columns": constraint.columns,
            "status": "evaluated",
            "evaluated_rows": evaluated_rows,
            "violated_rows": violation_count,
        }
        summaries.append(summary)
        if violation_count == 0:
            continue

        finding = Finding(
            finding_id=f"conflict.{constraint.constraint_id}",
            issue_type=IssueType.CONSTRAINT,
            severity=constraint.severity,
            scope=FindingScope.RELATION,
            columns=constraint.columns,
            affected_rows=violation_count,
            affected_rate=(
                round(violation_count / len(frame), 8) if len(frame) else 0.0
            ),
            evidence={
                **summary,
                **evaluation.evidence,
                "message": constraint.message,
                "violation_rate_among_evaluated": (
                    round(violation_count / evaluated_rows, 8)
                    if evaluated_rows
                    else 0.0
                ),
                "sample_values": _sample_values(
                    frame,
                    violated_mask,
                    constraint.columns,
                    schema,
                    min(options.sample_limit, 5),
                ),
            },
            sample_row_ids=sample_row_ids(
                frame, violated_mask, schema, options.sample_limit
            ),
            suggested_actions=evaluation.suggested_actions,
            confidence=1.0,
            rule_id=constraint.constraint_id,
            metadata={"tolerance": constraint.tolerance},
        )
        artifact = attach_selection_artifact(
            finding,
            row_selection_table(
                frame,
                schema,
                finding_id=finding.finding_id,
                rule_id=finding.rule_id or finding.finding_id,
                mask=violated_mask,
                reason=evaluation.reason,
                value_columns=constraint.columns,
            ),
            artifact_dir=options.artifact_dir,
            dataset_id=schema.dataset_id,
            kind=ArtifactKind.ROW_LIST,
            description=(
                f"Complete violating-row selection for {finding.finding_id}; "
                "row_values_json preserves the compared field values."
            ),
            metadata={
                "constraint_id": constraint.constraint_id,
                "constraint_kind": constraint.kind.value,
                "columns": constraint.columns,
                "flagged_rows": violation_count,
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
            "artifact_dir was not provided; complete row selections were not persisted"
        )

    return ToolResult(
        tool_name="check_conflicts",
        tool_version="0.2.0",
        dataset_id=schema.dataset_id,
        status=ToolStatus.PARTIAL if not_evaluated else ToolStatus.SUCCESS,
        findings=findings,
        artifacts=artifacts,
        summary={
            "has_conflicts": bool(union_mask.any()),
            "row_count": int(len(frame)),
            "constraints_configured": len(constraints),
            "constraints_checked": len(summaries),
            "constraints_not_evaluated": len(not_evaluated),
            "constraints_with_violations": sum(
                1 for summary in summaries if summary["violated_rows"] > 0
            ),
            "rows_flagged_union": int(union_mask.sum()),
            "constraints": summaries,
            "not_evaluated_constraints": not_evaluated,
        },
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={"options": options.model_dump(mode="json")},
    )


def check_conflicts(
    dataset: LoadedDataset,
    options: ConflictCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    return detect_conflicts(
        dataset.frame,
        dataset.schema,
        dataset.config.conflicts,
        options,
    )


__all__ = [
    "CONFLICT_CHECK_TOOL_SPEC",
    "ConflictCheckOptions",
    "check_conflicts",
    "detect_conflicts",
]
