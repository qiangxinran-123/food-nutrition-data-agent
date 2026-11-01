from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..config import ImbalancePolicy, LabelPolicy
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
from .common import (
    failed_tool_result,
    missing_schema_columns,
    write_table_artifact,
)
from .labels import analyse_label_column, label_specs


class ImbalanceCheckOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    artifact_dir: str | None = None


IMBALANCE_CHECK_TOOL_SPEC = ToolSpec(
    name="check_imbalance",
    version="0.2.0",
    category=ToolCategory.DETECTION,
    description=(
        "Measure label-class balance while excluding null and unavailable labels."
    ),
    mutates_data=False,
    input_schema=ImbalanceCheckOptions.model_json_schema(),
    output_schema=ToolResult.model_json_schema(),
)


def detect_imbalance(
    frame: pd.DataFrame,
    schema: SchemaProfile,
    label_policy: LabelPolicy,
    policy: ImbalancePolicy,
    options: ImbalanceCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    started_at = perf_counter()
    if options is None:
        options = ImbalanceCheckOptions()
    elif isinstance(options, dict):
        options = ImbalanceCheckOptions.model_validate(options)
    missing_columns = missing_schema_columns(frame, schema)
    if missing_columns:
        return failed_tool_result(
            tool_name="check_imbalance",
            tool_version="0.2.0",
            dataset_id=schema.dataset_id,
            started_at=started_at,
            code="schema_mismatch",
            message=f"dataframe is missing schema columns: {missing_columns}",
        )

    specs = label_specs(schema, label_policy)
    if policy.columns:
        selected = set(policy.columns)
        specs = [spec for spec in specs if spec.name in selected]

    findings: list[Finding] = []
    artifacts: list[Artifact] = []
    summaries: list[dict[str, Any]] = []
    warnings: list[str] = []

    for spec in specs:
        column_policy = label_policy.columns.get(spec.name)
        if column_policy is None:
            warnings.append(
                f"skipped imbalance check for {spec.name!r}: no label policy"
            )
            continue
        analysis = analyse_label_column(frame, spec, column_policy)
        counts = analysis.normalised[analysis.valid_mask].value_counts()
        usable_rows = int(counts.sum())
        if usable_rows == 0:
            warnings.append(
                f"skipped imbalance check for {spec.name!r}: no usable labels"
            )
            continue

        class_count = int(len(counts))
        majority_count = int(counts.max())
        minority_count = int(counts.min())
        minority_share = minority_count / usable_rows
        ratio = majority_count / minority_count if minority_count else float("inf")
        distribution = {
            str(label): {
                "count": int(count),
                "share": round(int(count) / usable_rows, 8),
            }
            for label, count in counts.items()
        }
        imbalanced = (
            class_count < policy.minimum_classes
            or minority_share < policy.minimum_minority_share
            or ratio > policy.maximum_majority_minority_ratio
        )
        summary = {
            "column": spec.name,
            "usable_rows": usable_rows,
            "excluded_unavailable_rows": int(len(frame) - usable_rows),
            "class_count": class_count,
            "class_distribution": distribution,
            "minority_share": round(minority_share, 8),
            "majority_minority_ratio": (
                round(ratio, 8) if ratio != float("inf") else None
            ),
            "is_imbalanced": imbalanced,
        }
        summaries.append(summary)

        distribution_table = pd.DataFrame.from_records(
            [
                {
                    "column": spec.name,
                    "class_label": str(label),
                    "count": int(count),
                    "share": round(int(count) / usable_rows, 8),
                    "is_minority": int(count) == minority_count,
                    "is_majority": int(count) == majority_count,
                }
                for label, count in counts.items()
            ]
        )
        distribution_artifact = write_table_artifact(
            distribution_table,
            artifact_dir=options.artifact_dir,
            dataset_id=schema.dataset_id,
            artifact_id=f"distribution.{schema.dataset_id}.imbalance.{spec.name}",
            kind=ArtifactKind.TABLE,
            description=(
                f"Usable label-class distribution for {spec.name!r}; unavailable "
                "labels are excluded."
            ),
            metadata={
                "role": "class_distribution",
                "column": spec.name,
                "usable_rows": usable_rows,
            },
        )
        if distribution_artifact is not None:
            artifacts.append(distribution_artifact)
        if not imbalanced:
            continue

        severity = (
            Severity.HIGH
            if minority_share < policy.high_minority_share
            or ratio > policy.high_majority_minority_ratio
            else Severity.MEDIUM
        )
        finding = Finding(
            finding_id=f"imbalance.{spec.name}",
            issue_type=IssueType.IMBALANCE,
            severity=severity,
            scope=FindingScope.DATASET,
            columns=[spec.name],
            affected_rows=minority_count,
            affected_rate=round(minority_share, 8),
            evidence={
                **summary,
                "minimum_minority_share": policy.minimum_minority_share,
                "maximum_majority_minority_ratio": (
                    policy.maximum_majority_minority_ratio
                ),
                "high_minority_share": policy.high_minority_share,
                "high_majority_minority_ratio": (
                    policy.high_majority_minority_ratio
                ),
            },
            suggested_actions=[
                "use_class_weights",
                "stratified_split",
                "collect_more_minority_samples",
            ],
            confidence=1.0,
            rule_id=f"imbalance.{spec.name}",
            metadata=(
                {
                    "distribution_artifact_id": distribution_artifact.artifact_id
                }
                if distribution_artifact is not None
                else {}
            ),
        )
        findings.append(finding)

    return ToolResult(
        tool_name="check_imbalance",
        tool_version="0.2.0",
        dataset_id=schema.dataset_id,
        status=ToolStatus.SUCCESS,
        findings=findings,
        artifacts=artifacts,
        summary={
            "has_imbalance": bool(findings),
            "row_count": int(len(frame)),
            "label_columns_checked": len(summaries),
            "imbalanced_columns": len(findings),
            "labels": summaries,
        },
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={"policy": policy.model_dump(mode="json")},
    )


def check_imbalance(
    dataset: LoadedDataset,
    options: ImbalanceCheckOptions | dict[str, Any] | None = None,
) -> ToolResult:
    return detect_imbalance(
        dataset.frame,
        dataset.schema,
        dataset.config.labels,
        dataset.config.imbalance,
        options,
    )


__all__ = [
    "IMBALANCE_CHECK_TOOL_SPEC",
    "ImbalanceCheckOptions",
    "check_imbalance",
    "detect_imbalance",
]
