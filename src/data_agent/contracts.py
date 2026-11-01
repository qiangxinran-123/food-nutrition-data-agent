from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


CONTRACT_VERSION = "0.1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContractBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    contract_version: Literal["0.1"] = CONTRACT_VERSION


class DataFormat(StrEnum):
    XLSX = "xlsx"
    CSV = "csv"
    PARQUET = "parquet"
    JSONL = "jsonl"
    OTHER = "other"


class LogicalType(StrEnum):
    IDENTIFIER = "identifier"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class IssueType(StrEnum):
    MISSING = "missing"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    FORMAT = "format"
    CONSTRAINT = "constraint"
    LABEL = "label"
    IMBALANCE = "imbalance"
    ENTITY_MATCH = "entity_match"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStatus(StrEnum):
    OPEN = "open"
    CONFIRMED = "confirmed"
    IGNORED = "ignored"
    RESOLVED = "resolved"


class FindingScope(StrEnum):
    DATASET = "dataset"
    COLUMN = "column"
    ROW = "row"
    CELL = "cell"
    RELATION = "relation"


class RiskLevel(StrEnum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolCategory(StrEnum):
    PROFILING = "profiling"
    DETECTION = "detection"
    PREPROCESSING = "preprocessing"
    VALIDATION = "validation"
    CLUSTERING = "clustering"


class ConstraintKind(StrEnum):
    RANGE = "range"
    RELATION = "relation"
    FORMULA = "formula"
    UNIQUE = "unique"
    ALLOWED_VALUES = "allowed_values"
    CONDITIONAL = "conditional"
    CUSTOM = "custom"


class ArtifactKind(StrEnum):
    ROW_LIST = "row_list"
    CELL_LIST = "cell_list"
    GROUP_LIST = "group_list"
    TABLE = "table"
    REPORT = "report"
    LOG = "log"
    FEATURE_TABLE = "feature_table"
    OTHER = "other"


class ChangeType(StrEnum):
    UPDATE = "update"
    ADD = "add"
    DROP = "drop"
    IMPUTE = "impute"
    NORMALIZE = "normalize"
    CONVERT = "convert"


class NumericRange(ContractBase):
    minimum: float | None = None
    maximum: float | None = None
    inclusive_minimum: bool = True
    inclusive_maximum: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> NumericRange:
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum must not be greater than maximum")
        return self


class DatasetRef(ContractBase):
    dataset_id: str = Field(min_length=1)
    version: str = Field(default="1", min_length=1)
    uri: str = Field(min_length=1)
    format: DataFormat
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    source: str | None = None
    parent_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ColumnSpec(ContractBase):
    name: str = Field(min_length=1)
    role: str | None = None
    logical_type: LogicalType
    physical_dtype: str = Field(min_length=1)
    unit: str | None = None
    nullable: bool = True
    unique: bool = False
    missing_tokens: list[str] = Field(default_factory=list)
    allowed_values: list[Any] = Field(default_factory=list)
    range: NumericRange | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SchemaProfile(ContractBase):
    dataset_id: str = Field(min_length=1)
    schema_version: str = Field(default="0.1", min_length=1)
    domain_pack: str | None = None
    columns: list[ColumnSpec] = Field(min_length=1)
    primary_key_candidates: list[list[str]] = Field(default_factory=list)
    row_id_column: str | None = None
    source_row_column: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_column_references(self) -> SchemaProfile:
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("column names must be unique")

        known = set(names)
        for key_candidate in self.primary_key_candidates:
            unknown = set(key_candidate) - known
            if unknown:
                raise ValueError(
                    f"primary key candidate references unknown columns: {sorted(unknown)}"
                )

        for column_name in (self.row_id_column, self.source_row_column):
            if column_name is not None and column_name not in known:
                raise ValueError(
                    f"row metadata column is not described by schema: {column_name}"
                )
        return self


class ConstraintSpec(ContractBase):
    constraint_id: str = Field(min_length=1)
    kind: ConstraintKind
    columns: list[str] = Field(min_length=1)
    expression: str | None = None
    tolerance: float | None = Field(default=None, ge=0)
    parameters: dict[str, Any] = Field(default_factory=dict)
    severity: Severity = Severity.MEDIUM
    message: str = Field(min_length=1)
    enabled: bool = True
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_definition(self) -> ConstraintSpec:
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("constraint columns must be unique")
        if self.kind in {ConstraintKind.FORMULA, ConstraintKind.RELATION}:
            if not self.expression or not self.expression.strip():
                raise ValueError(f"{self.kind} constraint requires an expression")
        return self


class Finding(ContractBase):
    finding_id: str = Field(min_length=1)
    issue_type: IssueType
    severity: Severity
    status: FindingStatus = FindingStatus.OPEN
    scope: FindingScope
    columns: list[str] = Field(default_factory=list)
    affected_rows: int = Field(ge=0)
    affected_cells: int | None = Field(default=None, ge=0)
    affected_rate: float = Field(ge=0, le=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    sample_row_ids: list[str] = Field(default_factory=list, max_length=50)
    suggested_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0, le=1)
    rule_id: str | None = None
    selection_artifact_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sample_count(self) -> Finding:
        if len(self.sample_row_ids) > self.affected_rows and self.affected_rows > 0:
            raise ValueError("sample_row_ids cannot exceed affected_rows")
        return self


class Artifact(ContractBase):
    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    row_count: int | None = Field(default=None, ge=0)
    dataset_id: str | None = None
    finding_ids: list[str] = Field(default_factory=list)
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(ContractBase):
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(default="0.1", min_length=1)
    dataset_id: str = Field(min_length=1)
    status: ToolStatus = ToolStatus.SUCCESS
    findings: list[Finding] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_failure(self) -> ToolResult:
        if self.status is ToolStatus.FAILED and not self.error_message:
            raise ValueError("failed ToolResult requires error_message")
        return self


class CleaningAction(ContractBase):
    action_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    target_columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    source_findings: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.CAUTION
    requires_confirmation: bool = False
    expected_effect: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_confirmation(self) -> CleaningAction:
        if self.risk_level is RiskLevel.DANGEROUS and not self.requires_confirmation:
            raise ValueError("dangerous actions must require confirmation")
        return self


class CleaningDecision(ContractBase):
    decision_id: str = Field(min_length=1)
    issue_type: IssueType
    strategy_id: str = Field(min_length=1)
    finding_id: str = Field(min_length=1)
    selection_artifact_id: str | None = None
    target_columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    expected_effect: str | None = None
    risk_level: RiskLevel = RiskLevel.CAUTION
    requires_confirmation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_confirmation(self) -> CleaningDecision:
        if self.risk_level is RiskLevel.DANGEROUS and not self.requires_confirmation:
            raise ValueError("dangerous cleaning decisions must require confirmation")
        return self


class CleaningPlan(ContractBase):
    plan_id: str = Field(min_length=1)
    before_dataset: DatasetRef
    decisions: list[CleaningDecision] = Field(default_factory=list)
    dry_run: bool = True
    output_dataset_id: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ChangeRecord(ContractBase):
    row_id: str = Field(min_length=1)
    column: str | None = None
    old_value: Any = None
    new_value: Any = None
    change_type: ChangeType
    reason: str = Field(min_length=1)
    is_imputed: bool = False


class ChangeSet(ContractBase):
    change_id: str = Field(min_length=1)
    before_dataset: DatasetRef
    after_dataset: DatasetRef
    actions: list[CleaningAction] = Field(default_factory=list)
    rows_added: int = Field(default=0, ge=0)
    rows_removed: int = Field(default=0, ge=0)
    cells_changed: int = Field(default=0, ge=0)
    changes_uri: str | None = None
    sample_changes: list[ChangeRecord] = Field(default_factory=list, max_length=50)
    description: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_row_accounting(self) -> ChangeSet:
        expected = (
            self.before_dataset.row_count + self.rows_added - self.rows_removed
        )
        if expected != self.after_dataset.row_count:
            raise ValueError(
                "after_dataset.row_count must equal before_dataset.row_count + rows_added - rows_removed"
            )
        change_events = self.cells_changed + self.rows_added + self.rows_removed
        if len(self.sample_changes) > change_events:
            raise ValueError(
                "sample_changes cannot exceed cells_changed + rows_added + rows_removed"
            )
        return self


class CleaningResult(ContractBase):
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(default="0.1", min_length=1)
    dataset_id: str = Field(min_length=1)
    status: ToolStatus = ToolStatus.SUCCESS
    before_dataset: DatasetRef
    after_dataset: DatasetRef
    decision: CleaningDecision | None = None
    change_set: ChangeSet | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    dry_run: bool = True
    validation: "ValidationReport | None" = None
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    output_frame: Any = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_failure(self) -> CleaningResult:
        if self.status is ToolStatus.FAILED and not self.error_message:
            raise ValueError("failed CleaningResult requires error_message")
        return self


class CleaningPlanResult(ContractBase):
    plan_id: str = Field(min_length=1)
    status: ToolStatus = ToolStatus.SUCCESS
    before_dataset: DatasetRef
    after_dataset: DatasetRef
    results: list[CleaningResult] = Field(default_factory=list)
    change_set: ChangeSet | None = None
    dry_run: bool = True
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_failure(self) -> CleaningPlanResult:
        if self.status is ToolStatus.FAILED and not self.error_message:
            raise ValueError("failed CleaningPlanResult requires error_message")
        return self


class CheckDelta(ContractBase):
    check_type: IssueType
    before_count: int = Field(ge=0)
    after_count: int = Field(ge=0)
    passed: bool
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(ContractBase):
    validation_id: str = Field(min_length=1)
    before_dataset: DatasetRef
    after_dataset: DatasetRef
    checks: list[CheckDelta] = Field(default_factory=list)
    passed: bool
    retained_rows: int = Field(ge=0)
    retained_rate: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_retention(self) -> ValidationReport:
        if self.retained_rows != self.after_dataset.row_count:
            raise ValueError("retained_rows must equal after_dataset.row_count")
        if self.before_dataset.row_count > 0:
            expected_rate = self.retained_rows / self.before_dataset.row_count
            if abs(expected_rate - self.retained_rate) > 1e-6:
                raise ValueError("retained_rate does not match dataset row counts")
        return self


class ToolSpec(ContractBase):
    name: str = Field(min_length=1)
    version: str = Field(default="0.1", min_length=1)
    category: ToolCategory
    description: str = Field(min_length=1)
    mutates_data: bool = False
    risk_level: RiskLevel = RiskLevel.SAFE
    idempotent: bool = True
    requires_confirmation: bool = False
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_confirmation(self) -> ToolSpec:
        if self.risk_level is RiskLevel.DANGEROUS and not self.requires_confirmation:
            raise ValueError("dangerous tools must require confirmation")
        return self


__all__ = [
    "Artifact",
    "ArtifactKind",
    "ChangeRecord",
    "ChangeSet",
    "ChangeType",
    "CheckDelta",
    "CleaningAction",
    "CleaningDecision",
    "CleaningPlan",
    "CleaningPlanResult",
    "CleaningResult",
    "ColumnSpec",
    "ConstraintKind",
    "ConstraintSpec",
    "DataFormat",
    "DatasetRef",
    "Finding",
    "FindingScope",
    "FindingStatus",
    "IssueType",
    "LogicalType",
    "NumericRange",
    "RiskLevel",
    "SchemaProfile",
    "Severity",
    "ToolCategory",
    "ToolResult",
    "ToolSpec",
    "ToolStatus",
    "ValidationReport",
]
