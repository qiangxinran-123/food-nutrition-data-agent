from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ConstraintSpec, DataFormat, LogicalType, NumericRange, Severity


class ConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReadOptions(ConfigBase):
    format: DataFormat | None = None
    engine: str | None = None
    dtype: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


class IngestionPolicy(ConfigBase):
    column_aliases: dict[str, str] = Field(default_factory=dict)
    identifier_columns: list[str] = Field(default_factory=list)
    preserve_raw_columns: list[str] = Field(default_factory=list)
    raw_column_prefix: str = Field(default="__raw__", min_length=1)
    drop_unnamed_columns: bool = False
    strip_bom: bool = True
    case_sensitive_columns: bool = True
    encoding: str | None = None
    ensure_row_id: bool = False
    row_id_prefix: str = Field(default="row", min_length=1)
    row_id_width: int = Field(default=6, ge=1, le=20)

    @model_validator(mode="after")
    def validate_policy(self) -> IngestionPolicy:
        if len(self.identifier_columns) != len(set(self.identifier_columns)):
            raise ValueError("ingestion identifier columns must be unique")
        if len(self.preserve_raw_columns) != len(set(self.preserve_raw_columns)):
            raise ValueError("preserve_raw_columns must be unique")
        if any(not source.strip() or not target.strip() for source, target in self.column_aliases.items()):
            raise ValueError("column aliases must not be empty")
        targets = list(self.column_aliases.values())
        if len(targets) != len(set(targets)):
            raise ValueError("column alias targets must be unique")
        if self.ensure_row_id and not self.row_id_prefix.strip():
            raise ValueError("row_id_prefix must not be blank")
        return self


class MissingPolicy(ConfigBase):
    tokens: list[str] = Field(default_factory=list)
    tokens_by_column: dict[str, list[str]] = Field(default_factory=dict)
    treat_empty_string_as_missing: bool = True
    trim_whitespace: bool = True
    case_sensitive: bool = False
    severity_medium_threshold: float = Field(default=0.2, ge=0, le=1)
    severity_high_threshold: float = Field(default=0.5, ge=0, le=1)
    severity_critical_threshold: float = Field(default=0.9, ge=0, le=1)

    @model_validator(mode="after")
    def validate_tokens(self) -> MissingPolicy:
        all_tokens = list(self.tokens)
        for column_tokens in self.tokens_by_column.values():
            all_tokens.extend(column_tokens)
        if any(not token.strip() for token in all_tokens):
            raise ValueError("missing-value tokens must not be empty or whitespace")
        if not (
            self.severity_medium_threshold
            <= self.severity_high_threshold
            <= self.severity_critical_threshold
        ):
            raise ValueError(
                "missing severity thresholds must satisfy medium <= high <= critical"
            )
        return self


class ColumnConfig(ConfigBase):
    role: str | None = None
    logical_type: LogicalType | None = None
    unit: str | None = None
    nullable: bool | None = None
    unique: bool | None = None
    missing_tokens: list[str] = Field(default_factory=list)
    allowed_values: list[Any] = Field(default_factory=list)
    range: NumericRange | None = None
    format_regex: str | None = None
    datetime_format: str | None = None
    dayfirst: bool | None = None
    yearfirst: bool | None = None
    boolean_values: list[str] | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_format_regex(self) -> ColumnConfig:
        if self.format_regex is not None:
            try:
                re.compile(self.format_regex)
            except re.error as error:
                raise ValueError(f"invalid format regex: {error}") from error
        return self


class DuplicateRuleConfig(ConfigBase):
    rule_id: str = Field(min_length=1)
    columns: list[str] = Field(default_factory=list)
    exclude_columns: list[str] = Field(default_factory=list)
    use_all_columns: bool = False
    normalize_text: bool = False
    case_sensitive: bool = True
    severity: Severity = Severity.MEDIUM
    description: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> DuplicateRuleConfig:
        if self.use_all_columns and self.columns:
            raise ValueError("duplicate rules cannot combine use_all_columns and columns")
        if not self.use_all_columns and not self.columns:
            raise ValueError("duplicate rules require columns or use_all_columns=true")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("duplicate rule columns must be unique")
        if len(self.exclude_columns) != len(set(self.exclude_columns)):
            raise ValueError("duplicate rule exclude_columns must be unique")
        return self


class OutlierPolicy(ConfigBase):
    statistical_columns: list[str] = Field(default_factory=list)
    iqr_multiplier: float = Field(default=5.0, gt=0)
    minimum_non_null: int = Field(default=20, ge=2)
    range_severity: Severity = Severity.HIGH
    statistical_severity: Severity = Severity.LOW


class TypeFormatPolicy(ConfigBase):
    check_logical_types: bool = True
    check_numeric_finite: bool = True
    regex_severity: Severity = Severity.MEDIUM
    numeric_parse_severity: Severity = Severity.HIGH
    non_finite_severity: Severity = Severity.HIGH
    datetime_severity: Severity = Severity.HIGH
    boolean_severity: Severity = Severity.HIGH
    boolean_values: list[str] = Field(
        default_factory=lambda: [
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
            "y",
            "n",
            "t",
            "f",
        ]
    )
    datetime_format: str | None = None
    dayfirst: bool = False
    yearfirst: bool = False


class LabelColumnPolicy(ConfigBase):
    unavailable_tokens: list[str] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    case_sensitive: bool = False
    trim_whitespace: bool = True

    @model_validator(mode="after")
    def validate_tokens(self) -> LabelColumnPolicy:
        if any(not token.strip() for token in self.unavailable_tokens):
            raise ValueError("label unavailable tokens must not be empty")
        if any(
            not source.strip() or not target.strip()
            for source, target in self.aliases.items()
        ):
            raise ValueError("label aliases must not be empty")
        return self


class LabelPolicy(ConfigBase):
    columns: dict[str, LabelColumnPolicy] = Field(default_factory=dict)
    severity_medium_threshold: float = Field(default=0.2, ge=0, le=1)
    severity_high_threshold: float = Field(default=0.5, ge=0, le=1)
    severity_critical_threshold: float = Field(default=0.9, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> LabelPolicy:
        if not (
            self.severity_medium_threshold
            <= self.severity_high_threshold
            <= self.severity_critical_threshold
        ):
            raise ValueError(
                "label severity thresholds must satisfy medium <= high <= critical"
            )
        return self


class ImbalancePolicy(ConfigBase):
    columns: list[str] = Field(default_factory=list)
    minimum_minority_share: float = Field(default=0.1, ge=0, le=1)
    maximum_majority_minority_ratio: float = Field(default=5.0, ge=1)
    minimum_classes: int = Field(default=2, ge=2)
    high_minority_share: float = Field(default=0.05, ge=0, le=1)
    high_majority_minority_ratio: float = Field(default=10.0, ge=1)


class DatasetConfig(ConfigBase):
    contract_version: Literal["0.1"] = "0.1"
    dataset_id: str = Field(min_length=1)
    domain_pack: str | None = None
    manifest: str = Field(min_length=1)
    row_id_column: str | None = None
    source_row_column: str | None = None
    primary_key_candidates: list[list[str]] = Field(default_factory=list)
    read: ReadOptions = Field(default_factory=ReadOptions)
    ingestion: IngestionPolicy = Field(default_factory=IngestionPolicy)
    missing: MissingPolicy = Field(default_factory=MissingPolicy)
    duplicates: list[DuplicateRuleConfig] = Field(default_factory=list)
    outliers: OutlierPolicy = Field(default_factory=OutlierPolicy)
    types_formats: TypeFormatPolicy = Field(default_factory=TypeFormatPolicy)
    conflicts: list[ConstraintSpec] = Field(default_factory=list)
    labels: LabelPolicy = Field(default_factory=LabelPolicy)
    imbalance: ImbalancePolicy = Field(default_factory=ImbalancePolicy)
    columns: dict[str, ColumnConfig] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> DatasetConfig:
        configured_columns = set(self.columns)
        alias_targets = set(self.ingestion.column_aliases.values())
        unknown_targets = alias_targets - configured_columns
        if unknown_targets:
            raise ValueError(
                "column aliases reference unconfigured canonical columns: "
                f"{sorted(unknown_targets)}"
            )
        identifier_unknown = set(self.ingestion.identifier_columns) - configured_columns
        if identifier_unknown:
            raise ValueError(
                "ingestion identifier columns are not configured: "
                f"{sorted(identifier_unknown)}"
            )
        raw_columns = set(self.ingestion.preserve_raw_columns)
        if "*" not in raw_columns:
            raw_unknown = raw_columns - configured_columns
            if raw_unknown:
                raise ValueError(
                    "preserve_raw_columns references unconfigured columns: "
                    f"{sorted(raw_unknown)}"
                )
        for candidate in self.primary_key_candidates:
            unknown = set(candidate) - configured_columns
            if unknown:
                raise ValueError(
                    f"primary key candidate references unconfigured columns: {sorted(unknown)}"
                )

        for column_name in (self.row_id_column, self.source_row_column):
            if column_name is not None and column_name not in configured_columns:
                raise ValueError(
                    f"row metadata column is not configured: {column_name}"
                )

        unknown_token_columns = set(self.missing.tokens_by_column) - configured_columns
        if unknown_token_columns:
            raise ValueError(
                "missing tokens reference unconfigured columns: "
                f"{sorted(unknown_token_columns)}"
            )

        for column_name, column_config in self.columns.items():
            if any(not token.strip() for token in column_config.missing_tokens):
                raise ValueError(
                    f"missing-value tokens for {column_name!r} must not be empty"
                )

        duplicate_ids = [rule.rule_id for rule in self.duplicates]
        if len(duplicate_ids) != len(set(duplicate_ids)):
            raise ValueError("duplicate rule ids must be unique")
        for rule in self.duplicates:
            references = set(rule.columns) | set(rule.exclude_columns)
            unknown = references - configured_columns
            if unknown:
                raise ValueError(
                    f"duplicate rule {rule.rule_id!r} references unknown columns: "
                    f"{sorted(unknown)}"
                )

        outlier_unknown = set(self.outliers.statistical_columns) - configured_columns
        if outlier_unknown:
            raise ValueError(
                f"outlier policy references unknown columns: {sorted(outlier_unknown)}"
            )

        constraint_ids = [constraint.constraint_id for constraint in self.conflicts]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("conflict constraint ids must be unique")
        for constraint in self.conflicts:
            unknown = set(constraint.columns) - configured_columns
            if unknown:
                raise ValueError(
                    f"constraint {constraint.constraint_id!r} references unknown columns: "
                    f"{sorted(unknown)}"
                )
            if constraint.expression:
                referenced = set(
                    re.findall(r"col\(\s*[\"']([^\"']+)[\"']\s*\)", constraint.expression)
                )
                unknown_refs = referenced - configured_columns
                if unknown_refs:
                    raise ValueError(
                        f"constraint {constraint.constraint_id!r} expression references "
                        f"unknown columns: {sorted(unknown_refs)}"
                    )

        label_unknown = set(self.labels.columns) - configured_columns
        if label_unknown:
            raise ValueError(
                f"label policy references unknown columns: {sorted(label_unknown)}"
            )

        imbalance_unknown = set(self.imbalance.columns) - configured_columns
        if imbalance_unknown:
            raise ValueError(
                f"imbalance policy references unknown columns: {sorted(imbalance_unknown)}"
            )
        return self


def load_dataset_config(path: str | Path) -> DatasetConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return DatasetConfig.model_validate(payload)


__all__ = [
    "ColumnConfig",
    "DatasetConfig",
    "DuplicateRuleConfig",
    "ImbalancePolicy",
    "IngestionPolicy",
    "LabelColumnPolicy",
    "LabelPolicy",
    "MissingPolicy",
    "OutlierPolicy",
    "ReadOptions",
    "TypeFormatPolicy",
    "load_dataset_config",
]
