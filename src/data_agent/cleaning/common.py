from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..checks.conflicts import detect_conflicts
from ..checks.duplicates import detect_duplicates
from ..checks.imbalance import detect_imbalance
from ..checks.labels import detect_label_errors
from ..checks.missing import detect_missing
from ..checks.outliers import detect_outliers
from ..checks.types_formats import detect_type_format_issues
from ..contracts import (
    Artifact,
    ArtifactKind,
    ChangeRecord,
    ChangeSet,
    ChangeType,
    CheckDelta,
    CleaningAction,
    CleaningDecision,
    CleaningResult,
    DatasetRef,
    LogicalType,
    RiskLevel,
    ToolStatus,
    ValidationReport,
)
from ..io import LoadedDataset
from .strategies import (
    StrategyNotAvailable,
    StrategyParameterError,
    StrategySpec,
    get_strategy,
    validate_strategy_params,
)
from .writer import write_processed_dataset, write_processed_profile, write_table_artifact


class CleaningError(ValueError):
    pass


class CleaningConfirmationError(CleaningError):
    pass


class SelectionError(CleaningError):
    pass


@dataclass(frozen=True)
class CellTarget:
    position: int
    row_id: str
    column: str
    reason_code: str | None = None


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    return str(value)


def _values_equal(left: Any, right: Any) -> bool:
    left_missing = _json_value(left) is None
    right_missing = _json_value(right) is None
    if left_missing or right_missing:
        return left_missing and right_missing
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def load_selection_table(
    selection_artifact_id: str | None,
    *,
    artifact_dir: str | Path | None = None,
    selection_path: str | Path | None = None,
) -> pd.DataFrame:
    if selection_path is not None:
        path = Path(selection_path)
    elif selection_artifact_id:
        candidate = Path(selection_artifact_id)
        if candidate.exists():
            path = candidate
        elif artifact_dir is not None:
            path = Path(artifact_dir) / f"{selection_artifact_id}.parquet"
        else:
            raise SelectionError(
                "selection artifact directory is required when the artifact id is not a path"
            )
    else:
        raise SelectionError("a selection artifact is required for this strategy")
    if not path.exists():
        raise SelectionError(f"selection artifact not found: {path}")
    return pd.read_parquet(path, engine="fastparquet")


def _row_id_series(frame: pd.DataFrame, schema: Any) -> pd.Series:
    row_id_column = getattr(schema, "row_id_column", None)
    if row_id_column is not None and row_id_column in frame.columns:
        return frame[row_id_column].astype("string")
    return pd.Series(
        [f"row_index:{index}" for index in range(len(frame))],
        index=frame.index,
        dtype="string",
    )


def row_id_for_position(frame: pd.DataFrame, schema: Any, position: int) -> str:
    if position < 0 or position >= len(frame):
        raise SelectionError(f"row position out of range: {position}")
    row_id = _row_id_series(frame, schema).iloc[position]
    text = str(row_id)
    return text if text and text != "<NA>" else f"row_index:{position}"


def positions_from_row_ids(
    frame: pd.DataFrame,
    schema: Any,
    row_ids: Iterable[Any],
) -> list[int]:
    row_id_column = getattr(schema, "row_id_column", None)
    if row_id_column is None or row_id_column not in frame.columns:
        raise SelectionError("row_id_column is not available for row-id selection")
    lookup = {
        str(value): position
        for position, value in enumerate(frame[row_id_column].tolist())
    }
    positions: list[int] = []
    for value in row_ids:
        text = str(value)
        if text not in lookup:
            raise SelectionError(f"row id not present in current dataset: {value!r}")
        positions.append(lookup[text])
    return positions


def _table_positions(table: pd.DataFrame, frame: pd.DataFrame, schema: Any) -> list[int]:
    if "row_index" in table.columns:
        raw = table["row_index"].dropna().tolist()
        positions: list[int] = []
        for value in raw:
            try:
                position = int(value)
            except (TypeError, ValueError) as error:
                raise SelectionError(f"invalid row_index in selection: {value!r}") from error
            if position < 0 or position >= len(frame):
                raise SelectionError(f"row_index out of range in selection: {position}")
            positions.append(position)
    elif "row_id" in table.columns:
        positions = positions_from_row_ids(frame, schema, table["row_id"].tolist())
    else:
        raise SelectionError("selection table needs row_index or row_id")
    return list(dict.fromkeys(positions))


def cells_from_selection(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    schema: Any,
    *,
    target_columns: Iterable[str] = (),
) -> list[CellTarget]:
    if "column" not in table.columns:
        raise SelectionError("cell selection table needs a column field")
    allowed = set(target_columns)
    targets: list[CellTarget] = []
    seen: set[tuple[int, str]] = set()
    for row_index, row in table.iterrows():
        column = str(row["column"])
        if column not in frame.columns:
            raise SelectionError(f"selection references unknown column: {column!r}")
        if allowed and column not in allowed:
            continue
        if "row_index" in table.columns and not pd.isna(row["row_index"]):
            position = int(row["row_index"])
        elif "row_id" in table.columns:
            position = positions_from_row_ids(frame, schema, [row["row_id"]])[0]
        else:
            raise SelectionError(f"selection row {row_index} has no row locator")
        if position < 0 or position >= len(frame):
            raise SelectionError(f"row_index out of range: {position}")
        key = (position, column)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            CellTarget(
                position=position,
                row_id=row_id_for_position(frame, schema, position),
                column=column,
                reason_code=(
                    str(row["reason_code"])
                    if "reason_code" in table.columns and not pd.isna(row["reason_code"])
                    else None
                ),
            )
        )
    return targets


def rows_from_selection(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    schema: Any,
) -> list[int]:
    return _table_positions(table, frame, schema)


def _change_table(changes: list[ChangeRecord], decision: CleaningDecision) -> pd.DataFrame:
    records = []
    for change in changes:
        records.append(
            {
                "decision_id": decision.decision_id,
                "finding_id": decision.finding_id,
                "strategy_id": decision.strategy_id,
                "row_id": change.row_id,
                "column": change.column,
                "old_value": _json_value(change.old_value),
                "new_value": _json_value(change.new_value),
                "change_type": change.change_type.value,
                "reason": change.reason,
                "is_imputed": change.is_imputed,
            }
        )
    return pd.DataFrame.from_records(records)


@dataclass
class CleaningRun:
    dataset: LoadedDataset
    decision: CleaningDecision
    strategy: StrategySpec = field(init=False)
    before: pd.DataFrame = field(init=False)
    frame: pd.DataFrame = field(init=False)
    started_at: float = field(default_factory=perf_counter, init=False)
    dry_run: bool = True
    confirm: bool = False
    validate: bool = True
    selection_table: pd.DataFrame | None = None
    selection_path: str | Path | None = None
    selection_artifact_dir: str | Path | None = None
    artifact_dir: str | Path | None = None
    output_dir: str | Path | None = None

    changes: list[ChangeRecord] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.strategy = get_strategy(self.decision.issue_type, self.decision.strategy_id)
            validate_strategy_params(self.strategy, self.decision.params)
        except (StrategyNotAvailable, StrategyParameterError) as error:
            raise CleaningError(str(error)) from error
        if self.strategy.risk_level is RiskLevel.DANGEROUS and not self.dry_run and not self.confirm:
            raise CleaningConfirmationError(
                f"strategy {self.strategy.strategy_id!r} requires confirm=true "
                "before writing a processed dataset"
            )
        self.before = self.dataset.frame.copy(deep=True)
        self.frame = self.dataset.frame.copy(deep=True)
        if self.artifact_dir is None:
            self.artifact_dir = Path("logs") / "cleaning" / self.dataset.ref.dataset_id
        if self.output_dir is None:
            self.output_dir = Path("data") / "processed"
        self.artifact_dir = Path(self.artifact_dir)
        self.output_dir = Path(self.output_dir)

    @property
    def schema(self) -> Any:
        return self.dataset.schema

    def get_selection(self) -> pd.DataFrame:
        if self.selection_table is not None:
            return self.selection_table.copy()
        return load_selection_table(
            self.decision.selection_artifact_id,
            artifact_dir=self.selection_artifact_dir,
            selection_path=self.selection_path,
        )

    def cell_targets(self) -> list[CellTarget]:
        return cells_from_selection(
            self.get_selection(),
            self.frame,
            self.schema,
            target_columns=self.decision.target_columns,
        )

    def row_positions(self) -> list[int]:
        return rows_from_selection(self.get_selection(), self.frame, self.schema)

    def value(self, position: int, column: str) -> Any:
        return self.frame[column].iloc[position]

    def set_cell(
        self,
        position: int,
        column: str,
        value: Any,
        *,
        reason: str,
        change_type: ChangeType = ChangeType.UPDATE,
        is_imputed: bool = False,
    ) -> bool:
        if position < 0 or position >= len(self.frame):
            raise SelectionError(f"row position out of range: {position}")
        if column not in self.frame.columns:
            raise SelectionError(f"unknown column: {column!r}")
        old_value = self.frame[column].iloc[position]
        if _values_equal(old_value, value):
            return False
        row_id = row_id_for_position(self.frame, self.schema, position)
        self.changes.append(
            ChangeRecord(
                row_id=row_id,
                column=column,
                old_value=_json_value(old_value),
                new_value=_json_value(value),
                change_type=change_type,
                reason=reason,
                is_imputed=is_imputed,
            )
        )
        label = self.frame.index[position]
        try:
            self.frame.at[label, column] = value
        except (TypeError, ValueError):
            # A pandas string extension column cannot accept a numeric replacement
            # in place. Convert that column to object before applying the explicit
            # strategy result; a later profiling pass can infer the new dtype.
            self.frame[column] = self.frame[column].astype("object")
            self.frame.at[label, column] = value
        return True

    def drop_positions(self, positions: Iterable[int], *, reason: str) -> int:
        unique_positions = sorted(set(int(position) for position in positions))
        labels = []
        for position in unique_positions:
            if position < 0 or position >= len(self.frame):
                raise SelectionError(f"row position out of range: {position}")
            labels.append(self.frame.index[position])
            self.changes.append(
                ChangeRecord(
                    row_id=row_id_for_position(self.frame, self.schema, position),
                    change_type=ChangeType.DROP,
                    reason=reason,
                )
            )
        if labels:
            self.frame = self.frame.drop(index=labels)
        return len(labels)

    def add_rows(self, rows: pd.DataFrame, *, reason: str) -> int:
        if rows.empty:
            return 0
        start_index = len(self.frame)
        row_id_column = self.schema.row_id_column
        rows = rows.copy()
        if row_id_column is not None and row_id_column in self.frame.columns:
            if row_id_column not in rows.columns:
                rows[row_id_column] = None
            existing = set(self.frame[row_id_column].astype("string").tolist())
            generated = []
            counter = 1
            for _ in range(len(rows)):
                while True:
                    candidate = f"{self.dataset.ref.dataset_id}_added_{start_index + counter:06d}"
                    counter += 1
                    if candidate not in existing:
                        existing.add(candidate)
                        generated.append(candidate)
                        break
            rows[row_id_column] = generated
        self.frame = pd.concat([self.frame, rows], ignore_index=True, sort=False)
        for offset in range(len(rows)):
            position = start_index + offset
            self.changes.append(
                ChangeRecord(
                    row_id=row_id_for_position(self.frame, self.schema, position),
                    change_type=ChangeType.ADD,
                    reason=reason,
                )
            )
        return len(rows)

    def quarantine_rows(
        self,
        positions: Iterable[int],
        *,
        reason: str,
        artifact_name: str = "quarantine_rows",
    ) -> Artifact | None:
        unique_positions = sorted(set(int(position) for position in positions))
        if not unique_positions:
            return None
        table = self.before.iloc[unique_positions].copy()
        table["__row_id_for_quarantine"] = [
            row_id_for_position(self.before, self.schema, position)
            for position in unique_positions
        ]
        table["__reason"] = reason
        artifact = write_table_artifact(
            table,
            output_dir=self.artifact_dir,
            artifact_id=f"{self.decision.decision_id}.{artifact_name}",
            kind=ArtifactKind.ROW_LIST,
            dataset_id=self.dataset.ref.dataset_id,
            description=f"Quarantined rows for {self.decision.finding_id}",
            metadata={
                "finding_id": self.decision.finding_id,
                "decision_id": self.decision.decision_id,
                "row_count": len(unique_positions),
            },
        )
        self.artifacts.append(artifact)
        return artifact

    def quarantine_cells(
        self,
        targets: Iterable[CellTarget],
        *,
        reason: str,
        artifact_name: str = "quarantine_cells",
    ) -> Artifact | None:
        records = []
        for target in targets:
            records.append(
                {
                    "row_index": target.position,
                    "row_id": target.row_id,
                    "column": target.column,
                    "old_value": _json_value(self.before[target.column].iloc[target.position]),
                    "reason": reason,
                }
            )
        if not records:
            return None
        artifact = write_table_artifact(
            pd.DataFrame.from_records(records),
            output_dir=self.artifact_dir,
            artifact_id=f"{self.decision.decision_id}.{artifact_name}",
            kind=ArtifactKind.CELL_LIST,
            dataset_id=self.dataset.ref.dataset_id,
            description=f"Quarantined cells for {self.decision.finding_id}",
            metadata={
                "finding_id": self.decision.finding_id,
                "decision_id": self.decision.decision_id,
                "cell_count": len(records),
            },
        )
        self.artifacts.append(artifact)
        return artifact

    def attach_table(
        self,
        table: pd.DataFrame,
        *,
        artifact_id: str,
        kind: ArtifactKind = ArtifactKind.TABLE,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        artifact = write_table_artifact(
            table,
            output_dir=self.artifact_dir,
            artifact_id=artifact_id,
            kind=kind,
            dataset_id=self.dataset.ref.dataset_id,
            description=description,
            metadata={
                "decision_id": self.decision.decision_id,
                "finding_id": self.decision.finding_id,
                **(metadata or {}),
            },
        )
        self.artifacts.append(artifact)
        return artifact

    def finish(self, *, tool_name: str, tool_version: str = "0.1") -> CleaningResult:
        output = self._coerce_output_dtypes(self.frame).reset_index(drop=True)
        changes_table = _change_table(self.changes, self.decision)
        cells_changed = sum(
            1
            for change in self.changes
            if change.change_type
            in {ChangeType.UPDATE, ChangeType.IMPUTE, ChangeType.NORMALIZE, ChangeType.CONVERT}
        )
        rows_added = sum(1 for change in self.changes if change.change_type is ChangeType.ADD)
        rows_removed = sum(1 for change in self.changes if change.change_type is ChangeType.DROP)
        change_id = f"change.{self.decision.decision_id}"

        changes_uri: str | None = None
        if not changes_table.empty:
            change_artifact = write_table_artifact(
                changes_table,
                output_dir=self.artifact_dir,
                artifact_id=f"{change_id}.details",
                kind=ArtifactKind.TABLE,
                dataset_id=self.dataset.ref.dataset_id,
                description="Complete cell/row change log for this cleaning decision.",
                metadata={"decision_id": self.decision.decision_id},
            )
            self.artifacts.append(change_artifact)
            changes_uri = change_artifact.uri

        action = CleaningAction(
            action_id=self.decision.decision_id,
            tool_name=tool_name,
            target_columns=self.decision.target_columns,
            params=self.decision.params,
            reason=self.decision.reason,
            source_findings=[self.decision.finding_id],
            risk_level=self.strategy.risk_level,
            requires_confirmation=self.strategy.requires_confirmation
            or self.strategy.risk_level is RiskLevel.DANGEROUS,
            expected_effect=self.decision.expected_effect,
        )

        if self.dry_run:
            after = self.dataset.ref.model_copy(
                update={
                    "row_count": int(len(output)),
                    "column_count": int(len(output.columns)),
                    "metadata": {
                        **self.dataset.ref.metadata,
                        "dry_run": True,
                        "parent_id": self.dataset.ref.dataset_id,
                    },
                }
            )
        else:
            after, _path, _changes = write_processed_dataset(
                output,
                before=self.dataset.ref,
                output_dir=self.output_dir,
                dataset_id=self.decision.metadata.get("output_dataset_id"),
                change_id=change_id,
                changes_table=changes_table,
                description=self.decision.reason,
                metadata={
                    "strategy_id": self.strategy.strategy_id,
                    "finding_id": self.decision.finding_id,
                    **self.metadata,
                },
            )
            manifest_uri = after.metadata.get("manifest_uri")
            if manifest_uri and self.dataset.config_path is not None:
                profile_path = write_processed_profile(
                    self.dataset.config,
                    after=after,
                    manifest_path=str(manifest_uri),
                    profile_dir=Path(self.dataset.config_path).parent,
                )
                after = after.model_copy(
                    update={
                        "metadata": {
                            **after.metadata,
                            "profile_uri": profile_path.resolve().as_posix(),
                        }
                    }
                )

        change_set = ChangeSet(
            change_id=change_id,
            before_dataset=self.dataset.ref,
            after_dataset=after,
            actions=[action],
            rows_added=rows_added,
            rows_removed=rows_removed,
            cells_changed=cells_changed,
            changes_uri=changes_uri,
            sample_changes=self.changes[:50],
            description=self.decision.reason,
            metadata={
                "strategy_id": self.strategy.strategy_id,
                "finding_id": self.decision.finding_id,
            },
        )
        validation = (
            self._build_validation(output, after)
            if self.validate
            else None
        )
        if validation is not None and not validation.passed:
            self.warnings.append(
                "post-cleaning validation did not improve the selected issue count"
            )
        status = (
            ToolStatus.PARTIAL
            if validation is not None and not validation.passed
            else ToolStatus.SUCCESS
        )
        return CleaningResult(
            tool_name=tool_name,
            tool_version=tool_version,
            dataset_id=self.dataset.ref.dataset_id,
            status=status,
            before_dataset=self.dataset.ref,
            after_dataset=after,
            decision=self.decision,
            change_set=change_set,
            artifacts=self.artifacts,
            dry_run=self.dry_run,
            validation=validation,
            warnings=self.warnings,
            duration_ms=(perf_counter() - self.started_at) * 1000,
            output_frame=output,
            metadata={
                "strategy": self.strategy.model_dump(mode="json"),
                "rows_before": int(len(self.before)),
                "rows_after": int(len(output)),
                "rows_added": rows_added,
                "rows_removed": rows_removed,
                "cells_changed": cells_changed,
                **self.metadata,
            },
        )

    def _issue_count(self, frame: pd.DataFrame) -> int:
        issue = self.decision.issue_type
        try:
            if issue.value == "missing":
                result = detect_missing(frame, self.schema)
                return int(result.summary.get("missing_cells", 0))
            if issue.value == "duplicate":
                result = detect_duplicates(
                    frame,
                    self.schema,
                    self.dataset.config.duplicates,
                )
                return int(result.summary.get("rows_flagged_union", 0))
            if issue.value == "outlier":
                result = detect_outliers(frame, self.schema, self.dataset.config.outliers)
                return int(result.summary.get("rows_flagged_union", 0))
            if issue.value == "format":
                result = detect_type_format_issues(
                    frame,
                    self.schema,
                    self.dataset.config.types_formats,
                )
                return int(result.summary.get("rows_flagged_union", 0))
            if issue.value == "constraint":
                result = detect_conflicts(
                    frame,
                    self.schema,
                    self.dataset.config.conflicts,
                )
                return int(result.summary.get("rows_flagged_union", 0))
            if issue.value == "label":
                result = detect_label_errors(
                    frame,
                    self.schema,
                    self.dataset.config.labels,
                )
                return int(result.summary.get("rows_flagged_union", 0))
            if issue.value == "imbalance":
                result = detect_imbalance(
                    frame,
                    self.schema,
                    self.dataset.config.labels,
                    self.dataset.config.imbalance,
                )
                return int(result.summary.get("imbalanced_columns", 0))
        except Exception as error:  # validation must not hide the cleaning result
            self.warnings.append(
                f"post-cleaning validation for {issue.value!r} skipped: {error}"
            )
            return 0
        return 0

    def _build_validation(
        self,
        output: pd.DataFrame,
        after: DatasetRef,
    ) -> ValidationReport | None:
        before_count = self._issue_count(self.before)
        after_count = self._issue_count(output)
        total = int(self.dataset.ref.row_count)
        retained_rate = int(len(output)) / total if total else 1.0
        return ValidationReport(
            validation_id=f"validation.{self.decision.decision_id}",
            before_dataset=self.dataset.ref,
            after_dataset=after,
            checks=[
                CheckDelta(
                    check_type=self.decision.issue_type,
                    before_count=before_count,
                    after_count=after_count,
                    passed=after_count <= before_count,
                    notes="Counts are produced by the matching detection tool.",
                )
            ],
            passed=after_count <= before_count,
            retained_rows=int(len(output)),
            retained_rate=max(0.0, retained_rate),
            warnings=list(self.warnings),
            metrics={
                "before_count": before_count,
                "after_count": after_count,
                "rows_before": int(len(self.before)),
                "rows_after": int(len(output)),
            },
        )

    def _coerce_output_dtypes(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for spec in self.schema.columns:
            if spec.name not in result.columns:
                continue
            series = result[spec.name]
            if spec.logical_type is LogicalType.NUMERIC:
                converted = pd.to_numeric(series, errors="coerce")
                present = series.notna()
                if bool(converted[present].notna().all()):
                    result[spec.name] = converted
            elif spec.logical_type is LogicalType.IDENTIFIER:
                result[spec.name] = series.astype("string")
        return result


__all__ = [
    "CellTarget",
    "CleaningConfirmationError",
    "CleaningError",
    "CleaningRun",
    "SelectionError",
    "cells_from_selection",
    "load_selection_table",
    "positions_from_row_ids",
    "row_id_for_position",
    "rows_from_selection",
]
