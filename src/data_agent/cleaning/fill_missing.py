from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..contracts import ChangeType, CleaningDecision, CleaningResult, LogicalType, ToolCategory, ToolSpec
from ..expressions import ExpressionError, evaluate_expression
from ..io import LoadedDataset
from ..masking import build_missing_mask, missing_policy_from_schema
from .common import CellTarget, CleaningError, CleaningRun


FILL_MISSING_TOOL_SPEC = ToolSpec(
    name="clean_missing",
    version="0.1.0",
    category=ToolCategory.PREPROCESSING,
    description=(
        "Execute one registered missing-value strategy against a complete detection "
        "selection artifact. The strategy must be chosen by the caller."
    ),
    mutates_data=True,
    risk_level="caution",
    input_schema=CleaningDecision.model_json_schema(),
    output_schema=CleaningResult.model_json_schema(),
)


def _spec_for(run: CleaningRun, column: str):
    for spec in run.schema.columns:
        if spec.name == column:
            return spec
    raise CleaningError(f"column is not present in schema: {column!r}")


def _forbid_identifier_imputation(run: CleaningRun, columns: Iterable[str], strategy: str) -> None:
    for column in set(columns):
        if _spec_for(run, column).logical_type is LogicalType.IDENTIFIER:
            raise CleaningError(
                f"strategy {strategy!r} cannot impute identifier column {column!r}; "
                "use an explicit source or leave it unchanged"
            )


def _missing_mask(run: CleaningRun, column: str, frame: pd.DataFrame | None = None) -> pd.Series:
    source = frame if frame is not None else run.frame
    spec = _spec_for(run, column)
    return build_missing_mask(
        source[column],
        spec,
        **missing_policy_from_schema(run.schema),
    ).total


def _only_currently_missing(
    run: CleaningRun,
    targets: list[CellTarget],
) -> list[CellTarget]:
    """Do not let a later missing-value step overwrite an earlier derivation."""

    keep: list[CellTarget] = []
    for target in targets:
        mask = _missing_mask(run, target.column)
        if bool(mask.iloc[target.position]):
            keep.append(target)
    return keep


def _statistic(series: pd.Series, method: str, quantile: float = 0.5) -> Any:
    values = series.dropna()
    if values.empty:
        raise CleaningError("cannot compute an imputation statistic from an empty column")
    if method == "mode":
        modes = values.value_counts()
        return modes.index[0]
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        raise CleaningError("imputation statistic requires numeric values")
    if method == "mean":
        return float(numeric.mean())
    if method == "median":
        return float(numeric.median())
    if method == "quantile":
        return float(numeric.quantile(float(quantile)))
    raise CleaningError(f"unsupported statistic method: {method!r}")


def _apply_cells(run: CleaningRun, targets: list[CellTarget], value: Any, *, reason: str, imputed: bool) -> None:
    for target in targets:
        run.set_cell(
            target.position,
            target.column,
            value,
            reason=reason,
            change_type=ChangeType.IMPUTE if imputed else ChangeType.UPDATE,
            is_imputed=imputed,
        )


def clean_missing(
    dataset: LoadedDataset,
    decision: CleaningDecision | dict[str, Any],
    *,
    selection_table: pd.DataFrame | None = None,
    selection_path: str | Path | None = None,
    selection_artifact_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = True,
    confirm: bool = False,
) -> CleaningResult:
    if isinstance(decision, dict):
        decision = CleaningDecision.model_validate(decision)
    run = CleaningRun(
        dataset,
        decision,
        dry_run=dry_run,
        confirm=confirm,
        selection_table=selection_table,
        selection_path=selection_path,
        selection_artifact_dir=selection_artifact_dir,
        artifact_dir=artifact_dir,
        output_dir=output_dir,
    )
    strategy = run.strategy.strategy_id
    params = decision.params
    targets = run.cell_targets()
    if strategy != "keep_missing":
        targets = _only_currently_missing(run, targets)
    columns = sorted({target.column for target in targets})
    if not targets and strategy not in {"keep_missing"}:
        run.warnings.append("selection contains no cells; no data was changed")

    if strategy == "keep_missing":
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    if strategy == "drop_row":
        positions = sorted({target.position for target in targets})
        max_rows = params.get("max_rows")
        if max_rows is not None and len(positions) > int(max_rows):
            raise CleaningError(f"drop_row would remove {len(positions)} rows, above max_rows={max_rows}")
        run.drop_positions(positions, reason="missing value policy: drop row")
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    if strategy == "quarantine_row":
        positions = sorted({target.position for target in targets})
        run.quarantine_rows(
            positions,
            reason="missing value policy: quarantine row",
            artifact_name=str(params.get("artifact_name", "quarantine_rows")),
        )
        run.drop_positions(positions, reason="missing value policy: quarantine row")
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    if strategy == "quarantine_cell":
        run.quarantine_cells(
            targets,
            reason="missing value policy: quarantine cell",
            artifact_name=str(params.get("artifact_name", "quarantine_cells")),
        )
        _apply_cells(run, targets, None, reason="quarantine missing cell", imputed=False)
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    if strategy == "fill_constant":
        _apply_cells(
            run,
            targets,
            params["value"],
            reason="missing value policy: fill constant",
            imputed=True,
        )
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    if strategy == "impute_statistic":
        _forbid_identifier_imputation(run, columns, strategy)
        method = str(params["method"])
        quantile = float(params.get("quantile", 0.5))
        for column in columns:
            valid_mask = ~_missing_mask(run, column)
            value = _statistic(run.frame.loc[valid_mask, column], method, quantile)
            _apply_cells(
                run,
                [target for target in targets if target.column == column],
                value,
                reason=f"missing value policy: {method} imputation",
                imputed=True,
            )
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    if strategy == "impute_grouped":
        _forbid_identifier_imputation(run, columns, strategy)
        method = str(params["method"])
        quantile = float(params.get("quantile", 0.5))
        group_columns = [str(column) for column in params["group_columns"]]
        missing_group = [column for column in group_columns if column not in run.frame.columns]
        if missing_group:
            raise CleaningError(f"group columns are absent: {missing_group}")
        min_group_size = int(params.get("min_group_size", 1))
        group_key: str | list[str] = (
            group_columns[0] if len(group_columns) == 1 else group_columns
        )
        groups = run.frame.groupby(group_key, dropna=False, observed=True).groups
        for column in columns:
            valid_mask = ~_missing_mask(run, column)
            for key, labels in groups.items():
                positions = [run.frame.index.get_loc(label) for label in labels]
                group_targets = [
                    target
                    for target in targets
                    if target.column == column and target.position in positions
                ]
                if not group_targets:
                    continue
                group_values = run.frame.iloc[positions][column]
                group_values = group_values[valid_mask.iloc[positions]]
                if len(group_values) < min_group_size:
                    fallback = _statistic(run.frame.loc[valid_mask, column], method, quantile)
                    run.warnings.append(
                        f"group {key!r} for {column!r} had fewer than {min_group_size} values; used global {method}"
                    )
                    value = fallback
                else:
                    value = _statistic(group_values, method, quantile)
                _apply_cells(
                    run,
                    group_targets,
                    value,
                    reason=f"missing value policy: grouped {method} imputation",
                    imputed=True,
                )
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    if strategy == "derive_from_formula":
        target_column = str(params["target_column"])
        if target_column not in run.frame.columns:
            raise CleaningError(f"formula target is absent: {target_column!r}")
        _forbid_identifier_imputation(run, [target_column], strategy)
        try:
            derived = evaluate_expression(str(params["expression"]), run.frame)
        except ExpressionError as error:
            raise CleaningError(f"invalid derivation expression: {error}") from error
        skipped = 0
        for target in targets:
            if target.column != target_column:
                continue
            value = derived.iloc[target.position]
            if pd.isna(value):
                skipped += 1
                continue
            if "round_digits" in params:
                value = round(float(value), int(params["round_digits"]))
            run.set_cell(
                target.position,
                target.column,
                value,
                reason="missing value policy: derived formula",
                change_type=ChangeType.IMPUTE,
                is_imputed=True,
            )
        if skipped:
            run.warnings.append(
                f"{skipped} selected cells could not be derived because the source expression was missing"
            )
        return run.finish(tool_name="clean_missing", tool_version="0.1.0")

    raise CleaningError(f"strategy {strategy!r} is not handled by clean_missing")


__all__ = ["FILL_MISSING_TOOL_SPEC", "clean_missing"]
