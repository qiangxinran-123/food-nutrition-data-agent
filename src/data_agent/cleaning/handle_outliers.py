from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import ChangeType, CleaningDecision, CleaningResult, ToolCategory, ToolSpec
from ..expressions import ExpressionError, evaluate_expression
from ..io import LoadedDataset
from .common import CellTarget, CleaningError, CleaningRun


HANDLE_OUTLIERS_TOOL_SPEC = ToolSpec(
    name="clean_outliers",
    version="0.1.0",
    category=ToolCategory.PREPROCESSING,
    description=(
        "Execute one registered outlier strategy against a complete outlier "
        "selection artifact. Statistical tails must not be treated as proven errors."
    ),
    mutates_data=True,
    risk_level="caution",
    input_schema=CleaningDecision.model_json_schema(),
    output_schema=CleaningResult.model_json_schema(),
)


def _column_spec(run: CleaningRun, column: str):
    for spec in run.schema.columns:
        if spec.name == column:
            return spec
    raise CleaningError(f"column is not present in schema: {column!r}")


def _set_null(run: CleaningRun, targets: list[CellTarget], reason: str) -> None:
    for target in targets:
        run.set_cell(
            target.position,
            target.column,
            None,
            reason=reason,
            change_type=ChangeType.UPDATE,
        )


def clean_outliers(
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
    if strategy == "flag_only":
        run.metadata["flagged_cells"] = len(run.cell_targets())
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    targets = run.cell_targets()
    positions = sorted({target.position for target in targets})
    run.metadata["flagged_cells"] = len(targets)
    if not targets:
        run.warnings.append("selection contains no cells; no data was changed")

    if strategy == "drop_row":
        max_rows = params.get("max_rows")
        if max_rows is not None and len(positions) > int(max_rows):
            raise CleaningError(f"drop_row would remove {len(positions)} rows, above max_rows={max_rows}")
        run.drop_positions(positions, reason="outlier policy: drop row")
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    if strategy == "quarantine_row":
        run.quarantine_rows(
            positions,
            reason="outlier policy: quarantine row",
            artifact_name=str(params.get("artifact_name", "quarantine_rows")),
        )
        run.drop_positions(positions, reason="outlier policy: quarantine row")
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    if strategy == "quarantine_cell":
        run.quarantine_cells(
            targets,
            reason="outlier policy: quarantine cell",
            artifact_name=str(params.get("artifact_name", "quarantine_cells")),
        )
        _set_null(run, targets, "outlier policy: quarantine cell")
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    if strategy == "set_null":
        _set_null(run, targets, "outlier policy: set null")
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    if strategy == "clip_value":
        for target in targets:
            spec = _column_spec(run, target.column)
            lower = params.get("minimum")
            upper = params.get("maximum")
            if spec.range is not None:
                if lower is None:
                    lower = spec.range.minimum
                if upper is None:
                    upper = spec.range.maximum
            if lower is None and upper is None:
                raise CleaningError(
                    f"clip_value needs a declared range or minimum/maximum for {target.column!r}"
                )
            old_value = run.value(target.position, target.column)
            numeric = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
            if pd.isna(numeric):
                run.warnings.append(f"skipped non-numeric outlier at {target.row_id}/{target.column}")
                continue
            value = float(numeric)
            if lower is not None:
                value = max(value, float(lower))
            if upper is not None:
                value = min(value, float(upper))
            run.set_cell(
                target.position,
                target.column,
                value,
                reason="outlier policy: clip to bounds",
                change_type=ChangeType.UPDATE,
            )
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    if strategy == "recompute_by_formula":
        target_column = str(params["target_column"])
        if target_column not in run.frame.columns:
            raise CleaningError(f"formula target is absent: {target_column!r}")
        try:
            derived = evaluate_expression(str(params["expression"]), run.frame)
        except ExpressionError as error:
            raise CleaningError(f"invalid recomputation expression: {error}") from error
        for target in targets:
            if target.column != target_column:
                continue
            value = derived.iloc[target.position]
            if pd.isna(value):
                continue
            if "round_digits" in params:
                value = round(float(value), int(params["round_digits"]))
            run.set_cell(
                target.position,
                target.column,
                value,
                reason="outlier policy: recompute formula",
                change_type=ChangeType.UPDATE,
            )
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    if strategy == "correct_unit_or_scale":
        column = str(params["column"])
        if column not in run.frame.columns:
            raise CleaningError(f"scale-correction column is absent: {column!r}")
        factor = float(params["factor"])
        offset = float(params.get("offset", 0.0))
        for target in targets:
            if target.column != column:
                continue
            old_value = pd.to_numeric(pd.Series([run.value(target.position, column)]), errors="coerce").iloc[0]
            if pd.isna(old_value):
                run.warnings.append(f"skipped non-numeric value at {target.row_id}/{column}")
                continue
            run.set_cell(
                target.position,
                column,
                float(old_value) * factor + offset,
                reason="outlier policy: correct unit or scale",
                change_type=ChangeType.CONVERT,
            )
        return run.finish(tool_name="clean_outliers", tool_version="0.1.0")

    raise CleaningError(f"strategy {strategy!r} is not handled by clean_outliers")


__all__ = ["HANDLE_OUTLIERS_TOOL_SPEC", "clean_outliers"]
