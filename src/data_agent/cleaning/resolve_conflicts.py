from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import ChangeType, CleaningDecision, CleaningResult, ToolCategory, ToolSpec
from ..expressions import ExpressionError, evaluate_expression
from ..io import LoadedDataset
from .common import CleaningError, CleaningRun


RESOLVE_CONFLICTS_TOOL_SPEC = ToolSpec(
    name="resolve_conflicts",
    version="0.1.0",
    category=ToolCategory.PREPROCESSING,
    description=(
        "Execute one registered cross-field conflict strategy against a complete "
        "row selection artifact."
    ),
    mutates_data=True,
    risk_level="caution",
    input_schema=CleaningDecision.model_json_schema(),
    output_schema=CleaningResult.model_json_schema(),
)


def resolve_conflicts(
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
    if strategy in {"flag_only", "no_action"}:
        return run.finish(tool_name="resolve_conflicts", tool_version="0.1.0")

    positions = run.row_positions()
    if not positions:
        run.warnings.append("selection contains no rows; no data was changed")
        return run.finish(tool_name="resolve_conflicts", tool_version="0.1.0")

    if strategy == "quarantine_row":
        run.quarantine_rows(
            positions,
            reason="constraint policy: quarantine row",
            artifact_name=str(params.get("artifact_name", "quarantine_rows")),
        )
        run.drop_positions(positions, reason="constraint policy: quarantine row")
        return run.finish(tool_name="resolve_conflicts", tool_version="0.1.0")

    if strategy == "recompute_derived":
        target_column = str(params["target_column"])
        if target_column not in run.frame.columns:
            raise CleaningError(f"formula target is absent: {target_column!r}")
        try:
            derived = evaluate_expression(str(params["expression"]), run.frame)
        except ExpressionError as error:
            raise CleaningError(f"invalid conflict recomputation expression: {error}") from error
        for position in positions:
            value = derived.iloc[position]
            if pd.isna(value):
                continue
            if "round_digits" in params:
                value = round(float(value), int(params["round_digits"]))
            run.set_cell(
                position,
                target_column,
                value,
                reason="constraint policy: recompute derived field",
                change_type=ChangeType.CONVERT,
            )
        return run.finish(tool_name="resolve_conflicts", tool_version="0.1.0")

    if strategy == "trust_primary_source":
        target_column = str(params["target_column"])
        source_column = str(params["source_column"])
        for column in (target_column, source_column):
            if column not in run.frame.columns:
                raise CleaningError(f"source/target column is absent: {column!r}")
        for position in positions:
            source_value = run.frame[source_column].iloc[position]
            if pd.isna(source_value):
                run.warnings.append(f"source value is missing at row position {position}; left unchanged")
                continue
            run.set_cell(
                position,
                target_column,
                source_value,
                reason=f"constraint policy: trust source column {source_column!r}",
                change_type=ChangeType.CONVERT,
            )
        return run.finish(tool_name="resolve_conflicts", tool_version="0.1.0")

    if strategy == "null_conflicting_cells":
        columns = [str(column) for column in params.get("columns", decision.target_columns)]
        if not columns:
            raise CleaningError("null_conflicting_cells requires params.columns or target_columns")
        for column in columns:
            if column not in run.frame.columns:
                raise CleaningError(f"conflict column is absent: {column!r}")
        for position in positions:
            for column in columns:
                run.set_cell(
                    position,
                    column,
                    None,
                    reason="constraint policy: null conflicting cell",
                    change_type=ChangeType.UPDATE,
                )
        return run.finish(tool_name="resolve_conflicts", tool_version="0.1.0")

    raise CleaningError(f"strategy {strategy!r} is not handled by resolve_conflicts")


__all__ = ["RESOLVE_CONFLICTS_TOOL_SPEC", "resolve_conflicts"]
