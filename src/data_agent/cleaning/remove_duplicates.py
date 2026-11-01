from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import ChangeType, CleaningDecision, CleaningResult, ToolCategory, ToolSpec
from ..io import LoadedDataset
from .common import CleaningError, CleaningRun


REMOVE_DUPLICATES_TOOL_SPEC = ToolSpec(
    name="clean_duplicates",
    version="0.1.0",
    category=ToolCategory.PREPROCESSING,
    description=(
        "Execute one registered duplicate-group strategy against a complete duplicate "
        "selection table. The strategy must be chosen by the caller."
    ),
    mutates_data=True,
    risk_level="caution",
    input_schema=CleaningDecision.model_json_schema(),
    output_schema=CleaningResult.model_json_schema(),
)


def _group_positions(table: pd.DataFrame) -> dict[str, list[int]]:
    if "group_id" not in table.columns or "row_index" not in table.columns:
        raise CleaningError("duplicate selection table needs group_id and row_index")
    groups: dict[str, list[int]] = {}
    for _, row in table.iterrows():
        group_id = str(row["group_id"])
        position = int(row["row_index"])
        groups.setdefault(group_id, []).append(position)
    return {key: sorted(set(value)) for key, value in groups.items()}


def _ranked_positions(table: pd.DataFrame, group_id: str, positions: list[int]) -> list[int]:
    if "group_rank" not in table.columns:
        return positions
    rows = table[table["group_id"].astype(str) == group_id]
    ranks = {
        int(row["row_index"]): int(row["group_rank"])
        for _, row in rows.iterrows()
        if not pd.isna(row.get("group_rank"))
    }
    return sorted(positions, key=lambda position: (ranks.get(position, position), position))


def _completeness(frame: pd.DataFrame, positions: list[int]) -> list[int]:
    values = frame.iloc[positions].notna().sum(axis=1).tolist()
    return [int(value) for value in values]


def _choose_keep(
    run: CleaningRun,
    table: pd.DataFrame,
    group_id: str,
    positions: list[int],
    strategy: str,
    params: dict[str, Any],
) -> int:
    ordered = _ranked_positions(table, group_id, positions)
    if strategy == "keep_first":
        return ordered[0]
    if strategy == "keep_last":
        return ordered[-1]
    if strategy == "keep_most_complete":
        scores = _completeness(run.frame, ordered)
        return ordered[max(range(len(ordered)), key=lambda index: (scores[index], -index))]
    if strategy in {"keep_by_quality", "keep_by_priority"}:
        parameter = "quality_column" if strategy == "keep_by_quality" else "priority_column"
        column = str(params[parameter])
        if column not in run.frame.columns:
            raise CleaningError(f"ranking column is absent: {column!r}")
        values = pd.to_numeric(run.frame[column].iloc[ordered], errors="coerce")
        ascending = bool(params.get("ascending", False))
        candidates = [
            (index, value)
            for index, value in enumerate(values.tolist())
            if not pd.isna(value)
        ]
        if not candidates:
            run.warnings.append(f"group {group_id!r} has no numeric ranking values; kept first row")
            return ordered[0]
        if ascending:
            relative = min(candidates, key=lambda item: (item[1], item[0]))[0]
        else:
            relative = max(candidates, key=lambda item: (item[1], -item[0]))[0]
        return ordered[relative]
    if strategy == "merge_group":
        keep_row_id = params.get("keep_row_id")
        if keep_row_id is not None:
            rows = table[table["group_id"].astype(str) == group_id]
            if "row_id" in rows.columns:
                matches = rows[rows["row_id"].astype(str) == str(keep_row_id)]
                if not matches.empty:
                    candidate = int(matches.iloc[0]["row_index"])
                    if candidate in positions:
                        return candidate
            run.warnings.append(
                f"keep_row_id {keep_row_id!r} is not in group {group_id!r}; used completeness ranking"
            )
        prefer_column = params.get("prefer_column")
        if prefer_column is not None and prefer_column in run.frame.columns:
            present = [
                position
                for position in ordered
                if not pd.isna(run.frame[prefer_column].iloc[position])
            ]
            if present:
                return present[0]
        scores = _completeness(run.frame, ordered)
        return ordered[max(range(len(ordered)), key=lambda index: (scores[index], -index))]
    if strategy == "quarantine_group":
        return ordered[0]
    raise CleaningError(f"strategy {strategy!r} has no duplicate keep rule")


def _merge_into(
    run: CleaningRun,
    group_id: str,
    ordered: list[int],
    keep_position: int,
) -> None:
    for column in run.frame.columns:
        if column == run.schema.row_id_column:
            continue
        if not pd.isna(run.frame[column].iloc[keep_position]):
            continue
        for position in ordered:
            if position == keep_position:
                continue
            value = run.frame[column].iloc[position]
            if not pd.isna(value):
                run.set_cell(
                    keep_position,
                    column,
                    value,
                    reason=f"duplicate merge in group {group_id}",
                    change_type=ChangeType.UPDATE,
                )
                break


def clean_duplicates(
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
    if strategy == "no_action":
        return run.finish(tool_name="clean_duplicates", tool_version="0.1.0")

    table = run.get_selection()
    groups = _group_positions(table)
    drops: list[int] = []
    for group_id, positions in groups.items():
        if len(positions) < 2:
            continue
        ordered = _ranked_positions(table, group_id, positions)
        keep = _choose_keep(run, table, group_id, positions, strategy, decision.params)
        if strategy == "merge_group":
            _merge_into(run, group_id, ordered, keep)
        if strategy == "quarantine_group" and not bool(decision.params.get("keep_candidate", False)):
            drops.extend(ordered)
        else:
            drops.extend(position for position in ordered if position != keep)

    if drops:
        if strategy == "quarantine_group":
            run.quarantine_rows(
                drops,
                reason="duplicate policy: quarantine group",
                artifact_name=str(decision.params.get("artifact_name", "quarantine_groups")),
            )
        run.drop_positions(drops, reason=f"duplicate policy: {strategy}")
    else:
        run.warnings.append("no duplicate group had more than one row")
    return run.finish(tool_name="clean_duplicates", tool_version="0.1.0")


__all__ = ["REMOVE_DUPLICATES_TOOL_SPEC", "clean_duplicates"]
