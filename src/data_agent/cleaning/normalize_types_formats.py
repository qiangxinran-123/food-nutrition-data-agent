from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import ChangeType, CleaningDecision, CleaningResult, ToolCategory, ToolSpec
from ..io import LoadedDataset
from .common import CellTarget, CleaningError, CleaningRun


NORMALIZE_TYPES_FORMATS_TOOL_SPEC = ToolSpec(
    name="normalize_types_formats",
    version="0.1.0",
    category=ToolCategory.PREPROCESSING,
    description=(
        "Execute one registered type/format strategy against a complete format "
        "selection artifact. The strategy must be chosen by the caller."
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


def _identifier_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _invalid_value(run: CleaningRun, target: CellTarget, invalid: str) -> None:
    if invalid == "keep":
        return
    if invalid == "null":
        run.set_cell(
            target.position,
            target.column,
            None,
            reason="format policy: invalid value set to null",
            change_type=ChangeType.CONVERT,
        )
        return
    raise CleaningError(f"unsupported invalid-value policy: {invalid!r}")


def _map_boolean(value: Any, true_values: set[str], false_values: set[str]) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in true_values:
        return True
    if text in false_values:
        return False
    return None


def normalize_types_formats(
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
    if strategy == "leave_unchanged":
        return run.finish(tool_name="normalize_types_formats", tool_version="0.1.0")

    targets = run.cell_targets()
    if not targets:
        run.warnings.append("selection contains no cells; no data was changed")
        return run.finish(tool_name="normalize_types_formats", tool_version="0.1.0")

    if strategy == "quarantine_invalid":
        run.quarantine_cells(
            targets,
            reason="format policy: quarantine invalid",
            artifact_name=str(params.get("artifact_name", "quarantine_cells")),
        )
        for target in targets:
            _invalid_value(run, target, "null")
        return run.finish(tool_name="normalize_types_formats", tool_version="0.1.0")

    if strategy == "set_invalid_null":
        for target in targets:
            _invalid_value(run, target, "null")
        return run.finish(tool_name="normalize_types_formats", tool_version="0.1.0")

    for target in targets:
        spec = _column_spec(run, target.column)
        old_value = run.value(target.position, target.column)
        if pd.isna(old_value):
            continue

        if strategy == "trim_whitespace":
            value = str(old_value).strip()
            run.set_cell(
                target.position,
                target.column,
                value,
                reason="format policy: trim whitespace",
                change_type=ChangeType.NORMALIZE,
            )
            continue

        if strategy == "normalize_case":
            case = str(params["case"]).lower()
            text = str(old_value)
            if case == "lower":
                value = text.lower()
            elif case == "upper":
                value = text.upper()
            elif case == "title":
                value = text.title()
            elif case == "casefold":
                value = text.casefold()
            else:
                raise CleaningError(f"unsupported case mode: {case!r}")
            run.set_cell(
                target.position,
                target.column,
                value,
                reason=f"format policy: normalize case ({case})",
                change_type=ChangeType.NORMALIZE,
            )
            continue

        if strategy == "cast_numeric":
            parsed = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
            if pd.isna(parsed):
                _invalid_value(run, target, str(params.get("invalid", "null")))
                continue
            run.set_cell(
                target.position,
                target.column,
                float(parsed),
                reason="format policy: cast numeric",
                change_type=ChangeType.CONVERT,
            )
            continue

        if strategy == "cast_datetime":
            datetime_format = params.get("datetime_format", spec.metadata.get("datetime_format"))
            dayfirst = bool(params.get("dayfirst", spec.metadata.get("dayfirst", False)))
            yearfirst = bool(params.get("yearfirst", spec.metadata.get("yearfirst", False)))
            parsed = pd.to_datetime(
                old_value,
                errors="coerce",
                format=datetime_format,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
            )
            if pd.isna(parsed):
                _invalid_value(run, target, str(params.get("invalid", "null")))
                continue
            run.set_cell(
                target.position,
                target.column,
                parsed.isoformat(),
                reason="format policy: cast datetime",
                change_type=ChangeType.CONVERT,
            )
            continue

        if strategy == "map_boolean":
            true_values = {
                str(value).strip().casefold()
                for value in params.get("true_values", ["true", "1", "yes", "y", "是"])
            }
            false_values = {
                str(value).strip().casefold()
                for value in params.get("false_values", ["false", "0", "no", "n", "否"])
            }
            value = _map_boolean(old_value, true_values, false_values)
            if value is None:
                _invalid_value(run, target, str(params.get("invalid", "null")))
                continue
            run.set_cell(
                target.position,
                target.column,
                value,
                reason="format policy: map boolean",
                change_type=ChangeType.CONVERT,
            )
            continue

        if strategy == "preserve_identifier_as_text":
            run.set_cell(
                target.position,
                target.column,
                _identifier_text(old_value),
                reason="format policy: preserve identifier as text",
                change_type=ChangeType.CONVERT,
            )
            continue

        if strategy == "normalize_date_format":
            datetime_format = params.get("datetime_format", spec.metadata.get("datetime_format"))
            dayfirst = bool(params.get("dayfirst", spec.metadata.get("dayfirst", False)))
            yearfirst = bool(params.get("yearfirst", spec.metadata.get("yearfirst", False)))
            parsed = pd.to_datetime(
                old_value,
                errors="coerce",
                format=datetime_format,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
            )
            if pd.isna(parsed):
                run.warnings.append(f"could not parse date at {target.row_id}/{target.column}")
                continue
            run.set_cell(
                target.position,
                target.column,
                parsed.strftime(str(params["format"])),
                reason="format policy: normalize date format",
                change_type=ChangeType.NORMALIZE,
            )
            continue

        raise CleaningError(f"strategy {strategy!r} is not handled by normalize_types_formats")

    return run.finish(tool_name="normalize_types_formats", tool_version="0.1.0")


__all__ = ["NORMALIZE_TYPES_FORMATS_TOOL_SPEC", "normalize_types_formats"]
