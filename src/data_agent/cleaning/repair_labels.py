from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import ChangeType, CleaningDecision, CleaningResult, ToolCategory, ToolSpec
from ..expressions import ExpressionError, evaluate_boolean_expression
from ..io import LoadedDataset
from .common import CellTarget, CleaningError, CleaningRun


REPAIR_LABELS_TOOL_SPEC = ToolSpec(
    name="repair_labels",
    version="0.1.0",
    category=ToolCategory.PREPROCESSING,
    description=(
        "Execute one registered label strategy against a complete label selection "
        "artifact. Labels are never treated as ordinary numeric missing values."
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


def _normalise(value: Any) -> str:
    return str(value).strip().casefold()


def _allowed_values(run: CleaningRun, column: str) -> set[str]:
    spec = _column_spec(run, column)
    return {_normalise(value) for value in spec.allowed_values}


def _set_label(
    run: CleaningRun,
    target: CellTarget,
    value: Any,
    *,
    reason: str,
    change_type: ChangeType = ChangeType.UPDATE,
) -> None:
    run.set_cell(
        target.position,
        target.column,
        value,
        reason=reason,
        change_type=change_type,
    )


def repair_labels(
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
    if strategy == "no_action":
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    targets = run.cell_targets()
    if not targets:
        run.warnings.append("selection contains no cells; no data was changed")
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    if strategy == "exclude_from_training":
        table = run.get_selection().copy()
        table["excluded_from_training"] = True
        table["exclusion_reason"] = decision.reason
        run.attach_table(
            table,
            artifact_id=f"{decision.decision_id}.training_exclusions",
            description="Rows/cells excluded from supervised training; labels were not invented.",
            metadata={"excluded_cells": len(targets)},
        )
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    if strategy == "quarantine_invalid":
        run.quarantine_cells(
            targets,
            reason="label policy: quarantine invalid label",
            artifact_name=str(params.get("artifact_name", "quarantine_labels")),
        )
        for target in targets:
            _set_label(
                run,
                target,
                None,
                reason="label policy: set invalid label to null",
                change_type=ChangeType.UPDATE,
            )
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    if strategy == "mark_unavailable":
        marker = str(params.get("marker", "unknown"))
        for target in targets:
            _set_label(
                run,
                target,
                marker,
                reason="label policy: mark unavailable",
                change_type=ChangeType.NORMALIZE,
            )
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    if strategy == "canonicalize_alias":
        for target in targets:
            column_policy = dataset.config.labels.columns.get(target.column)
            aliases = column_policy.aliases if column_policy is not None else {}
            old_value = run.value(target.position, target.column)
            old_text = _normalise(old_value)
            mapped = None
            for source, destination in aliases.items():
                if _normalise(source) == old_text:
                    mapped = destination
                    break
            if mapped is None:
                continue
            _set_label(
                run,
                target,
                mapped,
                reason="label policy: canonicalize alias",
                change_type=ChangeType.NORMALIZE,
            )
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    if strategy == "map_deterministic_value":
        mapping = {str(key): str(value) for key, value in params["mapping"].items()}
        for target in targets:
            old_value = run.value(target.position, target.column)
            if str(old_value) not in mapping:
                continue
            new_value = mapping[str(old_value)]
            allowed = _allowed_values(run, target.column)
            if allowed and _normalise(new_value) not in allowed:
                raise CleaningError(
                    f"mapped label {new_value!r} is not in allowed_values for {target.column!r}"
                )
            _set_label(
                run,
                target,
                new_value,
                reason="label policy: deterministic mapping",
                change_type=ChangeType.NORMALIZE,
            )
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    if strategy == "relabel_by_rule":
        rules = params["rules"]
        if not isinstance(rules, list) or not rules:
            raise CleaningError("relabel_by_rule requires a non-empty rules list")
        compiled_rules = []
        for rule in rules:
            if not isinstance(rule, dict) or "when" not in rule or "value" not in rule:
                raise CleaningError("each relabel rule needs when and value")
            compiled_rules.append(
                (evaluate_boolean_expression(str(rule["when"]), run.frame), str(rule["value"]))
            )
        for target in targets:
            for mask, value in compiled_rules:
                if bool(mask.iloc[target.position]):
                    allowed = _allowed_values(run, target.column)
                    if allowed and _normalise(value) not in allowed:
                        raise CleaningError(
                            f"rule label {value!r} is not in allowed_values for {target.column!r}"
                        )
                    _set_label(
                        run,
                        target,
                        value,
                        reason="label policy: relabel by rule",
                        change_type=ChangeType.NORMALIZE,
                    )
                    break
            else:
                run.warnings.append(f"no relabel rule matched {target.row_id}/{target.column}")
        return run.finish(tool_name="repair_labels", tool_version="0.1.0")

    raise CleaningError(f"strategy {strategy!r} is not handled by repair_labels")


__all__ = ["REPAIR_LABELS_TOOL_SPEC", "repair_labels"]
