from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import pandas as pd

from ..contracts import (
    ChangeSet,
    CleaningPlan,
    CleaningPlanResult,
    RiskLevel,
    ToolStatus,
)
from ..io import LoadedDataset
from ..profiling import build_schema_profile
from .common import CleaningConfirmationError, CleaningError, load_selection_table
from .strategies import get_strategy, validate_strategy_params
from .writer import write_processed_dataset, write_processed_profile


def _remap_selection(
    table: pd.DataFrame,
    dataset: LoadedDataset,
    *,
    positions_may_have_shifted: bool,
) -> tuple[pd.DataFrame, int]:
    row_id_column = dataset.schema.row_id_column
    if row_id_column is None or row_id_column not in dataset.frame.columns:
        if positions_may_have_shifted:
            raise CleaningError(
                "a multi-step plan changed row positions but the dataset has no stable row_id"
            )
        return table, 0
    if "row_id" not in table.columns:
        if positions_may_have_shifted:
            raise CleaningError(
                "a multi-step plan changed row positions but the selection has no row_id"
            )
        return table, 0

    lookup = {
        str(value): position
        for position, value in enumerate(dataset.frame[row_id_column].tolist())
    }
    keep = table["row_id"].map(lambda value: str(value) in lookup)
    remapped = table.loc[keep].copy()
    remapped["row_index"] = remapped["row_id"].map(lambda value: lookup[str(value)])
    return remapped, int((~keep).sum())


def _next_dataset(dataset: LoadedDataset, result: Any) -> LoadedDataset:
    if result.output_frame is None:
        raise CleaningError("cleaning result did not expose an output frame")
    frame = result.output_frame.reset_index(drop=True)
    after = result.after_dataset
    config = dataset.config.model_copy(update={"dataset_id": after.dataset_id})
    schema = build_schema_profile(after, frame, config)

    if result.dry_run:
        return LoadedDataset(
            ref=after,
            schema=schema,
            frame=frame,
            manifest=dataset.manifest,
            manifest_path=dataset.manifest_path,
            data_path=dataset.data_path,
            config=config,
            config_path=dataset.config_path,
        )

    manifest_path = Path(str(after.metadata["manifest_uri"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile_uri = after.metadata.get("profile_uri")
    return LoadedDataset(
        ref=after,
        schema=schema,
        frame=frame,
        manifest=manifest,
        manifest_path=manifest_path,
        data_path=Path(after.uri),
        config=config,
        config_path=Path(profile_uri) if profile_uri else None,
    )


def _failed_plan(
    *,
    plan: CleaningPlan,
    dataset: LoadedDataset,
    current: LoadedDataset,
    results: list[Any],
    warnings: list[str],
    started_at: float,
    code: str,
    message: str,
) -> CleaningPlanResult:
    return CleaningPlanResult(
        plan_id=plan.plan_id,
        status=ToolStatus.FAILED,
        before_dataset=dataset.ref,
        after_dataset=current.ref,
        results=results,
        dry_run=plan.dry_run,
        warnings=warnings,
        error_code=code,
        error_message=message,
        duration_ms=(perf_counter() - started_at) * 1000,
    )


def _write_plan_result(
    *,
    dataset: LoadedDataset,
    plan: CleaningPlan,
    results: list[Any],
    current: LoadedDataset,
    output_dir: str | Path | None,
) -> tuple[Any, ChangeSet | None]:
    change_frames = []
    for result in results:
        changes_uri = result.change_set.changes_uri if result.change_set else None
        if changes_uri:
            change_frames.append(pd.read_parquet(changes_uri, engine="fastparquet"))
    changes_table = (
        pd.concat(change_frames, ignore_index=True)
        if change_frames
        else pd.DataFrame()
    )
    change_id = f"change.{plan.plan_id}"
    directory = Path(output_dir) if output_dir is not None else Path("data") / "processed"
    after, _path, changes_path = write_processed_dataset(
        current.frame.reset_index(drop=True),
        before=dataset.ref,
        output_dir=directory,
        dataset_id=plan.output_dataset_id,
        change_id=change_id,
        changes_table=changes_table,
        description=plan.description or "Final dataset produced by a cleaning plan.",
        metadata={
            "plan_id": plan.plan_id,
            "persist_mode": "final",
            "decision_count": len(plan.decisions),
        },
    )
    if dataset.config_path is not None:
        profile_path = write_processed_profile(
            dataset.config,
            after=after,
            manifest_path=str(after.metadata["manifest_uri"]),
            profile_dir=Path(dataset.config_path).parent,
        )
        after = after.model_copy(
            update={
                "metadata": {
                    **after.metadata,
                    "profile_uri": profile_path.resolve().as_posix(),
                }
            }
        )

    actions = [
        action
        for result in results
        if result.change_set is not None
        for action in result.change_set.actions
    ]
    rows_added = sum(
        result.change_set.rows_added
        for result in results
        if result.change_set is not None
    )
    rows_removed = sum(
        result.change_set.rows_removed
        for result in results
        if result.change_set is not None
    )
    cells_changed = sum(
        result.change_set.cells_changed
        for result in results
        if result.change_set is not None
    )
    samples = [
        change
        for result in results
        if result.change_set is not None
        for change in result.change_set.sample_changes
    ][:50]
    change_set = ChangeSet(
        change_id=change_id,
        before_dataset=dataset.ref,
        after_dataset=after,
        actions=actions,
        rows_added=rows_added,
        rows_removed=rows_removed,
        cells_changed=cells_changed,
        changes_uri=changes_path.as_posix() if changes_path else None,
        sample_changes=samples,
        description=plan.description,
        metadata={"plan_id": plan.plan_id, "persist_mode": "final"},
    )
    return after, change_set


def execute_plan(
    dataset: LoadedDataset,
    plan: CleaningPlan | dict[str, Any],
    *,
    selection_artifact_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    confirm: bool = False,
    persist: Literal["per_step", "final"] = "per_step",
) -> CleaningPlanResult:
    """Execute decisions in order, feeding each resulting frame into the next.

    ``per_step`` writes one dataset version per action. ``final`` runs every
    action in memory and writes only one final dataset with one aggregate change
    log. In both modes, selection tables are remapped through the stable row id.
    """

    if isinstance(plan, dict):
        plan = CleaningPlan.model_validate(plan)
    if persist not in {"per_step", "final"}:
        raise CleaningError(f"unsupported persist mode: {persist!r}")
    if plan.before_dataset.dataset_id != dataset.ref.dataset_id:
        raise CleaningError(
            "plan.before_dataset does not match the supplied dataset: "
            f"{plan.before_dataset.dataset_id!r} != {dataset.ref.dataset_id!r}"
        )
    if (
        plan.output_dataset_id is not None
        and len(plan.decisions) > 1
        and persist != "final"
    ):
        raise CleaningError(
            "output_dataset_id is only supported for a single-decision plan or "
            "persist='final'"
        )
    decision_ids = [decision.decision_id for decision in plan.decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise CleaningError("cleaning plan decision_ids must be unique")

    for decision in plan.decisions:
        strategy = get_strategy(decision.issue_type, decision.strategy_id)
        validate_strategy_params(strategy, decision.params)
        if (
            not plan.dry_run
            and strategy.risk_level is RiskLevel.DANGEROUS
            and not confirm
        ):
            raise CleaningConfirmationError(
                f"strategy {strategy.strategy_id!r} requires confirm=true"
            )

    from . import CLEANERS

    started_at = perf_counter()
    current = dataset
    results = []
    warnings: list[str] = []
    status = ToolStatus.SUCCESS
    rows_shifted = False
    internal_dry_run = plan.dry_run or persist == "final"

    for index, decision in enumerate(plan.decisions):
        cleaner = CLEANERS.get(decision.issue_type)
        if cleaner is None:
            return _failed_plan(
                plan=plan,
                dataset=dataset,
                current=current,
                results=results,
                warnings=warnings,
                started_at=started_at,
                code="unsupported_issue_type",
                message=(
                    f"no cleaning tool is registered for {decision.issue_type.value!r}"
                ),
            )

        step_decision = decision
        if (
            persist == "per_step"
            and plan.output_dataset_id is not None
            and index == len(plan.decisions) - 1
        ):
            step_decision = decision.model_copy(
                update={
                    "metadata": {
                        **decision.metadata,
                        "output_dataset_id": plan.output_dataset_id,
                    }
                }
            )

        try:
            selection_table = None
            if decision.selection_artifact_id:
                selection_table = load_selection_table(
                    decision.selection_artifact_id,
                    artifact_dir=selection_artifact_dir,
                )
                selection_table, removed = _remap_selection(
                    selection_table,
                    current,
                    positions_may_have_shifted=rows_shifted,
                )
                if removed:
                    warnings.append(
                        f"step {index + 1} skipped {removed} selection records whose row_id "
                        "was removed by an earlier step"
                    )

            result = cleaner(
                current,
                step_decision,
                selection_table=selection_table,
                selection_artifact_dir=selection_artifact_dir,
                artifact_dir=artifact_dir,
                output_dir=output_dir,
                dry_run=internal_dry_run,
                confirm=confirm,
            )
        except CleaningError as error:
            return _failed_plan(
                plan=plan,
                dataset=dataset,
                current=current,
                results=results,
                warnings=warnings,
                started_at=started_at,
                code="cleaning_step_failed",
                message=f"step {index + 1}: {error}",
            )

        results.append(result)
        warnings.extend(f"step {index + 1}: {warning}" for warning in result.warnings)
        if result.status is ToolStatus.PARTIAL:
            status = ToolStatus.PARTIAL
        if result.change_set is not None and (
            result.change_set.rows_added or result.change_set.rows_removed
        ):
            rows_shifted = True
        current = _next_dataset(current, result)

    after = current.ref
    change_set = None
    if persist == "final" and not plan.dry_run:
        after, change_set = _write_plan_result(
            dataset=dataset,
            plan=plan,
            results=results,
            current=current,
            output_dir=output_dir,
        )

    return CleaningPlanResult(
        plan_id=plan.plan_id,
        status=status,
        before_dataset=dataset.ref,
        after_dataset=after,
        results=results,
        change_set=change_set,
        dry_run=plan.dry_run,
        warnings=warnings,
        duration_ms=(perf_counter() - started_at) * 1000,
        metadata={
            "decision_count": len(plan.decisions),
            "persist_mode": persist,
        },
    )


__all__ = ["execute_plan"]
