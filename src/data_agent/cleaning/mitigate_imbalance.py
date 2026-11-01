from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..checks.labels import analyse_label_column
from ..contracts import ArtifactKind, ChangeType, CleaningDecision, CleaningResult, ToolCategory, ToolSpec
from ..io import LoadedDataset
from .common import CleaningError, CleaningRun


MITIGATE_IMBALANCE_TOOL_SPEC = ToolSpec(
    name="mitigate_imbalance",
    version="0.1.0",
    category=ToolCategory.PREPROCESSING,
    description=(
        "Execute a registered class-imbalance strategy. Reporting and weighting do "
        "not alter rows; resampling is explicit and high risk."
    ),
    mutates_data=True,
    risk_level="caution",
    input_schema=CleaningDecision.model_json_schema(),
    output_schema=CleaningResult.model_json_schema(),
)


def _label_spec(run: CleaningRun, column: str):
    for spec in run.schema.columns:
        if spec.name == column:
            return spec
    raise CleaningError(f"label column is absent from schema: {column!r}")


def _label_analysis(run: CleaningRun, column: str):
    policy = run.dataset.config.labels.columns.get(column)
    if policy is None:
        raise CleaningError(f"label column {column!r} has no label policy")
    spec = _label_spec(run, column)
    return analyse_label_column(run.frame, spec, policy)


def _row_ids(run: CleaningRun, positions: list[int]) -> list[str]:
    from .common import row_id_for_position

    return [row_id_for_position(run.frame, run.schema, position) for position in positions]


def _distribution(run: CleaningRun, column: str) -> pd.DataFrame:
    analysis = _label_analysis(run, column)
    counts = analysis.normalised[analysis.valid_mask].value_counts()
    total = int(counts.sum())
    return pd.DataFrame(
        [
            {
                "column": column,
                "class_label": str(label),
                "count": int(count),
                "share": round(int(count) / total, 8) if total else 0.0,
            }
            for label, count in counts.items()
        ]
    )


def _target_column(run: CleaningRun, params: dict[str, Any]) -> str:
    if params.get("target_column"):
        return str(params["target_column"])
    if run.decision.target_columns:
        return str(run.decision.target_columns[0])
    raise CleaningError("target_column is required for imbalance strategies")


def mitigate_imbalance(
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

    if strategy in {"no_action", "report_only"}:
        if strategy == "report_only":
            column = _target_column(run, params)
            run.attach_table(
                _distribution(run, column),
                artifact_id=f"{decision.decision_id}.imbalance_report",
                kind=ArtifactKind.TABLE,
                description="Class distribution for the imbalance report.",
            )
        return run.finish(tool_name="mitigate_imbalance", tool_version="0.1.0")

    column = _target_column(run, params)
    analysis = _label_analysis(run, column)
    counts = analysis.normalised[analysis.valid_mask].value_counts()
    if counts.empty:
        raise CleaningError(f"label column {column!r} has no usable labels")

    if strategy == "class_weight":
        total = int(counts.sum())
        class_count = len(counts)
        table = pd.DataFrame(
            [
                {
                    "column": column,
                    "class_label": str(label),
                    "count": int(count),
                    "weight": round(total / (class_count * int(count)), 8),
                }
                for label, count in counts.items()
            ]
        )
        run.attach_table(
            table,
            artifact_id=f"{decision.decision_id}.class_weights",
            description="Inverse-frequency class weights; no rows were changed.",
            metadata={"target_column": column},
        )
        return run.finish(tool_name="mitigate_imbalance", tool_version="0.1.0")

    if strategy == "collect_more_samples":
        target_share = float(params.get("target_share", 1.0 / len(counts)))
        total = int(counts.sum())
        target_count = int(np.ceil(total * target_share / max(1.0 - target_share, 1e-12)))
        table = pd.DataFrame(
            [
                {
                    "column": column,
                    "class_label": str(label),
                    "count": int(count),
                    "required_count": max(int(count), target_count),
                    "additional_needed": max(0, target_count - int(count)),
                }
                for label, count in counts.items()
            ]
        )
        run.attach_table(
            table,
            artifact_id=f"{decision.decision_id}.sample_gap",
            description="Suggested additional samples; no synthetic labels were created.",
            metadata={"target_column": column},
        )
        return run.finish(tool_name="mitigate_imbalance", tool_version="0.1.0")

    if strategy == "stratified_split":
        test_size = float(params.get("test_size", 0.2))
        if not 0 < test_size < 1:
            raise CleaningError("test_size must be strictly between 0 and 1")
        seed = int(params.get("random_state", 42))
        rng = np.random.default_rng(seed)
        split_rows: list[dict[str, Any]] = []
        for label, count in counts.items():
            positions = np.flatnonzero(
                analysis.valid_mask.to_numpy()
                & (analysis.normalised.astype("string").to_numpy() == str(label))
            ).tolist()
            rng.shuffle(positions)
            n_test = max(1, int(round(len(positions) * test_size))) if len(positions) > 1 else 0
            n_test = min(n_test, len(positions) - 1)
            for offset, position in enumerate(positions):
                split_rows.append(
                    {
                        "row_index": int(position),
                        "row_id": _row_ids(run, [int(position)])[0],
                        "class_label": str(label),
                        "split": "test" if offset < n_test else "train",
                    }
                )
        run.attach_table(
            pd.DataFrame.from_records(split_rows).sort_values("row_index"),
            artifact_id=f"{decision.decision_id}.stratified_split",
            description="Stratified train/test assignment; no rows were changed.",
            metadata={"target_column": column, "test_size": test_size, "seed": seed},
        )
        return run.finish(tool_name="mitigate_imbalance", tool_version="0.1.0")

    if strategy == "undersample_majority":
        target_count = int(counts.min())
        max_rows = params.get("max_rows")
        if max_rows is not None and int(counts.max()) - target_count > int(max_rows):
            raise CleaningError("undersampling exceeds max_rows")
        seed = int(params.get("random_state", 42))
        rng = np.random.default_rng(seed)
        drop_positions: list[int] = []
        for label, count in counts.items():
            if int(count) <= target_count:
                continue
            positions = np.flatnonzero(
                analysis.valid_mask.to_numpy()
                & (analysis.normalised.astype("string").to_numpy() == str(label))
            ).tolist()
            rng.shuffle(positions)
            drop_positions.extend(positions[target_count:])
        run.attach_table(
            run.before.iloc[drop_positions].assign(
                __row_id_for_drop=_row_ids(run, drop_positions),
                __reason="imbalance policy: undersample majority",
            ),
            artifact_id=f"{decision.decision_id}.dropped_majority",
            description="Rows removed by majority undersampling.",
            metadata={"target_column": column},
        )
        run.drop_positions(drop_positions, reason="imbalance policy: undersample majority")
        return run.finish(tool_name="mitigate_imbalance", tool_version="0.1.0")

    if strategy == "oversample_minority":
        target_share = params.get("target_share")
        target_count = int(counts.max()) if target_share is None else int(
            np.ceil(int(counts.sum()) * float(target_share))
        )
        max_copies = int(params.get("max_copies", 100000))
        seed = int(params.get("random_state", 42))
        rng = np.random.default_rng(seed)
        copies: list[pd.DataFrame] = []
        for label, count in counts.items():
            needed = max(0, target_count - int(count))
            if needed == 0:
                continue
            positions = np.flatnonzero(
                analysis.valid_mask.to_numpy()
                & (analysis.normalised.astype("string").to_numpy() == str(label))
            ).tolist()
            chosen = rng.choice(positions, size=needed, replace=True).tolist()
            copies.append(run.frame.iloc[chosen].copy())
        if not copies:
            run.warnings.append("no minority class needed oversampling")
            return run.finish(tool_name="mitigate_imbalance", tool_version="0.1.0")
        additions = pd.concat(copies, ignore_index=True)
        if len(additions) > max_copies:
            raise CleaningError(f"oversampling would add {len(additions)} rows, above max_copies={max_copies}")
        run.add_rows(additions, reason="imbalance policy: oversample minority")
        return run.finish(tool_name="mitigate_imbalance", tool_version="0.1.0")

    raise CleaningError(f"strategy {strategy!r} is not handled by mitigate_imbalance")


__all__ = ["MITIGATE_IMBALANCE_TOOL_SPEC", "mitigate_imbalance"]
