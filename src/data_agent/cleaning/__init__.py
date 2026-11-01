from __future__ import annotations

from typing import Any, Callable

from ..contracts import CleaningDecision, CleaningPlan, CleaningResult, IssueType
from ..io import LoadedDataset
from .common import (
    CellTarget,
    CleaningConfirmationError,
    CleaningError,
    CleaningRun,
    SelectionError,
)
from .fill_missing import FILL_MISSING_TOOL_SPEC, clean_missing
from .handle_outliers import HANDLE_OUTLIERS_TOOL_SPEC, clean_outliers
from .mitigate_imbalance import MITIGATE_IMBALANCE_TOOL_SPEC, mitigate_imbalance
from .normalize_types_formats import NORMALIZE_TYPES_FORMATS_TOOL_SPEC, normalize_types_formats
from .plan import execute_plan
from .remove_duplicates import REMOVE_DUPLICATES_TOOL_SPEC, clean_duplicates
from .repair_labels import REPAIR_LABELS_TOOL_SPEC, repair_labels
from .resolve_conflicts import RESOLVE_CONFLICTS_TOOL_SPEC, resolve_conflicts
from .strategies import (
    CATALOG,
    StrategyError,
    StrategyNotAvailable,
    StrategyParameterError,
    StrategySpec,
    get_strategy,
    list_strategies,
    validate_strategy_params,
)


CLEANERS: dict[IssueType, Callable[..., CleaningResult]] = {
    IssueType.MISSING: clean_missing,
    IssueType.DUPLICATE: clean_duplicates,
    IssueType.OUTLIER: clean_outliers,
    IssueType.FORMAT: normalize_types_formats,
    IssueType.CONSTRAINT: resolve_conflicts,
    IssueType.LABEL: repair_labels,
    IssueType.IMBALANCE: mitigate_imbalance,
}


def make_decision(
    *,
    issue_type: IssueType | str,
    strategy_id: str,
    finding_id: str,
    selection_artifact_id: str | None = None,
    target_columns: list[str] | None = None,
    params: dict[str, Any] | None = None,
    reason: str | None = None,
    expected_effect: str | None = None,
    decision_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CleaningDecision:
    issue = IssueType(issue_type)
    strategy = get_strategy(issue, strategy_id)
    return CleaningDecision(
        decision_id=decision_id or f"{issue.value}.{strategy.strategy_id}",
        issue_type=issue,
        strategy_id=strategy.strategy_id,
        finding_id=finding_id,
        selection_artifact_id=selection_artifact_id,
        target_columns=target_columns or [],
        params=params or {},
        reason=reason or strategy.description,
        expected_effect=expected_effect,
        risk_level=strategy.risk_level,
        requires_confirmation=strategy.requires_confirmation,
        metadata=metadata or {},
    )


def build_plan(
    dataset: LoadedDataset,
    decisions: list[CleaningDecision],
    *,
    plan_id: str = "cleaning_plan",
    dry_run: bool = True,
    output_dataset_id: str | None = None,
) -> CleaningPlan:
    return CleaningPlan(
        plan_id=plan_id,
        before_dataset=dataset.ref,
        decisions=decisions,
        dry_run=dry_run,
        output_dataset_id=output_dataset_id,
    )


def execute_cleaning(
    dataset: LoadedDataset,
    decision: CleaningDecision | dict[str, Any],
    **kwargs: Any,
) -> CleaningResult:
    if isinstance(decision, dict):
        decision = CleaningDecision.model_validate(decision)
    cleaner = CLEANERS.get(decision.issue_type)
    if cleaner is None:
        raise CleaningError(f"no cleaning tool is registered for {decision.issue_type.value!r}")
    return cleaner(dataset, decision, **kwargs)


__all__ = [
    "CATALOG",
    "CLEANERS",
    "CellTarget",
    "CleaningConfirmationError",
    "CleaningError",
    "CleaningRun",
    "FILL_MISSING_TOOL_SPEC",
    "HANDLE_OUTLIERS_TOOL_SPEC",
    "MITIGATE_IMBALANCE_TOOL_SPEC",
    "NORMALIZE_TYPES_FORMATS_TOOL_SPEC",
    "REMOVE_DUPLICATES_TOOL_SPEC",
    "REPAIR_LABELS_TOOL_SPEC",
    "RESOLVE_CONFLICTS_TOOL_SPEC",
    "SelectionError",
    "StrategyError",
    "StrategyNotAvailable",
    "StrategyParameterError",
    "StrategySpec",
    "build_plan",
    "clean_duplicates",
    "clean_missing",
    "clean_outliers",
    "execute_cleaning",
    "execute_plan",
    "get_strategy",
    "list_strategies",
    "make_decision",
    "mitigate_imbalance",
    "normalize_types_formats",
    "repair_labels",
    "resolve_conflicts",
    "validate_strategy_params",
]
