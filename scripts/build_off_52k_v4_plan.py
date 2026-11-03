from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if (PROJECT_ROOT / ".deps").exists():
    sys.path.insert(0, str(PROJECT_ROOT / ".deps"))

from data_agent.cleaning.strategies import get_strategy
from data_agent.contracts import CleaningDecision, CleaningPlan, IssueType
from data_agent.io import load_dataset


V3_PROFILE = PROJECT_ROOT / "config" / "profiles" / "off_52k_v3.json"
CHECKS_DIR = PROJECT_ROOT / "logs" / "checks_v3"
OUTLIER_REPORT = CHECKS_DIR / "off_52k_v3_outliers.json"
CONFLICT_REPORT = CHECKS_DIR / "off_52k_v3_conflicts.json"
PLAN_PATH = PROJECT_ROOT / "config" / "plans" / "off_52k_v4_convergence.json"


def _decision(
    *,
    decision_id: str,
    issue_type: IssueType,
    strategy_id: str,
    finding_id: str,
    selection_artifact_id: str,
    target_columns: list[str],
    params: dict[str, Any],
    reason: str,
    expected_effect: str,
    policy: str,
    created_at: datetime,
) -> CleaningDecision:
    strategy = get_strategy(issue_type, strategy_id)
    return CleaningDecision(
        decision_id=decision_id,
        issue_type=issue_type,
        strategy_id=strategy_id,
        finding_id=finding_id,
        selection_artifact_id=selection_artifact_id,
        target_columns=target_columns,
        params=params,
        reason=reason,
        expected_effect=expected_effect,
        risk_level=strategy.risk_level,
        requires_confirmation=strategy.requires_confirmation,
        metadata={"source_dataset": "off_52k_v3", "repair_policy": policy},
        created_at=created_at,
    )


def main() -> None:
    now = datetime.now(timezone.utc)
    outliers = json.loads(OUTLIER_REPORT.read_text(encoding="utf-8"))
    conflicts = json.loads(CONFLICT_REPORT.read_text(encoding="utf-8"))
    iqr_by_column = {
        str(finding["columns"][0]): finding
        for finding in outliers["findings"]
        if finding["rule_id"].startswith("outlier.iqr.")
    }
    decisions: list[CleaningDecision] = []

    for column in ["sugars_100g", "saturated-fat_100g", "fat_100g"]:
        finding = iqr_by_column.get(column)
        if finding is None:
            continue
        evidence = finding["evidence"]
        decisions.append(
            _decision(
                decision_id=f"off52k_v4.iqr_winsorize.{column}",
                issue_type=IssueType.OUTLIER,
                strategy_id="clip_value",
                finding_id=str(finding["finding_id"]),
                selection_artifact_id=str(finding["selection_artifact_id"]),
                target_columns=[column],
                params={
                    "minimum": evidence.get("lower_bound"),
                    "maximum": evidence.get("upper_bound"),
                },
                reason=(
                    "Apply a second winsorization pass using the boxplot bounds "
                    "recomputed after the first repair."
                ),
                expected_effect=(
                    "Cap the remaining statistical tail to the newly computed "
                    "lower/upper boundary without deleting rows."
                ),
                policy="iqr_second_pass",
                created_at=now,
            )
        )

    salt_finding = iqr_by_column.get("salt_100g")
    if salt_finding is not None:
        salt_upper = float(salt_finding["evidence"]["upper_bound"])
        sodium_upper = salt_upper / 2.5
        salt_selection = str(salt_finding["selection_artifact_id"])
        decisions.extend(
            [
                _decision(
                    decision_id="off52k_v4.salt_pair.cap_sodium",
                    issue_type=IssueType.CONSTRAINT,
                    strategy_id="recompute_derived",
                    finding_id=str(salt_finding["finding_id"]),
                    selection_artifact_id=salt_selection,
                    target_columns=["sodium_100g"],
                    params={
                        "target_column": "sodium_100g",
                        "expression": f'min(col("sodium_100g"), {sodium_upper!r})',
                        "round_digits": 6,
                    },
                    reason=(
                        "Cap sodium to the value implied by the salt winsorization "
                        "bound so the pair can stay consistent."
                    ),
                    expected_effect="Remove the salt tail without breaking salt/sodium consistency.",
                    policy="paired_relation_winsorization",
                    created_at=now,
                ),
                _decision(
                    decision_id="off52k_v4.salt_pair.derive_salt",
                    issue_type=IssueType.CONSTRAINT,
                    strategy_id="recompute_derived",
                    finding_id=str(salt_finding["finding_id"]),
                    selection_artifact_id=salt_selection,
                    target_columns=["salt_100g"],
                    params={
                        "target_column": "salt_100g",
                        "expression": 'col("sodium_100g") * 2.5',
                        "round_digits": 6,
                    },
                    reason=(
                        "Recompute salt from the capped sodium value so both the "
                        "tail cap and the declared conversion hold."
                    ),
                    expected_effect="Set salt at or below the boxplot boundary while preserving the salt/sodium relation.",
                    policy="paired_relation_winsorization",
                    created_at=now,
                ),
            ]
        )

    conflict_actions = {
        "conflict.sugars_not_above_carbohydrates": (
            "sugars_100g",
            'min(col("sugars_100g"), col("carbohydrates_100g") * 1.05 + 0.5)',
            "Cap sugars after winsorization so it remains within the carbohydrate tolerance.",
        ),
        "conflict.saturated_fat_not_above_total_fat": (
            "saturated-fat_100g",
            'min(col("saturated-fat_100g"), col("fat_100g") * 1.05 + 0.5)',
            "Cap saturated fat after winsorization so it remains within the total-fat tolerance.",
        ),
    }
    for finding in conflicts["findings"]:
        finding_id = str(finding["finding_id"])
        action = conflict_actions.get(finding_id)
        if action is None:
            continue
        target_column, expression, reason = action
        decisions.append(
            _decision(
                decision_id=f"off52k_v4.conflict_repair.{target_column}",
                issue_type=IssueType.CONSTRAINT,
                strategy_id="recompute_derived",
                finding_id=finding_id,
                selection_artifact_id=str(finding["selection_artifact_id"]),
                target_columns=[target_column],
                params={
                    "target_column": target_column,
                    "expression": expression,
                    "round_digits": 6,
                },
                reason=reason,
                expected_effect="Remove the remaining bounded-field conflict without dropping rows.",
                policy="conflict_second_pass",
                created_at=now,
            )
        )

    dataset = load_dataset(V3_PROFILE, project_root=PROJECT_ROOT, verify_hash=True)
    plan = CleaningPlan(
        plan_id="off_52k_v4_convergence",
        before_dataset=dataset.ref,
        decisions=decisions,
        dry_run=True,
        output_dataset_id="off_52k_v4",
        description=(
            "Convergence repair after the stratified v3 pass: second IQR winsorization, "
            "relation-aware salt/sodium cap, and final child-field conflict caps."
        ),
        metadata={
            "source_dataset": "off_52k_v3",
            "repair_scope": ["iqr_second_pass", "paired_relation", "conflict_finalization"],
        },
        created_at=now,
    )
    PLAN_PATH.write_text(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Plan: {PLAN_PATH}")
    print(f"Decisions: {len(plan.decisions)}")


if __name__ == "__main__":
    main()
