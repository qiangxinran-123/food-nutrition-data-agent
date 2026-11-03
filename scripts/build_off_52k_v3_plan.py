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


V2_PROFILE = PROJECT_ROOT / "config" / "profiles" / "off_52k_v2.json"
OUTLIER_REPORT = PROJECT_ROOT / "logs" / "checks_v2" / "off_52k_v2_outliers.json"
CONFLICT_REPORT = PROJECT_ROOT / "logs" / "checks_v2" / "off_52k_v2_conflicts.json"
PLAN_PATH = PROJECT_ROOT / "config" / "plans" / "off_52k_v3_stratified_repair.json"


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
        metadata={"source_dataset": "off_52k_v2", "repair_policy": policy},
        created_at=created_at,
    )


def _fill_decision(
    *,
    column: str,
    selection_artifact_id: str,
    strategy_id: str,
    params: dict[str, Any],
    suffix: str,
    reason: str,
    created_at: datetime,
) -> CleaningDecision:
    return _decision(
        decision_id=f"off52k_v3.range_repair.{column}.{suffix}",
        issue_type=IssueType.MISSING,
        strategy_id=strategy_id,
        finding_id=f"range_repair.{column}.{suffix}",
        selection_artifact_id=selection_artifact_id,
        target_columns=[column],
        params=params,
        reason=reason,
        expected_effect=(
            "Replace an invalid hard-range cell with a derived value, then a "
            "column statistic when derivation is unavailable."
        ),
        policy="hard_range_null_then_impute",
        created_at=created_at,
    )


def main() -> None:
    now = datetime.now(timezone.utc)
    outliers = json.loads(OUTLIER_REPORT.read_text(encoding="utf-8"))
    conflicts = json.loads(CONFLICT_REPORT.read_text(encoding="utf-8"))
    range_findings = [
        finding
        for finding in outliers["findings"]
        if finding["rule_id"].startswith("outlier.range.")
    ]
    iqr_findings = [
        finding
        for finding in outliers["findings"]
        if finding["rule_id"].startswith("outlier.iqr.")
    ]
    range_by_column = {
        str(finding["columns"][0]): finding for finding in range_findings
    }
    decisions: list[CleaningDecision] = []

    for finding in range_findings:
        column = str(finding["columns"][0])
        decisions.append(
            _decision(
                decision_id=f"off52k_v3.range_repair.{column}.quarantine",
                issue_type=IssueType.OUTLIER,
                strategy_id="quarantine_cell",
                finding_id=str(finding["finding_id"]),
                selection_artifact_id=str(finding["selection_artifact_id"]),
                target_columns=[column],
                params={"artifact_name": "hard_range_invalid_values"},
                reason=(
                    "Treat a value outside the declared physical/business range as "
                    "invalid, preserve the old value in quarantine, and set the cell "
                    "to null before imputation."
                ),
                expected_effect=(
                    "Remove impossible hard-range values without inventing a repair "
                    "value in the same step."
                ),
                policy="hard_range_quarantine_then_missing",
                created_at=now,
            )
        )

    fill_steps: list[tuple[str, str, dict[str, Any], str, str]] = []
    if "energy-kcal_100g" in range_by_column:
        fill_steps.append(
            (
                "energy-kcal_100g",
                "derive_from_formula",
                {
                    "target_column": "energy-kcal_100g",
                    "expression": 'col("energy_100g") / 4.184',
                    "round_digits": 6,
                },
                "kcal_from_kj",
                "Fill an invalid kcal cell from the paired kJ field.",
            )
        )
    if "energy_100g" in range_by_column:
        fill_steps.append(
            (
                "energy_100g",
                "derive_from_formula",
                {
                    "target_column": "energy_100g",
                    "expression": 'col("energy-kcal_100g") * 4.184',
                    "round_digits": 6,
                },
                "kj_from_kcal",
                "Fill an invalid kJ cell from the paired kcal field.",
            )
        )
    if "energy-kcal_100g" in range_by_column:
        fill_steps.append(
            (
                "energy-kcal_100g",
                "impute_statistic",
                {"method": "median"},
                "median_fallback",
                "Median-fill invalid kcal cells when neither energy field is usable.",
            )
        )
    if "energy_100g" in range_by_column:
        fill_steps.append(
            (
                "energy_100g",
                "derive_from_formula",
                {
                    "target_column": "energy_100g",
                    "expression": 'col("energy-kcal_100g") * 4.184',
                    "round_digits": 6,
                },
                "kj_after_median",
                "Derive kJ again after the kcal fallback so the pair stays consistent.",
            )
        )

    if "sodium_100g" in range_by_column:
        fill_steps.append(
            (
                "sodium_100g",
                "derive_from_formula",
                {
                    "target_column": "sodium_100g",
                    "expression": 'col("salt_100g") / 2.5',
                    "round_digits": 6,
                },
                "sodium_from_salt",
                "Fill an invalid sodium cell from the paired salt field.",
            )
        )
    if "salt_100g" in range_by_column:
        fill_steps.append(
            (
                "salt_100g",
                "derive_from_formula",
                {
                    "target_column": "salt_100g",
                    "expression": 'col("sodium_100g") * 2.5',
                    "round_digits": 6,
                },
                "salt_from_sodium",
                "Fill an invalid salt cell from the paired sodium field.",
            )
        )
    if "sodium_100g" in range_by_column:
        fill_steps.append(
            (
                "sodium_100g",
                "impute_statistic",
                {"method": "median"},
                "median_fallback",
                "Median-fill invalid sodium cells when neither salt field is usable.",
            )
        )
    if "salt_100g" in range_by_column:
        fill_steps.append(
            (
                "salt_100g",
                "derive_from_formula",
                {
                    "target_column": "salt_100g",
                    "expression": 'col("sodium_100g") * 2.5',
                    "round_digits": 6,
                },
                "salt_after_median",
                "Derive salt again after the sodium fallback so the pair stays consistent.",
            )
        )

    median_columns = [
        "proteins_100g",
        "fat_100g",
        "saturated-fat_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "fiber_100g",
    ]
    for column in median_columns:
        if column in range_by_column:
            fill_steps.append(
                (
                    column,
                    "impute_statistic",
                    {"method": "median"},
                    "median_fallback",
                    "Median-fill an invalid hard-range cell when no reliable paired-field formula exists.",
                )
            )

    for column, strategy_id, params, suffix, reason in fill_steps:
        finding = range_by_column[column]
        decisions.append(
            _fill_decision(
                column=column,
                selection_artifact_id=str(finding["selection_artifact_id"]),
                strategy_id=strategy_id,
                params=params,
                suffix=suffix,
                reason=reason,
                created_at=now,
            )
        )

    for finding in iqr_findings:
        column = str(finding["columns"][0])
        evidence = finding["evidence"]
        lower = evidence.get("lower_bound")
        upper = evidence.get("upper_bound")
        if lower is None and upper is None:
            continue
        decisions.append(
            _decision(
                decision_id=f"off52k_v3.iqr_winsorize.{column}",
                issue_type=IssueType.OUTLIER,
                strategy_id="clip_value",
                finding_id=str(finding["finding_id"]),
                selection_artifact_id=str(finding["selection_artifact_id"]),
                target_columns=[column],
                params={"minimum": lower, "maximum": upper},
                reason=(
                    "Winsorize a statistical boxplot tail to the detected lower/upper "
                    "IQR boundary instead of deleting the row."
                ),
                expected_effect=(
                    "Keep the observation while capping an extreme tail to the "
                    "boxplot boundary."
                ),
                policy="iqr_winsorization",
                created_at=now,
            )
        )

    conflict_rules: dict[str, tuple[str, str, str]] = {
        "conflict.energy_kcal_kj_consistency": (
            "energy-kcal_100g",
            'col("energy_100g") / 4.184',
            "Treat kJ as the primary energy field and recompute kcal to enforce the unit relation.",
        ),
        "conflict.salt_sodium_consistency": (
            "salt_100g",
            'col("sodium_100g") * 2.5',
            "Treat sodium as the primary salt field and recompute salt to enforce the conversion.",
        ),
        "conflict.sugars_not_above_carbohydrates": (
            "sugars_100g",
            'min(col("sugars_100g"), col("carbohydrates_100g") * 1.05 + 0.5)',
            "Cap sugars at the maximum value allowed by total carbohydrates plus measurement tolerance.",
        ),
        "conflict.saturated_fat_not_above_total_fat": (
            "saturated-fat_100g",
            'min(col("saturated-fat_100g"), col("fat_100g") * 1.05 + 0.5)',
            "Cap saturated fat at the maximum value allowed by total fat plus measurement tolerance.",
        ),
    }
    for finding in conflicts["findings"]:
        finding_id = str(finding["finding_id"])
        rule = conflict_rules.get(finding_id)
        if rule is None:
            continue
        target_column, expression, reason = rule
        decisions.append(
            _decision(
                decision_id=f"off52k_v3.conflict_repair.{target_column}",
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
                expected_effect=(
                    "Make the conflicting pair satisfy its declared formula or "
                    "inequality without dropping the row."
                ),
                policy="conflict_recompute_or_cap",
                created_at=now,
            )
        )

    dataset = load_dataset(V2_PROFILE, project_root=PROJECT_ROOT, verify_hash=True)
    plan = CleaningPlan(
        plan_id="off_52k_v3_stratified_repair",
        before_dataset=dataset.ref,
        decisions=decisions,
        dry_run=True,
        output_dataset_id="off_52k_v3",
        description=(
            "Stratified repair: quarantine impossible hard-range values and impute "
            "them, winsorize IQR tails, recompute paired energy/salt fields, and cap "
            "bounded child fields instead of treating all anomalies the same way."
        ),
        metadata={
            "source_dataset": "off_52k_v2",
            "repair_scope": [
                "hard_range",
                "iqr_tails",
                "field_conflicts",
            ],
            "hard_range_policy": "quarantine then formula/median fill",
            "iqr_policy": "winsorize to detected lower/upper bound",
            "conflict_policy": "recompute paired fields or cap child field to relation",
        },
        created_at=now,
    )
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Plan: {PLAN_PATH}")
    print(f"Decisions: {len(plan.decisions)}")
    print(f"Hard-range findings: {len(range_findings)}")
    print(f"IQR findings: {len(iqr_findings)}")
    print(f"Conflict findings: {len(conflicts['findings'])}")


if __name__ == "__main__":
    main()
