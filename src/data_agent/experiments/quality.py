"""Compare existing checks, preserving their distinct counting units."""
from pathlib import Path
import json

import pandas as pd

from data_agent.config import load_dataset_config
from data_agent.io import sha256_file


def _comparison(before, after, *, same_rules=True, reason=None):
    if before is None or after is None or not same_rules:
        return {"status": "unavailable", "before": before, "after": after, "reason": reason or "missing value or incompatible rules", "absolute_change": None, "relative_change": None}
    delta = after - before
    return {"status": "available", "before": before, "after": after, "absolute_change": delta,
            "relative_change": delta / before if before else None,
            "relative_change_reason": "baseline is zero" if before == 0 else None,
            "direction": "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"}


def compare_quality(before, after, before_summary, after_summary, *, change_path=None):
    """Use verified dataset inputs and version-bound upstream summary dicts."""
    datasets = (before, after)
    summaries = (before_summary, after_summary)
    configs = [load_dataset_config(d.profile_path) for d in datasets]
    for data, summary in zip(datasets, summaries):
        if (summary.get("dataset_id"), summary.get("dataset_sha256")) != (data.ref.dataset_id, data.ref.sha256):
            raise ValueError("quality summary dataset identity or SHA256 mismatch")
    def check(i, name):
        s = summaries[i]
        item = s.get("checks", {}).get(name, {})
        if item.get("status") != "success" or s.get("overall_status") not in {"success", "partial", "failed"} or name in s.get("failed_checks", []):
            return None
        details = item.get("summary")
        if not isinstance(details, dict) or details.get("row_count") not in (None, len(datasets[i].frame)):
            return None
        return details
    def value(i, name, key):
        obj = check(i, name)
        return obj.get(key) if obj is not None else None
    def compatible(policy):
        return getattr(configs[0], policy) == getattr(configs[1], policy) and configs[0].columns == configs[1].columns
    columns_equal = tuple(before.frame.columns) == tuple(after.frame.columns)
    physical = [int(d.frame.isna().sum().sum()) for d in datasets]
    missing = [value(i, "missing", "missing_cells") for i in range(2)]
    missing_ok = compatible("missing") and columns_equal
    duplicates_ok = compatible("duplicates")
    outliers_ok = compatible("outliers")
    conflicts_ok = compatible("conflicts") and all((check(i, "conflicts") or {}).get("constraints_not_evaluated") == 0 for i in range(2))
    def compare_check(name, field, compatible_rules):
        return _comparison(value(0, name, field), value(1, name, field), same_rules=compatible_rules,
                           reason="check failed, incomplete, absent, or rules differ")
    completeness = {
        "physical_null_cells": _comparison(*physical, same_rules=columns_equal, reason="column sets/order differ"),
        "physical_null_rate": _comparison(*[p / d.frame.size if d.frame.size else None for p, d in zip(physical, datasets)], same_rules=columns_equal),
        "semantic_missing_including_physical_cells": _comparison(*missing, same_rules=missing_ok, reason="missing check failed/absent or policy differs"),
        "semantic_missing_including_physical_rate": compare_check("missing", "missing_cell_rate", missing_ok),
        "semantic_only_nonphysical_cells": _comparison(*[m - p if m is not None and m >= p else None for m, p in zip(missing, physical)], same_rules=missing_ok),
        "finding_count": _comparison(*[summaries[i].get("checks", {}).get("missing", {}).get("finding_count") if check(i, "missing") is not None else None for i in range(2)], same_rules=missing_ok),
        "affected_rows": compare_check("missing", "rows_with_missing", missing_ok),
    }
    duplicate_rows = [value(i, "duplicates", "rows_flagged_union") for i in range(2)]
    modifications = {"status": "unavailable", "reason": "verified change table not supplied"}
    if change_path is not None:
        path = Path(change_path)
        declared = after.manifest.get("change_file", {})
        if not declared.get("sha256") or sha256_file(path) != declared["sha256"]:
            raise ValueError("change table SHA256 mismatch")
        parent = after.manifest.get("parent", {})
        if (parent.get("dataset_id"), parent.get("version"), parent.get("sha256")) != (before.ref.dataset_id, before.ref.version, before.ref.sha256):
            raise ValueError("change table does not correspond to the before dataset")
        table = pd.read_parquet(path, engine="fastparquet")
        required = {"row_id", "column", "change_type", "is_imputed"}
        if not required.issubset(table.columns):
            raise ValueError("change table lacks required fields")
        imputed = table.loc[table["is_imputed"] == True]  # noqa: E712
        drops = table.loc[table["change_type"] == "drop"]
        modifications = {"status": "available", "change_file": str(path.resolve()), "change_sha256": declared["sha256"],
                         "imputation_events": len(imputed), "unique_imputed_cells": int(imputed[["row_id", "column"]].drop_duplicates().shape[0]),
                         "imputed_cell_fraction_of_before_cells": float(imputed[["row_id", "column"]].drop_duplicates().shape[0] / before.frame.size) if before.frame.size else None,
                         "deleted_row_ids": int(drops.row_id.nunique()), "deleted_fraction_of_before_rows": float(drops.row_id.nunique() / len(before.frame)) if len(before.frame) else None,
                         "note": "Change-log events, not proof that imputation was correct."}
    return {"datasets": [{"dataset_id": d.ref.dataset_id, "version": d.ref.version, "sha256": d.ref.sha256, "rows": len(d.frame), "columns": len(d.frame.columns)} for d in datasets],
            "check_statuses": [{name: item.get("status", "missing") for name, item in s.get("checks", {}).items()} for s in summaries],
            "completeness": completeness,
            "uniqueness": {"redundant_rows": _comparison(*duplicate_rows, same_rules=duplicates_ok),
                           "redundant_row_rate": _comparison(*[v/len(d.frame) if v is not None and len(d.frame) else None for v,d in zip(duplicate_rows, datasets)], same_rules=duplicates_ok),
                           "definition": "upstream rows_flagged_union: redundant rows, not all duplicate-group members"},
            "range_validity": {"range_flagged_cells": compare_check("outliers", "range_flagged_cells", outliers_ok),
                               "affected_rows": {"status": "unavailable", "reason": "upstream row union combines range and statistical flags"},
                               "finding_count": {"status": "unavailable", "reason": "finding count combines range and statistical rules"}},
            "field_consistency": {"affected_rows_union": compare_check("conflicts", "rows_flagged_union", conflicts_ok),
                                  "finding_count": _comparison(*[summaries[i].get("checks", {}).get("conflicts", {}).get("finding_count") if check(i,"conflicts") is not None else None for i in range(2)], same_rules=conflicts_ok),
                                  "evaluated_rows_per_rule": [[{k:r.get(k) for k in ("constraint_id", "evaluated_rows", "violated_rows")} for r in (check(i,"conflicts") or {}).get("constraints", [])] for i in range(2)],
                                  "note": "Union of affected rows, not number of cells or total rule violations; evaluated populations may differ."},
            "retention": {"before_rows": len(before.frame), "after_rows": len(after.frame), "row_count_ratio": len(after.frame)/len(before.frame) if len(before.frame) else None,
                          "common_row_ids": len(set(before.frame[before.row_id_column]) & set(after.frame[after.row_id_column])),
                          "retained_before_fraction": len(set(before.frame[before.row_id_column]) & set(after.frame[after.row_id_column])) / len(before.frame) if len(before.frame) else None},
            "modifications": modifications, "interpretation": "Changes in counts do not establish repair correctness."}
