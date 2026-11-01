"""Adapt existing complete selection artifacts to M3 evaluation keys."""
from pathlib import Path
import pandas as pd

from data_agent.config import load_dataset_config
from data_agent.io import sha256_file
from data_agent.checks.missing import detect_missing, MissingCheckOptions
from data_agent.checks.duplicates import detect_duplicates, DuplicateCheckOptions
from data_agent.checks.outliers import detect_outliers, OutlierCheckOptions
from data_agent.checks.types_formats import detect_type_format_issues, TypeFormatCheckOptions
from data_agent.checks.conflicts import detect_conflicts, ConflictCheckOptions


def detect_injected_errors(injection, data, artifact_dir):
    """Run upstream checks; each detector produces complete, hash-checked selections."""
    if (injection.config["source_dataset_id"], injection.config["source_version"], injection.config["source_sha256"]) != (data.ref.dataset_id, data.ref.version, data.ref.sha256):
        raise ValueError("injection source does not match detector dataset")
    directory = Path(artifact_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = load_dataset_config(data.profile_path)
    opts = {"artifact_dir": str(directory)}
    tasks = {
        "missing": lambda: detect_missing(injection.frame, data.schema, MissingCheckOptions.from_schema(data.schema).model_copy(update=opts)),
        "duplicate": lambda: detect_duplicates(injection.frame, data.schema, config.duplicates, DuplicateCheckOptions(**opts)),
        "range": lambda: detect_outliers(injection.frame, data.schema, config.outliers, OutlierCheckOptions(**opts)),
        "format": lambda: detect_type_format_issues(injection.frame, data.schema, config.types_formats, TypeFormatCheckOptions(**opts)),
        "conflict": lambda: detect_conflicts(injection.frame, data.schema, config.conflicts, ConflictCheckOptions(**opts)),
    }
    output = {}
    for kind, task in tasks.items():
        if not injection.config["counts"].get(kind):
            continue
        try:
            result = task()
            entry = {"status": result.status.value, "error": result.error_message, "positions": [], "warnings": result.warnings, "injection_id": injection.injection_id}
            if entry["status"] != "success":
                output[kind] = entry
                continue
            summary = result.summary
            evaluated = {
                "missing": lambda: "fat_100g" in injection.frame.columns,
                "duplicate": lambda: any(x.get("rule_id") == "primary_key" and x.get("columns_compared") == ["code"] for x in summary.get("rules", [])),
                "range": lambda: any(x.get("column") == "sugars_100g" and x.get("minimum") == 0 and x.get("maximum") == 100 for x in summary.get("range_results", [])),
                "format": lambda: any(x.get("column") == "proteins_100g" and x.get("check") == "numeric_parse" for x in summary.get("check_results", [])),
                "conflict": lambda: any(x.get("constraint_id") == "saturated_fat_not_above_total_fat" and x.get("status") == "evaluated" for x in summary.get("constraints", [])),
            }
            if not evaluated[kind]():
                entry["status"] = "unavailable"
                entry["error"] = "target rule was not evaluated with the required definition"
                output[kind] = entry
                continue
            for finding in result.findings:
                fid = finding.finding_id
                eligible = ((kind == "missing" and fid == "missing.fat_100g") or
                            (kind == "duplicate" and fid == "duplicate.primary_key") or
                            (kind == "range" and fid == "outlier.range.sugars_100g") or
                            (kind == "format" and fid.startswith("format.") and "proteins_100g" in fid) or
                            (kind == "conflict" and fid == "conflict.saturated_fat_not_above_total_fat"))
                if not eligible:
                    continue
                artifact = next((a for a in result.artifacts if a.artifact_id == finding.selection_artifact_id), None)
                if artifact is None or not artifact.sha256 or sha256_file(artifact.uri) != artifact.sha256:
                    raise ValueError(f"complete selection missing or invalid for {fid}")
                table = pd.read_parquet(artifact.uri, engine="fastparquet")
                if kind == "duplicate":
                    table = table[table["is_redundant"] == True]  # noqa: E712
                field = {"missing": "fat_100g", "duplicate": "__row__", "range": "sugars_100g", "format": "proteins_100g", "conflict": "saturated-fat_100g"}[kind]
                entry["positions"].extend({"row_id": str(row_id), "column": field} for row_id in table["row_id"])
            # A conflict rule may be skipped even though the tool reports success.
            if kind == "conflict" and any(x.get("constraint_id") == "saturated_fat_not_above_total_fat" for x in result.summary.get("not_evaluated_constraints", [])):
                entry["status"] = "unavailable"
                entry["error"] = "target constraint was not evaluated"
            output[kind] = entry
        except Exception as exc:
            output[kind] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "positions": [], "injection_id": injection.injection_id}
    return output
