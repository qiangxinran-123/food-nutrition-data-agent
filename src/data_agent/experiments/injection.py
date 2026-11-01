"""Controlled, isolated corruption with an explicit labeled evaluation universe."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ERROR_TYPES = ("missing", "duplicate", "range", "format", "conflict")


@dataclass
class InjectionResult:
    frame: pd.DataFrame
    truth: list[dict]
    universe: list[dict]
    injection_id: str
    config: dict


def inject_errors(data, *, counts: dict[str, int], seed: int = 42, sample_row_ids=None) -> InjectionResult:
    if set(counts) - set(ERROR_TYPES) or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
        raise ValueError("invalid injection counts or type")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    frame = data.frame.copy(deep=True)
    row_id = data.row_id_column
    if frame[row_id].isna().any() or frame[row_id].duplicated().any():
        raise ValueError("source row IDs must be unique and non-null")
    if sample_row_ids is not None:
        ids = list(sample_row_ids)
        if len(ids) != len(set(ids)) or not set(ids).issubset(set(frame[row_id])):
            raise ValueError("sample IDs must be unique source row IDs")
        frame = frame.set_index(row_id, drop=False).loc[ids].reset_index(drop=True)
    required = {row_id, "code", "fat_100g", "saturated-fat_100g", "proteins_100g", "sugars_100g"}
    if not required.issubset(frame.columns):
        raise ValueError("source lacks required injection fields")
    config = {"counts": {k: counts.get(k, 0) for k in ERROR_TYPES}, "seed": seed, "source_dataset_id": data.ref.dataset_id,
              "source_version": data.ref.version, "source_sha256": data.ref.sha256, "source_sample_rows": len(frame),
              "row_id_column": row_id,
              "source_row_ids_sha256": hashlib.sha256(json.dumps(frame[row_id].tolist(), ensure_ascii=False).encode()).hexdigest()}
    injection_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    rng = np.random.default_rng(seed)
    free = np.ones(len(frame), dtype=bool)
    numeric = {c: pd.to_numeric(frame[c], errors="coerce") for c in ("fat_100g", "saturated-fat_100g", "proteins_100g", "sugars_100g")}
    valid = {c: numeric[c].notna().to_numpy() & np.isfinite(numeric[c].to_numpy()) for c in numeric}
    fat, sat, protein, sugar = (numeric[c].to_numpy() for c in ("fat_100g", "saturated-fat_100g", "proteins_100g", "sugars_100g"))
    # Eligibility defines verified controls for each narrowly stated corruption rule.
    eligible = {
        "missing": valid["fat_100g"],
        "duplicate": frame.code.notna().to_numpy() & ~frame.code.duplicated(keep=False).to_numpy(),
        "range": valid["sugars_100g"] & (sugar >= 0) & (sugar <= 100),
        "format": valid["proteins_100g"] & (protein >= 0) & (protein <= 100),
        "conflict": valid["fat_100g"] & valid["saturated-fat_100g"] & (fat >= 0) & (fat <= 90) & (sat >= 0) & (sat <= fat),
    }
    fields = {"missing": "fat_100g", "duplicate": "__row__", "range": "sugars_100g", "format": "proteins_100g", "conflict": "saturated-fat_100g"}
    rules = {"missing": "hide a finite value", "duplicate": "append copy with new row ID", "range": "sugars=150 g/100g, above 100", "format": "append unit text ' g/100g' to a numeric value", "conflict": "saturated fat = total fat + 5 g/100g"}
    frame["proteins_100g"] = frame["proteins_100g"].astype(object)
    truth, universe, duplicates = [], [], []
    for kind in ERROR_TYPES:
        count = counts.get(kind, 0)
        if not count:
            continue
        candidates = np.flatnonzero(free & eligible[kind])
        if len(candidates) < 2 * count:
            raise ValueError(f"insufficient eligible disjoint records for {kind}: need {2*count}, found {len(candidates)}")
        picks = rng.choice(candidates, size=2*count, replace=False)
        free[picks] = False
        for pos in picks[count:]:
            universe.append({"error_type": kind, "row_id": str(frame.iloc[pos][row_id]), "column": fields[kind], "is_error": False})
        for pos in picks[:count]:
            original_id = str(frame.iloc[pos][row_id])
            field = fields[kind]
            old = None if kind == "duplicate" else float(numeric[field].iloc[pos])
            injected_id = original_id
            if kind == "duplicate":
                injected_id = f"injected_{injection_id}_{len(duplicates):06d}"
                if injected_id in set(frame[row_id]):
                    raise ValueError("generated duplicate row ID collides with source")
                copy = frame.iloc[pos].copy()
                old = {str(key): (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value) for key, value in copy.items()}
                copy[row_id] = injected_id
                duplicates.append(copy)
                new, expected = "copied row", "drop_duplicate"
            else:
                new = {"missing": None, "range": 150.0, "format": f"{old:.17g} g/100g", "conflict": float(fat[pos]+5)}[kind]
                frame.at[frame.index[pos], field] = np.nan if new is None else new
                expected = old
            truth.append({"injection_id": injection_id, "source_dataset_id": data.ref.dataset_id, "source_version": data.ref.version,
                          "source_sha256": data.ref.sha256, "original_row_id": original_id, "row_id": injected_id,
                          "column": field, "error_type": kind, "original_value": old, "injected_value": new,
                          "rule": rules[kind], "seed": seed, "expected_detection": {"row_id": injected_id, "column": field},
                          "expected_repair": expected})
            universe.append({"error_type": kind, "row_id": injected_id, "column": field, "is_error": True})
    if duplicates:
        frame = pd.concat([frame, pd.DataFrame(duplicates)], ignore_index=True)
    if frame[row_id].duplicated().any():
        raise ValueError("injection produced duplicate row IDs")
    return InjectionResult(frame, truth, universe, injection_id, config)


def load_injection(ground_truth_path) -> InjectionResult:
    """Reload the separately saved CSV and truth, verifying their link and hash."""
    from data_agent.io import sha256_file
    path = Path(ground_truth_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = payload["data_file"]
    if Path(name).name != name or sha256_file(path.parent / name) != payload["data_sha256"]:
        raise ValueError("injected data filename or SHA256 is invalid")
    config = payload["config"]
    expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
    if payload["injection_id"] != expected or any(t["injection_id"] != expected for t in payload["truth"]):
        raise ValueError("injection ID does not match configuration/truth")
    frame = pd.read_csv(path.parent / name, dtype={"code": "string", config["row_id_column"]: "string", "proteins_100g": "string"})
    # Preserve invalid unit-bearing strings while restoring parseable numeric cells.
    numeric = pd.to_numeric(frame["proteins_100g"], errors="coerce")
    column = frame["proteins_100g"].astype(object)
    column.loc[numeric.notna()] = numeric[numeric.notna()].astype(float)
    frame["proteins_100g"] = column
    if frame[config["row_id_column"]].isna().any() or frame[config["row_id_column"]].duplicated().any():
        raise ValueError("saved data contains invalid row IDs")
    return InjectionResult(frame, payload["truth"], payload["universe"], expected, config)


def save_injection(injection: InjectionResult, output_dir):
    """Save the dataset and ground truth separately without overwriting files."""
    from data_agent.io import sha256_file
    directory = Path(output_dir).resolve()
    if not directory.is_dir():
        raise ValueError("output directory must already exist")
    data_path = directory / "injected_data.csv"
    truth_path = directory / "ground_truth.json"
    if data_path.exists() or truth_path.exists():
        raise FileExistsError("injection output already exists")
    with data_path.open("x", encoding="utf-8", newline="") as handle:
        injection.frame.to_csv(handle, index=False)
    payload = {"injection_id": injection.injection_id, "config": injection.config, "data_file": data_path.name,
               "data_sha256": sha256_file(data_path), "truth": injection.truth, "universe": injection.universe}
    with truth_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return data_path, truth_path


def evaluate_injected_errors(injection: InjectionResult, detections: dict, *, repairs=None, tolerance=1e-6):
    """Evaluate unique (type,row_id,column) keys against known positives/controls only."""
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("invalid tolerance")
    universe = {(x["error_type"], x["row_id"], x["column"]): x["is_error"] for x in injection.universe}
    if len(universe) != len(injection.universe):
        raise ValueError("duplicate evaluation keys")
    truth = {(x["error_type"], x["row_id"], x["column"]): x for x in injection.truth}
    if set(truth) != {k for k, v in universe.items() if v}:
        raise ValueError("ground truth does not match universe")
    def ratio(a, b):
        return {"value": a/b if b else None, "status": "available" if b else "unavailable", "reason": None if b else "zero denominator"}
    types = {}
    total_tp = total_fp = total_fn = 0
    complete = True
    for kind in ERROR_TYPES:
        keys = {k for k in universe if k[0] == kind}
        if not keys:
            continue
        detection = detections.get(kind, {})
        if detection.get("injection_id", injection.injection_id) != injection.injection_id:
            raise ValueError("detection result belongs to a different injection")
        if detection.get("status") != "success" or not isinstance(detection.get("positions"), list):
            types[kind] = {"status": "unavailable", "reason": "detector failed, not run, or complete positions unavailable"}
            complete = False
            continue
        found = {(kind, x["row_id"], x["column"]) for x in detection["positions"]}
        positives = {k for k in keys if universe[k]}
        tp, fp, fn = len(found & positives), len(found & (keys-positives)), len(positives-found)
        total_tp += tp; total_fp += fp; total_fn += fn
        types[kind] = {"status": "available", "tp": tp, "fp": fp, "fn": fn, "known_controls": len(keys-positives),
                       "outside_universe_ignored": len(found - keys), "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn), "f1": ratio(2*tp, 2*tp+fp+fn)}
    micro = {"status": "available", "tp": total_tp, "fp": total_fp, "fn": total_fn,
             "precision": ratio(total_tp,total_tp+total_fp), "recall": ratio(total_tp,total_tp+total_fn), "f1": ratio(2*total_tp,2*total_tp+total_fp+total_fn)} if complete else {"status": "unavailable", "reason": "one or more detectors unavailable"}
    repair = {"status": "unavailable", "reason": "no repair changes supplied"}
    if repairs is not None:
        changes = {(r["error_type"], r["row_id"], r["column"]): r["new_value"] for r in repairs}
        if len(changes) != len(repairs):
            raise ValueError("duplicate repair change keys")
        correct = 0; errors = []; fixed = 0
        for key, target in truth.items():
            if key not in changes:
                continue
            fixed += 1
            new, expected = changes[key], target["expected_repair"]
            if expected == "drop_duplicate":
                correct += int(new == "drop_duplicate")
            elif isinstance(new, (int, float)) and not isinstance(new, bool) and np.isfinite(new):
                err = float(new) - float(expected)
                errors.append(err)
                correct += int(abs(err) <= tolerance)
        negatives = {k for k,v in universe.items() if not v}
        repair = {"status": "available", "tolerance_absolute": tolerance, "eligible_errors": len(truth), "changed_errors": fixed,
                  "correct_repairs": correct, "repair_accuracy": ratio(correct, len(truth)), "untreated_rate": ratio(len(truth)-fixed, len(truth)),
                  "false_modification_rate": ratio(len(set(changes) & negatives), len(negatives)), "numeric_changes_evaluated": len(errors),
                  "mae": float(np.mean(np.abs(errors))) if errors else None, "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                  "numeric_error_reason": None if errors else "no numeric repaired value supplied",
                  "outside_universe_changes": len(set(changes)-set(universe))}
    return {"injection_id": injection.injection_id, "evaluation_unit": "unique (error_type,row_id,column); duplicates use __row__", "scope": "known injected errors and rule-eligible negative controls only; other detections have unknown truth", "by_type": types, "micro": micro, "repair": repair}
