"""M4 orchestration: existing strategies and a fixed, common modeling cohort."""
from __future__ import annotations
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_agent.clustering import load_input
from data_agent.clustering.core import DEFAULT_FEATURES, select_features, scale_features, save_scaler, run_kmeans, evaluate_clusters
from data_agent.clustering.dataset_adapter import DatasetInput
from data_agent.io import load_dataset, sha256_file
from .clustering import compare_clustering, load_cluster_group
from .quality import compare_quality


class ExperimentProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    features: tuple[str, ...] = DEFAULT_FEATURES
    ks: tuple[int, ...] = tuple(range(2, 9))
    seeds: tuple[int, ...] = (42, 7, 21)
    n_init: int = Field(default=10, ge=1, strict=True)
    max_iter: int = Field(default=300, ge=1, strict=True)
    tol: float = Field(default=1e-4, gt=0)
    sample_size: int = Field(default=2000, ge=10, strict=True)
    sample_seed: int = Field(default=42, ge=0, le=2**32-1, strict=True)

    @model_validator(mode="after")
    def valid(self):
        if self.features != DEFAULT_FEATURES or self.ks != tuple(range(2, 9)):
            raise ValueError("M4 uses the fixed M1 features and k=2..8 protocol")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds) or any(not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError("provide at least two unique valid seeds")
        return self


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def identity(ref):
    if hasattr(ref, "model_dump"):
        ref = ref.model_dump()
    return tuple(ref.get(k) for k in ("dataset_id", "version", "sha256"))


def verify_fixed_strategy(root, before, after):
    plan_path = root / "config/plans/off_52k_v2_conservative.json"
    log_path = root / "logs/cleaning/off_52k_v2_conservative.plan_result.json"
    plan, log = read_json(plan_path), read_json(log_path)
    if identity(after.manifest.get("parent", {})) != identity(before.ref):
        raise ValueError("B parent does not match A original input")
    if identity(plan["before_dataset"]) != identity(before.ref) or identity(log["before_dataset"]) != identity(before.ref) or identity(log["after_dataset"]) != identity(after.ref):
        raise ValueError("B plan/log input or output identity mismatch")
    if log["status"] != "success" or log["dry_run"] or log["plan_id"] != plan["plan_id"] or plan["output_dataset_id"] != after.ref.dataset_id:
        raise ValueError("B final run is not successful/persisted or plan does not match")
    decisions = plan["decisions"]
    if len(log["results"]) != len(decisions) or any(r["decision"] != d or r["status"] not in ("success", "skipped") for r, d in zip(log["results"], decisions)):
        raise ValueError("B executed decisions do not match plan")
    change = root / after.manifest["change_file"]["path"]
    if sha256_file(change) != after.manifest["change_file"]["sha256"]:
        raise ValueError("B change table hash mismatch")
    return {"status": "verified", "plan_id": plan["plan_id"], "decision_count": len(decisions),
            "sources": [source(p) for p in (plan_path, log_path, change, after.manifest_path, after.profile_path)],
            "historical_cleaning_duration_ms": log.get("duration_ms"),
            "note": "Final persisted run is successful; per-step dry_run flags belong to final-persist mode. Historical plan identity is checked by exact recorded decisions."}


def import_agent_result(bundle_path, before, project_root):
    """Import a reviewed provider export plus linked decisions/executions, never synthesize C.

    Local hashes prove consistency, not provider authenticity; an external evidence
    review is mandatory and its assurance is retained in the result.
    """
    path = Path(bundle_path).resolve()
    from data_agent.cleaning import CLEANERS
    allowed_tools = {fn.__name__ for fn in CLEANERS.values()} | {"execute_plan", "execute_cleaning"}
    bundle = read_json(path)
    if identity(bundle["source_dataset"]) != identity(before.ref):
        raise ValueError("C source differs from the original input")
    def evidence(name):
        item = bundle[name]
        target = (path.parent / item["path"]).resolve()
        if not target.is_relative_to(path.parent) or sha256_file(target) != item["sha256"]:
            raise ValueError(f"C {name} path or hash mismatch")
        return target
    profile = evidence("output_profile")
    receipt_path, decision_path, tools_path, review_path = [evidence(k) for k in ("provider_export", "decisions", "tool_executions", "provenance_review")]
    receipts, decisions, executions, review = map(read_json, (receipt_path, decision_path, tools_path, review_path))
    if not review.get("reviewer") or not review.get("reviewed_at") or review.get("actual_provider_execution_verified") is not True or review.get("provider_export_sha256") != sha256_file(receipt_path):
        raise ValueError("C lacks a traceable external provenance review")
    if not receipts.get("run_id") or receipts.get("source_sha256") != before.ref.sha256 or not receipts.get("responses") or not decisions or not executions:
        raise ValueError("C evidence is incomplete")
    calls = {}
    for response in receipts["responses"]:
        if not response.get("id") or not response.get("model") or response.get("simulated") is True:
            raise ValueError("C provider response is incomplete or simulated")
        for call in response.get("tool_calls", []):
            key = (response["id"], call["id"])
            if key in calls:
                raise ValueError("duplicate C tool call")
            calls[key] = call
    for decision in decisions:
        call = calls.get((decision["response_id"], decision["tool_call_id"]))
        if call is None or (call["name"], call["arguments"]) != (decision["tool_name"], decision["arguments"]):
            raise ValueError("C decision does not originate from the provider trace")
        if decision["tool_name"] not in allowed_tools:
            raise ValueError("C references an unrecognized upstream tool")
    keys = [(d['response_id'], d['tool_call_id']) for d in decisions]
    if len(keys) != len(set(keys)):
        raise ValueError("C decisions repeat a tool call")
    if len(executions) != len(decisions):
        raise ValueError("C tool execution/decision counts differ")
    loaded = load_dataset(profile, project_root=project_root, verify_hash=True)
    previous_hash = before.ref.sha256
    for decision, execution in zip(decisions, executions):
        if any(execution.get(k) != decision.get(k) for k in ("response_id", "tool_call_id", "tool_name", "arguments")) or execution.get("status") != "success":
            raise ValueError("C tool executions do not match Agent decisions")
        if execution.get('input_sha256') != previous_hash or not execution.get('output_sha256'):
            raise ValueError("C tool data lineage is incomplete")
        previous_hash = execution['output_sha256']
    if executions[-1].get("output_sha256") != loaded.ref.sha256 or identity(bundle["output_dataset"]) != identity(loaded.ref):
        raise ValueError("C final tool output does not match its dataset")
    if identity(loaded.manifest.get("parent", {})) != identity(before.ref):
        raise ValueError("C dataset has no matching original parent")
    row_id = loaded.schema.row_id_column
    if row_id != before.row_id_column or loaded.frame[row_id].isna().any() or loaded.frame[row_id].duplicated().any() or not set(loaded.frame[row_id]).issubset(set(before.frame[row_id])):
        raise ValueError("C does not preserve unique source row IDs")
    data = DatasetInput(loaded.frame, loaded.ref, loaded.schema, loaded.manifest, profile, loaded.manifest_path, loaded.data_path, row_id, tuple(loaded.frame.columns))
    return data, {"status": "imported", "run_id": receipts["run_id"], "sources": [source(p) for p in (path, receipt_path, decision_path, tools_path, review_path, profile, loaded.data_path, loaded.manifest_path)],
                  "provider_responses": len(receipts["responses"]), "reported_usage": [r.get("usage") for r in receipts["responses"]],
                  "reported_cost": receipts.get("cost"), "review": review,
                  "assurance": "Evidence consistency verified locally; authenticity relies on the named external review."}


def _train(data, selected, scaler, output, protocol):
    output.mkdir(exist_ok=False)
    started = perf_counter()
    scaled = scale_features(selected, scaler=scaler)
    save_scaler(scaled, output / "scaler.json")
    pd.DataFrame({"row_id": selected.row_ids}).to_csv(output / "cohort.csv", index=False)
    rows = []
    for k in protocol.ks:
        tick = perf_counter()
        runs = [run_kmeans(scaled, k, random_state=seed, n_init=protocol.n_init, max_iter=protocol.max_iter, tol=protocol.tol) for seed in protocol.seeds]
        evaluation = evaluate_clusters(scaled, runs[0], stability_runs=runs[1:], sample_size=protocol.sample_size, sample_seed=protocol.sample_seed)
        pd.DataFrame({"row_id": scaled.row_ids, **{f"seed_{r.random_state}": r.labels for r in runs}}).to_csv(output / f"labels_k{k}.csv", index=False)
        r = runs[0]
        rows.append({"k": k, "status": r.status, "algorithm": "KMeans", "algorithm_variant": "lloyd", "random_state": r.random_state,
                     "n_init": r.n_init, "max_iter": r.max_iter, "tol": r.tol, "iterations": r.iterations,
                     "centers_standardized": r.centers.tolist(), "duration_seconds": perf_counter()-tick, **evaluation})
        print(f"{output.name} k={k} silhouette={evaluation['silhouette']:.6f}", flush=True)
    payload = {"dataset_id": data.ref.dataset_id, "dataset_version": data.ref.version, "data_sha256": data.ref.sha256,
               "features": list(selected.features), "input_rows": len(data.frame), "valid_rows": len(selected.row_ids),
               "matrix_sha256": scaled.matrix_sha256, "scaler_sha256": scaled.scaler_sha256, "results": rows,
               "profile_path": str(data.profile_path), "manifest_path": str(data.manifest_path), "cohort_file": "cohort.csv",
               "duration_seconds": perf_counter()-started, "artifact_sha256": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "metrics.json", payload)
    return payload


def compare_strategies(project_root, output_dir, *, protocol=None, m3_run, agent_bundle=None):
    started = perf_counter()
    protocol = protocol or ExperimentProtocol()
    root, output = Path(project_root).resolve(), Path(output_dir).resolve()
    candidates = sorted({str(p.resolve()) for folder in (root / 'logs', root / 'outputs', root / 'config') if folder.exists()
                         for p in folder.rglob('*.json') if any(word in p.name.lower() for word in ('agent', 'provider', 'trace'))})
    discovery = {"searched": [str(root / p) for p in ('logs', 'outputs', 'config')], "candidate_files": candidates,
                 "criterion": "Agent/provider/trace-named JSON; supplied bundle must pass full evidence validation"}
    bundles = [p for p in candidates if Path(p).name == 'agent_bundle.json']
    if agent_bundle is None and len(bundles) == 1:
        agent_bundle = bundles[0]
    before, after = load_input("1", project_root=root), load_input("2", project_root=root)
    b_evidence = verify_fixed_strategy(root, before, after)
    data = {"A": before, "B": after}
    c = {"status": "not_available", "reason": "No real Agent provider export, decisions, tool executions and external provenance review supplied."}
    if agent_bundle is not None:
        try:
            data["C"], c = import_agent_result(agent_bundle, before, root)
        except (ValueError, KeyError, OSError) as exc:
            c = {"status": "not_available", "reason": f"Agent evidence rejected: {exc}"}
    summaries = [root / "logs/checks/off_52k_v1_quality_summary.json", root / "logs/checks_v2/off_52k_v2_quality_summary.json"]
    quality = compare_quality(before, after, *(read_json(p) for p in summaries), change_path=root / after.manifest["change_file"]["path"])
    m3 = Path(m3_run).resolve()
    m3_manifest = read_json(m3 / "manifest.json")
    for name, digest in m3_manifest["artifacts"].items():
        if Path(name).name != name or sha256_file(m3 / name) != digest:
            raise ValueError("M3 artifact hash mismatch")
    if [identity(x) for x in m3_manifest["source_versions"]] != [identity(before.ref), identity(after.ref)]:
        raise ValueError("M3 source versions mismatch")
    truth = read_json(m3 / "ground_truth.json")
    injection = read_json(m3 / "injection_evaluation.json")
    if truth["injection_id"] != injection["injection_id"] or truth["config"]["source_sha256"] != after.ref.sha256:
        raise ValueError("M3 truth/evaluation provenance mismatch")
    eligible = {name: select_features(d, protocol.features) for name, d in data.items()}
    common = sorted(set.intersection(*(set(s.row_ids) for s in eligible.values())))
    if len(common) <= max(protocol.ks):
        raise ValueError("not enough common eligible samples for the fixed protocol")
    cohorts = {}
    for name, selected in eligible.items():
        positions = {row: i for i, row in enumerate(selected.row_ids)}
        cohorts[name] = replace(selected, frame=selected.frame.iloc[[positions[r] for r in common]].reset_index(drop=True), row_ids=tuple(common))
    reference = scale_features(cohorts["A"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol_input.json", protocol.model_dump(mode="json"))
    write_json(output / "protocol.json", {**protocol.model_dump(mode="json"), "fit_population": "common eligible row IDs, all groups", "scaler_fit": "A uncleaned common cohort only", "selection": "M1 complete finite numeric features; no imputation", "source": before.ref.model_dump(mode="json"), "common_rows": len(common)})
    group_results = {}
    for name, dataset in data.items():
        metrics = _train(dataset, cohorts[name], reference.scaler, output / name, protocol)
        group_results[name] = {"status": "executed", "dataset_id": dataset.ref.dataset_id, "version": dataset.ref.version,
                              "sha256": dataset.ref.sha256, "data_rows": len(dataset.frame), "eligible_rows": len(eligible[name].row_ids),
                              "eligible_fraction_of_original": len(eligible[name].row_ids)/len(before.frame),
                              "eligible_fraction_of_own_data": len(eligible[name].row_ids)/len(dataset.frame),
                              "excluded": eligible[name].exclusion_counts, "cohort_rows": len(common),
                              "cohort_fraction_of_original": len(common)/len(before.frame), "metrics": metrics,
                              "strategy": "minimum complete-case filtering" if name == "A" else "verified existing fixed-rule output" if name == "B" else "reviewed Agent tool output",
                              "metrics_file": source(output / name / "metrics.json")}
    group_results["B"]["evidence"] = b_evidence
    if "C" in group_results:
        group_results["C"]["evidence"] = c
        group_results["C"]["quality"] = {"status": "not_available", "reason": "separately version-bound seven-check quality summary required"}
    else:
        group_results["C"] = c
    comparisons = []
    for k in protocol.ks:
        groups = []
        for name, dataset in data.items():
            subset = dataset.frame.set_index(dataset.row_id_column, drop=False).loc[common].reset_index(drop=True)
            group = load_cluster_group(replace(dataset, frame=subset), output / name, k)
            group["input_rows"] = len(dataset.frame)
            groups.append(group)
        comparisons.append({"k": k, **compare_clustering(groups, max_evaluation_rows=protocol.sample_size, sample_seed=protocol.sample_seed)})
    payload = {"run_id": output.name, "status": "completed_available_groups", "protocol": read_json(output / "protocol.json"),
               "groups": group_results, "quality": quality, "injection": injection, "injection_config": truth["config"],
               "clustering_comparisons": comparisons, "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
               "agent_discovery": discovery,
               "elapsed_seconds": perf_counter()-started, "agent_cost": c.get("reported_cost"),
               "limitations": ["Common-cohort metrics exclude records unavailable to A; coverage is reported separately.", "Controlled injection metrics are not population-wide accuracy.",
                               f"Repair evaluation status: {injection['repair']['status']}", "Local evidence hashes cannot authenticate an external provider."],
               "sources": [source(p) for p in [*summaries, before.data_path, after.data_path, before.profile_path, before.manifest_path, m3 / "manifest.json", *[m3 / name for name in m3_manifest['artifacts']]] ]}
    write_json(output / "strategies.json", payload)
    csv_rows = []
    for comparison in comparisons:
        for i, name in enumerate(data):
            metrics = (comparison.get("common_subset_metrics") or [None]*len(data))[i]
            csv_rows.append({"group": name, "k": comparison["k"], "status": comparison["status"], "common_rows": comparison["common_rows"],
                             "evaluated_rows": comparison.get("evaluation_sample", {}).get("count"),
                             **{key: metrics.get(key) if metrics else None for key in ("silhouette", "calinski_harabasz", "davies_bouldin")},
                             "mean_ari": metrics["stability"]["mean_ari"] if metrics else None})
    pd.DataFrame(csv_rows).to_csv(output / "comparison.csv", index=False)
    return payload
