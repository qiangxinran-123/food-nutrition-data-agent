"""Run-scoped working copy, artifact directories and the final dataset version."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data_agent.clustering.dataset_adapter import DatasetInput
from data_agent.contracts import CleaningResult, DatasetRef, ToolResult
from data_agent.cleaning.writer import (
    available_output_id,
    sha256_file,
    write_processed_profile,
)
from data_agent.io import LoadedDataset, load_dataset


PRIMARY_COUNTS = {
    "missing": ("missing_cells", "check_missing"),
    "duplicate": ("rows_flagged_union", "check_duplicates"),
    "outlier": ("rows_flagged_union", "check_outliers"),
    "format": ("rows_flagged_union", "check_types_formats"),
    "constraint": ("rows_flagged_union", "check_conflicts"),
    "label": ("rows_flagged_union", "check_labels"),
    "imbalance": ("imbalanced_columns", "check_imbalance"),
}


@dataclass
class CleaningStep:
    index: int
    tool_name: str
    issue_type: str
    strategy_id: str
    decision_id: str
    finding_id: str
    response_id: str
    tool_call_id: str
    arguments_raw: str
    input_sha256: str
    output_sha256: str
    scratch_path: Path
    rows_before: int
    rows_after: int
    cells_changed: int
    rows_removed: int
    duration_ms: float
    changes_uri: str | None
    warnings: list[str] = field(default_factory=list)

    def execution_record(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments_raw,
            "status": "success",
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "step_index": self.index,
        }

    def decision_record(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments_raw,
            "step_index": self.index,
            "issue_type": self.issue_type,
            "strategy_id": self.strategy_id,
            "decision_id": self.decision_id,
            "finding_id": self.finding_id,
        }


class AgentSession:
    """Keeps the original dataset immutable and all changes in a working copy."""

    def __init__(self, project_root: str | Path, run_dir: str | Path, dataset_version: str = "1"):
        from data_agent.clustering import load_input

        self.root = Path(project_root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.source = load_input(dataset_version, project_root=self.root)
        self.loaded = load_dataset(
            self.source.profile_path, project_root=self.root, verify_hash=True
        )
        self.frame: pd.DataFrame = self.source.frame.copy(deep=True)
        self.schema = self.source.schema
        self.original_ref: DatasetRef = self.source.ref
        self.data_sha256: str = self.source.ref.sha256 or sha256_file(self.source.data_path)
        self.source_row_ids = set(self.frame[self.source.row_id_column].astype("string"))

        self.checks: dict[str, ToolResult] = {}
        self.baseline_counts: dict[str, int] = {name: 0 for name in PRIMARY_COUNTS}
        self.steps: list[CleaningStep] = []
        self.validations: list[dict[str, Any]] = []
        self.cluster_selected = None
        self.cluster_scaled = None
        self.cluster_runs: dict[tuple[int, int], Any] = {}
        self.cluster_metrics: dict[int, dict[str, Any]] = {}
        self.cluster_analysis: dict[int, dict[str, Any]] = {}
        self.clustering_started = False
        self.final_report: dict[str, Any] | None = None
        self.final_dataset: dict[str, Any] | None = None

        for name in ("checks", "checks/selections", "cleaning_artifacts", "work", "plots", "provider", "provider/requests", "provider/responses"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    @property
    def selection_dir(self) -> Path:
        return self.run_dir / "checks" / "selections"

    @property
    def cleaning_artifact_dir(self) -> Path:
        return self.run_dir / "cleaning_artifacts"

    @property
    def work_dir(self) -> Path:
        return self.run_dir / "work"

    # ---------------------------------------------------------------- dataset
    def current_ref(self) -> DatasetRef:
        return self.original_ref.model_copy(
            update={
                "sha256": self.data_sha256,
                "row_count": int(len(self.frame)),
                "column_count": int(len(self.frame.columns)),
            }
        )

    def working_loaded(self) -> LoadedDataset:
        return LoadedDataset(
            ref=self.current_ref(),
            schema=self.schema,
            frame=self.frame,
            manifest=self.loaded.manifest,
            manifest_path=self.loaded.manifest_path,
            data_path=self.loaded.data_path,
            config=self.loaded.config,
            config_path=self.loaded.config_path,
        )

    def dataset_input(self) -> DatasetInput:
        ref = self.current_ref()
        return DatasetInput(
            self.frame.copy(deep=True),
            ref,
            self.schema,
            self.loaded.manifest,
            self.loaded.config_path,
            self.loaded.manifest_path,
            self.loaded.data_path,
            self.source.row_id_column,
            tuple(self.frame.columns),
        )

    def row_ids(self) -> list[str]:
        values = self.frame[self.source.row_id_column].astype("string").tolist()
        return [str(value) for value in values]

    # ----------------------------------------------------------------- checks
    def record_check(self, name: str, result: ToolResult) -> None:
        if name not in self.checks:
            self.baseline_counts[name] = self.primary_count(name, result)
        self.checks[name] = result

    @staticmethod
    def primary_count(name: str, result: ToolResult) -> int:
        key = PRIMARY_COUNTS[name][0]
        value = result.summary.get(key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    # ---------------------------------------------------------------- cleaning
    def write_scratch(self, frame: pd.DataFrame, label: str) -> tuple[Path, str]:
        path = self.work_dir / f"{label}.parquet"
        frame.to_parquet(path, index=False, engine="fastparquet", compression="snappy")
        return path, sha256_file(path)

    def append_step(self, step: CleaningStep) -> None:
        self.frame = self.frame_from_step(step)
        self.data_sha256 = step.output_sha256
        self.steps.append(step)

    def frame_from_step(self, step: CleaningStep) -> pd.DataFrame:
        return pd.read_parquet(step.scratch_path, engine="fastparquet")

    def last_step_index(self) -> int:
        return self.steps[-1].index if self.steps else -1

    def last_validation_index(self) -> int:
        return self.validations[-1]["after_step"] if self.validations else -1

    # ---------------------------------------------------------------- persist
    def safety_audit(self) -> dict[str, Any]:
        ids = self.row_ids()
        unique = len(set(ids)) == len(ids)
        subset = set(ids).issubset(self.source_row_ids)
        return {
            "unique_row_ids": unique,
            "row_ids_subset_of_source": subset,
            "rows": len(self.frame),
            "source_rows": len(self.source.frame),
            "retained_rate": len(self.frame) / len(self.source.frame),
            "steps": len(self.steps),
            "cells_changed": sum(step.cells_changed for step in self.steps),
            "rows_removed": sum(step.rows_removed for step in self.steps),
        }

    def persist_final(self, requested_id: str | None = None) -> dict[str, Any]:
        if not self.steps:
            raise RuntimeError("no cleaning step was executed; nothing to persist")
        audit = self.safety_audit()
        if not audit["unique_row_ids"] or not audit["row_ids_subset_of_source"]:
            raise RuntimeError(f"row identifier audit failed: {audit}")
        last = self.steps[-1]
        if last.output_sha256 != self.data_sha256:
            raise RuntimeError("last cleaning step hash does not match working data")

        processed = self.root / "data" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        dataset_id = available_output_id(processed, self.source.ref.dataset_id, requested_id)
        final_path = processed / f"{dataset_id}.parquet"
        digest = last.output_sha256
        match = re.match(r"^(.*)_v(\d+)$", dataset_id)
        version = match.group(2) if match else str(int(self.source.ref.version) + 1)
        shutil.copyfile(last.scratch_path, final_path)
        if sha256_file(final_path) != digest:
            raise RuntimeError("persisted dataset bytes differ from the recorded tool output")

        changes_path = None
        frames = []
        for step in self.steps:
            if not step.changes_uri:
                continue
            change_table = pd.read_parquet(step.changes_uri, engine="fastparquet")
            change_table.insert(0, "step_index", step.index)
            change_table.insert(1, "tool_name", step.tool_name)
            change_table.insert(2, "issue_type", step.issue_type)
            frames.append(change_table)
        if frames:
            change_table = pd.concat(frames, ignore_index=True)
            changes_path = processed / f"change.{dataset_id}.changes.parquet"
            change_table.to_parquet(changes_path, index=False, engine="fastparquet")

        ref = DatasetRef(
            dataset_id=dataset_id,
            version=version,
            uri=f"data/processed/{dataset_id}.parquet",
            format=self.source.ref.format,
            sha256=digest,
            row_count=int(len(self.frame)),
            column_count=int(len(self.frame.columns)),
            source=self.source.ref.uri,
            parent_id=self.source.ref.dataset_id,
            metadata={"agent_run": self.run_dir.name},
        )
        manifest = {
            "contract_version": "0.1",
            "dataset_id": dataset_id,
            "version": ref.version,
            "status": "processed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent": {
                "dataset_id": self.source.ref.dataset_id,
                "version": self.source.ref.version,
                "sha256": self.source.ref.sha256,
                "uri": self.source.ref.uri,
            },
            "base_file": {
                "path": f"data/processed/{dataset_id}.parquet",
                "format": "parquet",
                "engine": "fastparquet",
                "compression": "snappy",
                "sha256": digest,
                "row_count": int(len(self.frame)),
                "column_count": int(len(self.frame.columns)),
            },
            "change_file": (
                {
                    "path": f"data/processed/{changes_path.name}",
                    "sha256": sha256_file(changes_path),
                }
                if changes_path
                else None
            ),
            "processing": {
                "cleaning_applied": True,
                "description": "Agent-selected cleaning steps; see run trace and decisions.",
                "change_id": f"change.{dataset_id}",
            },
            "metadata": {
                "agent_run": str(self.run_dir),
                "step_count": len(self.steps),
                "tools": [step.tool_name for step in self.steps],
                "safety_audit": audit,
            },
        }
        manifest_path = processed / f"{dataset_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        profile_path = write_processed_profile(
            self.loaded.config,
            after=ref,
            manifest_path=manifest_path,
            profile_dir=self.root / "config" / "profiles",
        )
        self.final_dataset = {
            "dataset_id": dataset_id,
            "version": ref.version,
            "sha256": digest,
            "profile_path": str(profile_path),
            "manifest_path": str(manifest_path),
            "change_path": str(changes_path) if changes_path else None,
            "rows": int(len(self.frame)),
            "columns": int(len(self.frame.columns)),
            "retained_rate": audit["retained_rate"],
        }
        return self.final_dataset


__all__ = ["AgentSession", "CleaningStep", "PRIMARY_COUNTS"]
