"""Compose the M4 C-group evidence bundle from a real agent run.

Local hashes prove consistency only. `provenance_review.json` is emitted as a
template with verification unset; M4 rejects the bundle until a named reviewer
confirms the provider execution.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from data_agent.io import sha256_file

from .session import AgentSession
from .trace import TraceWriter


def _file_entry(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256_file(path)}


def build_bundle(
    session: AgentSession,
    trace: TraceWriter,
    run_id: str,
    final_ref: dict[str, Any],
    *,
    blocking_reason: str | None = None,
) -> dict[str, Any]:
    bundle_dir = session.run_dir / "agent_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=False)

    export_path = bundle_dir / "provider_export.json"
    export_path.write_text(
        json.dumps(trace.provider_export(run_id, session.source.ref.sha256 or ""), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decisions_path = bundle_dir / "decisions.json"
    decisions_path.write_text(
        json.dumps([step.decision_record() for step in session.steps], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    executions_path = bundle_dir / "tool_executions.json"
    executions_path.write_text(
        json.dumps([step.execution_record() for step in session.steps], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    profile_source = Path(session.final_dataset["profile_path"])
    profile_path = bundle_dir / "output_profile.json"
    shutil.copyfile(profile_source, profile_path)

    review = {
        "reviewer": "",
        "reviewed_at": "",
        "actual_provider_execution_verified": False,
        "provider_export_sha256": sha256_file(export_path),
        "method": "",
        "instructions": (
            "该文件必须由未参与本次运行的同学/导师完成外部审查后填写：核对 DeepSeek 控制台的请求记录、"
            "provider_export.json 中的响应 id 与时间、以及 decisions.json 是否确实由这些响应产生。"
            "填写 reviewer、reviewed_at、method 并把 actual_provider_execution_verified 改为 true 后，"
            "M4 才能导入本 bundle 作为 C 组。仅本地哈希一致性不能证明提供方真实性。"
        ),
    }
    review_path = bundle_dir / "provenance_review.json"
    review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

    bundle = {
        "bundle_version": "1.0",
        "run_id": run_id,
        "created_from": str(session.run_dir),
        "source_dataset": session.source.ref.model_dump(mode="json"),
        "output_dataset": final_ref,
        "output_profile": _file_entry(profile_path),
        "provider_export": _file_entry(export_path),
        "decisions": _file_entry(decisions_path),
        "tool_executions": _file_entry(executions_path),
        "provenance_review": _file_entry(review_path),
        "ready_for_import": blocking_reason is None and review["actual_provider_execution_verified"],
        "blocking_reason": blocking_reason
        or "provenance_review.json 未完成外部审查（actual_provider_execution_verified=false）",
        "note": "第三部分只负责产生并自检证据；是否构成有效的 C 组由外部审查决定。",
    }
    bundle_path = bundle_dir / "agent_bundle.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "bundle_dir": str(bundle_dir),
        "bundle_path": str(bundle_path),
        "ready_for_import": bundle["ready_for_import"],
        "blocking_reason": bundle["blocking_reason"],
        "files": {
            name: bundle[name]["sha256"]
            for name in ("output_profile", "provider_export", "decisions", "tool_executions", "provenance_review")
        },
    }


__all__ = ["build_bundle"]
