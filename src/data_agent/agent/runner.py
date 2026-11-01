"""ReAct-style tool-calling loop with fixed validation gates and audit outputs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_agent.io import load_dataset, sha256_file

from .bundle import build_bundle
from .control import ControlLayer, RunPolicy
from .models import ModelCatalog, ModelProfile, ModelKeyMissing, project_paths
from .prompts import PromptSet, load_prompts
from .provider import ChatProvider, ProviderError
from .session import AgentSession
from .tools import specs
from .trace import TraceWriter


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _usage_table(totals: dict[str, Any]) -> str:
    usage = totals["usage"]
    return (
        f"请求次数 {usage['requests']}，输入 Token {usage['prompt_tokens']}，"
        f"输出 Token {usage['completion_tokens']}（其中推理 {usage['reasoning_tokens']}），"
        f"缓存命中 {usage['cache_hit_tokens']}，费用 {totals['cost_usd']} {totals['currency'] or ''}。"
    )


def _report_markdown(
    *,
    run_id: str,
    status: str,
    profile: ModelProfile,
    prompts: PromptSet,
    policy: RunPolicy,
    session: AgentSession,
    control: ControlLayer,
    trace: TraceWriter,
    elapsed: float,
    final_dataset: dict[str, Any] | None,
    bundle: dict[str, Any] | None,
    error: str | None,
) -> str:
    lines = [f"# Agent 运行报告：{run_id}", ""]
    lines.append("## 一、运行信息")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 运行状态 | {status} |")
    lines.append(f"| 模型 | {profile.provider} / {profile.model}（{profile.documented_version}） |")
    lines.append(f"| 接口 | {profile.base_url} |")
    lines.append(f"| thinking | {profile.thinking} |")
    lines.append(f"| temperature / max_tokens | {profile.temperature} / {profile.max_tokens} |")
    lines.append(f"| 系统提示词 | {prompts.system_version}，SHA256 `{prompts.system_sha256[:16]}…` |")
    lines.append(f"| 任务提示词 | {prompts.task_file}，SHA256 `{prompts.task_sha256[:16]}…` |")
    lines.append(f"| 预算 | {policy.max_steps} 轮模型调用 / {policy.max_tool_calls} 次工具调用 |")
    lines.append(f"| 危险操作策略 | {policy.dangerous_policy} |")
    lines.append(f"| 数据输入 | {session.source.ref.dataset_id} v{session.source.ref.version} `{session.source.ref.sha256}` |")
    lines.append(f"| 总耗时 | {elapsed:.1f} 秒 |")
    lines.append("")

    if session.final_report:
        report = session.final_report
        lines.append("## 二、Agent 自述")
        lines.append("")
        lines.append(f"- **总结**：{report['summary']}")
        lines.append(f"- **清洗理由**：{report['cleaning_rationale']}")
        lines.append(f"- **聚类解释**：{report['cluster_interpretation']}")
        lines.append(f"- **局限**：{report['limitations']}")
        lines.append("")

    lines.append("## 三、工具调用序列")
    lines.append("")
    lines.append("| # | 工具 | 结果 | 耗时(ms) | 摘要 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for record in control.records:
        summary = json.dumps(record["summary"], ensure_ascii=False)
        if len(summary) > 120:
            summary = summary[:117] + "…"
        summary = summary.replace("|", "/")
        lines.append(
            f"| {record['seq']} | {record['tool']} | {record['status']}"
            f"{(' / ' + str(record['error_code'])) if record['error_code'] else ''} | "
            f"{record['duration_ms']} | {summary} |"
        )
    lines.append("")

    lines.append("## 四、数据质量变化（最后一次验证）")
    lines.append("")
    if session.validations:
        last = session.validations[-1]
        lines.append("| 问题类型 | 清洗前 | 清洗后 | 变化 | 状态 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in last["rows"]:
            lines.append(
                f"| {row['issue_type']} | {row['before']} | {row['after']} | {row['delta']} | {row['status']} |"
            )
    else:
        lines.append("未执行验证。")
    lines.append("")

    audit = session.safety_audit()
    lines.append("## 五、清洗与保留情况")
    lines.append("")
    lines.append("| 步骤 | 策略 | 删除行 | 修改单元格 |")
    lines.append("| --- | --- | --- | --- |")
    for step in session.steps:
        lines.append(f"| {step.index} | {step.tool_name} / {step.strategy_id} | {step.rows_removed} | {step.cells_changed} |")
    lines.append("")
    lines.append(
        f"保留 {audit['rows']}/{audit['source_rows']} 行（{audit['retained_rate']:.2%}），"
        f"row id 唯一：{audit['unique_row_ids']}，均为原始 row id 子集：{audit['row_ids_subset_of_source']}。"
    )
    lines.append("")

    lines.append("## 六、聚类结果")
    lines.append("")
    if session.cluster_metrics:
        lines.append("| k | Silhouette（抽样） | CH | DBI | 平均稳定性 ARI | 簇大小 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for k in sorted(session.cluster_metrics):
            metrics = session.cluster_metrics[k]
            sizes = ", ".join(f"{item['label']}:{item['count']}" for item in metrics["clusters"])
            lines.append(
                f"| {k} | {metrics['silhouette']:.6f} | {metrics['calinski_harabasz']:.3f} | "
                f"{metrics['davies_bouldin']:.6f} | {metrics['stability']['mean_ari']} | {sizes} |"
            )
    else:
        lines.append("未完成聚类评价。")
    lines.append("")
    if session.cluster_analysis:
        k = sorted(session.cluster_analysis)[-1]
        lines.append(f"最近一次簇解释（k={k}）：")
        lines.append("")
        lines.append("| 簇 | 数量 | 能量均值 | 脂肪均值 | 糖均值 | 钠均值 | 代表食品 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for cluster in session.cluster_analysis[k]["clusters"]:
            nutrition = cluster["nutrition"]
            names = "；".join(str(item["product_name"]) for item in cluster["representatives"][:2])
            lines.append(
                f"| {cluster['label']} | {cluster['count']} | {nutrition['energy-kcal_100g']['mean']:.1f} | "
                f"{nutrition['fat_100g']['mean']:.2f} | {nutrition['sugars_100g']['mean']:.2f} | "
                f"{nutrition['sodium_100g']['mean']:.2f} | {names} |"
            )
    lines.append("")

    lines.append("## 七、成本与产物")
    lines.append("")
    lines.append(_usage_table(trace.totals()))
    lines.append("")
    if final_dataset:
        lines.append(
            f"最终数据版本：{final_dataset['dataset_id']} v{final_dataset['version']}，"
            f"{final_dataset['rows']} 行 × {final_dataset['columns']} 列，"
            f"SHA256 `{final_dataset['sha256']}`。"
        )
    if bundle:
        lines.append("")
        lines.append(
            f"C 组证据包：{bundle['bundle_path']}；可直接导入：{bundle['ready_for_import']}；"
            f"阻塞原因：{bundle['blocking_reason']}。"
        )
    if error:
        lines.append("")
        lines.append(f"失败信息：{error}")
    lines.append("")
    lines.append("## 八、说明与局限")
    lines.append("")
    lines.append("- 轮廓系数为固定种子的 2000 行抽样值，不是全量精确值；k 不代表业务最优。")
    lines.append("- 簇是营养相似群体，不能解释为健康等级。")
    lines.append("- 受控演示只覆盖本次实际执行的策略；未执行的清洗不填入指标。")
    lines.append("- 本地哈希只能证明一致性，不能独立证明提供方真实性；C 组需外部审查。")
    return "\n".join(lines)


def run_agent(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    profile_name: str | None = None,
    catalog_path: str | Path | None = None,
    prompt_version: str = "v1",
    task_file: str = "task_v1.md",
    max_steps: int = 36,
    max_tool_calls: int = 80,
    dangerous_policy: str = "deny",
    dataset_version: str = "1",
    run_id: str | None = None,
    make_bundle: bool = True,
) -> dict[str, Any]:
    paths = project_paths(project_root)
    root = paths["root"]
    catalog = ModelCatalog.load(catalog_path or paths["models"])
    profile = catalog.profile(profile_name)
    prompts = load_prompts(paths["prompts"], version=prompt_version, task_file=task_file)

    run_id = run_id or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{profile.model.replace('/', '-')}"
    run_dir = Path(output_dir) if output_dir else paths["outputs"] / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    policy = RunPolicy(max_steps=max_steps, max_tool_calls=max_tool_calls, dangerous_policy=dangerous_policy)
    started_at = datetime.now(timezone.utc)
    timer = time.perf_counter()
    session = AgentSession(root, run_dir, dataset_version)
    trace = TraceWriter(run_dir, profile)
    control = ControlLayer(session, trace, policy)
    provider = ChatProvider(profile)

    system_text = prompts.render_system(
        dataset_id=session.source.ref.dataset_id,
        max_steps=policy.max_steps,
        max_tool_calls=policy.max_tool_calls,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompts.task_text},
    ]
    trace.event(
        "run_start",
        {
            "run_id": run_id,
            "model": profile.public_dict(),
            "prompt": {
                "system_version": prompts.system_version,
                "system_sha256": prompts.system_sha256,
                "task_file": prompts.task_file,
                "task_sha256": prompts.task_sha256,
            },
            "policy": policy.__dict__,
            "dataset": session.source.ref.model_dump(mode="json"),
        },
    )

    status = "incomplete"
    error_message: str | None = None
    final_dataset: dict[str, Any] | None = None
    bundle: dict[str, Any] | None = None
    final_ref: dict[str, Any] | None = None
    nudges = 0
    model_calls = 0
    try:
        while model_calls < policy.max_steps:
            model_calls += 1
            response = provider.complete(messages, specs(), request_index=model_calls)
            record = trace.record_provider_response(response, model_calls)
            response_key = record["id"]

            assistant_message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments_raw},
                    }
                    for call in response.tool_calls
                ]
            if response.reasoning_content and profile.thinking == "enabled":
                assistant_message["reasoning_content"] = response.reasoning_content
            messages.append(assistant_message)

            if not response.tool_calls:
                nudges += 1
                if nudges > 3:
                    status = "no_tool_calls"
                    error_message = "模型连续只回复文本而未调用工具"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "请不要只回复文字。继续调用工具完成任务；如已完成，请调用 submit_final_report。",
                    }
                )
                continue

            for call in response.tool_calls:
                payload = control.execute(call, response_key)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": json.dumps(payload, ensure_ascii=False, default=_json_default),
                    }
                )
            if response.finish_reason == "length":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一条回复因 max_tokens 被截断。请缩短文字：submit_final_report 的四个字段"
                            "各不超过 100 个汉字，总输出不超过 600 字。"
                        ),
                    }
                )
            if session.final_report is not None:
                status = "completed"
                break
        else:
            status = "budget_exhausted"
            error_message = f"达到 {policy.max_steps} 轮模型调用上限，任务未提交"
    except ModelKeyMissing as error:
        status = "provider_unavailable"
        error_message = str(error)
    except ProviderError as error:
        status = "provider_error"
        error_message = str(error)
    except Exception as error:  # keep the audit trail even on unexpected failures
        status = "failed"
        error_message = f"{type(error).__name__}: {error}"
        trace.event("run_error", {"message": error_message})

    final_ref = None
    if status == "completed":
        try:
            final_dataset = session.persist_final()
            loaded_final = load_dataset(final_dataset["profile_path"], project_root=root, verify_hash=True)
            final_ref = loaded_final.ref.model_dump(mode="json")
            trace.event("final_dataset", final_dataset)
        except Exception as error:
            status = "persist_failed"
            error_message = f"{type(error).__name__}: {error}"
            trace.event("run_error", {"message": error_message})

    elapsed = time.perf_counter() - timer
    finished_at = datetime.now(timezone.utc)

    _write_json(run_dir / "conversation.json", messages)
    _write_json(
        run_dir / "decisions.json",
        [step.decision_record() for step in session.steps],
    )
    _write_json(
        run_dir / "tool_executions.json",
        [step.execution_record() for step in session.steps],
    )
    _write_json(run_dir / "provider_export.json", trace.provider_export(run_id, session.source.ref.sha256 or ""))

    if status == "completed" and final_ref is not None and make_bundle:
        try:
            bundle = build_bundle(session, trace, run_id, final_ref)
        except Exception as error:
            bundle = {"bundle_path": None, "ready_for_import": False, "blocking_reason": f"bundle failed: {error}"}

    summary = {
        "run_id": run_id,
        "status": status,
        "error": error_message,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "model": profile.public_dict(),
        "prompts": {
            "system_version": prompts.system_version,
            "system_sha256": prompts.system_sha256,
            "task_file": prompts.task_file,
            "task_sha256": prompts.task_sha256,
        },
        "policy": policy.__dict__,
        "model_calls": model_calls,
        "tool_calls": control.call_count,
        "usage_and_cost": trace.totals(),
        "source_dataset": session.source.ref.model_dump(mode="json"),
        "final_dataset": final_dataset,
        "bundle": bundle,
        "agent_report": session.final_report,
        "safety_audit": session.safety_audit(),
        "steps": [
            {
                "index": step.index,
                "tool": step.tool_name,
                "strategy": step.strategy_id,
                "rows_removed": step.rows_removed,
                "cells_changed": step.cells_changed,
                "input_sha256": step.input_sha256,
                "output_sha256": step.output_sha256,
                "duration_ms": round(step.duration_ms, 1),
            }
            for step in session.steps
        ],
        "tool_records": control.records,
    }
    _write_json(run_dir / "run.json", summary)
    _write_json(run_dir / "quality_summary.json", {"validations": session.validations, "baseline": session.baseline_counts})
    _write_json(run_dir / "clustering_summary.json", {str(k): v for k, v in session.cluster_metrics.items()})
    (run_dir / "report.md").write_text(
        _report_markdown(
            run_id=run_id,
            status=status,
            profile=profile,
            prompts=prompts,
            policy=policy,
            session=session,
            control=control,
            trace=trace,
            elapsed=elapsed,
            final_dataset=final_dataset,
            bundle=bundle,
            error=error_message,
        ),
        encoding="utf-8",
    )
    trace.event("run_end", {"status": status, "model_calls": model_calls, "tool_calls": control.call_count, "error": error_message})

    artifacts = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    _write_json(run_dir / "manifest.json", {"run_id": run_id, "files": artifacts, "status": status})
    summary["output_dir"] = str(run_dir)
    return summary


__all__ = ["run_agent"]
