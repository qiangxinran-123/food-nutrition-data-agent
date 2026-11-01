from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if (PROJECT_ROOT / ".deps").exists():
    sys.path.insert(0, str(PROJECT_ROOT / ".deps"))

from data_agent.checks import (
    check_conflicts,
    check_duplicates,
    check_imbalance,
    check_labels,
    check_missing,
    check_outliers,
    check_types_formats,
)
from data_agent.contracts import ToolResult, ToolStatus
from data_agent.io import LoadedDataset, load_dataset


CHECKS: dict[str, Callable[..., ToolResult]] = {
    "missing": check_missing,
    "duplicates": check_duplicates,
    "outliers": check_outliers,
    "types_formats": check_types_formats,
    "conflicts": check_conflicts,
    "labels": check_labels,
    "imbalance": check_imbalance,
}


def _key_metric(name: str, result: ToolResult) -> str:
    summary = result.summary
    if name == "missing":
        return (
            f"缺失单元格 {summary.get('missing_cells')}；"
            f"受影响行 {summary.get('rows_with_missing')}"
        )
    if name == "duplicates":
        content = next(
            (
                item
                for item in summary.get("rules", [])
                if item.get("rule_id") == "content_signature"
            ),
            None,
        )
        if content:
            return (
                f"内容重复组 {content.get('duplicate_group_count')}；"
                f"冗余行 {content.get('redundant_rows')}"
            )
        return f"受影响行 {summary.get('rows_flagged_union')}"
    if name == "outliers":
        return (
            f"硬范围违规 {summary.get('range_flagged_cells')}；"
            f"统计尾部 {summary.get('statistical_flagged_cells')}；"
            f"涉及行 {summary.get('rows_flagged_union')}"
        )
    if name == "types_formats":
        return (
            f"格式/类型有问题列 {summary.get('columns_with_issues')}；"
            f"涉及行 {summary.get('rows_flagged_union')}"
        )
    if name == "conflicts":
        return (
            f"冲突规则 {summary.get('constraints_with_violations')}；"
            f"涉及行 {summary.get('rows_flagged_union')}"
        )
    if name == "labels":
        labels = summary.get("labels", [])
        if labels:
            item = labels[0]
            return (
                f"有效标签 {item.get('valid_rows')}；"
                f"不可用 {item.get('unavailable_rows')}；"
                f"非法 {item.get('invalid_rows')}"
            )
        return "无标签列"
    if name == "imbalance":
        labels = summary.get("labels", [])
        if labels:
            item = labels[0]
            return (
                f"少数类占比 {item.get('minority_share')}；"
                f"最大/最小比 {item.get('majority_minority_ratio')}"
            )
        return "无可用标签"
    return ""


def _write_markdown(
    path: Path,
    payload: dict,
    reports: dict[str, ToolResult],
    selection_dir: str,
) -> None:
    lines = [
        "# 数据质量检测总览",
        "",
        f"- 数据集：`{payload['dataset_id']}`",
        f"- 生成时间：{payload['generated_at']}",
        f"- 总体状态：`{payload['overall_status']}`",
        f"- 问题记录总数：{payload['total_findings']}",
        "",
        "## 七类检测",
        "",
        "| 检测项 | 状态 | 问题记录 | 选择表 | 关键结果 |",
        "|---|---|---:|---:|---|",
    ]
    for name, result in reports.items():
        lines.append(
            f"| {name} | {result.status.value} | {len(result.findings)} | "
            f"{len(result.artifacts)} | "
            f"{_key_metric(name, result)} |"
        )

    lines.extend(
        [
            "",
            "## 问题明细",
            "",
            "| 检测项 | Finding | 严重度 | 涉及行数 |",
            "|---|---|---|---:|",
        ]
    )
    for name, result in reports.items():
        for finding in result.findings:
            lines.append(
                f"| {name} | `{finding.finding_id}` | {finding.severity.value} | "
                f"{finding.affected_rows} |"
            )
    lines.extend(
        [
            "",
            "> 本流程只检测和报告，不修改数据。每个 finding 的选择表位于 "
            f"`{selection_dir}`，处理工具应通过 `selection_artifact_id` 获取，"
            "不要只依赖 JSON 中的样例。清理必须生成 `data/processed` 下的新版本。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all data-quality checks.")
    parser.add_argument(
        "--profile",
        default=str(PROJECT_ROOT / "config" / "profiles" / "off_52k_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "logs" / "checks"),
    )
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(
        args.profile,
        project_root=PROJECT_ROOT,
        verify_hash=not args.skip_hash_check,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = output_dir / "selections"
    selection_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, ToolResult] = {}
    report_records: dict[str, dict] = {}
    for name, check in CHECKS.items():
        result = check(dataset, {"artifact_dir": str(selection_dir)})
        reports[name] = result
        report_path = output_dir / f"{dataset.ref.dataset_id}_{name}.json"
        report_path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_records[name] = {
            "tool_name": result.tool_name,
            "status": result.status.value,
            "finding_count": len(result.findings),
            "artifact_count": len(result.artifacts),
            "summary": result.summary,
            "warnings": result.warnings,
            "report_path": report_path.as_posix(),
        }

    failed = [
        name
        for name, result in reports.items()
        if result.status is ToolStatus.FAILED
    ]
    payload = {
        "contract_version": "0.1",
        "dataset_id": dataset.ref.dataset_id,
        "dataset_sha256": dataset.ref.sha256,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "failed" if failed else "success",
        "failed_checks": failed,
        "total_findings": sum(len(result.findings) for result in reports.values()),
        "checks": report_records,
    }
    summary_path = output_dir / f"{dataset.ref.dataset_id}_quality_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path = output_dir / f"{dataset.ref.dataset_id}_quality_summary.md"
    try:
        selection_label = selection_dir.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        selection_label = selection_dir.as_posix()
    _write_markdown(markdown_path, payload, reports, selection_label)

    print(f"Dataset: {dataset.ref.dataset_id}")
    print(f"Shape: {dataset.frame.shape[0]} rows x {dataset.frame.shape[1]} columns")
    print(f"Overall status: {payload['overall_status']}")
    print(f"Total findings: {payload['total_findings']}")
    for name, result in reports.items():
        print(f"- {name}: {result.status.value}, findings={len(result.findings)}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary Markdown: {markdown_path}")
    print(f"Selection artifacts: {selection_dir}")


if __name__ == "__main__":
    main()
