"""Tool registry for the four project modules, exposed as LLM function schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from data_agent.checks.conflicts import detect_conflicts
from data_agent.checks.duplicates import detect_duplicates
from data_agent.checks.imbalance import detect_imbalance
from data_agent.checks.labels import detect_label_errors
from data_agent.checks.missing import detect_missing
from data_agent.checks.outliers import detect_outliers
from data_agent.checks.types_formats import detect_type_format_issues
from data_agent.cleaning import execute_cleaning, list_strategies, make_decision
from data_agent.cleaning.strategies import get_strategy
from data_agent.clustering.analysis import analyze_clusters
from data_agent.clustering.core import (
    DEFAULT_FEATURES,
    evaluate_clusters,
    run_kmeans,
    save_scaler,
    scale_features,
    select_features,
)
from data_agent.clustering.visualization import visualize_clusters
from data_agent.contracts import Finding, IssueType, ToolResult
from data_agent.io import sha256_file

from .session import AgentSession


class ToolError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class ToolContext:
    response_id: str
    tool_call_id: str
    arguments_raw: str
    max_removal_fraction: float = 0.1
    dangerous_allowed: bool = False


@dataclass(frozen=True)
class ToolDef:
    name: str
    category: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[AgentSession, dict[str, Any], ToolContext], dict[str, Any]]
    mutates_data: bool = False
    risk: str = "safe"

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


REGISTRY: dict[str, ToolDef] = {}


def tool(
    name: str,
    category: str,
    description: str,
    parameters: dict[str, Any],
    *,
    mutates_data: bool = False,
    risk: str = "safe",
):
    def decorator(handler):
        REGISTRY[name] = ToolDef(name, category, description, parameters, handler, mutates_data, risk)
        return handler

    return decorator


def specs() -> list[dict[str, Any]]:
    return [item.spec() for item in REGISTRY.values()]


def get(name: str) -> ToolDef:
    if name not in REGISTRY:
        raise ToolError("unknown_tool", f"tool {name!r} is not registered in this run")
    return REGISTRY[name]


# ---------------------------------------------------------------- schema bits
SAMPLE_LIMIT = {
    "type": "object",
    "properties": {
        "sample_limit": {"type": "integer", "minimum": 0, "maximum": 50, "default": 10}
    },
    "additionalProperties": False,
}

CLEANING_PARAMS = {
    "type": "object",
    "properties": {
        "strategy_id": {"type": "string", "minLength": 1},
        "finding_id": {"type": "string", "minLength": 1},
        "selection_artifact_id": {"type": "string"},
        "target_columns": {"type": "array", "items": {"type": "string"}},
        "params": {"type": "object"},
        "reason": {"type": "string", "minLength": 1},
        "expected_effect": {"type": "string"},
        "confirm": {"type": "boolean", "default": False},
    },
    "required": ["strategy_id", "finding_id", "reason"],
    "additionalProperties": False,
}

K_PARAM = {"type": "integer", "minimum": 2, "maximum": 8}


# ------------------------------------------------------------------- helpers
def _issue_name(tool_name: str) -> str:
    return {
        "clean_missing": "missing",
        "clean_duplicates": "duplicate",
        "clean_outliers": "outlier",
        "normalize_types_formats": "format",
        "resolve_conflicts": "constraint",
        "repair_labels": "label",
    }[tool_name]


def _finding_payload(finding: Finding) -> dict[str, Any]:
    return {
        "finding_id": finding.finding_id,
        "severity": finding.severity.value,
        "columns": list(finding.columns),
        "affected_rows": finding.affected_rows,
        "affected_rate": finding.affected_rate,
        "severity_scope": finding.scope.value,
        "selection_artifact_id": finding.selection_artifact_id,
        "suggested_actions": list(finding.suggested_actions),
        "evidence": finding.evidence,
    }


def _compact(result: ToolResult, *, limit: int = 20) -> dict[str, Any]:
    findings = [_finding_payload(item) for item in result.findings[:limit]]
    return {
        "tool_name": result.tool_name,
        "status": result.status.value,
        "summary": result.summary,
        "finding_count": len(result.findings),
        "findings": findings,
        "truncated_findings": max(0, len(result.findings) - limit),
        "artifact_count": len(result.artifacts),
        "warnings": result.warnings[:5],
        "duration_ms": round(result.duration_ms or 0.0, 1),
    }


def _options(session: AgentSession, args: dict[str, Any], *, persist: bool) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if persist:
        options["artifact_dir"] = str(session.selection_dir)
    if "sample_limit" in args:
        options["sample_limit"] = int(args["sample_limit"])
    return options


CHECKS: dict[str, Callable[[AgentSession, dict[str, Any]], ToolResult]] = {
    "check_missing": lambda s, a: detect_missing(s.frame, s.schema, _options(s, a, persist=True)),
    "check_duplicates": lambda s, a: detect_duplicates(
        s.frame, s.schema, s.loaded.config.duplicates, _options(s, a, persist=True)
    ),
    "check_outliers": lambda s, a: detect_outliers(
        s.frame, s.schema, s.loaded.config.outliers, _options(s, a, persist=True)
    ),
    "check_types_formats": lambda s, a: detect_type_format_issues(
        s.frame, s.schema, s.loaded.config.types_formats, _options(s, a, persist=True)
    ),
    "check_conflicts": lambda s, a: detect_conflicts(
        s.frame, s.schema, s.loaded.config.conflicts, _options(s, a, persist=True)
    ),
    "check_labels": lambda s, a: detect_label_errors(
        s.frame, s.schema, s.loaded.config.labels, _options(s, a, persist=True)
    ),
    "check_imbalance": lambda s, a: detect_imbalance(
        s.frame, s.schema, s.loaded.config.labels, s.loaded.config.imbalance, _options(s, a, persist=True)
    ),
}

ISSUE_FOR_CHECK = {
    "check_missing": "missing",
    "check_duplicates": "duplicate",
    "check_outliers": "outlier",
    "check_types_formats": "format",
    "check_conflicts": "constraint",
    "check_labels": "label",
    "check_imbalance": "imbalance",
}


def _run_check(name: str) -> Callable[[AgentSession, dict[str, Any], ToolContext], dict[str, Any]]:
    def handler(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = CHECKS[name](session, args)
        session.record_check(ISSUE_FOR_CHECK[name], result)
        payload = _compact(result)
        payload["next"] = "如需清洗，请调用对应 clean_* 工具，并引用上面的 finding_id。"
        return payload

    return handler


# ---------------------------------------------------------------- module 0/1
@tool(
    "profile_dataset",
    "detection",
    "读取当前工作副本概况：行列数、字段类型与单位、关键营养字段空值数、数据版本哈希。只读。",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
def profile_dataset(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    frame = session.frame
    numeric = [
        column.name
        for column in session.schema.columns
        if column.logical_type.value == "numeric"
    ]
    null_counts = {column: int(frame[column].isna().sum()) for column in numeric if column in frame}
    sample = [
        str(value)
        for value in frame.get("product_name", pd.Series(dtype="string")).dropna().head(5).tolist()
    ]
    return {
        "dataset_id": session.source.ref.dataset_id,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "working_sha256": session.data_sha256,
        "source_sha256": session.source.ref.sha256,
        "steps_completed": len(session.steps),
        "numeric_columns": numeric,
        "numeric_null_counts": null_counts,
        "column_units": {
            column.name: column.unit for column in session.schema.columns if column.unit
        },
        "sample_products": sample,
        "clustering_features_available": list(DEFAULT_FEATURES),
    }


for _name, _desc in (
    ("check_missing", "检测空值、空字符串和声明的缺失标记，统计缺失单元格与列级 finding。只读。"),
    ("check_duplicates", "检测完全重复、主键重复和内容级重复组。只读。"),
    ("check_outliers", "按物理范围和统计规则检测营养值异常，输出违规单元格与受影响行。只读。"),
    ("check_types_formats", "检测类型、单位、每100g/每份计量口径和文本格式问题。只读。"),
    ("check_conflicts", "检测字段逻辑冲突（饱和脂肪>总脂肪、糖>碳水、钠>盐、宏量之和>100g 等）。只读。"),
    ("check_labels", "核验 Nutri-Score 等标签的合法取值与别名问题。只读。"),
    ("check_imbalance", "统计标签类别分布与不平衡程度。只读。"),
):
    REGISTRY[_name] = ToolDef(_name, "detection", _desc, SAMPLE_LIMIT, _run_check(_name))

del _name, _desc


@tool(
    "list_cleaning_strategies",
    "detection",
    "列出某个问题类型已注册的清洗策略、风险等级、是否修改数据以及允许参数。只读。",
    {
        "type": "object",
        "properties": {
            "issue_type": {
                "type": "string",
                "enum": ["missing", "duplicate", "outlier", "format", "constraint", "label"],
            }
        },
        "required": ["issue_type"],
        "additionalProperties": False,
    },
)
def list_cleaning_strategies(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    issue = IssueType(args["issue_type"])
    items = [
        {
            "strategy_id": spec.strategy_id,
            "description": spec.description,
            "risk_level": spec.risk_level.value,
            "mutates_data": spec.mutates_data,
            "requires_confirmation": spec.requires_confirmation,
            "implemented": spec.implemented,
            "allowed_params": spec.allowed_params,
            "required_params": spec.required_params,
            "notes": spec.notes,
        }
        for spec in list_strategies(issue)
    ]
    return {
        "issue_type": issue.value,
        "strategies": items,
        "rule": "未实现或 requires_confirmation 的策略在此运行中不会被批准；优先选择 safe/caution 且会修改数据的策略。",
    }


# ---------------------------------------------------------------- module 2
def _find_finding(session: AgentSession, issue: str, finding_id: str) -> Finding:
    result = session.checks.get(issue)
    if result is None:
        raise ToolError(
            "check_required",
            f"尚未运行 {PRIMARY.get(issue, issue)}；请先检测，再引用 finding_id",
        )
    for finding in result.findings:
        if finding.finding_id == finding_id:
            return finding
    raise ToolError(
        "unknown_finding",
        f"最近一次 {PRIMARY.get(issue, issue)} 没有 finding_id={finding_id!r}；请使用检测输出中的 id",
        details={"available": [item.finding_id for item in result.findings][:20]},
    )


PRIMARY = {
    "missing": "check_missing",
    "duplicate": "check_duplicates",
    "outlier": "check_outliers",
    "format": "check_types_formats",
    "constraint": "check_conflicts",
    "label": "check_labels",
}


def _clean(session: AgentSession, tool_name: str, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    issue = _issue_name(tool_name)
    if session.clustering_started:
        raise ToolError("state_locked", "聚类已开始，不能再修改数据；如需重来请开启新的运行")
    if session.final_report is not None:
        raise ToolError("run_finished", "本次运行已提交报告，不能再清洗")

    # A previous cleaning changed the row set; stale selections would use old row
    # positions. Refresh the matching check deterministically before resolving.
    refreshed = False
    latest = session.checks.get(issue)
    if latest is not None and int(latest.summary.get("row_count", -1)) != len(session.frame):
        CHECKS[PRIMARY[issue]](session, {"sample_limit": 3})
        refreshed = True

    strategy_id = args["strategy_id"]
    try:
        strategy = get_strategy(issue, strategy_id)
    except Exception as error:
        raise ToolError("invalid_strategy", str(error)) from error
    if not strategy.implemented:
        raise ToolError("strategy_unavailable", f"策略 {strategy_id!r} 在当前环境中未实现")
    if not strategy.mutates_data:
        raise ToolError(
            "read_only_strategy",
            f"策略 {strategy_id!r} 不修改数据，无法形成可追溯的数据版本；"
            "若只需标记问题，请改用 validate_quality 记录，或选择会修改数据的策略",
        )
    if strategy.requires_confirmation and not (
        args.get("confirm") and context.dangerous_allowed
    ):
        raise ToolError(
            "confirmation_required",
            f"策略 {strategy_id!r} 属高风险操作，本次运行未授权；请改用 safe/caution 策略",
        )

    finding = _find_finding(session, issue, args["finding_id"])
    # finding_id is authoritative; a supplied artifact id is only a hint and may
    # be stale after earlier row deletions.
    selection_id = finding.selection_artifact_id
    if not selection_id:
        raise ToolError("missing_selection", "该 finding 没有可用的完整选择集，无法执行清洗")
    selection_note = None
    supplied = args.get("selection_artifact_id")
    if supplied and supplied != selection_id:
        selection_note = f"selection_artifact_id 已按 finding_id 解析为最新选择集 {selection_id}"

    decision = make_decision(
        issue_type=issue,
        strategy_id=strategy_id,
        finding_id=finding.finding_id,
        selection_artifact_id=selection_id,
        target_columns=list(args.get("target_columns") or []),
        params=dict(args.get("params") or {}),
        reason=args["reason"],
        expected_effect=args.get("expected_effect"),
        decision_id=f"agent.{issue}.{strategy_id}.{len(session.steps) + 1:02d}",
        metadata={"output_dataset_id": None},
    )
    started = perf_counter()
    result = execute_cleaning(
        session.working_loaded(),
        decision,
        dry_run=True,
        confirm=bool(args.get("confirm")),
        selection_artifact_dir=session.selection_dir,
        artifact_dir=session.cleaning_artifact_dir,
    )
    if result.status.value == "failed":
        raise ToolError("cleaning_failed", result.error_message or "cleaning failed")
    if result.status.value == "partial":
        raise ToolError(
            "no_improvement",
            "该策略执行后对应问题数量没有下降，控制层拒绝落盘；请换策略或先复核检测结果",
            details={"validation": result.validation.model_dump(mode="json") if result.validation else None},
        )
    assert result.change_set is not None
    cells_changed = result.change_set.cells_changed
    rows_removed = result.change_set.rows_removed
    rows_before = int(len(session.frame))
    if cells_changed + rows_removed == 0:
        raise ToolError("no_changes", "策略未产生任何变化；请检查 selected 值与参数后重试或换策略")
    if rows_removed > 0 and rows_removed / rows_before > context.max_removal_fraction:
        raise ToolError(
            "retention_guard",
            f"将删除 {rows_removed}/{rows_before} 行，超过单步上限 "
            f"{context.max_removal_fraction:.0%}；请改用保留行的策略",
        )
    output = result.output_frame
    assert output is not None
    index = len(session.steps) + 1
    path, digest = session.write_scratch(output.reset_index(drop=True), f"step{index:02d}_{tool_name}")
    step = _step_from_result(session, result, tool_name, issue, strategy_id, context, path, digest, index, rows_before, rows_removed, cells_changed, perf_counter() - started)
    session.append_step(step)
    return {
        "status": "success",
        "step_index": index,
        "strategy_id": strategy_id,
        "rows_before": rows_before,
        "rows_after": int(len(session.frame)),
        "cells_changed": int(cells_changed),
        "rows_removed": int(rows_removed),
        "retained_rate": round(len(session.frame) / len(session.source.frame), 6),
        "validation": {
            "issue": issue,
            "before": (result.validation.metrics.get("before_count") if result.validation else None),
            "after": (result.validation.metrics.get("after_count") if result.validation else None),
        },
        "working_sha256": session.data_sha256,
        "warnings": result.warnings[:3],
        "detection_refreshed": refreshed,
        **({"selection_note": selection_note} if selection_note else {}),
        "next": "请调用 validate_quality 复核问题计数，再决定下一步。",
    }


def _step_from_result(session, result, tool_name, issue, strategy_id, context, path, digest, index, rows_before, rows_removed, cells_changed, elapsed):
    from .session import CleaningStep

    assert result.decision is not None
    return CleaningStep(
        index=index,
        tool_name=tool_name,
        issue_type=issue,
        strategy_id=strategy_id,
        decision_id=result.decision.decision_id,
        finding_id=result.decision.finding_id,
        response_id=context.response_id,
        tool_call_id=context.tool_call_id,
        arguments_raw=context.arguments_raw,
        input_sha256=session.data_sha256,
        output_sha256=digest,
        scratch_path=path,
        rows_before=rows_before,
        rows_after=int(len(result.output_frame)),
        cells_changed=int(cells_changed),
        rows_removed=int(rows_removed),
        duration_ms=elapsed * 1000,
        changes_uri=result.change_set.changes_uri if result.change_set else None,
        warnings=list(result.warnings[:5]),
    )


for _tool, _issue_label in (
    ("clean_missing", "缺失值"),
    ("clean_duplicates", "重复行"),
    ("clean_outliers", "异常值"),
    ("normalize_types_formats", "格式/类型"),
    ("resolve_conflicts", "字段冲突"),
    ("repair_labels", "标签"),
):
    _desc = (
        f"对{_issue_label}执行一个已注册清洗策略，生成新的工作数据版本。"
        "必须引用最近一次检测的 finding_id；strategy_id 来自 list_cleaning_strategies。"
    )

    def _bind(tool_name: str):
        def handler(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
            return _clean(session, tool_name, args, context)

        return handler

    REGISTRY[_tool] = ToolDef(
        _tool, "preprocessing", _desc, CLEANING_PARAMS, _bind(_tool), mutates_data=True, risk="caution"
    )

del _tool, _issue_label, _desc


# ---------------------------------------------------------------- module 4
@tool(
    "validate_quality",
    "validation",
    "重新运行检测并与清洗前基线比较，返回每类问题的数量变化。只读。",
    {
        "type": "object",
        "properties": {
            "issue_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["missing", "duplicate", "outlier", "format", "constraint", "label"],
                },
                "minItems": 1,
            }
        },
        "required": ["issue_types"],
        "additionalProperties": False,
    },
)
def validate_quality(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if not session.steps:
        raise ToolError("no_cleaning", "尚未执行任何清洗，无法验证")
    name_by_issue = {value: key for key, value in ISSUE_FOR_CHECK.items()}
    rows: list[dict[str, Any]] = []
    for issue in dict.fromkeys(args["issue_types"]):
        name = name_by_issue[issue]
        options = {}  # no artifact_dir: validation must not overwrite cleaning selections
        if name == "check_missing":
            result = detect_missing(session.frame, session.schema, options)
        elif name == "check_duplicates":
            result = detect_duplicates(session.frame, session.schema, session.loaded.config.duplicates, options)
        elif name == "check_outliers":
            result = detect_outliers(session.frame, session.schema, session.loaded.config.outliers, options)
        elif name == "check_types_formats":
            result = detect_type_format_issues(session.frame, session.schema, session.loaded.config.types_formats, options)
        elif name == "check_conflicts":
            result = detect_conflicts(session.frame, session.schema, session.loaded.config.conflicts, options)
        elif name == "check_labels":
            result = detect_label_errors(session.frame, session.schema, session.loaded.config.labels, options)
        else:
            result = detect_imbalance(session.frame, session.schema, session.loaded.config.labels, session.loaded.config.imbalance, options)
        after = session.primary_count(issue, result)
        before = session.baseline_counts.get(issue)
        status = "no_baseline" if before is None else ("improved" if after <= before else "worsened")
        rows.append(
            {
                "issue_type": issue,
                "before": before,
                "after": after,
                "delta": None if before is None else after - before,
                "status": status if before is not None else "no_baseline",
            }
        )
    record = {
        "after_step": session.last_step_index(),
        "rows": rows,
        "retained_rate": round(len(session.frame) / len(session.source.frame), 6),
    }
    session.validations.append(record)
    return record


@tool(
    "retention_report",
    "validation",
    "汇总保留行数、累计删除行数、修改单元格数和每一步的清洗动作。只读。",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
def retention_report(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return {
        "audit": session.safety_audit(),
        "steps": [
            {
                "index": step.index,
                "tool": step.tool_name,
                "strategy": step.strategy_id,
                "rows_removed": step.rows_removed,
                "cells_changed": step.cells_changed,
                "duration_ms": round(step.duration_ms, 1),
            }
            for step in session.steps
        ],
    }


# ---------------------------------------------------------------- module 3
@tool(
    "select_features",
    "clustering",
    "选择营养特征并筛除缺失/非数值/无穷行。必须在清洗完成后调用。只读（构建分析视图）。",
    {
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "默认使用项目八项营养特征：energy-kcal, proteins, fat, saturated-fat, carbohydrates, sugars, fiber, sodium（每100g）",
            }
        },
        "additionalProperties": False,
    },
)
def select_feature_tool(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if not session.steps:
        raise ToolError("no_cleaning", "请先完成清洗，再进入聚类阶段")
    features = tuple(args.get("features") or DEFAULT_FEATURES)
    session.clustering_started = True
    selected = select_features(session.dataset_input(), features)
    session.cluster_selected = selected
    session.cluster_scaled = None
    session.cluster_runs.clear()
    session.cluster_metrics.clear()
    session.cluster_analysis.clear()
    return {
        "status": "selected",
        "features": list(selected.features),
        "input_rows": selected.input_count,
        "valid_rows": len(selected.row_ids),
        "excluded": selected.exclusion_counts,
        "valid_rate": round(len(selected.row_ids) / selected.input_count, 6),
        "working_sha256": session.data_sha256,
        "next": "请调用 scale_features 标准化。",
    }


@tool(
    "scale_features",
    "clustering",
    "用 StandardScaler 拟合当前特征矩阵并保存 scaler 参数。只读（不改变原始列）。",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
def scale_feature_tool(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if session.cluster_selected is None:
        raise ToolError("sequence", "请先调用 select_features")
    scaled = scale_features(session.cluster_selected)
    session.cluster_scaled = scaled
    scaler_path = session.run_dir / "clustering" / "scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    save_scaler(scaled, scaler_path)
    return {
        "status": "scaled",
        "rows": len(scaled.row_ids),
        "features": list(scaled.features),
        "matrix_sha256": scaled.matrix_sha256,
        "scaler_sha256": scaled.scaler_sha256,
        "scaler_file": str(scaler_path),
        "scaler_mean": dict(zip(scaled.features, [round(float(v), 6) for v in scaled.scaler.mean_])),
    }


@tool(
    "run_kmeans",
    "clustering",
    "按指定 k 运行 K-Means（lloyd，固定种子，n_init=10），返回簇大小与惯性。",
    {
        "type": "object",
        "properties": {
            "k": K_PARAM,
            "random_state": {"type": "integer", "minimum": 0, "maximum": 4294967295, "default": 42},
            "n_init": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "max_iter": {"type": "integer", "minimum": 1, "default": 300},
            "tol": {"type": "number", "minimum": 0.000001, "default": 0.0001},
        },
        "required": ["k"],
        "additionalProperties": False,
    },
)
def run_kmeans_tool(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if session.cluster_scaled is None:
        raise ToolError("sequence", "请先调用 scale_features")
    k = int(args["k"])
    seed = int(args.get("random_state", 42))
    result = run_kmeans(
        session.cluster_scaled,
        k,
        random_state=seed,
        n_init=int(args.get("n_init", 10)),
        max_iter=int(args.get("max_iter", 300)),
        tol=float(args.get("tol", 1e-4)),
    )
    session.cluster_runs[(k, seed)] = result
    counts = pd.Series(result.labels).value_counts().sort_index()
    return {
        "status": result.status,
        "k": k,
        "random_state": seed,
        "inertia": round(float(result.inertia), 6),
        "iterations": int(result.iterations),
        "cluster_sizes": {str(int(label)): int(count) for label, count in counts.items()},
        "next": "请调用 evaluate_clusters 计算轮廓系数与稳定性。",
    }


@tool(
    "evaluate_clusters",
    "clustering",
    "计算轮廓系数（固定抽样）、CH、DBI 与多种子 ARI 稳定性。",
    {
        "type": "object",
        "properties": {
            "k": K_PARAM,
            "stability_seeds": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 4294967295},
                "default": [7, 21],
            },
            "sample_size": {"type": "integer", "minimum": 100, "default": 2000},
        },
        "required": ["k"],
        "additionalProperties": False,
    },
)
def evaluate_clusters_tool(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if session.cluster_scaled is None:
        raise ToolError("sequence", "请先完成标准化与 run_kmeans")
    k = int(args["k"])
    main = session.cluster_runs.get((k, 42))
    if main is None:
        raise ToolError("sequence", f"请先以 random_state=42 运行 k={k} 的 run_kmeans")
    stability = []
    for seed in args.get("stability_seeds") or [7, 21]:
        run = session.cluster_runs.get((k, int(seed)))
        if run is None:
            run = run_kmeans(session.cluster_scaled, k, random_state=int(seed))
            session.cluster_runs[(k, int(seed))] = run
        stability.append(run)
    metrics = evaluate_clusters(
        session.cluster_scaled,
        main,
        stability_runs=stability,
        sample_size=int(args.get("sample_size", 2000)),
        sample_seed=42,
    )
    metrics.update(
        {
            "k": k,
            "dataset_id": session.cluster_scaled.dataset_id,
            "dataset_version": session.cluster_scaled.version,
            "data_sha256": session.cluster_scaled.data_sha256,
            "matrix_sha256": session.cluster_scaled.matrix_sha256,
            "scaler_sha256": session.cluster_scaled.scaler_sha256,
        }
    )
    session.cluster_metrics[k] = metrics
    return {
        "k": k,
        "silhouette": round(metrics["silhouette"], 6),
        "silhouette_sampling": metrics["silhouette_sampling"],
        "calinski_harabasz": round(metrics["calinski_harabasz"], 4),
        "davies_bouldin": round(metrics["davies_bouldin"], 6),
        "inertia": round(metrics["inertia"], 6),
        "clusters": metrics["clusters"],
        "stability": metrics["stability"],
        "note": "轮廓系数为固定种子抽样值；k 不代表业务最优，也不能解释为健康等级。",
    }


@tool(
    "analyze_clusters",
    "clustering",
    "输出各簇营养均值/中位数、类别分布和最接近中心的代表食品。",
    {
        "type": "object",
        "properties": {
            "k": K_PARAM,
            "representatives": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
        },
        "required": ["k"],
        "additionalProperties": False,
    },
)
def analyze_clusters_tool(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if session.cluster_scaled is None:
        raise ToolError("sequence", "请先完成标准化与 run_kmeans")
    k = int(args["k"])
    run = session.cluster_runs.get((k, 42))
    if run is None:
        raise ToolError("sequence", f"请先运行 k={k} 的 run_kmeans")
    analysis = analyze_clusters(
        session.dataset_input(), session.cluster_scaled, run,
        representatives=int(args.get("representatives", 3)),
    )
    session.cluster_analysis[k] = analysis
    path = session.run_dir / "clustering" / f"analysis_k{k}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = []
    for cluster in analysis["clusters"]:
        nutrition = cluster["nutrition"]
        compact.append(
            {
                "label": cluster["label"],
                "count": cluster["count"],
                "proportion": round(cluster["proportion"], 6),
                "kcal_mean": round(nutrition["energy-kcal_100g"]["mean"], 3),
                "fat_mean": round(nutrition["fat_100g"]["mean"], 3),
                "sugars_mean": round(nutrition["sugars_100g"]["mean"], 3),
                "sodium_mean": round(nutrition["sodium_100g"]["mean"], 3),
                "proteins_mean": round(nutrition["proteins_100g"]["mean"], 3),
                "top_categories": cluster["categories"][:3],
                "representatives": [
                    {"product_name": item["product_name"], "category": item["category"], "distance": round(item["distance_standardized"], 4)}
                    for item in cluster["representatives"]
                ],
            }
        )
    return {
        "k": k,
        "sample_count": analysis["sample_count"],
        "clusters": compact,
        "analysis_file": str(path),
        "note": "簇只代表营养相似群体，不代表健康等级。",
    }


@tool(
    "visualize_clusters",
    "clustering",
    "生成 PCA 投影、标准化簇均值热图、簇大小和 k 指标曲线（k=2..8）。",
    {
        "type": "object",
        "properties": {
            "k": K_PARAM,
            "sample_size": {"type": "integer", "minimum": 100, "default": 5000},
        },
        "required": ["k"],
        "additionalProperties": False,
    },
)
def visualize_clusters_tool(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if session.cluster_scaled is None:
        raise ToolError("sequence", "请先完成标准化与 run_kmeans")
    k = int(args["k"])
    run = session.cluster_runs.get((k, 42))
    if run is None:
        raise ToolError("sequence", f"请先运行 k={k} 的 run_kmeans")
    analysis = session.cluster_analysis.get(k)
    if analysis is None:
        raise ToolError("sequence", f"请先调用 analyze_clusters(k={k})")
    rows = []
    for item in range(2, 9):
        metrics = session.cluster_metrics.get(item)
        if metrics is None:
            cached = session.cluster_runs.get((item, 42))
            if cached is None:
                cached = run_kmeans(session.cluster_scaled, item, random_state=42)
                session.cluster_runs[(item, 42)] = cached
            metrics = evaluate_clusters(session.cluster_scaled, cached, sample_size=2000, sample_seed=42)
            metrics.update(
                {
                    "k": item,
                    "dataset_id": session.cluster_scaled.dataset_id,
                    "dataset_version": session.cluster_scaled.version,
                    "data_sha256": session.cluster_scaled.data_sha256,
                    "matrix_sha256": session.cluster_scaled.matrix_sha256,
                    "scaler_sha256": session.cluster_scaled.scaler_sha256,
                }
            )
            session.cluster_metrics[item] = metrics
        rows.append(
            {
                "k": metrics["k"],
                "silhouette": metrics["silhouette"],
                "silhouette_sampling": metrics["silhouette_sampling"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"],
            }
        )
    plots_dir = session.run_dir / "plots"
    if any(plots_dir.glob("*.png")):
        raise ToolError("exists", "本次运行已生成过可视化，避免覆盖")
    metrics_payload = {
        "dataset_id": session.cluster_scaled.dataset_id,
        "dataset_version": session.cluster_scaled.version,
        "data_sha256": session.cluster_scaled.data_sha256,
        "matrix_sha256": session.cluster_scaled.matrix_sha256,
        "scaler_sha256": session.cluster_scaled.scaler_sha256,
        "results": rows,
    }
    plot_data = visualize_clusters(
        session.cluster_scaled,
        run,
        analysis,
        metrics_payload,
        plots_dir,
        sample_size=int(args.get("sample_size", 5000)),
    )
    files = sorted(str(path) for path in plots_dir.glob("*.png"))
    return {
        "status": "visualized",
        "k": k,
        "files": files,
        "explained_variance_ratio": [round(float(v), 6) for v in plot_data["pca"]["explained_variance_ratio"]],
        "k_metric_rows": [
            {"k": row["k"], "silhouette": round(row["silhouette"], 6)} for row in rows
        ],
    }


# ---------------------------------------------------------------- termination
@tool(
    "submit_final_report",
    "validation",
    "提交最终结论并结束任务。控制层会校验：已清洗、清洗后已验证、已完成聚类评价。",
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 1},
            "cleaning_rationale": {"type": "string", "minLength": 1},
            "cluster_interpretation": {"type": "string", "minLength": 1},
            "limitations": {"type": "string", "minLength": 1},
        },
        "required": ["summary", "cleaning_rationale", "cluster_interpretation", "limitations"],
        "additionalProperties": False,
    },
)
def submit_final_report(session: AgentSession, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "cleaning_steps": len(session.steps),
        "last_step": session.last_step_index(),
        "last_validation_after_step": session.last_validation_index(),
        "evaluated_k": sorted(session.cluster_metrics),
        "analysis_k": sorted(session.cluster_analysis),
    }
    failures = []
    if len(session.steps) == 0:
        failures.append("尚未执行任何清洗")
    if session.last_validation_index() < session.last_step_index():
        failures.append("最后一次清洗后没有调用 validate_quality")
    if not session.cluster_metrics:
        failures.append("尚未完成聚类评价")
    if not session.cluster_analysis:
        failures.append("尚未调用 analyze_clusters 解释簇")
    if failures:
        raise ToolError(
            "finish_gate",
            "；".join(failures) + "。请先完成这些步骤再提交",
            details=gates,
        )
    audit = session.safety_audit()
    if not audit["unique_row_ids"] or not audit["row_ids_subset_of_source"]:
        raise ToolError("row_id_audit", "row id 完整性校验失败，不能提交", details=audit)
    if audit["retained_rate"] < 0.5:
        raise ToolError(
            "retention_gate",
            f"保留率仅 {audit['retained_rate']:.2%}，低于 50%，删除过多，不能提交",
            details=audit,
        )
    session.final_report = {
        "summary": args["summary"],
        "cleaning_rationale": args["cleaning_rationale"],
        "cluster_interpretation": args["cluster_interpretation"],
        "limitations": args["limitations"],
        "gates": gates,
        "audit": audit,
    }
    return {
        "status": "accepted",
        "message": "控制层校验通过，运行结束。",
        "audit": audit,
    }
