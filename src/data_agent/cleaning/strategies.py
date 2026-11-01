from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import IssueType, RiskLevel


class StrategyError(ValueError):
    pass


class StrategyNotAvailable(StrategyError):
    pass


class StrategyParameterError(StrategyError):
    pass


class StrategySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    strategy_id: str = Field(min_length=1)
    issue_type: IssueType
    description: str = Field(min_length=1)
    risk_level: RiskLevel = RiskLevel.CAUTION
    mutates_data: bool = True
    requires_confirmation: bool = False
    implemented: bool = True
    allowed_params: list[str] = Field(default_factory=list)
    required_params: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    notes: str | None = None


def _spec(
    strategy_id: str,
    issue_type: IssueType,
    description: str,
    *,
    risk: RiskLevel = RiskLevel.CAUTION,
    mutates_data: bool = True,
    requires_confirmation: bool = False,
    implemented: bool = True,
    allowed: tuple[str, ...] = (),
    required: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    notes: str | None = None,
) -> StrategySpec:
    return StrategySpec(
        strategy_id=strategy_id,
        issue_type=issue_type,
        description=description,
        risk_level=risk,
        mutates_data=mutates_data,
        requires_confirmation=requires_confirmation,
        implemented=implemented,
        allowed_params=list(allowed),
        required_params=list(required),
        aliases=list(aliases),
        notes=notes,
    )


MISSING_STRATEGIES = [
    _spec(
        "keep_missing",
        IssueType.MISSING,
        "保留缺失值，不改动数据；适合聚类无关或应保持原样的字段。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
    _spec(
        "drop_row",
        IssueType.MISSING,
        "删除存在选中缺失单元格的行。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("max_rows",),
    ),
    _spec(
        "fill_constant",
        IssueType.MISSING,
        "用一个明确的常量填充选中单元格。",
        allowed=("value",),
        required=("value",),
    ),
    _spec(
        "impute_statistic",
        IssueType.MISSING,
        "用均值、中位数、众数或指定分位数填充选中单元格。",
        allowed=("method", "quantile"),
        required=("method",),
    ),
    _spec(
        "impute_grouped",
        IssueType.MISSING,
        "按分组键计算组内统计量后填充。",
        allowed=("method", "group_columns", "quantile", "min_group_size"),
        required=("method", "group_columns"),
    ),
    _spec(
        "derive_from_formula",
        IssueType.MISSING,
        "用受控公式从其他字段推导目标字段。",
        allowed=("target_column", "expression", "round_digits"),
        required=("target_column", "expression"),
    ),
    _spec(
        "impute_model",
        IssueType.MISSING,
        "用模型或近邻方法做数值插补。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        implemented=False,
        allowed=("method", "neighbors", "features"),
        notes="当前运行环境没有 sklearn；保留策略位，但未实现，避免伪装成模型插补。",
    ),
    _spec(
        "quarantine_cell",
        IssueType.MISSING,
        "把选中单元格置空并写入隔离表，主表保留但不再携带可疑值。",
        allowed=("artifact_name",),
    ),
    _spec(
        "quarantine_row",
        IssueType.MISSING,
        "把含选中缺失值的整行移入隔离表，并从主表删除。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("artifact_name",),
    ),
]


DUPLICATE_STRATEGIES = [
    _spec(
        "keep_first",
        IssueType.DUPLICATE,
        "每组保留第一次出现的行。",
        risk=RiskLevel.CAUTION,
        aliases=("keep_first_occurrence",),
    ),
    _spec(
        "keep_last",
        IssueType.DUPLICATE,
        "每组保留最后一次出现的行。",
        risk=RiskLevel.CAUTION,
    ),
    _spec(
        "keep_most_complete",
        IssueType.DUPLICATE,
        "每组保留非空字段最多的行，并列时保留最先出现的行。",
        risk=RiskLevel.CAUTION,
    ),
    _spec(
        "keep_by_quality",
        IssueType.DUPLICATE,
        "按质量分数保留每组质量最高或最低的行。",
        allowed=("quality_column", "ascending"),
        required=("quality_column",),
    ),
    _spec(
        "keep_by_priority",
        IssueType.DUPLICATE,
        "按优先级字段保留每组优先级最高或最低的行。",
        allowed=("priority_column", "ascending"),
        required=("priority_column",),
    ),
    _spec(
        "merge_group",
        IssueType.DUPLICATE,
        "按字段合并重复组：优先保留非空值，必要时使用第一个值，并删除冗余行。",
        allowed=("prefer_column", "keep_row_id"),
    ),
    _spec(
        "quarantine_group",
        IssueType.DUPLICATE,
        "keep_candidate=true 时隔离冗余行并保留一条候选；false 时隔离整组。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("artifact_name", "keep_candidate"),
    ),
    _spec(
        "no_action",
        IssueType.DUPLICATE,
        "只记录重复问题，不做数据修改。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
]


OUTLIER_STRATEGIES = [
    _spec(
        "flag_only",
        IssueType.OUTLIER,
        "只保留异常标记，不修改数值。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
    _spec(
        "clip_value",
        IssueType.OUTLIER,
        "把越界数值截断到配置范围或指定上下界。",
        allowed=("minimum", "maximum"),
    ),
    _spec(
        "set_null",
        IssueType.OUTLIER,
        "把异常数值改为空值，交给后续缺失值策略处理。",
    ),
    _spec(
        "quarantine_cell",
        IssueType.OUTLIER,
        "把异常单元格置空并保存隔离明细。",
        allowed=("artifact_name",),
    ),
    _spec(
        "quarantine_row",
        IssueType.OUTLIER,
        "把含异常值的整行移入隔离表并从主表删除。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("artifact_name",),
    ),
    _spec(
        "drop_row",
        IssueType.OUTLIER,
        "直接删除含异常值的行。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("max_rows",),
    ),
    _spec(
        "recompute_by_formula",
        IssueType.OUTLIER,
        "用受控公式重算目标字段，并用结果替换异常值。",
        allowed=("target_column", "expression", "round_digits"),
        required=("target_column", "expression"),
    ),
    _spec(
        "correct_unit_or_scale",
        IssueType.OUTLIER,
        "按指定因子和偏移修正单位或量纲错误。",
        allowed=("column", "factor", "offset"),
        required=("column", "factor"),
    ),
]


TYPE_FORMAT_STRATEGIES = [
    _spec(
        "trim_whitespace",
        IssueType.FORMAT,
        "去除文本首尾空白。",
        risk=RiskLevel.SAFE,
    ),
    _spec(
        "normalize_case",
        IssueType.FORMAT,
        "统一文本大小写。",
        risk=RiskLevel.SAFE,
        allowed=("case",),
        required=("case",),
    ),
    _spec(
        "cast_numeric",
        IssueType.FORMAT,
        "把可解析文本转换为数值；无法解析的值按策略处理。",
        allowed=("invalid",),
    ),
    _spec(
        "cast_datetime",
        IssueType.FORMAT,
        "把文本转换为日期时间。",
        allowed=("datetime_format", "dayfirst", "yearfirst", "invalid"),
    ),
    _spec(
        "map_boolean",
        IssueType.FORMAT,
        "把声明的真/假词映射成布尔值。",
        allowed=("true_values", "false_values", "invalid"),
    ),
    _spec(
        "preserve_identifier_as_text",
        IssueType.FORMAT,
        "把标识符强制保存为字符串，保留前导零。",
        risk=RiskLevel.SAFE,
    ),
    _spec(
        "normalize_date_format",
        IssueType.FORMAT,
        "把日期统一成指定输出格式。",
        allowed=("format", "datetime_format", "dayfirst", "yearfirst"),
        required=("format",),
    ),
    _spec(
        "set_invalid_null",
        IssueType.FORMAT,
        "把无法修复的格式值置空。",
    ),
    _spec(
        "quarantine_invalid",
        IssueType.FORMAT,
        "把无效格式单元格移入隔离表并置空。",
        allowed=("artifact_name",),
    ),
    _spec(
        "leave_unchanged",
        IssueType.FORMAT,
        "保持原值，仅记录格式问题。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
]


CONFLICT_STRATEGIES = [
    _spec(
        "flag_only",
        IssueType.CONSTRAINT,
        "只记录冲突，不修改字段。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
    _spec(
        "recompute_derived",
        IssueType.CONSTRAINT,
        "按受控公式重算派生字段。",
        allowed=("target_column", "expression", "round_digits"),
        required=("target_column", "expression"),
        aliases=("recompute_derived_field",),
    ),
    _spec(
        "trust_primary_source",
        IssueType.CONSTRAINT,
        "把指定主字段的值复制到需要修正的字段。",
        allowed=("target_column", "source_column"),
        required=("target_column", "source_column"),
    ),
    _spec(
        "null_conflicting_cells",
        IssueType.CONSTRAINT,
        "把冲突行中选中的字段置空。",
        allowed=("columns",),
    ),
    _spec(
        "quarantine_row",
        IssueType.CONSTRAINT,
        "把冲突行移入隔离表并从主表删除。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("artifact_name",),
    ),
    _spec(
        "custom_resolver",
        IssueType.CONSTRAINT,
        "调用外部受控解析器处理冲突。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        implemented=False,
        notes="不允许大模型临时生成代码；需要显式注册的回调解算器。",
    ),
    _spec(
        "no_action",
        IssueType.CONSTRAINT,
        "只记录冲突，不做修改。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
]


LABEL_STRATEGIES = [
    _spec(
        "canonicalize_alias",
        IssueType.LABEL,
        "按标签别名表把写法统一成标准标签。",
    ),
    _spec(
        "mark_unavailable",
        IssueType.LABEL,
        "把不可用标签显式写成统一的不可用标记。",
        risk=RiskLevel.SAFE,
        allowed=("marker",),
    ),
    _spec(
        "exclude_from_training",
        IssueType.LABEL,
        "生成训练排除表；不伪造标签。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
        aliases=("exclude_from_supervised_training",),
    ),
    _spec(
        "map_deterministic_value",
        IssueType.LABEL,
        "按明确的一对一映射修正标签。",
        allowed=("mapping", "target_column"),
        required=("mapping",),
        aliases=("map_to_allowed_labels",),
    ),
    _spec(
        "quarantine_invalid",
        IssueType.LABEL,
        "把非法标签单元格置空并写入隔离表。",
        allowed=("artifact_name",),
    ),
    _spec(
        "relabel_by_rule",
        IssueType.LABEL,
        "按受控条件表达式重标标签。",
        allowed=("target_column", "rules"),
        required=("rules",),
    ),
    _spec(
        "no_action",
        IssueType.LABEL,
        "只记录标签问题，不做修改。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
]


IMBALANCE_STRATEGIES = [
    _spec(
        "report_only",
        IssueType.IMBALANCE,
        "只输出类别分布和不平衡报告。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
    _spec(
        "class_weight",
        IssueType.IMBALANCE,
        "计算并输出类别权重，不改变原始行。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
        allowed=("target_column",),
        aliases=("use_class_weights",),
    ),
    _spec(
        "stratified_split",
        IssueType.IMBALANCE,
        "生成分层训练/验证索引表，不改变原始行。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
        allowed=("target_column", "test_size", "random_state"),
    ),
    _spec(
        "oversample_minority",
        IssueType.IMBALANCE,
        "复制少数类行，使类别比例接近目标。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("target_column", "target_share", "random_state", "max_copies"),
    ),
    _spec(
        "undersample_majority",
        IssueType.IMBALANCE,
        "删除多数类行，使类别比例接近目标。",
        risk=RiskLevel.DANGEROUS,
        requires_confirmation=True,
        allowed=("target_column", "target_share", "random_state", "max_rows"),
    ),
    _spec(
        "collect_more_samples",
        IssueType.IMBALANCE,
        "只输出建议和缺口，不伪造样本。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
        allowed=("target_column", "target_share"),
        aliases=("collect_more_minority_samples",),
    ),
    _spec(
        "no_action",
        IssueType.IMBALANCE,
        "不做处理。",
        risk=RiskLevel.SAFE,
        mutates_data=False,
    ),
]


CATALOG: dict[IssueType, list[StrategySpec]] = {
    IssueType.MISSING: MISSING_STRATEGIES,
    IssueType.DUPLICATE: DUPLICATE_STRATEGIES,
    IssueType.OUTLIER: OUTLIER_STRATEGIES,
    IssueType.FORMAT: TYPE_FORMAT_STRATEGIES,
    IssueType.CONSTRAINT: CONFLICT_STRATEGIES,
    IssueType.LABEL: LABEL_STRATEGIES,
    IssueType.IMBALANCE: IMBALANCE_STRATEGIES,
}


def list_strategies(issue_type: IssueType | str | None = None) -> list[StrategySpec]:
    if issue_type is None:
        return [spec for specs in CATALOG.values() for spec in specs]
    issue = IssueType(issue_type)
    return list(CATALOG.get(issue, []))


def get_strategy(issue_type: IssueType | str, strategy_id: str) -> StrategySpec:
    issue = IssueType(issue_type)
    for spec in CATALOG.get(issue, []):
        if spec.strategy_id == strategy_id or strategy_id in spec.aliases:
            return spec
    raise StrategyNotAvailable(
        f"strategy {strategy_id!r} is not registered for issue type {issue.value!r}"
    )


def validate_strategy_params(spec: StrategySpec, params: dict[str, Any]) -> None:
    if not spec.implemented:
        raise StrategyNotAvailable(
            f"strategy {spec.strategy_id!r} is registered but not implemented"
        )
    unknown = set(params) - set(spec.allowed_params)
    if spec.allowed_params and unknown:
        raise StrategyParameterError(
            f"strategy {spec.strategy_id!r} does not accept parameters: {sorted(unknown)}"
        )
    missing = set(spec.required_params) - set(params)
    if missing:
        raise StrategyParameterError(
            f"strategy {spec.strategy_id!r} requires parameters: {sorted(missing)}"
        )


__all__ = [
    "CATALOG",
    "StrategyError",
    "StrategyNotAvailable",
    "StrategyParameterError",
    "StrategySpec",
    "get_strategy",
    "list_strategies",
    "validate_strategy_params",
]
