# 数据清洗策略矩阵

清洗工具不预设“默认正确做法”。检测工具负责找出问题并生成完整选择表，大模型根据字段含义、缺失率、业务规则和下游聚类任务，从本矩阵中选择一个策略并给出参数。最终由确定性代码执行，禁止执行大模型临时生成的代码。

## 异常处理不能一刀切

正式 `off_52k_v3` 到 `off_52k_v5` 采用分层策略：

| 问题性质 | 采用策略 | 原因 |
|---|---|---|
| 明显不可能的硬范围值，例如每 100 克超过 100 克、能量小于 0 | 先隔离并置空，再用配对字段公式或中位数补齐 | 原值不可能成立，但整行不一定无效 |
| 箱线图统计尾部 | 截断到该字段检测出的上下界；复检后如仍超界，再按新分布截断一轮 | 极端值可能是真实高糖/高盐食品，优先保留观测而不是删行 |
| 能量千卡/千焦、盐/钠冲突 | 确定主字段后重算派生字段 | 两列存在确定换算关系 |
| 糖/碳水、饱和脂肪/总脂肪冲突 | 把子字段压到 `父字段 * 1.05 + 0.5` 的允许上界 | 关系明确，但没有唯一精确换算公式 |
| 整行不可信 | 隔离整行 | 仅用于多条关键字段同时失效，不能默认使用 |

这些策略都必须使用 finding 的完整选择表，不能用 JSON 中的样例行代替。

## 统一约束

- 原始版本只读；清洗结果写入 `data/processed/` 下的新版本。
- 每个工具必须接收 `finding_id` 和 `selection_artifact_id`，不能只依赖报告里的样例行。
- 标识符默认禁止统计插补。
- 标签不能使用普通数值插补器；不可用标签与非法标签分开处理。
- 删除行、覆盖标签、批量插补等高风险动作必须 `dry_run` 或显式 `confirm=true`。
- 每个动作都产生 `ChangeSet`、完整变更明细和清洗后版本 manifest。
- 每个正式写入的数据版本同时生成可直接加载的 profile；下一步清洗和重新检测都使用这个新 profile。
- 多动作计划通过稳定 `row_id` 重新定位选择表，不能沿用上一版本的行号位置。

## 1. 缺失值 `clean_missing`

| 策略 | 风险 | 是否改数据 | 关键参数 | 适用场景 |
|---|---|---|---|---|
| `keep_missing` | safe | 否 | 无 | 字段与聚类无关，或应保留“未知”含义 |
| `drop_row` | dangerous | 是 | `max_rows` | 缺失集中在关键字段，删少量行可接受 |
| `fill_constant` | caution | 是 | `value` | 领域已有明确默认值，例如“未提供” |
| `impute_statistic` | caution | 是 | `method`, `quantile` | 数值用均值/中位数，类别用众数 |
| `impute_grouped` | caution | 是 | `method`, `group_columns`, `min_group_size` | 同品牌/同类别内更可比 |
| `derive_from_formula` | caution | 是 | `target_column`, `expression`, `round_digits` | 字段间存在确定换算关系 |
| `impute_model` | dangerous | 是 | — | 当前未实现；没有真实模型时不伪装执行 |
| `quarantine_cell` | caution | 是 | `artifact_name` | 不猜测值，把可疑单元格移入隔离表 |
| `quarantine_row` | dangerous | 是 | `artifact_name` | 整行不可信，需移出主表 |

## 2. 重复数据 `clean_duplicates`

| 策略 | 风险 | 是否改数据 | 关键参数 | 适用场景 |
|---|---|---|---|---|
| `keep_first` | caution | 是 | 无 | 重复组顺序有意义，保留原始首条 |
| `keep_last` | caution | 是 | 无 | 后出现记录更新 |
| `keep_most_complete` | caution | 是 | 无 | 保留信息最完整的记录 |
| `keep_by_quality` | caution | 是 | `quality_column`, `ascending` | 有质量分/完整度分 |
| `keep_by_priority` | caution | 是 | `priority_column`, `ascending` | 有来源优先级 |
| `merge_group` | caution | 是 | `prefer_column` | 组内各字段互补，需要合并 |
| `quarantine_group` | dangerous | 是 | `keep_candidate`, `artifact_name` | `keep_candidate=true` 时隔离冗余行并保留一条候选；`false` 时隔离整组 |
| `no_action` | safe | 否 | 无 | 仅记录，人工复核 |

当前 `off_52k_v1` 的重复定义是 `code` 相同；2,000 组、每组 2 行，候选删除 2,000 行。

## 3. 异常值/噪声 `clean_outliers`

| 策略 | 风险 | 是否改数据 | 关键参数 | 适用场景 |
|---|---|---|---|---|
| `flag_only` | safe | 否 | 无 | 统计尾部可能真实存在 |
| `clip_value` | caution | 是 | `minimum`, `maximum` | 已知业务边界，保留行但截断 |
| `set_null` | caution | 是 | 无 | 让后续缺失策略统一处理 |
| `quarantine_cell` | caution | 是 | `artifact_name` | 保留证据，主表置空 |
| `quarantine_row` | dangerous | 是 | `artifact_name` | 整行不可信 |
| `drop_row` | dangerous | 是 | `max_rows` | 确认是错误记录 |
| `recompute_by_formula` | caution | 是 | `target_column`, `expression` | 可由其他字段重算 |
| `correct_unit_or_scale` | caution | 是 | `column`, `factor`, `offset` | 单位/量纲错误 |

## 4. 类型与格式 `normalize_types_formats`

| 策略 | 风险 | 是否改数据 | 关键参数 | 适用场景 |
|---|---|---|---|---|
| `trim_whitespace` | safe | 是 | 无 | 文本首尾有空白 |
| `normalize_case` | safe | 是 | `case` | 大小写不统一 |
| `cast_numeric` | caution | 是 | `invalid` | 数字被存成文本 |
| `cast_datetime` | caution | 是 | `datetime_format`, `dayfirst`, `yearfirst`, `invalid` | 日期解析 |
| `map_boolean` | caution | 是 | `true_values`, `false_values`, `invalid` | 是/否写法不统一 |
| `preserve_identifier_as_text` | safe | 是 | 无 | 条码/编号前导零 |
| `normalize_date_format` | caution | 是 | `format` | 日期展示统一 |
| `set_invalid_null` | caution | 是 | 无 | 无法解析且不应猜测 |
| `quarantine_invalid` | caution | 是 | `artifact_name` | 保留无效值证据 |
| `leave_unchanged` | safe | 否 | 无 | 格式不标准但业务可接受 |

## 5. 字段冲突 `resolve_conflicts`

| 策略 | 风险 | 是否改数据 | 关键参数 | 适用场景 |
|---|---|---|---|---|
| `flag_only` | safe | 否 | 无 | 先人工确认哪边错 |
| `recompute_derived` | caution | 是 | `target_column`, `expression` | 派生字段可由公式重算 |
| `trust_primary_source` | caution | 是 | `target_column`, `source_column` | 已确定主字段可信 |
| `null_conflicting_cells` | caution | 是 | `columns` | 不猜测，冲突字段置空 |
| `quarantine_row` | dangerous | 是 | `artifact_name` | 整行不可信 |
| `custom_resolver` | dangerous | 是 | — | 当前未实现；必须显式注册回调 |
| `no_action` | safe | 否 | 无 | 只记录 |

## 6. 标签 `repair_labels`

| 策略 | 风险 | 是否改数据 | 关键参数 | 适用场景 |
|---|---|---|---|---|
| `canonicalize_alias` | caution | 是 | 使用 profile 的 `labels.columns.aliases` | 拼写/别名统一 |
| `mark_unavailable` | safe | 是 | `marker` | 明确标成“不可用”而非猜标签 |
| `exclude_from_training` | safe | 否 | 无 | 不可用/非法标签不参与有监督训练 |
| `map_deterministic_value` | caution | 是 | `mapping` | 有明确一对一映射 |
| `quarantine_invalid` | caution | 是 | `artifact_name` | 非法标签移入隔离表 |
| `relabel_by_rule` | caution | 是 | `rules` | 有明确规则，不靠模型猜 |
| `no_action` | safe | 否 | 无 | 保留现状 |

## 7. 类别不平衡 `mitigate_imbalance`

| 策略 | 风险 | 是否改数据 | 关键参数 | 适用场景 |
|---|---|---|---|---|
| `report_only` | safe | 否 | `target_column` | 先报告分布 |
| `class_weight` | safe | 否 | `target_column` | 训练时用类别权重 |
| `stratified_split` | safe | 否 | `target_column`, `test_size`, `random_state` | 分层训练/验证 |
| `oversample_minority` | dangerous | 是 | `target_column`, `target_share`, `max_copies`, `random_state` | 允许复制少数类 |
| `undersample_majority` | dangerous | 是 | `target_column`, `target_share`, `max_rows`, `random_state` | 允许删除多数类 |
| `collect_more_samples` | safe | 否 | `target_column`, `target_share` | 只输出缺口，不伪造样本 |
| `no_action` | safe | 否 | 无 | 不处理 |

## 代码入口

- 策略注册与校验：`src/data_agent/cleaning/strategies.py`
- 公共执行与变更记录：`src/data_agent/cleaning/common.py`
- 七类工具：`fill_missing.py`、`remove_duplicates.py`、`handle_outliers.py`、`normalize_types_formats.py`、`resolve_conflicts.py`、`repair_labels.py`、`mitigate_imbalance.py`
- 统一分发：`src/data_agent/cleaning/__init__.py`
- 多动作计划：`src/data_agent/cleaning/plan.py`
- 单步 CLI：`scripts/run_cleaning.py`
- 计划 CLI：`scripts/run_cleaning_plan.py`

## 执行方式

单步预览：

```powershell
python scripts/run_cleaning.py `
  --issue-type duplicate --strategy keep_first `
  --finding-id duplicate.primary_key `
  --selection-artifact-id selection.off_52k_v1.duplicate.primary_key
```

确认后写入新版本：

```powershell
python scripts/run_cleaning.py `
  --issue-type duplicate --strategy keep_first `
  --finding-id duplicate.primary_key `
  --selection-artifact-id selection.off_52k_v1.duplicate.primary_key `
  --no-dry-run
```

多动作计划：

```powershell
python scripts/run_cleaning_plan.py --plan my_plan.json
```

计划默认也是 dry-run。正式执行时加 `--no-dry-run`；包含危险策略时还要加 `--confirm`。多动作计划会形成 `v2 -> v3 -> ...` 的版本链，每一步都能单独审计。

多动作计划支持两种落盘方式：`--persist per_step`（默认）会形成
`v2 -> v3 -> ...` 的版本链，适合逐步调试；`--persist final` 会在内存中依次执行
全部动作，只写一个最终版本和一份汇总变更日志。正式交付建议使用
`--persist final --no-dry-run`。当前 `v3`、`v4`、`v5` 都是按这种方式生成的最终版本。

`ValidationReport.retained_rate` 是“清洗后行数 / 清洗前行数”，所以过采样时允许大于 1；它不再被错误地限制为留存比例。
