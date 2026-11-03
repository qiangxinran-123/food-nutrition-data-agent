# 清洗修改说明（给队友的同步清单）

更新时间：2026-11-01

## 1. 一句话结论

最终机器读取版本已经从 `off_52k_v2` 更新为 `off_52k_v5`。本次不是继续“只标记”，而是按异常性质分别处理：硬范围错误先置空再补，箱线图尾部做截断，字段冲突按关系重算或压到允许上界。

最终复检结果：

- 行数：50,000，未删除行
- 重复数据：0
- 硬范围违规：0
- 箱线图统计尾部：0
- 字段冲突：0
- 缺失单元格：173,016，和 `v2` 相同
- 非标准条码：197 行，和 `v2` 相同

## 2. 最终文件和版本链

### 2.1 当前正式交付文件

| 用途 | 文件 |
|---|---|
| 聚类/建模数据 | `data/processed/off_52k_v5.parquet` |
| 字段规则 | `config/profiles/off_52k_v5.json` |
| 版本清单 | `data/processed/off_52k_v5.manifest.json` |
| 最终变更日志 | `data/processed/change.off_52k_v5_precision_finalization.changes.parquet` |
| 清洗后复检 | `logs/checks_v5/off_52k_v5_quality_summary.md` |
| 最终 SHA256 | `7b68f485d65e143716e5f0ecd588755ee47a07f9c88763e97f0e5fff75179a70` |

### 2.2 版本链

| 版本 | 计划 | 作用 |
|---|---|---|
| `v1` | 冻结数据 | 52,000 行，含 2,000 行人工重复数据 |
| `v2` | `off_52k_v2_conservative` | 保守基线：去重、补数值空缺，异常和冲突只标记 |
| `v3` | `off_52k_v3_stratified_repair` | 第一次分层修复：硬范围置空后补、IQR 截断、冲突重算/压边界 |
| `v4` | `off_52k_v4_convergence` | 第二次 IQR 截断，盐/钠按关系同步截断 |
| `v5` | `off_52k_v5_precision_finalization` | 修正最后 1 个浮点边界冲突 |

注意：`v5` 的变更日志只有 1 个单元格。要追溯完整清洗过程，必须同时看 `v3`、`v4`、`v5` 三份变更日志，不能只看 `v5` 那一份。

## 3. 本次实际改了什么

### 3.1 硬范围异常

对每 100 克超过 100 克、能量小于 0、能量超过业务范围等“明显不可能”的值：

1. 把原值写入隔离表；
2. 把主表单元格置空；
3. 能由配对字段推导的，用公式补；
4. 不能推导的，用该字段中位数补。

涉及字段：能量、蛋白质、脂肪、饱和脂肪、碳水、糖、纤维、盐、钠。

### 3.2 箱线图异常

不删除行，只把高出上界的值截断到上界，低于下界的值截断到下界。

由于截断后分布会变化，阈值也会变化，所以做了两轮：

- `v3`：第一轮截断；
- `v4`：按新分布再截断一轮。

### 3.3 字段冲突

| 冲突 | 处理方式 |
|---|---|
| 能量千卡 vs 千焦 | 以千焦为主，重算千卡：`kcal = kJ / 4.184` |
| 盐 vs 钠 | 以钠为主，重算盐：`salt = sodium * 2.5` |
| 糖 vs 碳水 | 糖压到 `carbohydrates * 1.05 + 0.5` 的允许上界 |
| 饱和脂肪 vs 总脂肪 | 饱和脂肪压到 `fat * 1.05 + 0.5` 的允许上界 |

盐/钠在截断时还做了关系同步：先按盐的上界反推钠的上界，再重算盐，避免截断后重新产生冲突。

### 3.4 标签与类别不平衡

#### 标签错误

`nutriscore_grade` 的合法值只有 `a`、`b`、`c`、`d`、`e`。`v5` 复检结果：

| 项目 | 数量 | 处理 |
|---|---:|---|
| 合法标签 | 21,978 | 可用于有监督训练 |
| 不可用标签 | 28,022 | 排除出有监督训练 |
| 非法标签 | 0 | 无需隔离或映射 |

不可用标签包括 `unknown` 26,482 行、`not-applicable` 1,539 行、空值 1 行。处理原则是**不猜标签、不插补标签、不删除行**：生成训练排除表，训练和评价时按行过滤。排除表共 28,022 行，字段 `excluded_from_training` 全部为 `True`：

`logs/cleaning/off52k_v2.label.exclude_from_training.nutriscore_grade.training_exclusions.parquet`

`nutriscore_grade` 是外部评价标签，不是普通聚类特征。不能把 `unknown` 当成第六个类别，也不能用 `a` 到 `e` 的均值或众数去补。

#### 类别不平衡

排除不可用标签后，可用于分类的样本是 21,978 行，分布如下：

| 类别 | 数量 | 占比 |
|---|---:|---:|
| `e` | 6,196 | 28.19% |
| `d` | 5,675 | 25.82% |
| `c` | 4,507 | 20.51% |
| `a` | 3,203 | 14.57% |
| `b` | 2,397 | 10.91% |

少数类是 `b`，占比 10.91%，低于 15% 的告警阈值；最大类/最小类比例为 2.5849，未达到 10 倍的高风险阈值，因此定为中等严重度。处理原则是**不做过采样、不做欠采样、不增删行**，保留真实分布，并输出反频率类别权重供训练使用：

| 类别 | 权重 |
|---|---:|
| `e` | 0.709425 |
| `d` | 0.774555 |
| `c` | 0.975283 |
| `a` | 1.372338 |
| `b` | 1.833792 |

权重文件：

`logs/cleaning/off52k_v2.imbalance.class_weight.class_weights.parquet`

建模时应使用分层切分，并把上述权重传给损失函数；后续如果收集到更多 `b` 类样本，再重新计算分布和权重。

说明：这两个产物文件名仍带 `v2`，是因为 `v2` 到 `v5` 的标签列和行集合没有变化，文件内容对 `v5` 仍然有效。`v3` 到 `v5` 只修复数值范围和字段关系，没有修改标签，也没有改变类别分布。对应检测结果在 `logs/checks_v5/off_52k_v5_labels.json` 和 `logs/checks_v5/off_52k_v5_imbalance.json`。

### 3.5 改动量

| 阶段 | 决策数 | 修改单元格 |
|---|---:|---:|
| `v3` | 37 | 9,630 |
| `v4` | 7 | 3,278 |
| `v5` | 1 | 1 |
| 合计 | 45 | 12,909 |

行数始终为 50,000，没有删除行。

## 4. 文件变更清单

### 4.1 新增清洗计划和可复现脚本

| 文件 | 说明 |
|---|---|
| `config/plans/off_52k_v3_stratified_repair.json` | 37 步分层修复计划 |
| `config/plans/off_52k_v4_convergence.json` | 7 步收敛计划 |
| `config/plans/off_52k_v5_precision_finalization.json` | 1 步浮点精度收尾计划 |
| `scripts/build_off_52k_v3_plan.py` | 从 `v2` 检测结果生成 `v3` 计划 |
| `scripts/build_off_52k_v4_plan.py` | 从 `v3` 复检结果生成 `v4` 计划 |

### 4.2 新增数据和日志

每个版本都新增了 parquet、manifest、profile 和变更日志，没有覆盖旧版本：

- `data/processed/off_52k_v3.*`、`off_52k_v4.*`、`off_52k_v5.*`
- `config/profiles/off_52k_v3.json`、`off_52k_v4.json`、`off_52k_v5.json`
- `data/processed/change.off_52k_v3_stratified_repair.changes.parquet`
- `data/processed/change.off_52k_v4_convergence.changes.parquet`
- `data/processed/change.off_52k_v5_precision_finalization.changes.parquet`
- `logs/cleaning/off_52k_v3_stratified_repair.plan_result.json`
- `logs/cleaning/off_52k_v4_convergence.plan_result.json`
- `logs/cleaning/off_52k_v5_precision_finalization.plan_result.json`
- `logs/cleaning/` 下对应的 `*.details.parquet`、隔离表和汇总表
- `logs/checks_v3/`、`logs/checks_v4/`、`logs/checks_v5/`

### 4.3 更新文档

- `data/processed/README.md`：改为说明 `v5` 是当前正式版本，并列出 `v2 → v5` 版本链。
- `docs/检测与清洗工具交接说明.md`：新增 `0.4 2026-11-01 正式清洗修订版 v5`。
- `docs/cleaning_strategy_matrix.md`：新增“异常处理不能一刀切”的分层策略表。

### 4.4 此前已完成的画像调整

这部分不是本轮 `v3 → v5` 清洗，但如果队友负责检测/画像说明，需要同步：

- `src/data_agent/profiling.py`：不再重复扫描缺失率、唯一值、分位数等统计量，只负责把配置规则合并成运行时字段画像。
- `src/data_agent/ingestion.py`：删除未使用的统计辅助函数。
- `tests/test_profiling.py`：改为验证“画像不重复扫描统计量”。
- `src/data_agent/cleaning/common.py`：只调整了注释措辞。

## 5. 队友需要同步改的地方

### 5.1 代码和 Notebook

- 把所有 `off_52k_v2.parquet` 换成 `off_52k_v5.parquet`。
- 把所有 `config/profiles/off_52k_v2.json` 换成 `off_52k_v5.json`。
- 如果有硬编码的 dataset id，把 `off_52k_v2` 换成 `off_52k_v5`。
- 如果绘图、聚类结果或报告缓存是 `v2` 生成的，需要重新跑。
- 如果读取 manifest 或校验哈希，更新为 `off_52k_v5.manifest.json`。

### 5.2 PPT 和报告

- 把“保守清洗版只标记异常/冲突”改为“v2 是保守基线，v5 已完成分层修复”。
- 把异常值、范围违规、字段冲突的数字更新为 `v5` 的 0。
- 如果保留前后对比，建议写：`v1 原始 → v2 保守基线 → v5 最终清洗`，不要只写 `v2 → v5`。
- 缺失值 173,016、非标准条码 197、不可用标签 28,022、少数类占比 10.91% 这些数字没有因为本轮修复而消失，不要写成“全部问题清零”。

### 5.3 Excel 和演示

- `outputs/off_52k_v2_view/` 里的 Excel 还是旧的 `v2` 查看副本，不是 `v5`。
- 演示时不要拿旧 Excel 和 `v5` parquet 混用。
- 如果要用 `v5` 做 Excel 展示，需要重新导出 `v5` 查看副本；正式机器读取仍然只认 parquet + profile。

### 5.4 变更日志

- 不要只看 `change.off_52k_v5_precision_finalization.changes.parquet`，它只记录最后 1 个单元格。
- 完整追溯顺序是：`v3` 变更日志 → `v4` 变更日志 → `v5` 变更日志。
- 查某个值为什么被改时，看 `change_type`、`reason`、`old_value`、`new_value` 和 `is_imputed`。

## 6. 本次没有改的部分

- 没有改动七个检测器的检测规则，只新增清洗计划和复检。
- 没有新增清洗策略代码，全部使用现有策略：隔离单元格、公式重算、统计量填补、IQR 截断、关系压边界。
- 没有填补文本缺失：品牌、品类、商品名、份量等仍然保留空值。
- 没有修复 197 行非标准条码。
- 没有伪造 Nutri-Score 标签，不可用标签继续排除出训练。
- 没有对类别不平衡做过采样或欠采样。
- 没有删除任何行。

## 7. 复现与验收

重新生成计划：

```powershell
python scripts/build_off_52k_v3_plan.py
python scripts/build_off_52k_v4_plan.py
```

执行 v3：

```powershell
python scripts/run_cleaning_plan.py `
  --plan config/plans/off_52k_v3_stratified_repair.json `
  --profile config/profiles/off_52k_v2.json `
  --selection-dir logs/checks_v2/selections `
  --artifact-dir logs/cleaning `
  --output-dir data/processed `
  --persist final --no-dry-run
```

执行 v4：

```powershell
python scripts/run_cleaning_plan.py `
  --plan config/plans/off_52k_v4_convergence.json `
  --profile config/profiles/off_52k_v3.json `
  --selection-dir logs/checks_v3/selections `
  --artifact-dir logs/cleaning `
  --output-dir data/processed `
  --persist final --no-dry-run
```

执行 v5：

```powershell
python scripts/run_cleaning_plan.py `
  --plan config/plans/off_52k_v5_precision_finalization.json `
  --profile config/profiles/off_52k_v4.json `
  --selection-dir logs/checks_v4/selections `
  --artifact-dir logs/cleaning `
  --output-dir data/processed `
  --persist final --no-dry-run
```

最终验收：

```powershell
python scripts/run_quality_checks.py `
  --profile config/profiles/off_52k_v5.json `
  --output-dir logs/checks_v5
```

验收标准：`off_52k_v5` 的重复、硬范围违规、箱线图尾部、字段冲突均为 0；缺失值、条码格式、标签和类别不平衡按报告保留。自动化测试 65 个，全部通过。
