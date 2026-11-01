# 任务一：Open Food Facts 数据集选择报告

> 后续更新说明：本报告保留最初选择 50,000 条样本时的统计结果。项目当前
> 已使用 52,000 行数据集：原始 50,000 行加 2,000 行随机复制的重复数据，
> 当前文件为 `data/raw/openfoodfacts_nutrition_sample_52000.xlsx` 和
> `data/base/off_52k_v1.parquet`，重复判定只比较 `code`。

## 1. 最终选定数据集

- **数据集**：Open Food Facts 全球食品产品数据库（英文 CSV 导出）
- **原始文件**：`raw/en.openfoodfacts.org.products.csv/en.openfoodfacts.org.products.csv`
- **文件格式**：UTF-8、制表符分隔（TSV）
- **本地文件大小**：13,908,928,036 bytes
- **统计时间**：2026-10-28
- **官方数据页**：https://world.openfoodfacts.org/data
- **直接下载地址**：https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
- **数据字段说明**：https://static.openfoodfacts.org/data/data-fields.txt

本地原始文件保留不修改。项目分析副本采用固定规则从原始文件抽取，见
`openfoodfacts_nutrition_sample_50000.csv`。

## 2. 规模和字段核查

全量流式统计结果：

| 检查项 | 结果 | 课程要求 | 结论 |
|---|---:|---:|---|
| 产品记录数 | 4,879,512 | >= 5,000 | 通过 |
| 原始字段数 | 211 | >= 10 | 通过 |
| 有商品编码记录 | 4,879,512 | 有可用标识 | 通过 |
| 有产品名称记录 | 4,534,948 | 产品可识别 | 通过 |
| 分析副本记录数 | 50,000 | >= 5,000 | 通过 |
| 分析副本字段数 | 19 | >= 10 | 通过 |

分析副本字段包括商品编码、产品名称、品牌、食品分类、热量、蛋白质、脂肪、饱和脂肪、碳水、糖、纤维、盐、钠、Nutri-Score、每份规格等。

## 3. 数据质量问题证据

### 缺失值

全量统计中主要营养字段的缺失数：

| 字段 | 缺失数 |
|---|---:|
| `fiber_100g` | 3,004,149 |
| `energy-kcal_100g` | 1,371,343 |
| `proteins_100g` | 1,397,839 |
| `fat_100g` | 1,403,313 |
| `carbohydrates_100g` | 1,401,518 |
| `sugars_100g` | 1,609,003 |
| `salt_100g` | 1,828,667 |
| `sodium_100g` | 1,828,662 |

### 重复数据

- 完全重复（选定字段）：0 条。
- 商品编码重复：56 个编码，超出首条的记录共 56 条。
- 后续需要根据产品版本、更新时间和业务规则区分合法版本差异，不能直接全部删除。

### 异常值和逻辑冲突

全量宽范围筛查发现：

- `energy-kcal_100g` 超出 0--1000：16,564 条。
- `energy_100g` 超出 0--5000：12,449 条。
- `fat_100g` 超出 0--100：6,538 条。
- `carbohydrates_100g` 超出 0--100：14,804 条。
- `sugars_100g` 超出 0--100：6,673 条。
- `proteins_100g` 超出 0--100：2,053 条。

在固定 50,000 条分析副本中，发现：

- 饱和脂肪大于总脂肪：39 条。
- 糖大于碳水化合物：43 条。
- 钠大于盐：45 条。
- 蛋白质、脂肪、碳水三者之和大于 100g：525 条。
- 负营养值：3 条。

这些记录先标记为疑似问题，后续由质量检测模块结合单位、食品类别和来源进行复核，不直接当作错误自动删除。

### 格式和计量口径不一致

50,000 条分析副本中：

- `serving_size` 非空 13,298 条；
- 该字段有 4,836 种不同文本格式；
- 常见写法同时包括 `100g`、`100 g`、`100.0g`、`1 serving (100 g)`、`1 ONZ (28 g)` 等。

这为后续单位/口径规范化和格式一致性检测提供了真实样本。

## 4. Nutri-Score 和下游任务

全量 `nutriscore_grade` 统计：

| 标签 | 记录数 |
|---|---:|
| A | 209,267 |
| B | 164,838 |
| C | 299,336 |
| D | 358,448 |
| E | 400,081 |
| `not-applicable` | 109,590 |
| `unknown` | 3,271,504 |

Nutri-Score 的有效标签数量足以支持后续标签分布分析，但 `unknown` 比例较高。因此项目主下游任务建议采用营养数值特征聚类；Nutri-Score 作为可选的外部标签，用于结果解释或子集分析，不把缺失标签强行补齐。

## 5. 分析副本抽样规则

为了避免后续工具加载 14GB 原始文件，使用 `MD5(code) % 100 < 2`，并要求 `code` 和 `product_name` 非空，固定抽取 50,000 条记录。该规则写在 `analysis/analyze_openfoodfacts.py` 中，任何组员都可以从同一版本原始文件重新生成相同副本。

## 6. 选用结论

Open Food Facts 满足课程要求的记录规模和字段数量，并且真实包含缺失值、重复编码、异常营养值、字段逻辑冲突以及计量格式不一致问题。它同时提供丰富营养特征和部分 Nutri-Score 标签，能够支撑后续质量检测、数据清洗、营养特征聚类、固定规则与智能体对照实验，以及消融实验。因此选定 Open Food Facts 作为本项目数据集。

## 7. 当前阶段文件

- `analysis/openfoodfacts_quality_report.json`：全量质量统计。
- `analysis/openfoodfacts_field_dictionary.csv`：项目字段字典。
- `analysis/openfoodfacts_nutrition_sample_50000.csv`：固定 50,000 条分析副本。
- `analysis/analyze_openfoodfacts.py`：流式统计和抽样脚本。
- `analysis/check_sample_conflicts.py`：抽样副本逻辑冲突检查脚本。
