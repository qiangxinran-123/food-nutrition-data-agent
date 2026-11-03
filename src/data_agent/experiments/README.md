# M3 数据质量与聚类评价

此模块只消费 M0/M1/M2 和上游检测、清洗产物，不修改它们。Python 3.12 环境复用 M0–M2 已有依赖；无需新增依赖。

## 运行真实示例

在项目根目录的 PowerShell：

```powershell
$python = "C:\Users\27969\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$env:PYTHONPATH = "src;src/data_agent/clustering/.m1_deps;src/data_agent/clustering/.m2_deps;.deps"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:LOKY_MAX_CPU_COUNT = "1"
New-Item -ItemType Directory -Force src/data_agent/experiments/.tmp | Out-Null
$env:TEMP = (Resolve-Path src/data_agent/experiments/.tmp).Path
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = $env:TEMP
& $python -m unittest data_agent.experiments.test_m3 -v
& $python -m data_agent.experiments --project-root . --output outputs/experiments/my_new_m3_run --m1-run outputs/clustering/my_v5_m1_run --seed 42 --sample-size 2000 --count-per-type 5 --k 5
```

输出目录必须不存在。入口产生质量比较、独立注入 CSV、ground truth、上游检测及选择表、检测评价、聚类可比性结果和带哈希的 manifest；中途失败时保留新目录，重试使用另一个目录。CSV 混有数值和人为格式错误，读取时请用 `load_injection`，它验证与真值关联的哈希并恢复数值、ID 及格式错误。保存的 `ground_truth.json` 除正例还包含规则已知的负例控制与所有匹配键。

## 质量比较 API

```python
import json
from pathlib import Path
from data_agent.clustering import load_input
from data_agent.experiments.quality import compare_quality

root = Path.cwd()
before, after = load_input("1", project_root=root), load_input("5", project_root=root)
old = json.loads((root / "logs/checks/off_52k_v1_quality_summary.json").read_text(encoding="utf-8"))
new = json.loads((root / "logs/checks_v5/off_52k_v5_quality_summary.json").read_text(encoding="utf-8"))
comparison = compare_quality(before, after, old, new)
```

汇总必须与数据集 ID/哈希一致；相关检测未成功、信息缺失或规则不一致时，该维度不可计算。物理 null 单独通过 DataFrame 统计；语义缺失总数包含物理 null，语义专有数是总数减物理 null。重复使用上游冗余行数，而非重复组全部成员。范围数是范围违规单元格，不是行数；上游的异常行并集同时包含统计异常，因此范围专有的受影响行不可计算。冲突行是多个规则的行并集，不是 findings 数或单元格数；各规则有效检测行数随版本而变化。填充及删除来自经哈希和父数据版本核验的变更表，填充比例分母为清洗前总单元格数。数量下降不能证明修复正确。

## 错误注入、检测与修复评价 API

```python
from data_agent.experiments.injection import inject_errors, save_injection, load_injection, evaluate_injected_errors
from data_agent.experiments.detection import detect_injected_errors

# 可提供原始行 ID 列表进行抽样；否则使用整个数据集的副本。
injected = inject_errors(after, counts={"missing": 5, "duplicate": 5, "range": 5, "format": 5, "conflict": 5}, seed=42)
save_injection(injected, "new_empty_directory")
loaded = load_injection("new_empty_directory/ground_truth.json")
detections = detect_injected_errors(loaded, after, "new_empty_directory/selection_artifacts")
evaluation = evaluate_injected_errors(loaded, detections)
```

注入位置分开选择，每类注入等量已知负例控制：原本有限的脂肪值、样本中 code 唯一的行、0–100 的糖值、可解析的 0–100 蛋白值、原本饱和脂肪不大于总脂肪的行。副本拥有独立 row ID。格式错误是数值追加 ` g/100g`；范围错误是糖值 150；冲突是饱和脂肪设为总脂肪加 5。原值仅是数据输入中的已知值，并不证明该值真实或正确。

检测适配调用上游五种检测器，读取完整且哈希核验的 selection；不使用抽样 finding。匹配键为 `(error_type, row_id, column)`，重复使用特殊列 `__row__` 并只评价被标为冗余的复制行。每键最多计一次。TP/FP/FN 的评价范围严格限定为注入正例与明确规则下的负例控制；范围外位置真值未知，不视为 FP，并单独计数。因此该 precision 不是整个真实数据集上的总体 precision。失败、未执行、部分执行或缺少目标规则的检测对应指标不可计算，不能当作零错误；只要一类不可计算，微平均不可计算。分母为零时指标为 null 并附原因。

可选修复输入 `repairs` 是变更列表，每项包含 `error_type`、`row_id`、`column`、`new_value`；重复副本删除用 `"drop_duplicate"`。这是对提供的变更记录进行评价，不运行修复器。正确率分母是所有已知注入正例，未处理率是其中没有变更记录的比例；误修改率分母为已知负例控制数。数值 MAE/RMSE 仅针对有数值修复结果的正例，记录纳入数量；绝对容差默认为 1e-6。无修复输入时所有修复指标不可计算，无法评估的数值不假定为正确。

## 聚类比较 API

```python
from data_agent.experiments.clustering import load_cluster_group, compare_clustering
group_a = load_cluster_group(after, "outputs/clustering/my_v5_m1_run", 5)
# group_b = load_cluster_group(other_dataset, "other_existing_m1_run", 5)
comparison = compare_clustering([group_a])  # 只有一组时明确不可比较
# comparison = compare_clustering([group_a, group_b])
```

每组验证已有 M1 标签和 scaler 哈希、数据身份和重建矩阵。需要不同来源的两组、相同特征与顺序、scaler 参数哈希、k、种子集合和顺序、K-Means 算法与参数；跨版本需有可核验的直接父子或相同父数据来源，并依赖上游保留稳定 row ID 的约定。其他情况返回不可比，不排名。全样本指标保留为各自描述，不当作严格共同样本结论。公平评价使用共有 row ID 构造最多 2,000 条的同一固定抽样，保证每组每个种子的每簇均覆盖，再调用 M1 `evaluate_clusters()` 计算样本上的 Silhouette、CH、DBI 和稳定性。跨方案 ARI 在全部共有行 ID 上计算。共同样本太少、簇缺失或覆盖要求不能满足时明确不可计算。

各函数返回可 JSON 序列化的数据；`data_agent.experiments.__main__.write_json` 以独占方式保存，避免覆盖既有结果。M3 不执行多方案实验或生成最终报告。

## M4 最终交付：策略组织与动态报告

新增 API：

```python
from data_agent.experiments.strategies import compare_strategies, ExperimentProtocol, import_agent_result
from data_agent.experiments.report import generate_report

result = compare_strategies(project_root, new_output_dir,
    protocol=ExperimentProtocol(), m3_run=existing_m3_directory)
generate_report(new_output_dir / "strategies.json", existing_m1_directory,
                existing_m2_directory, new_output_dir / "report")
```

运行完整 M4（先执行本文前面的环境设置；无需额外安装依赖）：

```powershell
& $python -m data_agent.experiments.run_m4 --project-root . --output outputs/experiments/my_new_m4_run `
  --m1-run outputs/clustering/my_v5_m1_run `
  --m2-run outputs/clustering/my_v5_m2_run `
  --m3-run outputs/experiments/my_v5_m3_run
```

M4 必须显式传入同一 v5 数据版本生成的 `--m1-run`、`--m2-run`、`--m3-run`；旧的 `m1_20261029_run03`、`m2_20261029_k5_run02`、`m3_20261029_run03` 均为 v2 历史产物，不能与 v5 混用。输出根目录必须不存在。`protocol_input.json` 可用 `--protocol` 重新传入，包含可验证的完整参数；`protocol.json` 还记录实际队列、原始数据和标准化规则。`strategies.json` 是机器评价结果，`comparison.csv` 是共同样本指标表，`A/`、`B/` 保存分别真实训练的标签、中心、scaler 和评价；`report/` 保存中文 Markdown、统一 JSON 和便携图表副本；`delivery_manifest.json` 提供交付文件与实现代码的 SHA256。

协议固定 M1 的八特征和 k=2–8，默认种子 42/7/21、n_init=10、max_iter=300、tol=1e-4。A 只作 M1 完整有限数值筛选；B 复用 v5，并核验 `v2 → v3 → v4 → v5` 四份计划、运行记录、数据哈希、变更表和 profile 组成的完整链。A/B 主对照预先限制到共有可建模行 ID，按相同顺序各自真实训练；在 A 未清洗共同队列上 fit 一次 StandardScaler，B 只 transform。共有队列筛选属于实验范围，不属于额外清洗。各组全量数据质量、完整案例可用率和共同队列保留率分别保存。

共同队列的 k 搜索结果，以及 M3 在两组相同的最多 2,000 行上重算的 Silhouette/CH/DBI/稳定性，都保存并标明口径；后者三种内部指标均为该共同抽样上的值，跨组 ARI 使用全部共同 ID。旧快照的全样本指标只作为单独的描述性展示，不能混入 M4 主对照。M3 的共同子集惯性也已最小修正为对子集计算的平方距离和，避免复用全训练集惯性。

### C 组正式导入

启动前检查 logs/outputs/config 中 Agent、provider、trace 命名的 JSON，并记录候选列表；没有真实证据时 C 为 `not_available`。使用 `--agent-bundle path/to/agent_bundle.json` 导入，不能只给一份 B 组数据作为 C。此接口只核验证据、读取已经执行的输出，不调用 Agent 或清洗工具。

bundle 必须含 `source_dataset`、`output_dataset`（dataset_id/version/sha256）以及以下 `{path, sha256}` 条目，路径限制在 bundle 目录内：

- `output_profile`：现有 loader 可读取的真实输出 profile，manifest 的 parent 必须匹配 v1。
- `provider_export`：包含 run_id、source_sha256、responses；每个 response 包含 id、model、tool_calls（id/name/arguments），可选真实 usage。不能声明 simulated=true。
- `decisions`：列表，每项为 response_id、tool_call_id、tool_name、arguments，必须匹配提供方响应，工具必须属于现有工具集合。
- `tool_executions`：按决策顺序的实际成功执行记录，含相同标识/参数、input_sha256、output_sha256，形成从 v1 到最终数据的哈希链。
- `provenance_review`：reviewer、reviewed_at、actual_provider_execution_verified=true、provider_export_sha256，记录外部对实际执行真实性的核查。

本地文件哈希只能证明一致性，不能独立证明提供方记录真实。必须由其他组员提供实际来源证据和外部审查；单元测试中的合成 fixture 仅验证接口，绝不纳入真实 C 组实验。C 原始行 ID 必须保留且唯一。C 的额外质量检测汇总未提供时仍明确不可计算；已有可比聚类数据可参与主协议。没有真实日志的 token、费用不估算。

### 完整验收命令和范围

```powershell
& $python -m unittest data_agent.experiments.test_m4 -v
& $python -m unittest data_agent.experiments.test_m3 -v
& $python -m unittest data_agent.clustering.test_m0 data_agent.clustering.test_m1 data_agent.clustering.test_m2 -v
& $python -m unittest discover -s tests
```

M4 集成测试使用交付中的 `outputs/experiments/m4_20261029_run02` 真实开发验证产物；其他测试使用临时副本，不改原数据。此目录与最终目录都须保留供回归追溯。耗时为实际测量，重复运行的耗时和带耗时的 metrics 哈希可能变化；固定数据、环境、参数下的标签、scaler 和指标值可复现。

模块三六个函数位于 `clustering/core.py`、`clustering/analysis.py`、`clustering/visualization.py`。模块四五个函数位于 `experiments/quality.py`、`experiments/injection.py`、`experiments/clustering.py`、`experiments/strategies.py`、`experiments/report.py`。所有输出均可追溯，不把受控 F1 当作总体检测准确率，不把共同样本选择偏差或极端值隐藏，也不推断最佳业务 k、食品健康等级或不存在的 Agent 优势。
