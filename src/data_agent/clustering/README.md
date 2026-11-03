# 模块三、四 M0 接口与环境

## 边界和输入

模块三使用 `data_agent.clustering.load_input(version, project_root=...)`；模块四可从 `data_agent.experiments` 导入同一入口。下游默认使用版本字符串 `"5"`，即 `config/profiles/off_52k_v5.json`；`"1"` 到 `"4"` 只用于原始数据审计和版本链复核。根目录必须显式给出，入口不接受任意版本，也不回退。读取固定的 `config/profiles/off_52k_v{version}.json` 和其中 manifest，manifest 指向 Parquet；入口复用 `load_dataset(config_path, *, project_root=None, verify_hash=True)`，后者返回 `LoadedDataset(ref, schema, frame, manifest, manifest_path, data_path, config, config_path)`；失败可抛 `DatasetLoadError`（包括 `DatasetIntegrityError`、`DatasetShapeError`）、配置/解析/读取异常。适配器还拒绝版本或 ID 不匹配、行 ID 缺失或重复。

加载器对文件字节按 manifest 的 base_file.sha256 进行 SHA256 校验，并检查源文件行列数；不校验 profile 或 manifest 自身哈希。返回 `DatasetInput` 包含独立 DataFrame、原有 `DatasetRef` 和 `SchemaProfile`、完整 manifest、实际 profile/manifest/data 路径、行 ID 字段与列顺序。`ref.sha256` 是输入文件哈希；`ref.source` 是 manifest 的 source.path（v5 无 source，父来源在 `manifest['parent']`）。上游接口并不将 manifest.profile_file.sha256 用作完整性验证。数据加载和字段识别不执行质量检测、清洗、标准化或聚类。

已知 `v2` 到 `v5` profile 固定指向 `D:/data_mining/data/processed/off_52k_v*.manifest.json`。适配器只对这些精确路径使用显式根目录下的同名 manifest，通过临时 profile 调用原加载器，原文件不改动，仍校验数据哈希和 ID/version。其他路径照原加载器处理，不搜索、替换其他字符串或关闭校验。临时目录必须可写；临时 profile 仅生命周期内存在。

v1 冻结快照含 52,000 行，其中 2,000 为注入重复行；v5 为最终清洗后 50,000 行，均 20 列。`__row_id` 在冻结快照建立时生成并存入文件；v5 保留幸存行的 ID，适配器要求非空且唯一。`__source_row` 是源表行号，v1 重复行共享它，不能作为唯一 ID；`code` 在 v1 也不唯一。跨版本对齐请用 `__row_id`，并注意 v2 已删除重复行。

营养字段均为数值 float64：`energy-kcal_100g` 单位 kcal/100g，`energy_100g` 为 kJ/100g，`proteins_100g`、`fat_100g`、`saturated-fat_100g`、`carbohydrates_100g`、`sugars_100g`、`fiber_100g`、`salt_100g`、`sodium_100g` 为 g/100g。`serving_quantity` 单位由来源定义，不应直接视作每百克营养值。数值缺失以 null/NaN 表达，不等于零；profile 中 categories 的 `unknown`、categories_en 的 `en:null`、nutriscore_grade 的 `unknown` 属于对应列的质量检测缺失语义，并非加载时统一替换。v5 已消除硬范围、IQR 尾部和字段冲突，但文本缺失仍保留。

## 配置与输出

`ClusterConfig` 记录 dataset_id、dataset_version、data_sha256、profile_path、manifest_path、row_id_column、按序 features、sample_filter（描述性字典，不执行）、random_state、algorithm、algorithm_params、artifact_path 和可选 artifact_sha256。它只验证基本格式和特征名唯一性，不验证字段是否存在、实际数据是否与配置一致，也不解释筛选规则。M1 执行前应自行核对输入和配置，包括版本、哈希、行 ID 和特征；实际产物生成后记录真实哈希。M0 不生成产物；建议后续使用 `outputs/clustering/<run_id>/` 和 `outputs/experiments/<run_id>/`，创建新运行目录并避免覆盖。

现有全局 `DatasetRef` 包含 contract_version、dataset_id、version、uri、format、sha256、row_count、column_count、source、parent_id、created_at、metadata；`Artifact` 包含 contract_version、artifact_id、kind、uri、media_type、sha256、row_count、dataset_id、finding_ids、description、created_at、metadata；`ToolResult` 包含 contract_version、tool_name、tool_version、dataset_id、status、findings、summary、artifacts、warnings、error_code、error_message、duration_ms、created_at、metadata；`ValidationReport` 包含 contract_version、validation_id、before_dataset、after_dataset、checks、passed、retained_rows、retained_rate、warnings、metrics、created_at、metadata。模块四未来可消费这些契约及模块三产物，M0 不实现评价或报告；全局契约不变。

## 环境与验证

本机验证 Python 3.12.14、pandas 3.0.1、numpy 2.3.5、pydantic 2.14.1、fastparquet 2026.9.0；fastparquet 位于项目 `.deps`。建议后续使用 Python 3.12、pandas >=3.0,<4、numpy >=2.3,<3、pydantic >=2.14,<3、fastparquet >=2026.9；M1 另需 scikit-learn >=1.7,<2（尚未安装或验证兼容性），实现算法前应锁定并测试具体版本。未安装绘图库。系统默认 Python 3.14 缺少依赖，不用于本次测试。

在项目根目录 PowerShell：

```powershell
$env:PYTHONPATH = "src;.deps"
$env:PYTHONDONTWRITEBYTECODE = "1"
# TEMP/TMP 必须指向可写的临时目录
$python = "C:\Users\27969\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python -m unittest data_agent.clustering.test_m0 -v
& $python -m unittest discover -s tests
```

M1 调用示例：

```python
from pathlib import Path
from data_agent.clustering import load_input

data = load_input("5", project_root=Path.cwd())
features = ["proteins_100g", "fat_100g", "carbohydrates_100g"]
frame = data.frame[[data.row_id_column, *features]].copy()
print(data.ref.dataset_id, data.ref.version, data.ref.sha256, data.columns)
```

## M1 聚类接口

```python
from data_agent.clustering.core import (select_features, scale_features, run_kmeans,
                                        evaluate_clusters, save_scaler, load_scaler)
selected = select_features(data)                  # 默认 8 个营养字段；排除不完整或无效记录
scaled = scale_features(selected)                 # 在此参考集 fit StandardScaler
save_scaler(scaled, "new-scaler.json")            # 只创建新文件，不覆盖
transformed = scale_features(selected, scaler=load_scaler("new-scaler.json"))
first = run_kmeans(scaled, 3, random_state=42, n_init=10)
other = run_kmeans(scaled, 3, random_state=7, n_init=10)
metrics = evaluate_clusters(scaled, first, stability_runs=[other], sample_size=2000)
```

`select_features` 要求字段的 profile 指定可比较的每 100g 单位，禁止标识、标签及能量或钠盐冗余组合。无效行以 missing、non_numeric、infinite 的优先级归类，一行只计一次，`exclusions` 给出行 ID 和原因；不会插补。`frame` 与 `row_ids` 严格按源顺序对齐。可通过参数显式指定合法特征，不会自动更改特征集合。

`scale_features` 返回矩阵、行 ID、特征、版本和输入哈希、scaler、矩阵哈希与 scaler 参数哈希。fit 时拒绝常数列；transform 接受已保存的 scaler 并严格检查特征顺序。未来比较多个清洗策略应明确选定参考集，固定并复用同一个 scaler；如果特征不一致则失败。JSON scaler 含 mean、variance、scale、训练样本数和列顺序，不含可执行的 pickle。只应使用可信的本项目产物。

`run_kmeans` 接受单个 k 或一组 k（默认 2–8），显式固定 n_init、迭代数和 random_state；返回带行 ID、标签、标准化空间中心、惯性、运行参数和输入身份的数据对象。退化结果标记为 `degenerate`，评价时拒绝；运行及评价时检查矩阵及 scaler 是否被更改。`evaluate_clusters` 校验数据、k、缩放和非随机运行参数一致，并按行 ID 对齐稳定性标签；返回 CH、DBI、惯性、各簇数量比例和种子两两 ARI、均值及总体标准差。抽样 Silhouette 先固定随机种子保证每个实际簇有一条记录，再从其余样本均匀无放回抽取；指标 JSON 标记样本量、方法和是否抽样，CH/DBI 则用全部有效记录。没有多个种子时 ARI 为 null。不能据此直接判断食品健康程度。

本机额外安装 `requirements-m1.txt` 中固定版本到模块内 `.m1_deps`，不升级上游依赖。以下命令从项目根目录执行；首次安装需要网络。TEMP 需设置为用户可写的临时目录：

```powershell
$python = "C:\Users\27969\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python -m pip install --target src/data_agent/clustering/.m1_deps --no-deps -r src/data_agent/clustering/requirements-m1.txt
$env:PYTHONPATH = "src;src/data_agent/clustering/.m1_deps;.deps"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:LOKY_MAX_CPU_COUNT = "1"
& $python -m unittest data_agent.clustering.test_m1 data_agent.clustering.test_m0 -v
& $python -m unittest discover -s tests
& $python -m data_agent.clustering --project-root . --output outputs/clustering/my_new_run --n-init 10 --sample-size 2000
```

运行目录必须不存在；程序不会覆盖已有运行。输出包括 `metrics.json`、`scaler.json`、排除行 CSV、每个 k 的行 ID 与各种子标签 CSV；指标 JSON 记录中心、参数、种子、追踪信息及其他文件哈希。中途失败会保留部分新运行目录，重试须使用新路径。此输出用于复现和后续接口，不是模块四正式报告或 M2 簇画像。

## M2 描述性分析与绘图

绘图额外需要 `requirements-m2.txt`，其版本已在本机 Python 3.12 环境测试。安装到模块专用目录并加入现有搜索路径：

```powershell
& $python -m pip install --target src/data_agent/clustering/.m2_deps --no-deps -r src/data_agent/clustering/requirements-m2.txt
$env:PYTHONPATH = "src;src/data_agent/clustering/.m1_deps;src/data_agent/clustering/.m2_deps;.deps"
New-Item -ItemType Directory -Force src/data_agent/clustering/.tmp | Out-Null
$env:TEMP = (Resolve-Path src/data_agent/clustering/.tmp).Path
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = $env:TEMP
& $python -m data_agent.clustering --project-root . --m1-run outputs/clustering/my_new_run --analysis-k 5 --output outputs/clustering/my_new_m2_run
```

`--m1-run` 复用已有 M1 目录；必须显式传入 `--analysis-k`，这里 k=5 只是示例，不是自动选择的业务 k。它核对 scaler 与标签文件的 M1 哈希、数据版本、行 ID、重建矩阵与 scaler 的哈希、簇数量，使用 M1 主种子的标签和中心。`m1_20261029_run03` 是旧 v2 快照的历史产物，不能与当前 v5 混用；切换版本后必须重新运行 M1。若省略 `--m1-run` 而指定 `--analysis-k`，原 M1 运行会照常执行，并在同一个新目录接着生成 M2 产物；既有参数含义不变。两个路径都不覆盖已有目录。中断后应使用新目录重试。

接口 `load_m1_result(data, run_dir, k)` 返回 `(selected, scaled, result, metrics)`；`analyze_clusters(data, scaled, result)` 按 ID 关联原数据并输出各簇原始单位均值、中位数、与总体均值之差及按标准化空间中心距离排序的代表食品。每行类别只计一次：优先 `categories_en`，再取 `categories`，取逗号分隔的第一个非空且非 unknown/en:null/null 项，否则记为 `(missing)`；统计包含全部类别与缺失，分母为簇大小。商品名缺失记 null，不生成解释性标签。

`visualize_clusters(scaled, result, analysis, metrics, output_dir)` 以全量标准化矩阵拟合二维 PCA，再用固定种子抽样最多 5,000 行并确保每簇至少一条；PCA 仅作展示。热图是各簇均值按 M1 参考 scaler 转换的无量纲值，颜色饱和限制为 ±3，但格内数字不截断。簇大小图使用对数纵轴并标注实际数量。指标图直接使用 M1 的 k=2–8 指标；Silhouette 抽样信息在图和数据中注明。

`cluster_analysis.json` 保存原始单位统计和代表食品，`plot_data.json`、`pca_projection.csv` 保存重绘所需数据。可以只用它们重新绘制四图到一个新建空目录：

```python
from data_agent.clustering.visualization import render_visualizations
render_visualizations("existing/plot_data.json", "existing/pca_projection.csv", "new_empty_directory")
```

评估曲线依赖来源 M1 的 metrics 文件；分析 JSON 记录该文件哈希和路径，便于追溯。单样本簇照常保留。两个新函数在 `data_agent.clustering.analysis` 和 `data_agent.clustering.visualization` 中，不更改全局数据契约。
