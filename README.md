# 食品营养数据质量分析与聚类系统

基于 Open Food Facts 食品营养样本的 Python 数据挖掘项目，涵盖数据质量检测、可追溯清洗、K-means 聚类、实验评价，以及调用大语言模型协调工具的智能体。项目以配置文件描述数据字段和质量规则，并保存数据版本、行标识及处理记录，便于复现和对比。

## 功能

- 检测缺失值、重复、异常值、类型与格式问题、标签问题、类别不平衡和字段间冲突。
- 按决策或清洗计划处理数据，记录变更及数据版本。
- 对营养特征进行标准化和 K-means 聚类，计算轮廓系数、CH、DBI 等指标，生成簇画像与图表。
- 使用人工注入错误和多方案实验评价检测、清洗及聚类结果。
- 通过模型配置和工具调用运行数据处理智能体；调用模型需要相应服务的 API 密钥。

## 项目结构

```text
config/                  数据配置、清洗计划及模型配置
 data/raw/               原始 Excel 样本
 data/base/              冻结的 v1 数据及 manifest
 data/processed/         已处理的数据版本及变更记录
src/data_agent/          检测、清洗、聚类、实验和智能体代码
scripts/                 命令行入口
tests/                   测试
docs/                    质量规则和清洗策略说明
```

v1 包含 52,000 行（50,000 行样本和 2,000 行注入重复），v2 为清洗后的 50,000 行。数据来源与哈希见 `data/base/off_52k_v1.manifest.json`；样本来源为 [Open Food Facts](https://world.openfoodfacts.org/data)。原始 manifest 中未记录数据许可，使用或再分发前请核实来源许可及署名要求。

## 安装

建议使用 Python 3.12，在项目根目录的 PowerShell 中运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

`requirements.txt` 汇总了代码和模块文档中使用的依赖。不同平台或新版依赖可能有兼容性差异；原有模块文档记录了实验时使用的具体版本。

## 使用

从项目根目录运行全部质量检测，报告输出至 `logs/checks/`：

```powershell
python scripts/run_quality_checks.py
```

执行聚类（输出目录必须不存在）：

```powershell
python -m data_agent.clustering --project-root . --output outputs/clustering/my_run --n-init 10 --sample-size 2000
```

运行智能体前，先在环境中设置对应服务的密钥。默认模型配置在 `config/agent/models.json`，默认读取 `DEEPSEEK_API_KEY`；密钥不要写入仓库。模型调用会产生服务费用。

```powershell
$env:DEEPSEEK_API_KEY = "你的密钥"
python scripts/run_agent.py --output outputs/agent_runs/my_run
```

各入口的参数可通过 `--help` 查看。清洗计划默认支持试运行；实际写入前请检查计划及参数。聚类与实验的参数、输出和限制详见 [聚类模块说明](src/data_agent/clustering/README.md) 和 [实验模块说明](src/data_agent/experiments/README.md)。

## 测试

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m unittest data_agent.clustering.test_m0 data_agent.clustering.test_m1 data_agent.clustering.test_m2 data_agent.experiments.test_m3 data_agent.experiments.test_m4
```

部分真实数据测试需要仓库内的数据文件，实验测试或完整报告可能还需要先生成上游运行产物。`outputs/` 和 `logs/` 为生成目录，不包含历史运行结果；M4 默认引用的旧运行目录需要自行生成或通过参数指定。

## 已知限制

- 部分 v2–v5 配置包含原环境绝对路径。聚类适配器对文档中指定的 v2 路径有特殊处理；直接使用通用加载器或其他版本前，请检查并按实际项目目录调整 profile。详见聚类模块说明。
- 质量与聚类结果用于探索分析，不能直接作为食品健康程度的判断。
- 原项目未提供代码许可证；使用和再分发代码前请向权利人确认。

更多资料：[配置说明](config/README.md)、[质量定义](docs/data_quality_definitions.md)、[清洗策略](docs/cleaning_strategy_matrix.md)。
