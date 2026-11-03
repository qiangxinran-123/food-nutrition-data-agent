"""Generate a Chinese course report exclusively from saved, verified artifacts."""
from pathlib import Path
import shutil

from data_agent.io import sha256_file
from .strategies import read_json, write_json, source


def _verified(item):
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"artifact hash mismatch: {path}")
    return path


def generate_report(strategies_path, m1_run, m2_run, output_dir):
    strategy = read_json(strategies_path)
    for item in strategy["sources"]:
        _verified(item)
    for group in strategy["groups"].values():
        for item in group.get("evidence", {}).get("sources", []):
            _verified(item)
        if group.get("status") == "executed":
            path = _verified(group["metrics_file"])
            stored = read_json(path)
            if stored != group["metrics"]:
                raise ValueError("embedded group metrics differ from saved metrics")
            for name, digest in stored["artifact_sha256"].items():
                if Path(name).name != name or sha256_file(path.parent / name) != digest:
                    raise ValueError("group artifact hash mismatch")
    m1, m2 = Path(m1_run).resolve(), Path(m2_run).resolve()
    original = read_json(m1 / "metrics.json")
    analysis = read_json(m2 / "cluster_analysis.json")
    plot = read_json(m2 / "plot_data.json")
    if analysis["m1_metrics_sha256"] != sha256_file(m1 / "metrics.json") or analysis["matrix_sha256"] != original["matrix_sha256"] or plot["matrix_sha256"] != original["matrix_sha256"]:
        raise ValueError("M1/M2 lineage mismatch")
    if original["data_sha256"] != strategy["groups"]["B"]["sha256"]:
        raise ValueError("M1 descriptive experiment is not the verified B dataset")
    for name, digest in original["artifact_sha256"].items():
        if Path(name).name != name or sha256_file(m1 / name) != digest:
            raise ValueError("M1 artifact hash mismatch")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    original_files = {"m1_metrics.json": m1 / "metrics.json", "cluster_analysis.json": m2 / "cluster_analysis.json",
                      "plot_data.json": m2 / "plot_data.json", "pca_projection.csv": m2 / "pca_projection.csv"}
    for name in ("pca_projection.png", "cluster_heatmap.png", "cluster_sizes.png", "k_metrics.png"):
        original_files[name] = m2 / name
    references = {}
    for name, path in original_files.items():
        shutil.copyfile(path, evidence / name)
        references[name] = {"source": source(path), "copy": source(evidence / name)}
    bundle = {"strategies": strategy, "descriptive_m1": original, "descriptive_m2": analysis,
              "plot_data": plot, "sources": references, "strategy_source": source(strategies_path)}
    write_json(output / "report.json", bundle)
    def fmt(x):
        return "不可计算/未记录" if x is None else f"{x:.6g}" if isinstance(x, (int, float)) else str(x).replace("|", "／").replace("\n", " ")
    def table(headers, rows):
        return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |", *["| " + " | ".join(fmt(x) for x in row) + " |" for row in rows], ""]
    protocol = strategy["protocol"]
    quality = strategy["quality"]
    lines = ["# 食品营养特征聚类与数据质量实验评价报告", "", f"运行编号：{strategy['run_id']}。本报告由保存的结构化结果生成；不预先指定最佳业务 k。", "",
             "## 一、实验概况", "", "目标是验证营养特征聚类和数据质量评价流程，描述固定规则处理前后的变化。模块三实现特征选择、标准化、K-Means、评价、簇分析、可视化；模块四实现质量比较、真值评价、聚类比较、策略组织和报告生成。", "",
             f"环境：{strategy['environment']}。特征顺序：{', '.join(protocol['features'])}。", "",
             f"K-Means：k={protocol['ks']}，种子={protocol['seeds']}，n_init={protocol['n_init']}，max_iter={protocol['max_iter']}，tol={protocol['tol']}。", "",
             f"原始输入取完整 {quality['datasets'][0]['dataset_id']}；B 使用其经核验的固定规则 {quality['datasets'][1]['dataset_id']}。策略主对照在共同可建模行 ID 上分别重新训练，两组使用在 A 共同数据上拟合的同一 StandardScaler，不做新的插补、去重或异常修复。", ""]
    lines += table(["数据集", "版本", "行数", "SHA256"], [[d['dataset_id'],d['version'],d['rows'],d['sha256']] for d in quality['datasets']])
    lines += ["## 二、数据质量分析", "", f"以下统计对应完整的 {quality['datasets'][0]['dataset_id']} 与 {quality['datasets'][1]['dataset_id']}，而非仅共同建模子集。语义缺失包含物理空值；findings、违规单元格、受影响行分别记录。", ""]
    quality_rows = [("物理空值单元格",quality['completeness']['physical_null_cells']), ("语义缺失单元格（含物理）",quality['completeness']['semantic_missing_including_physical_cells']), ("语义缺失率",quality['completeness']['semantic_missing_including_physical_rate']), ("冗余重复行",quality['uniqueness']['redundant_rows']), ("范围违规单元格",quality['range_validity']['range_flagged_cells']), ("冲突行并集",quality['field_consistency']['affected_rows_union'])]
    lines += table(["指标", quality['datasets'][0]['dataset_id'], quality['datasets'][1]['dataset_id'], "绝对变化", "计算状态"], [[name,x.get('before'),x.get('after'),x.get('absolute_change'),x['status']] for name,x in quality_rows])
    mods=quality['modifications']
    lines += [f"共同保留行数：{quality['retention']['common_row_ids']}；保留率：{quality['retention']['retained_before_fraction']:.4%}。变更表填充单元格数：{fmt(mods.get('unique_imputed_cells'))}；填充比例（清洗前总单元格为分母）：{fmt(mods.get('imputed_cell_fraction_of_before_cells'))}；删除行数：{fmt(mods.get('deleted_row_ids'))}。", "", "范围专有的受影响行数无法从混合异常行并集拆分。冲突规则可评价行数随版本改变，不能把数量下降直接认定为修复正确。", "", "## 三、受控错误注入实验", ""]
    inj=strategy['injection']
    lines += [f"配置：{strategy['injection_config']}。Ground Truth 来源及哈希见 report.json 的 strategies.sources。", "", "评价只覆盖已知注入正例和规则明确的负例控制；范围外检测位置真值未知，不计为假阳性。", ""]
    lines += table(["错误类型", "TP", "FP", "FN", "Precision", "Recall", "F1", "状态"], [[kind,x.get('tp'),x.get('fp'),x.get('fn'),x.get('precision',{}).get('value'),x.get('recall',{}).get('value'),x.get('f1',{}).get('value'),x['status']] for kind,x in inj['by_type'].items()])
    lines += [f"修复评价：{inj['repair']}。未运行的修复不能填入 MAE/RMSE 或正确率。即使受控 F1 为 1，也不代表全数据集准确率 100%。", "", "## 四、食品营养聚类实验（既有 M1/M2）", "", f"这一节为 {original['dataset_id']} 全部 {original['valid_rows']} 条可建模记录的描述性实验，使用原 M1 的 B 数据拟合 scaler；不能与后文统一 scaler 的实验混为一谈。", ""]
    lines += table(["k", "抽样 Silhouette", "全样本 CH", "全样本 DBI", "平均稳定性 ARI"], [[r['k'],r['silhouette'],r['calinski_harabasz'],r['davies_bouldin'],r['stability']['mean_ari']] for r in original['results']])
    sampling=original['results'][0]['silhouette_sampling']
    lines += [f"Silhouette 使用固定种子 {sampling['seed']}、{sampling['sample_size']} 条覆盖各簇的抽样，并非全量精确值。PCA 只用于展示，不能将簇编号解释为食品健康等级。下面展示的 k={analysis['k']} 是既有分析示例，不是业务最优选择。", ""]
    lines += table(["簇", "数量", "占比", "能量均值 kcal/100g", "脂肪均值 g/100g", "糖均值 g/100g", "最近中心食品"], [[c['label'],c['count'],c['proportion'],c['nutrition']['energy-kcal_100g']['mean'],c['nutrition']['fat_100g']['mean'],c['nutrition']['sugars_100g']['mean'],c['representatives'][0]['product_name']] for c in analysis['clusters']])
    lines += ["完整八特征均值、中位数、类别分布和代表食品原始信息见 evidence/cluster_analysis.json。单样本簇及极端输入值均保留，可能影响聚类。热图为标准化簇均值，原始含量请查看 JSON 单位。", ""]
    for title,name in [("PCA 投影","pca_projection.png"),("标准化特征画像","cluster_heatmap.png"),("簇大小（对数轴）","cluster_sizes.png"),("原 M1 指标曲线","k_metrics.png")]:
        lines += [f"![{title}](evidence/{name})", ""]
    lines += ["## 五、策略对照实验", "", "A 是最低限度有限值/完整案例过滤；B 是既有固定规则结果；C 仅接受真实 Agent 证据和已执行工具产物。共有行训练限制是预设实验范围，不是对 A 增加清洗。", ""]
    lines += table(["组", "状态", "数据行数", "可建模行数", "占原始 v1 比例", "主实验行数", "聚类实际耗时秒"], [[name,g['status'],g.get('data_rows'),g.get('eligible_rows'),g.get('eligible_fraction_of_original'),g.get('cohort_rows'),g.get('metrics',{}).get('duration_seconds')] for name,g in strategy['groups'].items()])
    lines += [f"C 组信息：{strategy['groups']['C'].get('reason',strategy['groups']['C'].get('evidence',{}))}。", "", f"B 历史清洗耗时（既有日志，不是本次重跑）：{fmt(strategy['groups']['B']['evidence'].get('historical_cleaning_duration_ms'))} ms；本次策略流程总耗时：{strategy['elapsed_seconds']:.3f} 秒。Agent token/费用无真实日志时为不可计算；A/B 不发生模型调用。", "", "下面所有指标使用每个 k 下两组相同的共同样本抽样；CH/DBI 也在该样本上计算。训练使用全部共同队列，跨组 ARI 使用全部共同 ID。各 k 的抽样可能不同，不用于自动选最优 k。", ""]
    rows=[]
    for comp in strategy['clustering_comparisons']:
        if comp['status'] != 'comparable':
            rows.append([comp['k'],'—',None,None,None,None,comp['reason']]); continue
        for name,m in zip([n for n,g in strategy['groups'].items() if g['status']=='executed'],comp['common_subset_metrics']):
            rows.append([comp['k'],name,m['silhouette'],m['calinski_harabasz'],m['davies_bouldin'],m['stability']['mean_ari'],comp['evaluation_sample']['count']])
    lines += table(["k", "组", "共同抽样 Silhouette", "共同抽样 CH", "共同抽样 DBI", "样本稳定性 ARI", "评价行数/限制"], rows)
    lines += table(["k", "A/B 共有行", "A/B 跨组 ARI", "状态"], [[c['k'],c['common_rows'],(c.get('cross_group_ari') or [{}])[0].get('ari'),c['status']] for c in strategy['clustering_comparisons']])
    available = [c for c in strategy['clustering_comparisons'] if c['status']=='comparable']
    if available and all(all(abs(c['common_subset_metrics'][0][key]-c['common_subset_metrics'][1][key]) < 1e-10 for key in ('silhouette','calinski_harabasz','davies_bouldin')) for c in available):
        lines += ['实际观察：可比较的 k 上 A/B 共同样本指标在数值容差内一致。本次结果支持覆盖范围发生变化，不支持宣称共同样本聚类质量提升。', '']
    lines += ["## 六、实验局限与结论", "", "已经验证固定规则产物的来源、只读输入、共同样本聚类、受控真值评价、图表和报告生成。缺失/重复数量变化见第二节；这些统计不能证明每处修复正确。A 的完整案例筛选有选择偏差，主实验结果只适用于共同可建模样本，不能据此评价 B 新增覆盖样本的聚类质量。", "", "C 未接入时不能得出智能体优于固定规则的结论。接入需同一原始输入、真实提供方响应、由响应产生的决策、匹配的既有工具执行、经哈希校验的数据及外部真实性审查。", "", "本报告不作显著性判断、不自动选择最佳 k，也不将轮廓系数提升解释为数据更真实或食品更健康。所有数值和图表来源、路径和 SHA256 均收录于 report.json；本目录的文件哈希见 manifest.json。", ""]
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    write_json(output / "manifest.json", {"files": {str(p.relative_to(output)): sha256_file(p) for p in output.rglob('*') if p.is_file()}})
    return output / "report.md"
