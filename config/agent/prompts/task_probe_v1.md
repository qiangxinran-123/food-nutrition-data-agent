# 模型对照基线任务 v1（相同测试任务）

数据集：off_52k_v1。目标是在固定预算内完成一条最小但完整的“检测 → 清洗 → 验证 → 聚类 → 报告”链路，
用于比较不同模型的工具调用正确性、指令遵循、轮数、Token 与费用。

请严格按以下顺序执行，不要增加额外清洗或可视化：
1. profile_dataset。
2. check_missing、check_duplicates、check_outliers（三项各一次）。
3. clean_duplicates：strategy_id=keep_first，finding_id=duplicate.primary_key。
4. clean_missing：从检测输出中任选一个缺失的营养数值列，strategy_id=impute_statistic，params={"method": "median"}。
5. validate_quality，issue_types=["missing", "duplicate"]。
6. select_features → scale_features → run_kmeans(k=5, random_state=42) → evaluate_clusters(k=5) → analyze_clusters(k=5)。
7. submit_final_report：四个字段各不超过 80 个汉字。

约束：不调用 visualize_clusters、retention_report、list_cleaning_strategies；不处理格式、冲突、标签问题；
不删除行；不修改 Nutri-Score；聚类只解释为营养相似群体。
