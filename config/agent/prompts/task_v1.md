# 用户任务 v1：Open Food Facts 营养数据质量处理与聚类

数据集：Open Food Facts 食品营养分析副本 {dataset_id}（52,000 行，含 2,000 行人为复制的重复记录）。
字段基准：所有营养值均为每 100g；核心字段为 energy-kcal_100g、proteins_100g、fat_100g、saturated-fat_100g、
carbohydrates_100g、sugars_100g、fiber_100g、sodium_100g，另有 code、product_name、brands、categories_en、nutriscore_grade。

本次任务要求：
1. 先用 profile_dataset 了解数据，再检测缺失、重复、异常值和字段冲突；其他检测按需执行。
2. 针对最影响后续聚类的问题选择清洗策略。重复记录必须处理；核心营养字段的缺失应按列选择可核验的填充方式，
   允许使用字段间的换算关系（如 salt 与 sodium 的 2.5 倍关系）或中位数，但必须说明依据。
3. 每完成一步清洗后调用 validate_quality，确认问题数量下降且保留率合理。
4. 清洗完成后进行营养特征聚类：
   - 使用项目默认八项营养特征；
   - 至少运行 k=3、k=5 两个取值，并报告轮廓系数、CH、DBI 与多种子稳定性；
   - 至少对一个 k 调用 analyze_clusters，解释各簇的营养特征与代表食品。
5. 最后调用 submit_final_report，用中文给出：数据质量主要问题与处理理由、聚类结果解释、局限与未做事项。

约束提醒：不要删除超过 10% 的行；不要修改 Nutri-Score 标签；不要把缺失填成 0；不要声称簇等于健康等级。
