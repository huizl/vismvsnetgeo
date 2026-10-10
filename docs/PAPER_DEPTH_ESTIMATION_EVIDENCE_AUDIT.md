# 本版论文数据与实验身份核对记录

该文件独立于论文正文，保留数据出处和投稿前必须核定的事实，不改写原始 CSV。

新稿只讨论深度估计，Base 是当前数值参照配置；不将当前映射写成已经核定的原始 Vis-MVSNet 复现。

当前模型/CSV 的原始 B 开关与论文分析 B 位取反。显示名不会改变检查点的网络。因而表格可作为现有实验的配置比较，但模块启用与原始基线的因果身份仍需核对训练命令、Git commit 和检查点。该问题不可用图像效果或配置改名解决。

方法章节继承现有稿中已核对的数学定义；损失权重和训练超参数未由推理元数据推断。完整方法设计不等于已证明各模块在每个组合中均有效。

全测试集来源：eval/ablation_test_light3 与 eval/ablation_test_view3_light3。两套各八组、1078样本，已核对配对键/区域像素和逐图像素加权的十项指标。

定性来源：outputs/paper_visualizations_ranked。View3、View5各六例；按原全图 Abs 收益排序的案例选择是收益导向，不代表随机或无偏抽样。案例新指标从数组核算，不替换旧表。

原稿 PAPER_COMPLETE_WITH_RESULTS.md 保持不变。新稿图片包复制现有框架/统计图和十二例真实导出 PNG/PDF，未修改预测或误差颜色。

全部区域案例指标：paper_depth_visual_assets/case_region_metrics.csv；原始误差分布：case_error_distribution.csv；逐项核对摘要：validation.json。

投稿前需统一实验身份、确认训练超参数，增加统一协议相关方法对比和运行开销；若只主张深度估计，不要求以点云指标补足该任务。
