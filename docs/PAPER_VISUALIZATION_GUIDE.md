# 论文可视化与服务器运行指南

## 1. 需要哪些图，以及各自支持什么结论

| 图 | 必须包含的内容 | 配合的数字 | 可支持的结论 |
| --- | --- | --- | --- |
| 总体架构与模块结构 | 三级尺度、两次正则化、A 的训练支路、B 的阶段候选、C 的十一通道输入与源维归一化 | 不对应效果指标 | 交代方法与原 Vis-MVSNet 的关系 |
| 困难区域定义图 | RGB、GT、大位移图、遮挡比例、交集掩码 | 掩码像素数 | 证明评测的困难区域来自固定 GT/几何而非按模型误差筛选 |
| 八组深度/误差全图和共同 ROI | 同一图像、同一裁剪、同一深度范围和误差色标；ROI 标框 | 全图与 ROI 的 Abs、Acc2/4/8；ROI 交集区域指标 | 直接呈现大视差与遮挡处的实际误差变化 |
| 逐像素误差差值图 | $|D_{Base}-G|-|D_{config}-G|$，围绕零的统一双向色标 | 同一掩码上的误差均值与相对降幅 | 展示改善在哪里发生，也保留退化像素 |
| B 的范围诊断 | S1/S2/S3 覆盖图、归一化宽度、固定像素候选点与 GT 位置 | coverage、width、候选间距及 Abs | 解释真实深度是否进入搜索区间；覆盖提高不自动证明最终误差下降 |
| A 的遮挡诊断 | 每个源视图 GT 可见/遮挡与训练过的可见性头概率 | 有效像素数、可见/遮挡组概率、准确率、可见精确率/召回率和遮挡召回率 | 检查显式监督是否学到了几何遮挡；预测概率不是实际融合权重 |
| C 的融合诊断 | 最近 GT 候选处的逐源权重图、固定像素权重随候选深度曲线 | 权重和=1、可见/遮挡源的平均权重 | 候选相关融合在不同深度假设处怎样调整源视图证据；机制图不能替代模块消融 |
| 误差 CDF | 全图与大位移∩遮挡的经验累计分布，标出 2/4/8 mm | Acc2/4/8 与 Abs | 解释阈值准确率和平均误差的取舍；局部图像 CDF 不是全测试集 CDF |
| 每场景改善/退化及八组统计图 | 全部场景，包含负降幅；八组七区域矩阵 | 全测试集像素加权指标、各场景指标 | 检查平均提升是否由少数场景主导 |

正文建议使用总体图、模块图、2–3 个困难场景的误差对比、一个范围诊断与一个融合诊断；八组大拼图和更多案例放补充材料。失败案例至少展示一个；图不能只挑提升最大的位置。三视图训练、五视图训练分别放在 `View3`、`View5`，均使用原 CSV 的五视图测试协议。

## 2. 命名与数据来源

| 输入模型 | 输出目录 | 分析 ABC | 原 CSV ABC |
| --- | --- | --- | --- |
| range | Base | 000 | 010 |
| oa_range | Base+A | 100 | 110 |
| vis | Base+B | 010 | 000 |
| range_hyp | Base+C | 001 | 011 |
| oa | Base+A+B | 110 | 100 |
| oa_full | Base+A+C | 101 | 111 |
| hyp | Base+B+C | 011 | 001 |
| oa_hyp | Base+A+B+C | 111 | 101 |

`tools/paper_configs.py` 是这套命名的唯一映射。加载网络仍使用检查点所属的原始 model_type；分析名仅用于文件夹、图题和表格。原始编码记录在元数据中，不覆盖原 CSV。此命名与实际模块身份之间的投稿核对见 `PAPER_FORMULA_AUDIT.md`。

## 3. 服务器运行

从仓库中的 `vismvsnetgeo` 目录执行。只需原项目依赖以及 matplotlib；图采用 Agg 后端，无需桌面环境。先拉取你提交的修改，再运行下面三步。

也可一次执行全部步骤（第三个参数可改为 `View5` 或 `View3`）：

```bash
bash tools/run_paper_visualizations.sh /path/to/DTU_training checkpoints/dtu all
```

用 `PAPER_OUTPUT`、`PAPER_PYTHON`、`PAPER_EVAL_ARGS_JSON` 环境变量可分别指定输出目录、Python 路径和实际评测参数文件；输出默认仍按八组分析配置归档。

### 第一步：生成现有数字的统计图与案例选择

```bash
python tools/visualize_paper_results.py stats \
  --series all \
  --outdir outputs/paper_visualizations
```

读取 `eval/ablation_test_light3` 和 `eval/ablation_test_view3_light3`。验证八组逐图样本、掩码像素数、汇总值一致后，生成八组统计图、逐场景比较、完整样本排名和 `selection.json`。

默认案例包含大视差、遮挡、二者交集与边界遮挡交集的 Base 误差中位数样本，额外包含交集区域 Base 误差第 90 百分位样本，以及联合方法退化最大的样本。每个选择保存理由。这样案例选择不会全部依赖提升幅度。

### 第二步：导出实际检查点预测

```bash
python tools/visualize_paper_results.py export \
  --series View5 \
  --testpath /path/to/DTU_training \
  --testlist lists/dtu/test.txt \
  --checkpoint_root checkpoints/dtu \
  --selection outputs/paper_visualizations/View5/selection.json \
  --eval_args_json tools/paper_eval_args.example.json \
  --outdir outputs/paper_visualizations
```

将 `--testpath` 改为包含 `Rectified/`、`Depths/`、`Cameras/` 的 DTU 根目录。`checkpoint_root` 下使用 CSV 原检查点子目录，例如 `range_view5/best_2mm.ckpt`；脚本只把文件保存到 `Base`。三视图训练结果把 `--series` 改为 `View3`，同时选择相应 View3 的 selection 文件。

`paper_eval_args.example.json` 是评测器默认参数示例，应核对它与原 CSV 的评测命令一致。不要把示例当作已核定的训练超参数。范围和残差倍率、推理视图数、区域视图数从原 CSV 读取。单次导出使用同一光照；当前现有结果为 light 3。

若检查点目录不同，可使用 `--checkpoint_manifest`，JSON 以分析配置为键，值为实际文件路径。例：

```json
{
  "Base": "/path/to/range_view5/best_2mm.ckpt",
  "Base+A": "/path/to/oa_range_view5/best_2mm.ckpt",
  "Base+B": "/path/to/vis_view5/best_2mm.ckpt",
  "Base+C": "/path/to/range_hyp_view5/best_2mm.ckpt",
  "Base+A+B": "/path/to/oa_view5/best_2mm.ckpt",
  "Base+A+C": "/path/to/oa_full_view5/best_2mm.ckpt",
  "Base+B+C": "/path/to/hyp_view5/best_2mm.ckpt",
  "Base+A+B+C": "/path/to/oa_hyp_view5/best_2mm.ckpt"
}
```

`--dry_run` 只打印将运行的命令。`--configs Base Base+B` 可只导出指定组合以调试，但论文完整图需八组齐全。新预测的逐图区域指标与原 CSV 对照，默认绝对容差为 $10^{-3}$、相对容差为 $10^{-4}$；超出容差停止并留下 reproduction_check.csv，不用新的结果冒充旧数字。

### 第三步：渲染图像并计算共同裁剪指标

```bash
python tools/visualize_paper_results.py render \
  --series all \
  --outdir outputs/paper_visualizations \
  --error_max 20 \
  --gain_max 10 \
  --cdf_max 30
```

误差图显示 0–20 mm，差值图显示 ±10 mm。超出范围只截断显示颜色，不截断指标；实际原数组与 CSV 保留原值。若画局部细节需要较小色标，在相同 scene 的全部模型上统一修改 `--error_max`。

## 4. 手动确定共同 ROI 与探针像素

先运行统计和默认导出查看图，在 selection.json 里添加坐标，八组共用同一份选择，重新 export 和 render：

```json
[
  {
    "scan": "scan1",
    "view": 0,
    "light": 3,
    "reason": "manual fixed ROI: foreground/background occlusion boundary",
    "roi": [40, 50, 200, 170],
    "probe": [100, 100]
  }
]
```

坐标相对于导出 GT/预测图分辨率；ROI 为 `[x0,y0,x1,y1]`，右下端不包含，probe 为 `[x,y]`。上述 scan 与坐标仅是格式示例，需选择实际测试集中存在且图像内有效的位置。留空时从 Base 的 GT 交集掩码自动确定 ROI 和探针。探针用于候选级权重和区间曲线；不要根据各模型预测分别选探针。

## 5. 输出目录

```text
outputs/paper_visualizations/
  View5/
    mapping.json
    selection.json
    Base/
      region_abs.png / region_abs.pdf
      exported_metrics/
        all_metrics.csv
        summary_metrics.csv
        reproduction_check.csv
      scanX_viewYY_light3/
        arrays.npz
        metadata.json
        region_metrics.csv
        source_visibility_metrics.csv
        figures/
          depth_error.png / .pdf
          stage_coverage_width.png / .pdf
          source_fusion_diagnostics.png / .pdf
          candidate_source_weights.png / .pdf
    Base+A/ ...
    Base+B/ ...
    Base+C/ ...
    Base+A+B/ ...
    Base+A+C/ ...
    Base+B+C/ ...
    Base+A+B+C/ ...
    comparison/
      global_metrics.csv
      per_scan_abs.csv
      sample_ranking.csv
      region_gain.png / .pdf
      threshold_accuracy.png / .pdf
      coverage_width.png / .pdf
      scan_improvements_regressions.png / .pdf
      scanX_viewYY_light3/
        region_definition.png / .pdf
        eight_config_depth_error.png / .pdf
        error_gain.png / .pdf
        stage_coverage.png / .pdf
        probe_search_intervals.png / .pdf
        error_cdf.png / .pdf
        metrics_full_and_roi.csv
        single_factor_comparisons.csv
        display_settings.json
  View3/ ...
```

## 6. 图与数字一起分析

1. `global_metrics.csv` 的全测试集 pixel-weighted 指标用于主表和结论；每场景图使用 `per_scan_abs.csv`。不要用逐图平均代替像素加权汇总。
2. `metrics_full_and_roi.csv` 为这张图的全图与共同 ROI，按七个掩码计算全部十项指标；`abs_reduction_pct` 相对 Base，阈值准确率与 coverage 的差值单位是百分点。
3. `single_factor_comparisons.csv` 列出 A/B/C 各四组共十二组匹配比较。支持某模块时应对照仅改变该模块的两个配置；仅展示 Base 对完整模型只能说明联合效果。
4. CDF 的纵轴在 2/4/8 mm 处对应该图区域的阈值准确率。误差图色标截断后看不到长尾完整幅度，因此同时看 Abs 和 CDF，不从颜色面积推出平均误差最优。
5. 候选区间覆盖 GT 与最终预测正确是两个事件。扩大搜索范围可能提高覆盖率、降低离散精度；应联合 width 和 Acc2 判断。
6. 最近 GT 候选处的源权重只用于机制诊断，标题明确标为 GT-based diagnostic。实际推理对全部深度候选计算权重，不读取 GT。可见性分类头未接受 A 监督的模型不显示该头的概率，避免把未训练头当作遮挡预测。
7. 所有实际预测必须通过 reproduction_check，若未通过先核对检查点和评测命令。上述流程不生成不存在的网络结果，不把统计 CSV 逆造成像素图。

## 7. 数据补充与稿件衔接

目前已有的统计图位于 `docs/paper_visualizations/`。服务器跑出真实图像后，优先选择代表性大位移∩遮挡案例，将八组大拼图中的必要配置（Base、单模块、关键组合、完整方法）整理为正文图，并保留所有八组到补充材料。图注写明 scan、参考视图、光照、同一 ROI、色标单位和该区域指标。完整论文现在的图1–6已经有连续编号；新增定性图应接续编号并在分析对应段落引用。

若论文宣称三维重建质量，还需一致的深度过滤/融合以及 DTU 点云 Accuracy、Completeness、Overall。当前像素深度指标与彩色点云截图不能代替这些三维评价。
