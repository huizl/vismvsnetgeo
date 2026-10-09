# 论文可视化与服务器运行指南（历史版）

当前默认流程已改为三类困难场景的整图对比，只比较 Base 与 Base+A+B+C，不再生成 R1/R2。请使用 [新版运行指南](PAPER_SCENE_VISUALIZATION_GUIDE.md)。以下 ROI 与两行五列内容记录旧版流程，不适用于当前默认渲染。

## 1. 需要哪些图，以及各自支持什么结论

| 图 | 必须包含的内容 | 配合的数字 | 可支持的结论 |
| --- | --- | --- | --- |
| 总体架构与模块结构 | 三级尺度、两次正则化、A 的训练支路、B 的阶段候选、C 的十一通道输入与源维归一化 | 不对应效果指标 | 交代方法与原 Vis-MVSNet 的关系 |
| 困难区域定义图 | RGB、GT、大位移图、遮挡比例、交集掩码 | 掩码像素数 | 证明评测的困难区域来自固定 GT/几何而非按模型误差筛选 |
| Base 与完整方法的两行五列总览及共同 ROI | RGB、GT、大视差掩码、遮挡比例、困难区域叠加、两组深度、两组误差、误差改善；每个子图单独保存 | 全图与 ROI 的 Abs、Acc2/4/8；ROI 交集区域指标 | 直接呈现大视差与遮挡处的实际误差变化 |
| 逐像素误差差值图 | $|D_{Base}-G|-|D_{config}-G|$，围绕零的统一双向色标 | 同一掩码上的误差均值与相对降幅 | 展示改善在哪里发生，也保留退化像素 |
| B 的范围诊断 | S1/S2/S3 覆盖图、归一化宽度、固定像素候选点与 GT 位置 | coverage、width、候选间距及 Abs | 解释真实深度是否进入搜索区间；覆盖提高不自动证明最终误差下降 |
| A 的遮挡诊断 | 每个源视图 GT 可见/遮挡与训练过的可见性头概率 | 有效像素数、可见/遮挡组概率、准确率、可见精确率/召回率和遮挡召回率 | 检查显式监督是否学到了几何遮挡；预测概率不是实际融合权重 |
| C 的融合诊断 | 最近 GT 候选处的逐源权重图、固定像素权重随候选深度曲线 | 权重和=1、可见/遮挡源的平均权重 | 候选相关融合在不同深度假设处怎样调整源视图证据；机制图不能替代模块消融 |
| 误差 CDF | 全图与大位移∩遮挡的经验累计分布，标出 2/4/8 mm | Acc2/4/8 与 Abs | 解释阈值准确率和平均误差的取舍；局部图像 CDF 不是全测试集 CDF |
| 每场景改善/退化及八组统计图 | 全部场景，包含负降幅；八组七区域矩阵 | 全测试集像素加权指标、各场景指标 | 检查平均提升是否由少数场景主导 |

默认定性图只比较 Base 与完整方法 Ours（目录名 `Base+A+B+C`），不逐模块绘制消融。八组统计图仍用于分析既有 CSV；旧版八组定性与机制诊断可用 `--layout ablation` 单独生成。正文建议使用总体图、模块图、2–3 个困难场景的误差对比和局部放大；更多案例放补充材料。失败案例至少展示一个；图不能只挑提升最大的位置。三视图训练、五视图训练分别放在 `View3`、`View5`，均使用原 CSV 的五视图测试协议。

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
CUDA_VISIBLE_DEVICES=0 bash tools/run_paper_visualizations.sh \
  /home/disk_10T/lzh_data/dtu_training/mvs_training/dtu checkpoints/dtu View5
```

`CUDA_VISIBLE_DEVICES=0` 使用物理 GPU 0，可改为实际编号；第三个参数改为 `all` 时处理 View5 和 View3。用 `PAPER_OUTPUT`、`PAPER_PYTHON`、`PAPER_EVAL_ARGS_JSON` 环境变量可分别指定输出目录、Python 路径和实际评测参数文件。批处理先生成八组 CSV 统计，再只导出 Base 与完整方法的真实预测，最后生成新版总览、独立子图和局部放大。

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
  --configs Base 'Base+A+B+C' \
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

`--dry_run` 只打印将运行的命令。新版定性图只需 `--configs Base 'Base+A+B+C'`；直接调用 export 时若省略 configs 则保留导出八组的兼容行为。新预测的逐图区域指标与原 CSV 对照，默认绝对容差为 $10^{-3}$、相对容差为 $10^{-4}$；超出容差停止并留下 reproduction_check.csv。

### 第三步：渲染图像并计算共同裁剪指标

```bash
python tools/visualize_paper_results.py render \
  --series all \
  --layout paper \
  --outdir outputs/paper_visualizations \
  --error_max 20 \
  --gain_max 10 \
  --cdf_max 30
```

常规总览误差图显示 0–20 mm，差值图显示 ±10 mm。差值为 Base 绝对误差减 Ours 绝对误差，蓝色表示改善、红色表示退化、白色表示无变化；GT 无效位置为黑色。超出范围只截断显示颜色，不截断指标；实际原数组与 CSV 保留原值。总览、常规独立子图和 `zoom_2x5` 使用相同颜色映射，局部图直接裁剪全图栅格。另存的 `detail_comparison` 使用明确标注的更细色标：误差 0–5 mm、差值 ±2 mm，同一条图中的 Base/Ours 始终共享色标，指标仍使用未截断的误差。

已有 `Base/样本/arrays.npz` 和 `Base+A+B+C/样本/arrays.npz` 时，只运行 render 即可，无需重新 export，不需要 GPU。`--layout paper` 是默认值，不要求八组，也不需要 `--allow_partial`。旧版定性图使用 `--layout ablation`，部分配置还需 `--allow_partial`。

## 4. 手动确定共同 ROI 与探针像素

先运行统计和默认导出查看图，在 selection.json 里添加共同坐标。支持旧版单个 `roi`，也支持多个 `rois`：

```json
[
  {
    "scan": "scan1",
    "view": 0,
    "light": 3,
    "reason": "manual fixed ROI: foreground/background occlusion boundary",
    "rois": [
      {"name": "large_disparity", "bounds": [40, 50, 200, 170]},
      {"name": "occlusion_boundary", "bounds": [210, 80, 370, 200]}
    ],
    "probe": [100, 100]
  }
]
```

坐标相对于导出 GT/预测图分辨率；ROI 为整数 `[x0,y0,x1,y1]`，右下端不包含，probe 为 `[x,y]`。上述 scan 与坐标仅是格式示例，需选择实际测试集中存在且图像内有效的位置。R1、R2、R3 分别是第 1、2、3 个 ROI（局部观察框），不是模型模块或阶段。

没有指定 ROI 时，裁剪宽、高默认取导出图像对应尺寸的 30%，上限分别为 160、120 像素。例如 160×128 的预测图，默认裁剪为 48×38，而不是覆盖几乎全图的 160×120。先在固定 GT 困难区域与 GT 边界的交集中找连通区域，按区域大小选代表像素；没有边界交集则使用困难区域本身。最多选三个框，IoU 超过 0.5 时尝试其他连通区域，避免高度重叠。选区不使用预测误差或改善幅度排序，GT 图、困难掩码及评测区域不改变。GT 边界不等于物体分割，自动框仍需人工检查。

可以用 `--roi_fraction 0.25` 进一步缩小自动框。常规局部图标题明确区分“ROI 所有有效像素指标”和“ROI 内目标困难掩码指标”，避免把整个框的指标误认为纯遮挡或纯大视差指标。额外的 `detail_comparison.png/.pdf` 为一行五列：RGB、GT、Base 误差、Ours 误差、误差改善；独立细节子图位于 `detail_panels/`。其较细色标通过 `--detail_error_max`、`--detail_gain_max` 调整，并记录到 `display_settings.json`。

调整 ROI 后只需以下命令，render 会读取 selection 覆盖缓存中的裁剪位置，并仅处理 selection 中的样本：

```bash
python tools/visualize_paper_results.py render --series View5 \
  --outdir outputs/paper_visualizations \
  --selection outputs/paper_visualizations/View5/selection.json
```

探针只用于旧版候选级诊断，调整探针需要重新 export；调整裁剪不需要。不要根据各模型预测分别选探针或裁剪。

## 5. 输出目录

新版默认定性输出如下。`panels/` 内的 PNG 保持导出预测的原生像素尺寸，无标题和边框，便于论文自行排版；`panels/with_legend/` 提供带标题和色标的 PNG/PDF。每个 ROI 同样输出十个独立子图和两行五列放大组合图。RGB 叠加中红色表示仅大视差、黄色表示仅多数遮挡、橙色表示两者交集，绿色/粉色/蓝色框标出 R1/R2/R3。

```text
outputs/paper_visualizations/View5/
  Base/scanX_viewYY_light3/figures/paper/
    base_depth.png
    base_abs_error.png
    with_legend/...
  Base+A+B+C/scanX_viewYY_light3/figures/paper/
    ours_depth.png
    ours_abs_error.png
    with_legend/...
  comparison/scanX_viewYY_light3/
    overview_2x5.png / .pdf
    panels/
      reference_rgb.png
      gt_depth.png
      large_disparity_mask.png
      source_occlusion_ratio.png
      difficulty_overlay.png
      base_depth.png
      ours_depth.png
      base_abs_error.png
      ours_abs_error.png
      error_gain.png
      with_legend/各子图.png / .pdf
    roi_01/
      zoom_2x5.png / .pdf
      detail_comparison.png / .pdf
      panels/上述十张局部子图.png
      panels/with_legend/上述十张局部子图.png / .pdf
      detail_panels/五张细节子图.png
      detail_panels/with_legend/五张细节子图.png / .pdf
      metrics.csv
    roi_02/...
    roi_03/...
    metrics_full_and_roi.csv
    display_settings.json
```

`display_settings.json` 保存色标上下限、每个子图的路径与排版位置、ROI 坐标及选择方式。区域 CSV 保存七个固定区域的像素数、十项指标及相对 Base 的误差降幅和百分点变化，scope 区分 `full_image`、`roi_01`、`roi_02` 等。一个 ROI 的七个区域会重叠，不应相加当作总像素数。不同 ROI 的全图指标只保存一次。

八组 CSV 统计、导出数组及 `--layout ablation` 的旧版输出结构如下：

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
2. `metrics_full_and_roi.csv` 为这张图的全图与共同 ROI，按七个掩码计算全部十项指标；`abs_reduction_pct` 相对 Base，阈值准确率与 coverage 的差值单位是百分点。局部表格与局部图坐标完全一致；局部提升不能替代全测试集提升。
3. 旧版 ablation 输出的 `single_factor_comparisons.csv` 列出 A/B/C 各四组共十二组匹配比较。支持某模块时应对照仅改变该模块的两个配置；新版 Base 对完整模型图用于说明联合方法在大视差和遮挡区域的效果，模块贡献用已有消融统计表论证。
4. CDF 的纵轴在 2/4/8 mm 处对应该图区域的阈值准确率。误差图色标截断后看不到长尾完整幅度，因此同时看 Abs 和 CDF，不从颜色面积推出平均误差最优。
5. 候选区间覆盖 GT 与最终预测正确是两个事件。扩大搜索范围可能提高覆盖率、降低离散精度；应联合 width 和 Acc2 判断。
6. 最近 GT 候选处的源权重只用于机制诊断，标题明确标为 GT-based diagnostic。实际推理对全部深度候选计算权重，不读取 GT。可见性分类头未接受 A 监督的模型不显示该头的概率，避免把未训练头当作遮挡预测。
7. 所有实际预测必须通过 reproduction_check，若未通过先核对检查点和评测命令。上述流程不生成不存在的网络结果，不把统计 CSV 逆造成像素图。

## 7. 数据补充与稿件衔接

目前已有的统计图位于 `docs/paper_visualizations/`。服务器跑出真实图像后，优先选择代表性大视差、遮挡以及交集案例，正文排入 Base/Ours 两行五列总览或从独立子图重排，并用局部放大突出边缘和遮挡细节。消融配置保留在统计表中。图注写明 scan、参考视图、光照、同一 ROI、色标单位和该区域指标。完整论文现在的图1–6已经有连续编号；新增定性图应接续编号并在分析对应段落引用。

若论文宣称三维重建质量，还需一致的深度过滤/融合以及 DTU 点云 Accuracy、Completeness、Overall。当前像素深度指标与彩色点云截图不能代替这些三维评价。

## 8. 导出后提示 New predictions differ from CSV

这表示真实预测已经导出，但数值复核没有通过；严格模式会在当前配置处停止，所以后续完整模型推理及绘图尚未执行。先检查新生成的差异报告，不能仅凭异常提示断定是检查点或参数错误。

更新代码后，可以直接检查现有缓存，不需要再次推理：

```bash
python tools/visualize_paper_results.py verify --series View5 --configs Base \
  --outdir outputs/paper_visualizations --verify_mode warn
```

报告位于 `View5/Base/exported_metrics/`：`reproduction_check.csv` 为所有比较，`reproduction_failures.csv` 只列失败比较，`reproduction_summary.csv` 按十项指标列出失败数量、最大绝对差值及对应样本，`reproduction_status.json` 保存复核状态。命令也直接打印最大差值。通过后删除旧的 failures 文件。

应核对检查点是否被更新、输入预处理是否变化、推理深度候选和间隔、原评测 batch size，以及 CUDA/cuDNN 的数值波动。仓库的 `eval_regions_view5.sh` 默认 batch size 为 4，而可视化导出默认 1；这只是一个可核对的差异，不足以判断本次不一致的原因。CSV 没有保存全部运行参数和检查点哈希，需结合原运行记录确认。

若当前先生成本次真实预测的图，显式开启告警模式：

```bash
CUDA_VISIBLE_DEVICES=3 PAPER_VERIFY_MODE=warn bash tools/run_paper_visualizations.sh \
  /home/disk_10T/lzh_data/dtu_training/mvs_training/dtu checkpoints/dtu all
```

告警模式仍计算并保存全部数值差异，只允许数值复核失败后继续；模型身份、源视图协议、区域像素数、缺失区域或非有限指标不符仍停止。八组统计图仍来自原 CSV；新版深度图、误差图和局部表全部来自当前导出数组，不替换成旧数字。每个样本的 `display_settings.json` 包含复核记录。若不一致尚未解释，不应声称这次图像严格复现了旧表。

若要减少重复推理，可先用 verify 检查已有 Base，再仅 export 完整模型，最后 render。若确认旧评测确实使用 batch size 4，可通过 `PAPER_BATCH_SIZE=4` 在相同条件下重试；默认仍为 1。告警模式没有扩大容差：绝对容差 `1e-3`、相对容差 `1e-4` 保持不变，严格模式仍是默认值。
