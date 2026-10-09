# 三类困难场景的论文可视化

当前默认可视化只比较 `Base` 和 `Base+A+B+C`，展示整张场景，不生成 R1/R2 或局部裁剪。两组读取各自真实检查点，保存目录仍使用分析配置名。

## 1. 三类场景与选择依据

| 目录 | 场景 | 固定 GT 几何筛选条件 | 排序依据 |
| --- | --- | --- | --- |
| `large_disparity_dominant` | 大视差为主 | 大视差至少 30 像素；至少两视角遮挡占有效像素不超过 10%；交集占大视差区域不超过 25% | GT 位移第 80 百分位，从大到小 |
| `occlusion_dominant` | 复杂遮挡为主 | 至少两视角遮挡至少 30 像素，且占有效像素至少 10%；交集占复杂遮挡区域不超过 35% | 至少两视角遮挡的有效像素比例，从大到小 |
| `joint_difficulty` | 交集明显 | 大视差与至少两视角遮挡交集至少 30 像素，占有效像素至少 5%，且占大视差区域至少 25% | 交集占有效像素的比例，从大到小 |

这些阈值是可视化案例选择规则，不修改评测掩码。满足交集条件的案例优先归入交集类，避免同一案例被当成不同主要困难反复展示。每类默认选两个不同 scan 的案例。筛选不使用完整模型的改善幅度；改善和退化都如实保留。几何类别不能直接代替重建误差结论，还要查看对应区域指标。

`select` 在 CPU 上读取原 CSV 对应测试集的 GT、相机和源视图，检查参考有效像素和大视差像素数与 CSV 一致，保存所有候选的几何统计及入选列表。缺少合格类别时明确报错，不把其他场景强行归入该类。

## 2. 每个案例的三行五列图

| 行 | 第 1 列 | 第 2 列 | 第 3 列 | 第 4 列 | 第 5 列 |
| --- | --- | --- | --- | --- | --- |
| 第一行 | 参考 RGB | GT 深度 | GT 大视差位移热图 | 源视角遮挡比例 | RGB 上的困难区域叠加 |
| 第二行 | Base 深度 | Base+A+B+C 深度 | Base 绝对误差 | Base+A+B+C 绝对误差 | 误差改善图 |
| 第三行 | 任一源视角遮挡掩码 | 至少两个源视角遮挡掩码 | 大视差掩码 | 大视差与任一遮挡交集 | 大视差与至少两视角遮挡交集 |

第一行的位移热图是 GT 点在源视图中的最大投影位移，单位像素；第三行的大视差掩码仍采用原评测的百分位定义，默认取较大位移的 20%。

遮挡按源 GT、相机投影和深度容差判断，投影越界或源 GT 无效不直接算遮挡。任一遮挡为 `occluded_count >= 1`，复杂遮挡为 `occluded_count >= 2`；“至少两个视角”与原来的多数比例掩码不是同一条件。遮挡比例仍是“被遮挡视角数 / 可比较源视角数”，沿用原评测定义。

RGB 叠加：红色为仅大视差，绿色为仅复杂遮挡，黄色为两者交集。误差图上的黄色轮廓分别沿固定 GT 大视差区域和复杂遮挡区域绘制，不按预测误差圈选，不加 R1/R2 标签。

误差改善定义：

$$\Delta E = |D_{\mathrm{Base}}-D_{\mathrm{GT}}|-|D_{\mathrm{Base+A+B+C}}-D_{\mathrm{GT}}|.$$

红色表示改善（正值），蓝色表示退化（负值），白色为零；有效 GT 外的深度和误差区域为黑色。Base 与完整模型使用相同的深度、误差颜色范围。图上不绘制坐标刻度、色条或色条刻度，颜色范围及单位保存在 `display_settings.json` 和 `figure_caption.txt`，便于论文图注使用。黄色仅为边界标记，不改变原始误差或区域指标，超过显示范围的误差仍完整参与数值计算。

## 3. 服务器一键运行

在项目根目录执行，默认每类两个 scan，同时处理 View5 和 View3：

```bash
CUDA_VISIBLE_DEVICES=3 PAPER_VERIFY_MODE=warn \
bash tools/run_paper_visualizations.sh \
  /home/disk_10T/lzh_data/dtu_training/mvs_training/dtu \
  checkpoints/dtu \
  all
```

流程为：读取并核对原 CSV → 全测试集固定 GT 几何选例 → 两组真实模型推理 → 三类整场景绘图。一键脚本不再自动重画八组统计图；既有表格保留，若需要原统计图仍可单独调用 stats。GT 筛选阶段不需要 GPU，会每处理 100 个样本打印一次进度。`PAPER_EXAMPLES_PER_CATEGORY=3` 可改成每类三个 scan。

保留严格复核作为默认值；上面的 `PAPER_VERIFY_MODE=warn` 是针对已出现小幅复现差异时，显式保留报告并继续画本次真实预测。身份、协议、区域像素数等结构错误仍停止。八组统计图来自原 CSV，新定性图与新局部区域表来自本次数组，复现状态记录在 JSON。

## 4. 用已有缓存先查看新版样式

```bash
python tools/visualize_paper_results.py render \
  --series all --outdir outputs/paper_visualizations --layout paper \
  --error_max 20 --gain_max 10
```

只重新绘图，不需要 GPU。旧 NPZ 的 `source_occluded` 可计算遮挡数量，只要缓存包含原区域评测使用的全部源视图。新版导出会直接保存区域遮挡数，支持区域视图数与推理视图数不同的情形。

已有缓存不一定包含三类合格场景。render 会按固定几何条件归类，并在 `category_coverage.json` 列出各类数量；不符合三类条件的旧案例放 `mixed_other`。要覆盖全部三类，应执行一键流程重新筛选，而不是强行改目录名字。

分步筛选命令：

```bash
python tools/visualize_paper_results.py select \
  --series View5 \
  --testpath /home/disk_10T/lzh_data/dtu_training/mvs_training/dtu \
  --testlist lists/dtu/test.txt \
  --eval_args_json tools/paper_eval_args.example.json \
  --outdir outputs/paper_visualizations --examples_per_category 2
```

后续 export、render 都使用 `--selection outputs/paper_visualizations/View5/scene_selection.json`；新案例缺少缓存时须先 export。View3 同理。

## 5. 文件保存

```text
outputs/paper_visualizations/View5/
  scene_selection.json
  Base/scanX_viewYY_light3/arrays.npz
  Base/scanX_viewYY_light3/figures/scenes/base_depth.png / base_abs_error.png
  Base+A+B+C/scanX_viewYY_light3/arrays.npz
  Base+A+B+C/scanX_viewYY_light3/figures/scenes/full_model_depth.png / full_model_abs_error.png
  scenes/
    scene_candidates.csv
    rendered_scene_index.csv
    category_coverage.json
    large_disparity_dominant/
      scanX_viewYY_light3/
        overview_3x5.png
        overview_3x5.pdf
        panels/15 张独立 PNG
        panels/with_title/15 张带标题、无色条和刻度的 PNG 与 PDF
        region_metrics.csv
        comparison_arrays.npz
        display_settings.json
        figure_caption.txt
    occlusion_dominant/...
    joint_difficulty/...
  View3 同样结构
```

独立 PNG 保持原预测/GT 图像分辨率，便于自行排版；整图与独立子图采用同一颜色映射。新输出统一写入 `scenes/`，已有 `comparison/` 中的 R1/R2 图片属于旧版历史输出，不会被当作新版结果。

## 6. 数值与图片配合

每个案例的 `region_metrics.csv` 只含两种模型，保留原七个区域并增加 `occluded_ge2`、`large_disp_and_occluded_ge2` 两个区域，总计两组九区域。

每行保存像素数、Abs、Acc2/4/8、三级搜索范围覆盖率与宽度、相对 Base 的平均误差降幅和阈值/覆盖率百分点差。新增改善、退化、无变化的像素比例，用于对照红蓝差值图；边界的黄色显示不参与指标计算。

掩码彼此有重叠，不相加作为总像素数。原 CSV 的 `occluded_majority` 指标仍保留原定义，新增的“至少两个视角”指标只从本次像素数组计算，不把原多数遮挡数据改名替代。单案例指标也不能替代完整测试集指标。

当前默认色标误差 0–20 mm、改善 ±10 mm。若细微变化不明显，可重新 render 并统一改为 `--error_max 10 --gain_max 5`；两组色标仍相同，原始数字不改变。图是否支持遮挡改善，要看固定遮挡掩码内的实际误差变化及对应指标。
