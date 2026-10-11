# BlendedMVS 跨数据集深度评测

使用已有 DTU 检查点比较 Base 与 Base+A+B+C，不重新训练，不读取旧 DTU 指标。
下面的命令均在服务器项目根目录执行。

## 数据结构

```text
/home/disk_10T/lzh_data/BlendedMVS/
  场景ID/
    blended_images/00000000.jpg
    cams/00000000_cam.txt
    cams/pair.txt
    rendered_depth_maps/00000000.pfm
```

读取未加 `_masked` 后缀的 RGB 图像。GT 必须与图像原始尺寸一致。
源视图按配对文件排序取前四个，不根据 GT 或模型结果改选。
这五个视图的图像、相机和 GT 缺失时会报告具体路径。
源视图不足、参考相机范围无效、参考 GT 为空的排除记录保存到
`skipped_samples.csv`。无源 GT 和越界位置不计为遮挡。

## 数据检查

```bash
python tools/eval_blendedmvs.py \
  --testpath /home/disk_10T/lzh_data/BlendedMVS \
  --check_only
```

不需要 GPU 或检查点。检查选定样本的文件存在性、相机及配对格式，
并实际读取默认三个样本的图像、GT 和几何信息。
保存 `outputs/cross_dataset/BlendedMVS/View5/data_check.json`。
不是对所有 GT 内容的完整扫描，完整评测会继续逐样本检查。

## 少量样本验证

```bash
CUDA_VISIBLE_DEVICES=3 \
BLENDED_MAX_SAMPLES=4 \
BLENDED_OUTPUT=outputs/cross_dataset/BlendedMVS_smoke \
bash tools/run_blendedmvs_paper.sh \
  /home/disk_10T/lzh_data/BlendedMVS \
  checkpoints/dtu \
  View5
```

少量样本可能不能覆盖三类困难场景，或者没有符合条件的改善案例。
程序报告选图不足，已经计算的指标仍会保存。不能用调试子集代替全量评价。

## 全量评价

先运行五视图训练检查点：

```bash
CUDA_VISIBLE_DEVICES=3 bash tools/run_blendedmvs_paper.sh \
  /home/disk_10T/lzh_data/BlendedMVS checkpoints/dtu View5
```

两套训练检查点都评价：

```bash
CUDA_VISIBLE_DEVICES=3 bash tools/run_blendedmvs_paper.sh \
  /home/disk_10T/lzh_data/BlendedMVS checkpoints/dtu all
```

View3/View5 指检查点的训练设置，两套评价默认均为一个参考视图和四个源视图。
推理 batch size 为1，每个样本顺序执行两个模型。

按现有 `tools/paper_configs.py` 显示配置映射，默认检查点如下：

| 显示配置 | 原始 model_type | View5 检查点 |
| --- | --- | --- |
| Base | range | checkpoints/dtu/range_view5/best_2mm.ckpt |
| Base+A+B+C | oa_hyp | checkpoints/dtu/oa_hyp_view5/best_2mm.ckpt |

View3 对应 `range_view3` 和 `oa_hyp_view3`。
显示名称不改变模型开关，实际 model_type、ABC 编码、检查点路径和 SHA256
会写入元数据。检查点严格加载，不自动替换未匹配层。

检查点目录不同可显式指定：

```bash
CUDA_VISIBLE_DEVICES=3 python tools/eval_blendedmvs.py \
  --testpath /home/disk_10T/lzh_data/BlendedMVS \
  --series View5 \
  --base_checkpoint /实际目录/range_view5/best_2mm.ckpt \
  --full_checkpoint /实际目录/oa_hyp_view5/best_2mm.ckpt
```

若要运行原始代码中的另一组开关，可同时传
`--base_model_type vis --full_model_type oa_full` 及对应检查点。
不能只改显示标签来改变实验身份。

## 评价名单

未传 `--testlist` 时，评价根目录下全部包含 `cams/pair.txt` 的场景，
不声称这是某个官方测试划分。只有未用于检查点训练或调参的场景才能作为
跨数据集评价。指定固定名单：

```bash
CUDA_VISIBLE_DEVICES=3 \
BLENDED_TESTLIST=/实际目录/blended_eval_scenes.txt \
bash tools/run_blendedmvs_paper.sh \
  /home/disk_10T/lzh_data/BlendedMVS checkpoints/dtu View5
```

名单每行一个场景目录名。评价名单和参数应在查看结果前确定。

## 尺度与指标

只使用相机文件，为每个场景计算统一尺度：

```text
scene_scale = 1000 / median(该场景有效相机 depth_max)
normalized_depth = native_depth * scene_scale
normalized_translation = native_translation * scene_scale
```

尺度不依赖 GT、预测、检查点或所选参考样本，两个模型相同。
默认最多768×576，缩放后尺寸为64的倍数。图像、GT 与内参同步缩放，
平移随深度尺度变化。网络使用四分之一图像尺度的投影矩阵，
GT 遮挡计算使用完整评价网格的内参。

- `abs`、`rmse`、`acc2_normalized/acc4_normalized/acc8_normalized`
  及图像着色使用归一化场景单位，不能写成 mm。
- `abs_native` 在逐图及逐场景表中报告，不将不同场景原始单位直接合并。
- `abs_rel` 是平均绝对相对误差；
  `rel_acc1/2/5` 是相对误差严格小于1%/2%/5%的像素比例。
- 准确率变化输出百分点，Abs/AbsRel 下降输出相对百分比。
- 保存三个阶段的 GT-in-range 覆盖率及范围宽度/基础间隔。

评价域固定为参考有效 GT，不根据任一模型结果重筛。有效 GT 上的非有限
预测会中止评价，不删除这些像素。源 GT 最近邻采样，避免跨深度边缘及
缺失位置插值。遮挡容差为 `max(2个归一化单位, 0.01×源GT深度)`，不是2 mm。

九类区域：整体、边界、大视差、任一遮挡、至少一半可比较源遮挡、
至少两个源遮挡、大视差与任一遮挡、大视差与至少两个源遮挡、
边界与任一遮挡。大视差为每图二维重投影位移较大的约20%。

总体表输出像素加权、逐图平均及逐场景平均。像素加权 RMSE 先汇总平方误差
再开平方。不同数据集分别报告，新协议不替换或声称复现已有 DTU CSV。

## 可视化

全部指定样本的定量结果先保存，再在符合 GT 分类规则的样本内按
完整模型相对 Base 的整体 Abs 降幅降序选图，每类默认两个不同场景的改善案例。
数量不足时报告实际数量，不强行凑图。三类条件与现有论文工具一致：

- 大视差为主：大视差像素至少30，至少两源遮挡占比不超过10%，
  两者交集占大视差区域不超过25%。
- 遮挡为主：至少两源遮挡像素至少30，占有效域至少10%，
  交集占该遮挡区域不超过35%。
- 交集明显：交集像素至少30，占有效域至少5%，占大视差区域至少25%。
  满足交集条件时优先归到交集类。

每例输出3×5拼图、15张独立PNG、带标题PNG/PDF、区域指标、着色说明、
模型预测PFM。没有R1/R2、图边刻度或颜色条。RGB叠加用红/绿/黄表示
大视差/至少两源遮挡/交集；误差改善图红色改善、蓝色变差，
黄色轮廓标出 GT 困难区域。着色截断不改变指标，预测PFM恢复原始场景单位。

```text
outputs/cross_dataset/BlendedMVS/View5/
  protocol.json
  skipped_samples.csv
  scene_selection.json
  Base/exported_metrics/
    all_metrics.csv
    summary_metrics.csv
    per_scene_metrics.csv
  Base+A+B+C/exported_metrics/
    all_metrics.csv
    summary_metrics.csv
    per_scene_metrics.csv
  comparison/
    summary_comparison.csv
    scene_candidates.csv
    large_disparity_dominant/样本/overview_3x5.png
    occlusion_dominant/样本/overview_3x5.png
    joint_difficulty/样本/overview_3x5.png
    各样本目录/panels/*.png
  Base/所选样本/arrays.npz、depth_native.pfm、metadata.json
  Base+A+B+C/所选样本/arrays.npz、depth_native.pfm、metadata.json
```

总体结论看 `comparison/summary_comparison.csv`，挑选的改善图只作为典型案例。
重复运行或变更名单后，有效图片以最新 `scene_selection.json` 为准。
不删除旧图片。只为所选案例保存完整预测数组，避免全量预测文件占据大量磁盘。

## 本地验证

```bash
python -m unittest discover -s tests -p test_blendedmvs.py -v
```

测试数据接口、相机缩放、遮挡及统计，不需要PyTorch或GPU。
真实检查点推理与出图仍需在服务器运行少量样本命令验证。
