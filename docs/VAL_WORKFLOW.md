# 只使用 Val 的后续实验流程

## 固定设置

- 训练数据：`lists/dtu/train.txt`。
- 训练每个 epoch 后验证，以及训练结束后的区域评估：`lists/dtu/val.txt`。
- 后续不运行 `test.txt`，不使用已有 Test 指标选模块、参数或 checkpoint。
- 先使用五视图训练模型，推理与区域划分均固定五视图，区域评估 `light=3`。
- 首轮统一比较 `best_2mm.ckpt`；若分析 `best_abs.ckpt`，所有模型一起切换，单独输出一轮结果。
- `view3` 是三视图训练模型，后续复验时仍用五视图推理；它不是三视图推理实验。
- 训练期间原有 Val 光照范围不变；区域评估的 light=3 与训练日志的总体指标不直接混用。

## 第一轮：已有权重做推理诊断

在服务器的 `vismvsnetgeo` 目录运行。无需先重训。默认读取：

```text
checkpoints/dtu/vis_view5/best_2mm.ckpt
checkpoints/dtu/m2_visibility_view5/best_2mm.ckpt
checkpoints/dtu/m3_hybrid_view5/best_2mm.ckpt
```

一次运行全部九组：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_val_diagnostics.py \
  --train_nviews 5 --suite all \
  --outdir eval/val_diagnostics_view5_round1
```

数据或权重路径不同时，加 `--datapath /实际/DTU路径 --checkpoint_root /实际/checkpoints/dtu`。可先附加 `--dry_run` 检查命令；该模式不创建输出、不读取数据或 checkpoint、不启动 GPU 推理。

只跑 M2 或 M3 时分别指定 `--suite m2`、`--suite m3`，各五组（含一个 vis）。输出目录必须新建或为空；失败后的日志保留，再次运行请使用新的输出目录。

### M2：同一个 M2 checkpoint，四组 beta

| beta | 含义 |
| --- | --- |
| 0.0 | 关闭推理门控，保留训练过可见性监督的权重 |
| 0.1 | 较弱门控 |
| 0.2 | 复现当前设置 |
| 0.3 | 略增强门控 |

beta=0 相对 beta=0.2 的差异用于分析门控贡献；beta=0 仍不是原始 vis，也不能完全分离训练期间门控与辅助监督的贡献。不要将这四组标作分别训练的消融。

### M3：同一个 M3 checkpoint，四组采样设置

其他参数固定 `sigma_scale=2`，Stage 2/3 的 wide_num 固定 8/4。

| 标签 | clip_mode | max_scale | 用途 |
| --- | --- | --- | --- |
| m3_global_scale1.0 | global | 1 | 均匀采样加全局裁剪 |
| m3_global_scale2.0 | global | 2 | 复现当前 M3 |
| m3_none_scale1.0 | none | 1 | 原始均匀采样边界策略，使用 M3 训练权重 |
| m3_none_scale2.0 | none | 2 | 只保留尾部扩展，取消全局裁剪 |

先在相同 scale 下比较 global/none，再在相同 clip_mode 下比较 scale=1/2。none+scale1 也不是重新训练的 vis。切换采样导致训练/推理设置不同，所以首轮用于定位问题，最终效果由一致设置的训练确认。

输出内容：

```text
comparison.md            七个区域的指标、相对 vis 的差值与 Stage 2/3 覆盖率/宽度
manifest.json            完整命令、checkpoint 路径、代码和 val 清单的 SHA256
<setting>.log            每次推理的完整日志
<setting>/all_metrics.csv
<setting>/summary_metrics.csv
```

宽度指标已除以基础深度间隔，不是 mm。CSV 新增 `hybrid_clip_mode`，用于区分新旧采样策略。比较表不自动选冠军。

## 第二轮：按 Val 结果决定重训设置

### M2

1. 若 beta=0.1/0.2/0.3 有目标区域改善、全图基本保持的候选，下一轮只重训这个候选。
2. 若 beta=0 最好，说明当前推理门控没有带来净收益。优先检查训练日志中的可见性准确率、遮挡召回率，再决定是否调整监督；不继续盲目增大 beta。
3. 若所有 beta 差异都很小，先重复当前训练，确认收益是否超过训练波动，再投入更大搜索。

### M3

1. 若 none 相对相同 scale 的 global 恢复了大视差 coverage 并降低 Abs，下一轮验证取消全局裁剪的训练。
2. 若 none+scale2 又优于 none+scale1，才有证据继续验证尾部扩展；若相反，先考虑更保守的 `max_scale=1.5` 或只在后级扩展，暂不训练组合模型。
3. 若 coverage 提升但 Abs 仍退化，不能仅凭 coverage 保留 M3。下一步应检查采样间距与代价正则化、前一阶段深度偏差；需要新增逐像素诊断，而不是直接继续扩大范围。

下面是“第一轮支持取消裁剪并保留扩展后”的重训命令示例，不代表该设置已验证有效：

```bash
GPU=0 MODEL_TYPE=m3_hybrid \
TRAINLIST=lists/dtu/train.txt VALLIST=lists/dtu/val.txt \
TRAIN_NVIEWS=5 EVAL_NVIEWS=5 \
HYBRID_CLIP_MODE=none HYBRID_MAX_SCALE=2.0 \
LOGDIR=./checkpoints/val_round2/m3_none_scale2_view5 \
bash tools/train_view5.sh
```

训练结束后，用完全相同采样设置评估 `val.txt`：

```bash
GPU=0 MODEL_TYPE=m3_hybrid \
CHECKPOINT=./checkpoints/val_round2/m3_none_scale2_view5/best_2mm.ckpt \
LABEL=m3_none_scale2_view5 \
TESTLIST=lists/dtu/val.txt \
HYBRID_CLIP_MODE=none HYBRID_MAX_SCALE=2.0 \
OUTDIR=./eval/val_round2/m3_none_scale2_view5 \
bash tools/eval_regions_view5.sh
```

`TESTLIST` 是历史变量名，以上命令读取的是 Val。训练脚本已经默认 Val，区域评估脚本也已默认 Val；M3 裁剪默认仍为 global，保证旧 checkpoint 的默认评估行为不被悄然改变。

## 决策口径

先看 full、large_disparity、occluded_any、large_disp_and_occluded 的 Abs 和 Acc2，再看其余三个区域与 coverage。优先保留目标区域 Abs 同向下降且全图没有明显损失的设置；出现指标权衡时保留完整表，不只挑单项最优。

M2 当前五视图 Val 参照（像素加权）：

| 区域 | vis Abs | M2 Abs |
| --- | --- | --- |
| full | 4.5423 | 4.4811 |
| large_disparity | 10.9524 | 10.7467 |
| occluded_any | 21.1716 | 20.7237 |
| large_disp_and_occluded | 31.7535 | 30.7240 |

首轮复现项应与现有 CSV 接近；若差异大，先核查 checkpoint、服务器代码和数据设置，不立即解释成参数收益。最终候选应重复训练，并在 Val 逐 scan 检查是否由少数场景主导。确认 M2/M3 单模块后再跑组合，最后复验三视图训练。
