# Val 第二轮：分离训练权重与推理策略的影响

## 第一轮得到的依据

数据来自 `eval/val_diagnostics_view5_round1`，只使用 Val，像素加权：

- M2 beta=0 的全图 Abs 为 4.4808，beta=0.2 为 4.4811；关闭推理门控仍保留几乎全部收益。因此优先补训练期关闭门控的实验，而不是继续细调 beta。
- 原 M3 global+scale2 全图 Abs 为 5.8281，取消裁剪后为 5.6157；只修复了一部分差距。
- M3 none+scale1 的全图 Abs 为 5.5400；同权重下开启扩展后，七个区域 Abs 都变差，尽管 coverage 提升。
- 上述 M3 四组均使用原 M3 训练权重；下一步先用同一个 vis checkpoint 做对照，避免把训练历史与推理行为混为一谈。

训练仍用 `train.txt`，训练期间和训练结束后都只用 `val.txt`。区域评估固定 light=3、五视图推理、五视图划分区域。

## A：基线权重上的 M3 对照（先运行，无需训练）

在服务器 `vismvsnetgeo` 目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validate_val_diagnostics.py \
  --train_nviews 5 --suite m3_vis \
  --outdir eval/val_m3_vis_weights_view5_round2
```

三组均读取 `checkpoints/dtu/vis_view5/best_2mm.ckpt`：

| 输出标签 | 模型配置 | 目的 |
| --- | --- | --- |
| vis | 原始基线 | 同轮参照 |
| vis_weights_m3_none_scale1.0 | 不裁剪、不扩展 | 验证采样等价性 |
| vis_weights_m3_none_scale2.0 | 不裁剪、启用扩展 | 检查基线权重上的扩展影响 |

`--suite all` 保持第一轮原有九组；第二轮使用独立的 `m3_vis` 套件。输出 `comparison.md` 会列出每组的实际 checkpoint 来源，避免与原 M3 权重的结果混淆。

判读顺序：

1. none+scale1 应与 vis 接近。先检查七个区域 Abs/Acc2/coverage 和逐图记录；若出现明显差异，应排查实现或加载配置，不急于解释 scale2 的效果。本地增加了共享同一 state_dict 的三级完整前向等价测试，CPU 测试不替代服务器检查。
2. 若 none+scale2 仍增加目标区域误差，则现有扩展对基线权重也无即时收益；下一步先检查最近假设距离、尾部概率质量及 Stage 2 深度误差，暂缓组合训练。
3. 若 none+scale2 改善目标区域，而旧 M3 权重没有，则训练过程与采样适配更值得排查。此时再设计训练与推理一致的重训；单次推理对照不证明最终训练效果。

## B：M2 训练时关闭门控、保留可见性监督

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_m2_supervision_val.py \
  --train_nviews 5 --seed 1
```

脚本固定以下内容：

| 项目 | 设置 |
| --- | --- |
| 模型 | m2_visibility，M1/M3 均关闭 |
| 初始化 | 从头训练，不加载原 M2 权重 |
| 训练/验证列表 | train.txt / val.txt |
| 训练门控 | beta=0.0 |
| 可见性监督 | 保留，loss weight=0.2，Focal gamma=2.0 |
| pair depth / uncertainty loss | 权重均为 1.0，保持原设置 |
| 默认训练设置 | 5 视图、16 epochs、batch size=4、seed=1 |
| 优化器参数 | lr=0.001，10/12/14 epoch 按 2 倍衰减，wd=0 |
| 训练完成后的权重选择 | best_2mm.ckpt |
| 最终区域评估 | beta=0.0，Val light=3，5 视图 |

`MODEL_TYPE=m2_visibility` 保证可见性 GT 和辅助损失仍启用；beta=0 只让融合门控成为常数 1。训练期间保持原有 Val 光照范围，训练后单独做 light=3 区域评估。

默认输出：

```text
checkpoints/val_round2/m2_supervision_only_view5_seed1/
  experiment.json
  logs.txt
  model_*.ckpt
  best_2mm.ckpt
  best_abs.ckpt
  ...

eval/val_round2/m2_supervision_only_view5_seed1/
  manifest.json
  validation.log
  all_metrics.csv
  summary_metrics.csv
```

manifest 记录训练和最终验证命令、seed、源码与数据列表校验值。训练失败时不会继续评估；目录要求新建或为空。若训练成功但最终验证中断，可从 manifest 的 validation_command 单独恢复评估，不必重训。

比较三种结果：现有 vis、原 M2 的 beta=0 推理结果、新训练的 M2 beta=0。新对照用于区分训练期门控的影响；仍需用相同训练条件和重复 seed 确认辅助监督的收益。

## 路径、显存与命令预览

两个脚本都支持 `--datapath /实际/DTU路径`、`--batch_size` 和 `--dry_run`。M3 脚本支持 `--checkpoint_root`；M2 脚本支持 `--logdir`、`--outdir`、`--train_workers`、`--eval_workers`。

`--dry_run` 不写文件、不启动 GPU。使用同一张 GPU 时，先完成 A，再运行 B，避免两个进程争用显存。若改变训练 batch size，应记录并在相同 batch size 的基线下比较；不要把它当作没有影响的设置。

复验三视图训练时使用 `--train_nviews 3`，推理仍为五视图。重复训练可改变 `--seed`，默认输出目录会随 seed 变化。当前不自动运行多 seed 或组合训练。
