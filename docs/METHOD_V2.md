# v2：三个独立候选模块与验证路线

2026-09-08 更新：本文件保留 v2 设计和首轮实验计划。三视图训练的 v2 四组结果现已齐全，见 [联合审计](V2_RESULT_AUDIT.md)。下文“尚无结果”为设计时状态；后续研究以 [局限驱动路线](LIMITATION_DRIVEN_PLAN.md) 为准，不再预设必须得到三个有效模块。

目标是最终得到三个可独立消融、具有正向作用的改进。当前只把已验证的现象与待验证的候选分开记录，不能预设三个模块都会提高指标。训练使用 train.txt，所有模型选择与评估使用 val.txt。

## 现有证据对设计的约束

- M2 在训练和推理均关闭可见性门控时，仍改善了 Val 七个区域的像素加权 Abs 与 Acc2；保留辅助监督方向，停止把 beta 搜索作为主线。
- 原 M3 的全局裁剪损害部分搜索覆盖；取消裁剪后，扩展仍存在 coverage 提升但 Abs 变差的问题。相同 vis 权重下的扩展收益很小且区域方向不一致，因此 v2 不继续沿用尾部扩展。
- M1 旧假设融合尚没有当前定义下完整的正向证据。旧 hyp checkpoint 结构兼容不等于新 M1 已验证有效。v2 在假设融合中增加可解释的投影有效性约束，仍需独立训练确认。

## M1：投影有效性约束的假设级源视图融合

原融合可能给投影到图像外、或位于源相机后方的假设位置分配权重。此时 warped feature 不能提供正常的匹配证据，但后续正则化仍可能产生非零特征。

对每个参考像素、源视图和深度假设计算投影有效性 V：投影坐标有限、源相机深度为正、投影落在图像内。此标签来自相机和当前深度假设，不需要源视图 GT。

融合权重：

```text
r(s,d) = lambda * tanh(H(pair_features, score, probability, depth_coordinate))
g_proj(s,d) = 0.05 + 0.95 * V(s,d)
w(s,d) = exp(-u(s) + r(s,d)) * g_proj(s,d)
```

按源视图归一化后融合代价特征。正下界保持所有源视图均无效时的分母有定义。投影有效性不等于遮挡可见性：图像内部被遮挡的位置仍可能 V=1，不能把它当成遮挡标签。

保留已有轻量假设残差头，零初始化输出层；v2 队列使用 lambda=0.5。v2 M1 不使用旧的 GT 平面可见性辅助损失，与 M2 的监督职责分开。

有效投影区域起始残差为零；无效投影的处理从开始就改变，因此 M1 不保证整体严格退化到 vis。旧 warping 的坐标约定保持不变，只有新路径返回有效掩码并清理无效投影。

## M2：训练期的逐源可见性辅助监督

沿用已经出现正向结果的设置：几何逐源可见性标签、平衡 Focal BCE，权重 0.2、gamma=2。M2 不启用显式融合门控，训练和推理 beta 均为 0。

```text
L = L_fused + L_pair + L_uncertainty + 0.2 * L_visibility
```

可见性头和 uncertainty 头共享前面的熵特征网络，辅助监督通过共享特征及上游网络影响深度估计。M1 的假设融合与 M3 的中心选择均可独立开启。

M2 的已有结果支持其作为候选，但尚需匹配训练条件与重复 seed，不能将一次结果解释为所有场景稳定提升。

## M3：参考特征引导的级联深度中心上采样

普通双线性上采样会混合相邻粗深度。在前景和背景交界处，混合后的中心可能落在两层表面之间，影响下一级局部搜索。

新 M3 在 Stage 1→2、Stage 2→3 的中心传递处，使用目标分辨率的参考图特征与四个粗深度邻居的相对深度，预测四个插值权重。令 b_i 为 align_corners=False 的双线性权重：

```text
a_i = 2 * tanh(G(reference_features, relative_neighbor_depths)_i)
q_i = b_i * exp(a_i) / sum_j(b_j * exp(a_j))
center_next = sum_i(q_i * depth_neighbor_i)
```

输出层零初始化，因此起点为双线性插值。权重非负、总和为 1，中心保持在四邻域深度的最小值与最大值之间。后续仍使用原始均匀假设：Stage 2 为 32 个，Stage 3 为 16 个，间隔和范围宽度不变；不做旧 M3 的尾部扩展或全局裁剪。

该设计改变搜索中心，不保证 coverage 永远不下降，也不能凭零初始化或凸组合性质认定指标会提高。目标是学习减少跨表面混合，收益由 Val 深度误差与覆盖率共同检验。

## 八组独立配置

| MODEL_TYPE | code | M1 投影约束融合 | M2 仅监督 | M3 引导中心 |
| --- | --- | --- | --- | --- |
| v2_vis | 000 | 0 | 0 | 0 |
| v2_m1 | 100 | 1 | 0 | 0 |
| v2_m2 | 010 | 0 | 1 | 0 |
| v2_m3 | 001 | 0 | 0 | 1 |
| v2_m1_m2 | 110 | 1 | 1 | 0 |
| v2_m1_m3 | 101 | 1 | 0 | 1 |
| v2_m2_m3 | 011 | 0 | 1 | 1 |
| v2_full | 111 | 1 | 1 | 1 |

旧八个 MODEL_TYPE 保持原含义。CSV 新增 method_revision、factor_m1_projection_validity、factor_m3_guided_centers、visibility_gate_enabled，避免将旧采样 M3 和新中心 M3 混为一组。

v2 M1 单独不要求可见性 GT；只有含 M2 的配置需要该 GT。v2 所有配置都关闭可见性门控，输入的 visibility_fusion_beta 被规范为 0。v2 M1 的假设残差尺度在队列中固定为 0.5，单独使用旧 shell 入口时需要显式传入 HYPOTHESIS_RESIDUAL_SCALE=0.5。

## 首轮运行：基线与三个单模块

在服务器的 vismvsnetgeo 目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_v2_val.py --train_nviews 5 --seed 1
```

依次从头训练四组，不使用旧 M3 checkpoint。默认保留原训练参数：16 epochs、batch size=4、lr=0.001、10/12/14 epoch 衰减、相同 seed。每组训练后自动加载自己的 best_2mm.ckpt，在 Val light=3 上评估。相同 seed 不代表不同架构必定有逐参数相同的初始化；最终需重复训练。

```text
checkpoints/v2/v2_vis_view5_seed1/
checkpoints/v2/v2_m1_view5_seed1/
checkpoints/v2/v2_m2_view5_seed1/
checkpoints/v2/v2_m3_view5_seed1/

eval/v2_val_view5_seed1/manifest.json
eval/v2_val_view5_seed1/comparison.md
eval/v2_val_view5_seed1/<model>/all_metrics.csv
eval/v2_val_view5_seed1/<model>/summary_metrics.csv
eval/v2_val_view5_seed1/<model>/validation.log
```

先预览命令可追加 `--dry_run`。路径不同时使用 `--datapath`、`--log_root`、`--out_root`。只训练某个候选可指定 `--models v2_m3`，但该轮没有 v2_vis 时对比表只列绝对指标，不生成相对基线差值。

单模块通过后，再显式训练三个两两组合和完整模型：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_v2_val.py \
  --models v2_m1_m2 v2_m1_m3 v2_m2_m3 v2_full \
  --train_nviews 5 --seed 1 --out_root eval/v2_combinations_view5_seed1
```

已存在的非空权重目录和评估根目录会被拒绝，避免重跑覆盖。训练失败不会继续评估；后续阶段失败时，manifest 保留命令，可以单独恢复该阶段。

## 如何认定三个改进具有正向作用

1. 每个单模块相对同条件 v2_vis，检查全图、大视差、任一遮挡、大视差∩遮挡的 Abs/Acc2，边界及其他遮挡区域作为补充。不能只选最有利的区域或聚合方式。
2. 记录像素加权与图像平均、逐 scan 胜负和退化场景；收益不能仅由单个场景解释。
3. 对候选做配对 seed 复验。M1/M3 尚无新训练结果，先不赋予“有效模块”的结论。
4. 通过单模块后检查组合相对组成模块的增量：独立有效不代表组合必然互补。
5. 若某个候选无效，修改或替换该候选，保留失败记录；目标是获得三个有证据支持的改进，而不是预设三个名字都必须成功。
