# 面向大视差与遮挡的 Vis-MVSNet 改进方案

## 1. 研究目标

目标是在不替换 Vis-MVSNet 主干的前提下，提高以下区域的深度估计质量：

- 大视差区域；
- 任一源视图遮挡区域；
- 多数源视图遮挡区域；
- 大视差与遮挡交集；
- 深度边界与遮挡交集。

方法沿 Vis-MVSNet 的三个关键环节展开：源视图融合、逐源可见性建模和级联深度假设生成。三个模块可以独立启用，因此能够形成完整的八组消融。

## 2. 基线

对参考视图和第 s 个源视图，Vis-MVSNet 得到成对代价体特征、成对深度和不确定性 u_s。原始 soft 融合权重为：

```text
w_s = exp(-u_s)
```

该权重对一个源视图的全部深度假设相同；原模型虽然包含 `occ_head`，但默认损失没有正确的逐源遮挡标签，也没有在融合中使用该输出。

## 3. M1：深度假设感知源视图融合

M1 对每个源视图 s 和每个深度假设 d 预测残差 r_(s,d)。输入包含：

```text
8-channel pair cost features
pair matching score
pair probability
normalized depth coordinate
```

网络使用轻量 3D 卷积并输出一个假设级 logit：

```text
r_(s,d) = lambda_h * tanh(H(F_(s,d)))
```

融合权重变为：

```text
w_(s,d) = exp(-u_s + r_(s,d))
```

输出层零初始化，因此刚开始训练时 `r=0`，模型严格退化为原始不确定性融合。M1 还在最接近 GT 深度的假设平面监督该 logit，使其学习当前源视图对正确深度假设是否可见。

代码位置：

```text
models/vismvsnet_oa.py::HypothesisWeightNet
models/vismvsnet_oa.py::SingleStage.forward
models/vismvsnet_oa.py::VisMVSLoss._hypothesis_logit_at_gt
```

## 4. M2：几何监督的源视图可见性建模

### 4.1 逐源标签

使用参考视图 GT 深度将参考像素反投影到三维，再投影到第 s 个源视图。设投影点在源相机坐标系中的深度为 z_proj，源视图 GT 深度为 z_src，则：

```text
tau = max(tau_abs, tau_rel * z_src)
visible = |z_proj - z_src| <= tau
occluded = z_proj > z_src + tau
```

只在投影有效、源深度有效且位于图像范围内的像素监督。标签是逐参考-源图像对的，不会把参考图有效掩码误当成所有源视图都可见。

### 4.2 可见性损失

原 `occ_head` 输出可见性 logit o_s：

```text
p_vis(s) = sigmoid(o_s)
```

使用类别平衡 Focal BCE：

```text
L_vis = FocalBCE(o_s, M_vis_s)
```

### 4.3 保守软门控

可见性不进行硬筛除，而是形成有下界的软门控：

```text
g_s = (1 - beta) + beta * p_vis(s)
```

默认 `beta=0.2`，所以 `g_s` 位于 `[0.8, 1.0]`。即使可见性预测错误，也不会完全删除一个源视图。零可见性 logit 对所有源视图产生相同门控，归一化后与原始融合完全一致。

M2 不改变原始 pair-depth 和 uncertainty 损失：

- 遮挡像素仍然具有 pair-depth 梯度；
- 不对遮挡区域执行 detach；
- 不使用 hard mask。

代码位置：

```text
datasets/dtu_yao.py
models/dynamic_visibility.py::ground_truth_pair_visibility
models/vismvsnet_oa.py::UncertNet
models/vismvsnet_oa.py::SingleStage.forward
models/vismvsnet_oa.py::VisMVSLoss
```

## 5. M3：覆盖保持的局部-扩展混合采样

### 5.1 目标

Stage 2/3 需要同时满足：

- 中央区域保持较小的深度采样间隔；
- 前一阶段误差较大时有少量假设覆盖更远深度；
- 总深度假设数量和显存开销保持不变。

### 5.2 假设划分

设阶段总假设数为 D，尾部扩展假设数为 K：

```text
D = D_local + K_wide
```

默认：

```text
Stage 2: D=32, D_local=24, K_wide=8
Stage 3: D=16, D_local=12, K_wide=4
```

中央 `D_local` 个假设始终使用原始间隔 delta。其余假设平均分配到左右尾部。

### 5.3 不确定性控制尾部跨度

设上一阶段预测标准差为 sigma_d，原始半范围为 h_base：

```text
s = clip(gamma * sigma_d / h_base, 1, s_max)
```

只把最外侧尾部位置乘以 s，并在外端点与中央相邻位置之间线性布置尾部假设。`hybrid_clip_mode=global` 时最后裁剪到 DTU 全局深度上下界，保留旧实现；`none` 时不进行该裁剪，以便单独验证尾部扩展的影响。

裁剪前，`s=1` 时拼接后的全部假设与原始均匀采样逐项相同；`s>1` 时中央局部假设不变，只有尾部向外扩展。全局裁剪可能改变中央假设并导致边界处重复采样；因此覆盖保持的性质针对裁剪前，或 `none` 模式。用 `hybrid_max_scale=1` 可以关闭扩展，同时独立控制裁剪。

因此 M3 不是把固定数量的假设均匀摊到更宽区间，而是在保持局部分辨率的同时补充搜索覆盖。

代码位置：

```text
models/vismvsnet_oa.py::_build_hybrid_depth_range
models/vismvsnet_oa.py::VisMVSModel.forward
```

## 6. 三模块联合公式

完整模型的源视图-深度假设融合权重为：

```text
log w_(s,d) = -u_s + r_(s,d) + log(g_s)
w_(s,d) = exp(log w_(s,d))
```

随后对所有源视图进行归一化：

```text
F_fused(d) = sum_s(w_(s,d) * F_(s,d)) / sum_s(w_(s,d))
```

三项职责不同：

- M1 判断某个源视图对具体深度假设的支持程度；
- M2 判断该像素在该源视图中的可见程度；
- M3 决定后续阶段在哪些深度位置建立假设。

## 7. 总损失

每个阶段保留原始融合深度、pair depth 和 uncertainty 项，并按启用模块加入辅助监督：

```text
L_stage = L_fused
        + lambda_pair * L_pair
        + lambda_uncert * L_uncert
        + lambda_vis * L_vis
        + lambda_hyp * L_hyp_vis
```

其中：

- `L_vis` 只在 M2 启用；
- `L_hyp_vis` 只在 M1 启用；
- M3 不增加损失和可学习参数；
- 三级损失权重仍为 `(0.5, 1.0, 2.0)`。

## 8. 正式八组

| `MODEL_TYPE` | code | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| `vis` | 000 | 0 | 0 | 0 |
| `m1_hyp` | 100 | 1 | 0 | 0 |
| `m2_visibility` | 010 | 0 | 1 | 0 |
| `m3_hybrid` | 001 | 0 | 0 | 1 |
| `m1_m2` | 110 | 1 | 1 | 0 |
| `m1_m3` | 101 | 1 | 0 | 1 |
| `m2_m3` | 011 | 0 | 1 | 1 |
| `full` | 111 | 1 | 1 | 1 |

正式表只使用这八组，不混入其他模块定义。

## 9. 实验顺序

建议按以下顺序进行：

1. 当前 M2 在 Val 上有正向结果，保留候选；先用已有 checkpoint 诊断 M2 门控强度和 M3 裁剪/扩展，流程见 [VAL_WORKFLOW.md](VAL_WORKFLOW.md)。
2. 保留已有 `vis`；已有 `hyp` checkpoint 可作为 `m1_hyp` 评测。
3. 单模块通过后训练 `m1_m2`、`m1_m3`、`m2_m3`。
4. 最后训练 `full`，避免在单模块无效时浪费组合训练时间。
5. 训练只用 `train.txt`；训练期间和训练结束后只用 `val.txt` 验证、选 checkpoint 和参数。后续不运行 `test.txt`。
6. 三视图训练和五视图训练都使用五视图推理，分析稀疏训练与正常训练条件。

单模块进入正式论文的最低判断标准：

```text
large_disparity Abs 下降
occluded_any 或 occluded_majority Abs 下降
large_disp_and_occluded Abs 下降
Acc2 不明显退化
full 区域无不可接受退化
Val 上按相同 checkpoint 规则对比，并在重复训练中确认改善方向
```

## 10. 区域评测

评测脚本为：

```text
tools/eval_region_metrics_dtu_yao.py
```

输出：

```text
all_metrics.csv
summary_metrics.csv
```

每张图和汇总表都包含：

```text
full
boundary
large_disparity
occluded_any
occluded_majority
large_disp_and_occluded
boundary_and_occluded
```

除 `abs/acc2/acc4/acc8` 外，还记录 Stage 1/2/3 GT in-range 比例和 Stage 2/3 范围宽度，用于检查 M3 是否真正提高目标区域覆盖。

## 11. 当前实现状态

已经实现：

- M1 假设级融合和 GT 深度平面的可见性监督；
- M2 逐源几何标签、平衡 Focal BCE、保守软门控；
- M3 局部密集与不确定性尾部扩展采样；
- 八组统一模型配置；
- 训练、队列训练、Val 区域评测和 CSV 元数据；
- Val 固定列表的 M2 beta 与 M3 裁剪/扩展推理诊断队列；
- 采样退化性质、损失梯度、checkpoint 参数结构和三模块联合前向测试。

仍需服务器实验确认：

- M2 单模块是否稳定改善遮挡区域；
- M3 单模块是否提高大视差区域 Stage 2/3 coverage 并降低 Abs；
- M1+M2 是否消除旧式硬遮挡处理造成的组合冲突；
- `beta`、尾部数量、`gamma` 和最大扩展比例的 Val 参数选择；
- 最终三视图、五视图 Val 消融与定向可视化。
