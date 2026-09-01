# 方法、完整消融与下一步

## 1. 研究目标

目标是在不更换 Vis-MVSNet 主干的前提下，改善大视差、遮挡和遮挡边界区域的深度估计。当前方法针对三个可定位的内部问题：

1. 原始 pair-depth 监督没有区分“该参考点在某个源视图中可见还是被遮挡”；
2. 级联 Stage 2/3 的固定窄范围可能把粗阶段错误永久锁在搜索区间之外；
3. 原始源视图可靠性 `exp(-uncertainty)` 对所有深度假设共享，不能表达“同一源图在不同候选深度上的匹配可靠性不同”。

因此定义三个独立因素：

- A：source-specific occlusion-aware supervision；
- B：uncertainty-adaptive cascade range；
- C：depth-hypothesis-aware source fusion。

## 2. 完整三因素设计

三个二值因素必须形成 (2^3=8) 组，而不是只做逐步叠加：

| `MODEL_TYPE` | code | A | B | C |
| --- | --- | ---: | ---: | ---: |
| `vis` | 000 | 0 | 0 | 0 |
| `oa` | 100 | 1 | 0 | 0 |
| `range` | 010 | 0 | 1 | 0 |
| `hyp` | 001 | 0 | 0 | 1 |
| `oa_range` | 110 | 1 | 1 | 0 |
| `oa_hyp` | 101 | 1 | 0 | 1 |
| `range_hyp` | 011 | 0 | 1 | 1 |
| `oa_full` | 111 | 1 | 1 | 1 |

配置只在 [model_variants.py](../models/model_variants.py) 中定义一次，训练与评测共用，避免两个入口对同一名称采用不同开关。

## 3. A：逐源遮挡监督

### 3.1 真实逐源标签

对参考像素 (x) 及源视图 (i)，使用参考 GT 深度恢复三维点，再投影至源视图。记投影深度为 (z_{rightarrow i})，采样到的源 GT 表面深度为 (z_i)，容差为：

```text
tau = max(occ_abs_tol, occ_rel_tol * z_i)
```

标签定义为：

```text
visible:  abs(z_ref_to_src - z_src) <= tau
occluded: z_ref_to_src - z_src > tau
```

投影出界、参考/源 GT 无效、投影点位于源表面明显前方的像素不参加该 pair 的可见性监督。这样每个参考像素对不同源图可以得到不同标签。

### 3.2 修正后的 pair 损失

原始 fused depth 监督不变：

```text
L_fused = mean_valid(abs(d_fused - d_gt) / depth_interval)
```

A 打开时，pair-depth L1 只在当前 source 可见处计算：

```text
L_pair_A = mean_visible(abs(d_pair - d_gt) / depth_interval)
```

uncertainty NLL 为：

```text
L_uncert = error * exp(-u) + u
```

可见像素的 `error` 同时更新 pair depth 与 uncertainty；遮挡像素的 `error.detach()` 只更新 uncertainty。原始 `occ_head` 使用 balanced focal BCE 学习真实逐源标签：

```text
L_A = L_fused
    + lambda_pair * L_pair_A
    + lambda_uncert * L_uncert_A
    + lambda_vis * L_occ
```

默认 `lambda_pair=1`、`lambda_uncert=1`、`lambda_vis=0.2`。

### 3.3 A 关闭时的严格行为

A=0 时：

- pair L1 使用原始参考有效掩码；
- uncertainty 使用原始、未 detach 的 pair error；
- 原 `occ_head` 不计算可见性损失；
- fused/pair/uncertainty 三项与原 `VisMVSLoss(occ_guide=False)` 数值一致。

这使 `hyp` 和 `range_hyp` 可以读取 C 所需的逐源标签，却不会隐式获得 A。

## 4. B：不确定性感知的级联范围

Stage 1 继续使用全局统一深度范围。每个阶段从融合概率体 (P_s(d,x)) 得到深度和标准差：

```text
mu_s(x) = sum_d P_s(d,x) * d
sigma_s(x) = sqrt(sum_d P_s(d,x) * (d - mu_s(x))^2)
```

下一阶段的 baseline 半宽与自适应半宽为：

```text
fixed_half_s = D_next / 2 * interval_next
adaptive_half_s(x) =
    clamp(k * sigma_s(x),
          min_scale * fixed_half_s,
          max_scale * fixed_half_s)
```

随后在以下端点间均匀采样固定数量的深度假设：

```text
lower(x) = clamp(mu_s(x) - adaptive_half_s(x), global_min, global_max)
upper(x) = clamp(mu_s(x) + adaptive_half_s(x), global_min, global_max)
```

默认：

```text
k = 2.0
min_scale = 1.0
max_scale = 2.0
```

因此第一版保持与 baseline 相同的最小范围，只允许高不确定像素最多扩大至两倍。深度假设数量不变，所以范围扩大不会增加 3D volume 的深度维度，但会降低该像素的深度采样分辨率。这是必须通过覆盖率与误差共同验证的代价。

B 不增加可学习参数，也不使用 source visibility GT。

## 5. C：深度假设相关源视图融合

### 5.1 原始融合的限制

原始 soft fusion 对 source (i) 使用二维 uncertainty：

```text
w_i(x) = exp(-u_i(x))
```

这个权重在整个深度轴共享。大视差与遮挡处，同一源视图可能只在部分深度假设上产生可靠匹配，因此共享权重表达能力不足。

### 5.2 三维 residual weight

C 使用轻量 3D head (H)，输入：

- 8 通道 pair-wise regularized volume feature；
- 匹配 score；
- 当前 pair probability；
- 归一化 depth coordinate。

输出受限残差：

```text
r_i(x,d) = residual_scale * tanh(H_i(x,d))
log w_i(x,d) = -u_i(x) + r_i(x,d)
```

融合为：

```text
F(x,d) =
    sum_i exp(log w_i(x,d)) * F_i(x,d)
    / sum_i exp(log w_i(x,d))
```

最后一层权重与 bias 均零初始化。初始时 (r_i=0)，所以接入 C 后不会在第一步随机破坏原融合。

### 5.3 C 自己的监督

在每个像素找到距离 GT 最近的深度假设 (d^*)，取对应 residual logit：

```text
d_star = argmin_d abs(D(d,x) - d_gt(x))
L_C = BalancedFocalBCE(H(x,d_star), M_vis)
```

默认 `lambda_C=0.1`。C 使用真实逐源可见性是为了训练自己的融合权重，并不改变 A=0 时的 baseline pair-depth 与 uncertainty 损失。

C 当前只在 `vismode=soft` 下启用。

## 6. 总损失

三个阶段权重保持 Vis-MVSNet 的 `0.5, 1.0, 2.0`。统一形式为：

```text
L_stage = L_fused
        + L_pair(A on/off)
        + L_uncert(A on/off)
        + I[A] * lambda_A * L_occ
        + I[C] * lambda_C * L_hyp_visibility

L_total = 0.5 * L_stage1 + 1.0 * L_stage2 + 2.0 * L_stage3
```

B 只改变实际深度假设，不额外添加损失。

## 7. 参数与 checkpoint 兼容性

- A 只改变监督，不增加参数；
- B 只改变深度范围，不增加参数；
- C 在三个阶段增加 hypothesis heads；
- `vis/oa/range/oa_range` 的参数键和形状一致；
- `hyp/oa_hyp/range_hyp/oa_full` 的参数键和形状一致；
- C 模型可加载无 C checkpoint，缺失的 head 参数保持零初始化；
- C checkpoint 不能用无 C 结构严格加载；
- B 虽没有参数，推理时仍必须启用正确开关。

主消融应全部从头训练。旧 checkpoint 初始化仅用于快速检查或额外 fine-tuning 实验。

## 8. 评测区域与指标

每个模型在相同 scan、view、light 和相同像素掩码上统计：

- `full`；
- `boundary`；
- `large_disparity`；
- `occluded_any`；
- `occluded_majority`；
- `large_disp_and_occluded`；
- `boundary_and_occluded`。

深度指标：

- Abs Error，越低越好；
- Acc2、Acc4、Acc8，越高越好。

范围诊断：

- `stage1/2/3_in_range`：GT 是否位于该阶段实际搜索区间；
- `stage1/2/3_range_width`：搜索宽度除以原始 base interval。

B 的有效性不能只看范围变宽。理想结果是目标区域的 Stage 2/3 coverage 提高，并同时带来 Abs/Acc 改善。

## 9. 如何读完整消融

单因素最直接比较：

- A：`oa - vis`；
- B：`range - vis`；
- C：`hyp - vis`。

在不同上下文中还要看四组成对比较：

- A：`oa vs vis`、`oa_range vs range`、`oa_hyp vs hyp`、`oa_full vs range_hyp`；
- B：`range vs vis`、`oa_range vs oa`、`range_hyp vs hyp`、`oa_full vs oa_hyp`；
- C：`hyp vs vis`、`oa_hyp vs oa`、`range_hyp vs range`、`oa_full vs oa_range`。

如果某模块单独有效、组合后无效，说明存在交互作用，不能只凭 `oa_full` 一组否定该模块。论文表格应保留八组，重点报告目标区域与 full 的共同变化。

## 10. 成功判据

A 的预期证据：

- `occluded_any/majority` 与交集区域 Abs 降低或 Acc 提高；
- pair visible/occluded 比例合理；
- occluded recall 不长期为 0；
- full 区域不能出现明显系统性退化。

B 的预期证据：

- Stage 2/3 target-region coverage 提升；
- width 增幅受控；
- coverage 提升转化为深度指标改善，而不是只扩大范围。

C 的预期证据：

- hypothesis visible recall 与 occluded recall 都非退化解；
- C 的四组成对比较中多数方向一致；
- 对大视差与遮挡交集的收益高于 full 区域随机波动。

主方法不必强行保留三个模块。如果八组结果显示某因素主效应持续为负，应删除该因素，把有效因素组合作为最终方法。

## 11. 当前训练与后续步骤

当前 `vis=000` 与 `oa=100` 已开始训练时，GPU 2、3 可运行剩余六组。命令见项目 [README.md](../README.md)。

全部训练完成后：

1. 检查每个 `logs.txt` 是否有 NaN、空监督或异常 recall；
2. 每组只用 `val.txt` 选择 checkpoint，记录选择规则；
3. 用完全相同的评测参数生成八组 `summary_metrics.csv`；
4. 先分析 full 与七个目标区域的八组表，再分析 A/B/C 主效应与交互；
5. 模型和超参数冻结后，再对 `test.txt` 运行最终报告；
6. 从逐图 `all_metrics.csv` 中挑选改善显著且有代表性的 scan/view；
7. 最后生成干净深度图、误差图、区域 mask 和局部放大，不再盲目遍历全部可视化。
