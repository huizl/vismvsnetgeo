# 同一 GT 网格的搜索范围来源诊断

模型：v2_vis；checkpoint：/home/disk_10T/lzh/triple/vismvsnetgeo/checkpoints/v2/v2_vis_view3_seed1/best_2mm.ckpt。
GT 和区域掩码固定；各阶段区间端点与 S3 深度用 nearest 选择到 GT 网格，不混合邻居深度。
误差是该 common-grid S3 预测的误差，与原生阶段/常规 bilinear 评估分开命名。
覆盖转移是这个固定对齐规则下的像素联合统计，不等于已证明的因果来源。

## 主要分解：初始漏覆盖还是后续丢失

所有占比的分母分别为该区域全部像素或 S3 绝对误差总和；各区域相互重叠，不能跨区域相加。

| 区域 | S1内→S3外 像素% | 其S3误差贡献% | S1外→S3外 像素% | 其S3误差贡献% | S1外→S3内 像素% |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 3.322 | 46.625 | 1.409 | 26.741 | 1.478 |
| boundary | 18.847 | 63.335 | 3.139 | 21.352 | 1.284 |
| large_disparity | 5.200 | 37.485 | 6.321 | 48.636 | 6.327 |
| occluded_any | 17.372 | 60.054 | 4.950 | 28.105 | 2.946 |
| occluded_majority | 25.648 | 67.710 | 5.159 | 23.562 | 2.320 |
| large_disp_and_occluded | 16.519 | 47.179 | 11.885 | 45.747 | 7.211 |
| boundary_and_occluded | 26.424 | 65.662 | 4.410 | 23.336 | 1.378 |

## 定位实际采样端点和级联阶段

in_out 表示源区间覆盖 GT、目标区间未覆盖 GT。其 gap 是 GT 到目标区间的距离。

| 区域 | 比较 | in_out像素% | 其S3误差贡献% | 平均越界mm |
| --- | --- | ---: | ---: | ---: |
| full | original_to_s1 | 0.210 | 1.035 | 3.973 |
| full | s1_to_s2 | 0.923 | 26.332 | 69.736 |
| full | s2_to_s3 | 2.563 | 21.638 | 24.038 |
| boundary | original_to_s1 | 0.220 | 0.905 | 3.878 |
| boundary | s1_to_s2 | 4.580 | 32.218 | 68.019 |
| boundary | s2_to_s3 | 14.560 | 31.787 | 24.625 |
| large_disparity | original_to_s1 | 0.879 | 1.784 | 3.978 |
| large_disparity | s1_to_s2 | 1.792 | 24.522 | 95.806 |
| large_disparity | s2_to_s3 | 4.098 | 15.178 | 27.006 |
| occluded_any | original_to_s1 | 0.510 | 1.230 | 3.989 |
| occluded_any | s1_to_s2 | 5.408 | 36.189 | 75.725 |
| occluded_any | s2_to_s3 | 12.551 | 24.981 | 25.769 |
| occluded_majority | original_to_s1 | 0.490 | 1.173 | 3.957 |
| occluded_majority | s1_to_s2 | 8.994 | 43.773 | 81.705 |
| occluded_majority | s2_to_s3 | 17.361 | 24.909 | 27.278 |
| large_disp_and_occluded | original_to_s1 | 1.200 | 1.938 | 3.999 |
| large_disp_and_occluded | s1_to_s2 | 6.247 | 32.231 | 101.555 |
| large_disp_and_occluded | s2_to_s3 | 11.637 | 16.599 | 29.454 |
| boundary_and_occluded | original_to_s1 | 0.319 | 1.038 | 3.918 |
| boundary_and_occluded | s1_to_s2 | 7.090 | 35.487 | 72.527 |
| boundary_and_occluded | s2_to_s3 | 19.742 | 30.851 | 26.332 |

## 初始区间越界方向

| 区域 | 区间 | 方向 | 像素% | 平均越界mm | 最大越界mm |
| --- | --- | --- | ---: | ---: | ---: |
| full | original | below | 0.006 | 2.313 | 6.738 |
| full | original | above | 2.672 | 74.368 | 311.873 |
| full | s1 | below | 0.006 | 2.313 | 6.738 |
| full | s1 | above | 2.882 | 76.601 | 319.823 |
| boundary | original | below | 0.011 | 2.723 | 6.294 |
| boundary | original | above | 4.192 | 90.755 | 311.873 |
| boundary | s1 | below | 0.011 | 2.723 | 6.294 |
| boundary | s1 | above | 4.412 | 93.981 | 319.823 |
| large_disparity | original | below | 0.028 | 2.313 | 6.738 |
| large_disparity | original | above | 11.741 | 77.463 | 311.873 |
| large_disparity | s1 | below | 0.028 | 2.313 | 6.738 |
| large_disparity | s1 | above | 12.620 | 79.738 | 319.823 |
| occluded_any | original | below | 0.008 | 2.238 | 6.294 |
| occluded_any | original | above | 7.378 | 79.559 | 311.873 |
| occluded_any | s1 | below | 0.008 | 2.238 | 6.294 |
| occluded_any | s1 | above | 7.888 | 82.105 | 319.823 |
| occluded_majority | original | below | 0.012 | 2.066 | 6.294 |
| occluded_majority | original | above | 6.977 | 75.318 | 311.873 |
| occluded_majority | s1 | below | 0.012 | 2.066 | 6.294 |
| occluded_majority | s1 | above | 7.468 | 78.061 | 319.823 |
| large_disp_and_occluded | original | below | 0.022 | 2.238 | 6.294 |
| large_disp_and_occluded | original | above | 17.874 | 82.978 | 311.873 |
| large_disp_and_occluded | s1 | below | 0.022 | 2.238 | 6.294 |
| large_disp_and_occluded | s1 | above | 19.074 | 85.459 | 319.823 |
| boundary_and_occluded | original | below | 0.006 | 2.889 | 6.294 |
| boundary_and_occluded | original | above | 5.463 | 90.826 | 311.873 |
| boundary_and_occluded | s1 | below | 0.006 | 2.889 | 6.294 |
| boundary_and_occluded | s1 | above | 5.782 | 93.543 | 319.823 |

决策：

- original 已漏覆盖：先核查相机/深度单位/掩码和初始范围规则。不能按每张 Val GT 动态扩展正式推理范围。
- original 覆盖而 S1 漏覆盖：优先检查实际采样端点。
- S1 内→S3 外误差贡献显著：优先做候选保留/恢复改动，并由 S1→S2、S2→S3 定位阶段。
- 相邻阶段的 in_out 组可能重叠或涉及后来找回的像素，其误差贡献不能直接相加。
- 原始区间是数据实际提供的 depth_values 的 min/max，并非从 GT 推导。空组均值为 nan。
