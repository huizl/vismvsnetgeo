# 同一 GT 网格的搜索范围来源诊断

模型：v2_vis；checkpoint：/home/disk_10T/lzh/triple/vismvsnetgeo/checkpoints/v2/v2_vis_view3_seed1/best_2mm.ckpt。
GT 和区域掩码固定；各阶段区间端点与 S3 深度用 nearest 选择到 GT 网格，不混合邻居深度。
误差是该 common-grid S3 预测的误差，与原生阶段/常规 bilinear 评估分开命名。
覆盖转移是这个固定对齐规则下的像素联合统计，不等于已证明的因果来源。

## 主要分解：初始漏覆盖还是后续丢失

所有占比的分母分别为该区域全部像素或 S3 绝对误差总和；各区域相互重叠，不能跨区域相加。

| 区域 | S1内→S3外 像素% | 其S3误差贡献% | S1外→S3外 像素% | 其S3误差贡献% | S1外→S3内 像素% |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 10.293 | 88.672 | 0.162 | 1.291 | 0.081 |
| boundary | 23.427 | 83.872 | 0.380 | 3.041 | 0.141 |
| large_disparity | 24.738 | 92.849 | 0.801 | 2.516 | 0.404 |
| occluded_any | 26.950 | 88.701 | 0.629 | 2.932 | 0.260 |
| occluded_majority | 35.012 | 90.774 | 0.507 | 2.409 | 0.153 |
| large_disp_and_occluded | 36.196 | 90.300 | 1.773 | 5.127 | 0.740 |
| boundary_and_occluded | 32.111 | 86.638 | 0.549 | 3.439 | 0.197 |

## 定位实际采样端点和级联阶段

in_out 表示源区间覆盖 GT、目标区间未覆盖 GT。其 gap 是 GT 到目标区间的距离。

| 区域 | 比较 | in_out像素% | 其S3误差贡献% | 平均越界mm |
| --- | --- | ---: | ---: | ---: |
| full | original_to_s1 | 0.000 | 0.000 | nan |
| full | s1_to_s2 | 5.889 | 69.196 | 68.115 |
| full | s2_to_s3 | 4.528 | 19.968 | 36.831 |
| boundary | original_to_s1 | 0.000 | 0.000 | nan |
| boundary | s1_to_s2 | 7.410 | 50.387 | 79.686 |
| boundary | s2_to_s3 | 16.175 | 33.834 | 28.745 |
| large_disparity | original_to_s1 | 0.000 | 0.000 | nan |
| large_disparity | s1_to_s2 | 14.669 | 70.468 | 69.601 |
| large_disparity | s2_to_s3 | 10.673 | 23.336 | 49.411 |
| occluded_any | original_to_s1 | 0.000 | 0.000 | nan |
| occluded_any | s1_to_s2 | 11.597 | 61.690 | 79.351 |
| occluded_any | s2_to_s3 | 15.739 | 27.719 | 33.325 |
| occluded_majority | original_to_s1 | 0.000 | 0.000 | nan |
| occluded_majority | s1_to_s2 | 15.247 | 65.044 | 88.063 |
| occluded_majority | s2_to_s3 | 20.007 | 26.075 | 31.755 |
| large_disp_and_occluded | original_to_s1 | 0.000 | 0.000 | nan |
| large_disp_and_occluded | s1_to_s2 | 18.414 | 66.579 | 92.778 |
| large_disp_and_occluded | s2_to_s3 | 18.840 | 24.951 | 42.693 |
| boundary_and_occluded | original_to_s1 | 0.000 | 0.000 | nan |
| boundary_and_occluded | s1_to_s2 | 10.653 | 53.954 | 85.425 |
| boundary_and_occluded | s2_to_s3 | 21.656 | 33.022 | 29.191 |

## 初始区间越界方向

| 区域 | 区间 | 方向 | 像素% | 平均越界mm | 最大越界mm |
| --- | --- | --- | ---: | ---: | ---: |
| full | original | below | 0.006 | 2.313 | 6.738 |
| full | original | above | 2.672 | 74.368 | 311.873 |
| full | s1 | below | 0.006 | 2.313 | 6.738 |
| full | s1 | above | 0.237 | 44.153 | 150.223 |
| boundary | original | below | 0.011 | 2.723 | 6.294 |
| boundary | original | above | 4.192 | 90.755 | 311.873 |
| boundary | s1 | below | 0.011 | 2.723 | 6.294 |
| boundary | s1 | above | 0.511 | 58.423 | 150.223 |
| large_disparity | original | below | 0.028 | 2.313 | 6.738 |
| large_disparity | original | above | 11.741 | 77.463 | 311.873 |
| large_disparity | s1 | below | 0.028 | 2.313 | 6.738 |
| large_disparity | s1 | above | 1.177 | 44.366 | 150.223 |
| occluded_any | original | below | 0.008 | 2.238 | 6.294 |
| occluded_any | original | above | 7.378 | 79.559 | 311.873 |
| occluded_any | s1 | below | 0.008 | 2.238 | 6.294 |
| occluded_any | s1 | above | 0.880 | 50.268 | 150.223 |
| occluded_majority | original | below | 0.012 | 2.066 | 6.294 |
| occluded_majority | original | above | 6.977 | 75.318 | 311.873 |
| occluded_majority | s1 | below | 0.012 | 2.066 | 6.294 |
| occluded_majority | s1 | above | 0.648 | 53.016 | 150.223 |
| large_disp_and_occluded | original | below | 0.022 | 2.238 | 6.294 |
| large_disp_and_occluded | original | above | 17.874 | 82.978 | 311.873 |
| large_disp_and_occluded | s1 | below | 0.022 | 2.238 | 6.294 |
| large_disp_and_occluded | s1 | above | 2.490 | 50.625 | 150.223 |
| boundary_and_occluded | original | below | 0.006 | 2.889 | 6.294 |
| boundary_and_occluded | original | above | 5.463 | 90.826 | 311.873 |
| boundary_and_occluded | s1 | below | 0.006 | 2.889 | 6.294 |
| boundary_and_occluded | s1 | above | 0.739 | 62.640 | 150.223 |

决策：

- original 已漏覆盖：先核查相机/深度单位/掩码和初始范围规则。不能按每张 Val GT 动态扩展正式推理范围。
- original 覆盖而 S1 漏覆盖：优先检查实际采样端点。
- S1 内→S3 外误差贡献显著：优先做候选保留/恢复改动，并由 S1→S2、S2→S3 定位阶段。
- 相邻阶段的 in_out 组可能重叠或涉及后来找回的像素，其误差贡献不能直接相加。
- 原始区间是数据实际提供的 depth_values 的 min/max，并非从 GT 推导。空组均值为 nan。
