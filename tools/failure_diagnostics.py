"""Native-resolution range failure accounting; no model changes or GT intervention."""

from collections import defaultdict

import numpy as np


TOTALS = ("pixels", "region_pixels", "abs_sum", "region_abs_sum",
          "acc2_pixels", "bad8_pixels", "nearest_sum", "range_floor_sum")
DERIVED = ("pixel_fraction", "abs", "acc2", "bad8", "nearest_hyp_abs",
           "range_floor_abs", "error_share", "range_floor_error_share")


def ratios(row):
    def divide(a, b):
        return float(a / b) if b else float("nan")
    n = row["pixels"]
    return {
        "pixel_fraction": divide(n, row["region_pixels"]),
        "abs": divide(row["abs_sum"], n),
        "acc2": divide(row["acc2_pixels"], n),
        "bad8": divide(row["bad8_pixels"], n),
        "nearest_hyp_abs": divide(row["nearest_sum"], n),
        "range_floor_abs": divide(row["range_floor_sum"], n),
        "error_share": divide(row["abs_sum"], row["region_abs_sum"]),
        "range_floor_error_share": divide(row["range_floor_sum"], row["region_abs_sum"]),
    }


def measure_stage(prediction, gt, hypotheses, regions, stage):
    """Inputs use exactly the same native H,W; hypotheses are D or D,H,W.

    Nearest-hypothesis distance diagnoses sampling density, not a regression
    error lower bound. Range-floor distance IS a lower bound for estimates
    constrained to the min/max interval. Empty groups retain counts and NaNs.
    """
    prediction, gt, hypotheses = [np.asarray(x, dtype=np.float64)
                                  for x in (prediction, gt, hypotheses)]
    if prediction.ndim != 2 or prediction.shape != gt.shape:
        raise ValueError("prediction and GT must have the same native H,W")
    if hypotheses.ndim == 1:
        hypotheses = hypotheses[:, None, None]
    if hypotheses.ndim != 3 or hypotheses.shape[0] == 0 or hypotheses.shape[1:] not in ((1, 1), gt.shape):
        raise ValueError("hypotheses must be D or D,H,W at stage resolution")
    minimum, maximum = hypotheses.min(axis=0), hypotheses.max(axis=0)
    in_range = (gt >= minimum) & (gt <= maximum)
    error = np.abs(prediction - gt)
    nearest = np.min(np.abs(hypotheses - gt[None]), axis=0)
    floor = np.maximum(np.maximum(minimum - gt, gt - maximum), 0.0)
    finite = (np.isfinite(prediction) & np.isfinite(gt) & np.isfinite(hypotheses).all(axis=0))
    rows = []
    for region, mask in regions.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError("region masks must use the native stage resolution")
        if np.any(mask & ~finite):
            raise ValueError("Nonfinite prediction, GT or hypotheses inside a valid region")
        for group, selected in (("all", mask), ("in_range", mask & in_range),
                                ("out_of_range", mask & ~in_range)):
            row = {
                "stage": stage, "region": region, "group": group,
                "height": gt.shape[0], "width": gt.shape[1],
                "pixels": int(selected.sum()), "region_pixels": int(mask.sum()),
                "abs_sum": float(error[selected].sum()),
                "region_abs_sum": float(error[mask].sum()),
                "acc2_pixels": int((error[selected] < 2.0).sum()),
                "bad8_pixels": int((error[selected] > 8.0).sum()),
                "nearest_sum": float(nearest[selected].sum()),
                "range_floor_sum": float(floor[selected].sum()),
            }
            rows.append({**row, **ratios(row)})
    return rows


def summarize(rows):
    groups = defaultdict(lambda: {"images": 0, **{name: 0 for name in TOTALS}})
    for row in rows:
        key = (row["stage"], row["region"], row["group"])
        result = groups[key]
        # Images with region pixels but zero group pixels still count in the
        # region denominator. Never average per-image conditional means.
        result["images"] += int(row["region_pixels"] > 0)
        for name in TOTALS:
            result[name] += row[name]
    return [{"stage": stage, "region": region, "group": group,
             **values, **ratios(values)}
            for (stage, region, group), values in groups.items()]


def report(rows, metadata):
    lines = ["# 基线搜索范围失效诊断", "",
             f"模型：{metadata['model_type']}；checkpoint：{metadata['checkpoint']}。",
             "各阶段原始分辨率；GT 和固定区域掩码以 nearest 下采样；所有数值按该阶段像素加权。",
             "不同阶段像素集合/分辨率不同，不应把比例差直接解释为同一批像素的转移。",
             "该表定位关联和误差贡献，不证明修改采样或融合后的因果收益。", "",
             "| 区域 | 阶段 | 范围外像素 % | 范围内 Abs mm | 范围外 Abs mm | 范围外贡献的总绝对误差 % | 范围约束下界/总绝对误差 % |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    indexed = {(r["stage"], r["region"], r["group"]): r for r in rows}
    for row in rows:
        if row["group"] != "all":
            continue
        inside = indexed[row["stage"], row["region"], "in_range"]
        outside = indexed[row["stage"], row["region"], "out_of_range"]
        lines.append(f"| {row['region']} | S{row['stage']} | {100*outside['pixel_fraction']:.2f} | {inside['abs']:.4f} | {outside['abs']:.4f} | {100*outside['error_share']:.2f} | {100*row['range_floor_error_share']:.2f} |")
    lines += ["", "先看 S3 的 large_disparity、occluded_any、large_disp_and_occluded 和 boundary：", "",
              "- 范围外像素对总误差贡献较大，且区间约束误差下界较大：优先检查中心/候选丢失。",
              "- 主要误差发生在范围内：继续检查采样密度、pair 错误高权重和遮挡融合，不能仅凭此表确定是哪一个。",
              "- nearest_hyp_abs 是最近离散假设到 GT 的距离，不是 soft-argmin 的不可约误差：插值回归可落在两个假设之间。",
              "- range_floor_abs 是 GT 到当前 min/max 区间的距离；在预测限制于区间内时构成误差下界，不等于扩大范围可以实际获得的收益。",
              "- 空组的均值/零总误差的占比写为 nan，不作为零误差；CSV 的 bad8 使用严格 >8 mm。",
              "", "先检查原 summary_metrics.csv 是否接近既有基线，确认评估条件可比，再根据本表选择一个最小改动。", ""]
    return "\n".join(lines)


ORIGIN_TOTALS = ("pixels", "region_pixels", "abs_sum", "region_abs_sum",
                 "gap_sum", "s3_gap_sum")


def origin_ratios(row):
    def divide(a, b):
        return float(a / b) if b else float("nan")
    return {
        "pixel_fraction": divide(row["pixels"], row["region_pixels"]),
        "abs": divide(row["abs_sum"], row["pixels"]),
        "error_share": divide(row["abs_sum"], row["region_abs_sum"]),
        "gap_mean": divide(row["gap_sum"], row["pixels"]),
        "s3_gap_mean": divide(row["s3_gap_sum"], row["pixels"]),
    }


def measure_origins(prediction, gt, intervals, regions):
    """Compare intervals on ONE GT grid, using nearest-selected stage pixels.

    intervals contains (lower, upper) for original/s1/s2/s3, each scalar or H,W.
    All error sums use the same nearest-selected S3 prediction. Thus each
    transition is a true partition of the chosen common-grid pixel set.
    gap_* refers to the target interval of the comparison, or the named stage
    for directional groups. It is not a causal attribution of prediction error.
    """
    prediction, gt = np.asarray(prediction, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if gt.ndim != 2 or prediction.shape != gt.shape:
        raise ValueError("prediction and GT must share the common H,W grid")
    finite = np.isfinite(prediction) & np.isfinite(gt)
    masks, gaps, bounds = {}, {}, {}
    for name in ("original", "s1", "s2", "s3"):
        lo, hi = [np.broadcast_to(np.asarray(x, dtype=np.float64), gt.shape) for x in intervals[name]]
        finite &= np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
        masks[name] = (gt >= lo) & (gt <= hi)
        gaps[name] = np.maximum(np.maximum(lo - gt, gt - hi), 0)
        bounds[name] = (lo, hi)
    error = np.abs(prediction - gt)
    comparisons = []
    for source, target in (("original", "s1"), ("s1", "s2"), ("s2", "s3"), ("s1", "s3")):
        a, b = masks[source], masks[target]
        comparisons.append((f"{source}_to_{target}", target, {
            "all": np.ones(gt.shape, bool), "in_in": a & b,
            "in_out": a & ~b, "out_in": ~a & b, "out_out": ~a & ~b}))
    for name in ("original", "s1"):
        lo, hi = bounds[name]
        comparisons.append((f"direction_{name}", name, {
            "all": np.ones(gt.shape, bool), "below": gt < lo,
            "inside": masks[name], "above": gt > hi}))
    rows = []
    for region, region_mask in regions.items():
        region_mask = np.asarray(region_mask, dtype=bool)
        if region_mask.shape != gt.shape:
            raise ValueError("regions must share the common GT grid")
        if np.any(region_mask & ~finite):
            raise ValueError("Invalid values or reversed interval in a valid region")
        for comparison, target, selections in comparisons:
            for group, selected in selections.items():
                mask = region_mask & selected
                row = {
                    "region": region, "comparison": comparison, "group": group,
                    "pixels": int(mask.sum()), "region_pixels": int(region_mask.sum()),
                    "abs_sum": float(error[mask].sum()),
                    "region_abs_sum": float(error[region_mask].sum()),
                    "gap_sum": float(gaps[target][mask].sum()),
                    "s3_gap_sum": float(gaps["s3"][mask].sum()),
                    "gap_max": float(gaps[target][mask].max()) if mask.any() else float("nan"),
                }
                rows.append({**row, **origin_ratios(row)})
    return rows


def summarize_origins(rows):
    groups = defaultdict(lambda: {"images": 0, **{name: 0 for name in ORIGIN_TOTALS},
                                   "gap_max": float("nan")})
    for row in rows:
        result = groups[row["region"], row["comparison"], row["group"]]
        result["images"] += int(row["region_pixels"] > 0)
        for name in ORIGIN_TOTALS:
            result[name] += row[name]
        if np.isfinite(row["gap_max"]):
            result["gap_max"] = (max(result["gap_max"], row["gap_max"])
                                 if np.isfinite(result["gap_max"]) else row["gap_max"])
    return [{"region": region, "comparison": comparison, "group": group,
             **values, **origin_ratios(values)}
            for (region, comparison, group), values in groups.items()]


def origin_report(rows, metadata):
    indexed = {(r["region"], r["comparison"], r["group"]): r for r in rows}
    regions = list(dict.fromkeys(r["region"] for r in rows))
    lines = ["# 同一 GT 网格的搜索范围来源诊断", "",
             f"模型：{metadata['model_type']}；checkpoint：{metadata['checkpoint']}。",
             "GT 和区域掩码固定；各阶段区间端点与 S3 深度用 nearest 选择到 GT 网格，不混合邻居深度。",
             "误差是该 common-grid S3 预测的误差，与原生阶段/常规 bilinear 评估分开命名。",
             "覆盖转移是这个固定对齐规则下的像素联合统计，不等于已证明的因果来源。", "",
             "## 主要分解：初始漏覆盖还是后续丢失", "",
             "所有占比的分母分别为该区域全部像素或 S3 绝对误差总和；各区域相互重叠，不能跨区域相加。", "",
             "| 区域 | S1内→S3外 像素% | 其S3误差贡献% | S1外→S3外 像素% | 其S3误差贡献% | S1外→S3内 像素% |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for region in regions:
        lost = indexed[region, "s1_to_s3", "in_out"]
        persistent = indexed[region, "s1_to_s3", "out_out"]
        recovered = indexed[region, "s1_to_s3", "out_in"]
        lines.append(f"| {region} | {100*lost['pixel_fraction']:.3f} | {100*lost['error_share']:.3f} | {100*persistent['pixel_fraction']:.3f} | {100*persistent['error_share']:.3f} | {100*recovered['pixel_fraction']:.3f} |")
    lines += ["", "## 定位实际采样端点和级联阶段", "",
              "in_out 表示源区间覆盖 GT、目标区间未覆盖 GT。其 gap 是 GT 到目标区间的距离。", "",
              "| 区域 | 比较 | in_out像素% | 其S3误差贡献% | 平均越界mm |",
              "| --- | --- | ---: | ---: | ---: |"]
    for region in regions:
        for comparison in ("original_to_s1", "s1_to_s2", "s2_to_s3"):
            row = indexed[region, comparison, "in_out"]
            lines.append(f"| {region} | {comparison} | {100*row['pixel_fraction']:.3f} | {100*row['error_share']:.3f} | {row['gap_mean']:.3f} |")
    lines += ["", "## 初始区间越界方向", "",
              "| 区域 | 区间 | 方向 | 像素% | 平均越界mm | 最大越界mm |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for region in regions:
        for name in ("original", "s1"):
            for direction in ("below", "above"):
                row = indexed[region, f"direction_{name}", direction]
                lines.append(f"| {region} | {name} | {direction} | {100*row['pixel_fraction']:.3f} | {row['gap_mean']:.3f} | {row['gap_max']:.3f} |")
    lines += ["", "决策：", "",
              "- original 已漏覆盖：先核查相机/深度单位/掩码和初始范围规则。不能按每张 Val GT 动态扩展正式推理范围。",
              "- original 覆盖而 S1 漏覆盖：优先检查实际采样端点。",
              "- S1 内→S3 外误差贡献显著：优先做候选保留/恢复改动，并由 S1→S2、S2→S3 定位阶段。",
              "- 相邻阶段的 in_out 组可能重叠或涉及后来找回的像素，其误差贡献不能直接相加。",
              "- 原始区间是数据实际提供的 depth_values 的 min/max，并非从 GT 推导。空组均值为 nan。", ""]
    return "\n".join(lines)
