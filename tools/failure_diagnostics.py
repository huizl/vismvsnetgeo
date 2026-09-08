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
