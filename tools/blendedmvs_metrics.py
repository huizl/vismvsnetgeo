"""GT geometry, metrics, aggregation and example selection for BlendedMVS."""
import numpy as np


REGIONS = ("full", "boundary", "large_disparity", "occluded_any",
           "occluded_majority", "occluded_ge2", "large_disp_and_occluded",
           "large_disp_and_occluded_ge2", "boundary_and_occluded")
METRICS = ("abs", "abs_rel", "rmse", "acc2_normalized", "acc4_normalized",
           "acc8_normalized", "rel_acc1", "rel_acc2", "rel_acc5",
           "stage1_in_range", "stage2_in_range", "stage3_in_range",
           "stage1_range_width", "stage2_range_width", "stage3_range_width")
PERCENT_METRICS = tuple(m for m in METRICS if "acc" in m or "in_range" in m)
CATEGORIES = ("large_disparity_dominant", "occlusion_dominant", "joint_difficulty")


def percentile_mask(values, valid, percentile):
    if not valid.any():
        return valid.copy()
    return valid & (values >= np.percentile(values[valid], percentile))


def boundary_mask(gt, valid, pct=10):
    filled = np.where(valid, gt, np.median(gt[valid]))
    p = np.pad(filled, 1, mode="reflect")
    gx = (-p[:-2, :-2] + p[:-2, 2:] - 2*p[1:-1, :-2]
          + 2*p[1:-1, 2:] - p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:])
    return percentile_mask(np.hypot(gx, gy), valid, 100 - pct)


def geometry_maps(sample, occ_abs_tol=2.0, occ_rel_tol=.01, large_disp_pct=80):
    gt = sample["depth"]
    valid = sample["mask"].astype(bool)
    if not valid.any():
        raise ValueError("Reference image has no finite positive GT pixels")
    h, w = gt.shape
    y, x = np.indices((h, w), dtype=np.float64)
    pixels = np.stack((x.ravel(), y.ravel(), np.ones(h*w)))
    xyz = np.linalg.inv(sample["intrinsics"][0]) @ pixels * gt.reshape(1, -1)
    xyz_h = np.vstack((xyz, np.ones(h*w)))
    inverse_ref = np.linalg.inv(sample["extrinsics"][0])
    disparity = np.zeros(h*w, dtype=np.float64)
    count = np.zeros(h*w, np.uint16)
    comparable_count = np.zeros(h*w, np.uint16)
    for index in range(1, len(sample["gt_depths"])):
        xyz_src = (sample["extrinsics"][index] @ inverse_ref)[:3] @ xyz_h
        projected = sample["intrinsics"][index] @ xyz_src
        z = xyz_src[2]
        good_z = np.isfinite(z) & (z > 1e-8)
        safe_z = np.where(good_z, z, 1)
        sx, sy = projected[0] / safe_z, projected[1] / safe_z
        displacement = np.hypot(sx - pixels[0], sy - pixels[1])
        # Retain the existing definition based on all source displacements.
        disparity = np.maximum(disparity, np.where(good_z, displacement, 0))
        in_bounds = (valid.ravel() & good_z & np.isfinite(sx) & np.isfinite(sy)
                     & (sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1))
        # Nearest GT surface: no interpolation of invalid pixels or surfaces
        # across depth discontinuities. Record this explicitly in the protocol.
        ix = np.rint(np.clip(np.nan_to_num(sx), 0, w-1)).astype(np.int64)
        iy = np.rint(np.clip(np.nan_to_num(sy), 0, h-1)).astype(np.int64)
        source_gt = sample["gt_depths"][index][iy, ix]
        comparable = (in_bounds & sample["gt_masks"][index][iy, ix]
                      & np.isfinite(source_gt) & (source_gt > 0))
        occluded = comparable & (z > source_gt + np.maximum(
            occ_abs_tol, occ_rel_tol * source_gt))
        comparable_count += comparable.astype(np.uint16)
        count += occluded.astype(np.uint16)
    disparity = disparity.reshape(h, w).astype(np.float32)
    if not np.isfinite(disparity[valid]).all():
        raise ValueError("Non-finite GT reprojection displacement; check camera scale")
    count, comparable_count = count.reshape(h, w), comparable_count.reshape(h, w)
    ratio = np.divide(count, comparable_count, out=np.zeros((h, w), np.float32),
                      where=comparable_count > 0)
    large = percentile_mask(disparity, valid, large_disp_pct)
    boundary = boundary_mask(gt, valid)
    any_occ = valid & (count > 0)
    majority = valid & (comparable_count > 0) & (2*count >= comparable_count)
    ge2 = valid & (count >= 2)
    masks = dict(full=valid, boundary=boundary, large_disparity=large,
                 occluded_any=any_occ, occluded_majority=majority, occluded_ge2=ge2,
                 large_disp_and_occluded=large & any_occ,
                 large_disp_and_occluded_ge2=large & ge2,
                 boundary_and_occluded=boundary & any_occ)
    return dict(disparity=disparity, occluded_count=count,
                comparable_count=comparable_count, occlusion_ratio=ratio, masks=masks)


def region_metrics(pred, gt, mask, coverage, widths):
    count = int(mask.sum())
    if not count:
        return dict(pixels=0, **{name: float("nan") for name in METRICS})
    if not np.isfinite(pred[mask]).all():
        raise ValueError("Non-finite predictions on fixed GT pixels; refusing to remove pixels")
    error = np.abs(pred[mask].astype(np.float64) - gt[mask])
    relative = error / gt[mask]
    result = dict(pixels=count, abs=float(error.mean()), abs_rel=float(relative.mean()),
                  rmse=float(np.sqrt(np.square(error).mean())))
    result.update({f"acc{t}_normalized": float((error < t).mean()) for t in (2, 4, 8)})
    result.update({f"rel_acc{t}": float((relative < t/100).mean()) for t in (1, 2, 5)})
    for index in range(3):
        result[f"stage{index+1}_in_range"] = float(coverage[index][mask].mean())
        result[f"stage{index+1}_range_width"] = float(widths[index][mask].mean())
    return result


def aggregate(rows, scene_mean=False):
    """Pixel weighted RMSE pools squared error, rather than averaging RMSE."""
    results = []
    for region in REGIONS:
        items = [row for row in rows if row["region"] == region and row["pixels"]]
        if not items:
            continue
        scenes = sorted({row["scene"] for row in items})
        aggregations = ["pixel_weighted", "image_mean"]
        if scene_mean:
            aggregations.append("scene_mean")
        for aggregation in aggregations:
            groups = ([items] if aggregation != "scene_mean" else
                      [[row for row in items if row["scene"] == s] for s in scenes])
            values = {}
            for metric in METRICS:
                grouped_values = []
                for group in groups:
                    weight = (np.ones(len(group)) if aggregation == "image_mean"
                              else np.array([r["pixels"] for r in group], np.float64))
                    arr = np.array([r[metric] for r in group])
                    value = (np.sqrt(np.average(arr**2, weights=weight))
                             if metric == "rmse" and aggregation != "image_mean"
                             else np.average(arr, weights=weight))
                    grouped_values.append(value)
                values[metric] = float(np.mean(grouped_values))
            results.append(dict(aggregation=aggregation, region=region, images=len(items),
                                scenes=len(scenes), pixels=sum(r["pixels"] for r in items),
                                **values))
    return results


def describe_scene(sample, geometry):
    valid = sample["mask"].astype(bool)
    large, occ = geometry["masks"]["large_disparity"], geometry["masks"]["occluded_ge2"]
    n, l, o, j = (int(m.sum()) for m in (valid, large, occ, large & occ))
    return dict(valid_pixels=n, large_disparity_pixels=l, occluded_ge2_pixels=o,
                joint_ge2_pixels=j, occluded_ge2_fraction=o/n, joint_ge2_fraction=j/n,
                gt_comparable_fraction=float((geometry["comparable_count"][valid] > 0).mean()),
                mean_comparable_sources=float(geometry["comparable_count"][valid].mean()),
                joint_fraction_of_large=j/l if l else 0,
                joint_fraction_of_occlusion=j/o if o else 0,
                disparity_p80_px=float(np.percentile(geometry["disparity"][valid], 80)))


def category_scores(record):
    result = {}
    if (record["large_disparity_pixels"] >= 30 and record["occluded_ge2_fraction"] <= .10
            and record["joint_fraction_of_large"] <= .25):
        result["large_disparity_dominant"] = record["disparity_p80_px"]
    if (record["occluded_ge2_pixels"] >= 30 and record["occluded_ge2_fraction"] >= .10
            and record["joint_fraction_of_occlusion"] <= .35):
        result["occlusion_dominant"] = record["occluded_ge2_fraction"]
    if (record["joint_ge2_pixels"] >= 30 and record["joint_ge2_fraction"] >= .05
            and record["joint_fraction_of_large"] >= .25):
        result = {"joint_difficulty": record["joint_ge2_fraction"]}
    return result


def select_examples(records, per_category=2):
    selected, missing = [], []
    for category in CATEGORIES:
        candidates = [r for r in records if category in category_scores(r)
                      and np.isfinite(r["abs_reduction_pct"]) and r["abs_reduction_pct"] > 0]
        candidates.sort(key=lambda r: (-r["abs_reduction_pct"], -category_scores(r)[category],
                                       r["scene"], r["view"]))
        used = set()
        for record in candidates:
            if record["scene"] in used:
                continue
            selected.append(dict(record, category=category))
            used.add(record["scene"])
            if len(used) >= per_category:
                break
        if len(used) < per_category:
            missing.append(dict(category=category, requested=per_category, selected=len(used)))
    return selected, missing
