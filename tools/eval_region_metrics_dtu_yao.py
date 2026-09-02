"""DTU per-image region metrics and cascade depth-range diagnostics.

The evaluator runs a checkpoint on the same ``dtu_yao`` pipeline used during
training. It writes one row per image/region plus pixel-weighted and image-mean
summaries. Baseline and configurable M1/M2/M3 checkpoints can all be loaded by
this script.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import find_dataset_def  # noqa: E402
from datasets.data_io import read_pfm  # noqa: E402
from models.model_variants import (  # noqa: E402
    MODEL_TYPE_CHOICES,
    get_model_variant,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Per-image and aggregate DTU metrics for the M1/M2/M3 ablation."
        )
    )
    parser.add_argument(
        "--model_type", required=True,
        choices=MODEL_TYPE_CHOICES)
    parser.add_argument("--loadckpt", required=True)
    parser.add_argument("--label", default=None,
                        help="Model label; defaults to the checkpoint directory name.")
    parser.add_argument("--testpath", required=True,
                        help="DTU training root containing Rectified/, Depths/ and Cameras/.")
    parser.add_argument("--testlist", required=True, help="e.g. lists/dtu/val.txt")
    parser.add_argument("--outdir", default="./outputs/region_metrics")

    parser.add_argument("--vismode", default="soft",
                        choices=["soft", "hard", "average", "uwta", "maxpool"])
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--stage1_dnum", type=int, default=48)
    parser.add_argument("--stage1_iscale", type=int, default=4)
    parser.add_argument("--stage2_dnum", type=int, default=32)
    parser.add_argument("--stage2_iscale", type=int, default=2)
    parser.add_argument("--stage3_dnum", type=int, default=16)
    parser.add_argument("--stage3_iscale", type=int, default=1)
    parser.add_argument("--hypothesis_residual_scale", type=float, default=1.0)
    parser.add_argument("--visibility_fusion_beta", type=float, default=0.2)
    parser.add_argument("--hybrid_stage2_wide_num", type=int, default=8)
    parser.add_argument("--hybrid_stage3_wide_num", type=int, default=4)
    parser.add_argument("--hybrid_sigma_scale", type=float, default=2.0)
    parser.add_argument("--hybrid_max_scale", type=float, default=2.0)
    parser.add_argument("--eval_nviews", type=int, default=5,
                        help="Total model views, including the reference view.")
    parser.add_argument("--region_nviews", type=int, default=5,
                        help="Total views used to define disparity/occlusion masks.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--light", type=int, default=3,
                        help="DTU light in [0,6]; use -1 for all seven lights.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional debug limit. One sample is one scan/ref/light tuple.")
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--boundary_pct", type=float, default=10.0,
                        help="Top percentage of GT depth-gradient pixels.")
    parser.add_argument("--large_disp_pct", type=float, default=80.0,
                        help="Percentile threshold; 80 means the top 20 percent.")
    parser.add_argument("--occ_abs_tol", type=float, default=2.0,
                        help="Absolute source-depth tolerance in mm for GT occlusion classification.")
    parser.add_argument("--occ_rel_tol", type=float, default=0.01,
                        help="Relative source-depth tolerance for GT occlusion classification.")
    return parser.parse_args()


def build_model(args):
    # All variants use this inference-compatible model so the evaluator can
    # expose exact stage ranges. Factor C adds parameterized fusion heads.
    variant = get_model_variant(args.model_type)
    module = __import__("models.vismvsnet_oa", fromlist=["VisMVSModel"])
    return module.VisMVSModel(
        mode=args.vismode,
        stage1_depth_num=args.stage1_dnum,
        stage1_interval_scale=args.stage1_iscale,
        stage2_depth_num=args.stage2_dnum,
        stage2_interval_scale=args.stage2_iscale,
        stage3_depth_num=args.stage3_dnum,
        stage3_interval_scale=args.stage3_iscale,
        hypothesis_fusion=variant.hypothesis_fusion,
        hypothesis_residual_scales=(
            args.hypothesis_residual_scale,
            args.hypothesis_residual_scale,
            args.hypothesis_residual_scale,
        ),
        visibility_fusion=variant.visibility_modeling,
        visibility_fusion_betas=(
            args.visibility_fusion_beta,
            args.visibility_fusion_beta,
            args.visibility_fusion_beta,
        ),
        hybrid_sampling=variant.hybrid_sampling,
        hybrid_stage2_wide_num=args.hybrid_stage2_wide_num,
        hybrid_stage3_wide_num=args.hybrid_stage3_wide_num,
        hybrid_sigma_scale=args.hybrid_sigma_scale,
        hybrid_max_scale=args.hybrid_max_scale,
    )


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    target_has_module = next(iter(model.state_dict())).startswith("module.")
    source_has_module = next(iter(state_dict)).startswith("module.")
    if target_has_module and not source_has_module:
        state_dict = {"module." + key: value for key, value in state_dict.items()}
    elif source_has_module and not target_has_module:
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def to_cuda(sample):
    return {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
            for key, value in sample.items()}


def to_2d(array):
    """Convert one DataLoader item to a 2-D HxW numpy array."""
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 3:
        # The official DTU depth mask is normally one-channel.  This keeps the
        # script usable if a copy was saved as RGB without affecting that case.
        array = array.mean(axis=-1)
    if array.ndim != 2:
        raise ValueError("Expected a 2-D depth/mask/confidence array, got {}".format(array.shape))
    return array.astype(np.float32, copy=False)


def boundary_mask(depth_gt, valid, pct):
    if not valid.any():
        return valid.copy()
    filled = depth_gt.copy()
    filled[~valid] = np.median(depth_gt[valid])
    grad_x = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    threshold = np.percentile(gradient[valid], 100.0 - pct)
    return valid & (gradient >= threshold)


def percentile_mask(values, valid, percentile):
    if not valid.any():
        return valid.copy()
    threshold = np.percentile(values[valid], percentile)
    return valid & (values >= threshold)


def read_train_camera(datapath, view_id):
    filename = os.path.join(datapath, "Cameras", "train", "{:08d}_cam.txt".format(view_id))
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(4, 4)
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape(3, 3)
    return intrinsics, extrinsics


def read_dtu_depth(datapath, scan, view_id):
    filename = os.path.join(
        datapath, "Depths", scan + "_train", "depth_map_{:04d}.pfm".format(view_id)
    )
    if not os.path.exists(filename):
        raise FileNotFoundError("DTU GT depth not found: {}".format(filename))
    depth, _ = read_pfm(filename)
    return np.asarray(depth, dtype=np.float32)


def read_dtu_mask(datapath, scan, view_id, target_shape=None):
    filename = os.path.join(
        datapath, "Depths", scan + "_train", "depth_visual_{:04d}.png".format(view_id)
    )
    if not os.path.exists(filename):
        return None
    mask = np.asarray(Image.open(filename), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask.mean(axis=2)
    mask = mask > 10.0
    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def geometry_region_maps(datapath, scan, ref_view, src_views, depth,
                         occ_abs_tol=2.0, occ_rel_tol=0.01):
    """Build fixed GT/camera disparity and source-occlusion maps.

    Occlusion is defined only for source pixels with valid source GT.  A
    reference 3-D point is occluded in a source when its projected source Z is
    behind the sampled visible source surface by more than
    ``max(occ_abs_tol, occ_rel_tol * source_depth)``.  Out-of-image points are
    deliberately not counted as occlusion.
    """
    if not src_views:
        raise ValueError("At least one source view is required to define geometry regions.")

    height, width = depth.shape
    ref_k, ref_ext = read_train_camera(datapath, ref_view)
    y, x = np.indices((height, width), dtype=np.float32)
    homogeneous = np.stack((x.reshape(-1), y.reshape(-1), np.ones(height * width, dtype=np.float32)))
    rays_ref = np.linalg.inv(ref_k).astype(np.float32) @ homogeneous
    xyz_ref = rays_ref * depth.reshape(1, -1)
    xyz_ref_h = np.vstack((xyz_ref, np.ones((1, xyz_ref.shape[1]), dtype=np.float32)))

    max_disparity = np.zeros(height * width, dtype=np.float32)
    occluded_count = np.zeros(height * width, dtype=np.uint16)
    comparable_count = np.zeros(height * width, dtype=np.uint16)
    for src_view in src_views:
        src_k, src_ext = read_train_camera(datapath, src_view)
        transform = src_ext @ np.linalg.inv(ref_ext)
        xyz_src = transform[:3, :] @ xyz_ref_h
        projected = src_k @ xyz_src
        z = projected[2]
        safe_z = np.where(np.abs(z) > 1e-6, z, 1e-6)
        src_x = projected[0] / safe_z
        src_y = projected[1] / safe_z
        displacement = np.sqrt((src_x - homogeneous[0]) ** 2 + (src_y - homogeneous[1]) ** 2)
        max_disparity = np.maximum(max_disparity, displacement.astype(np.float32))

        map_x = src_x.reshape(height, width).astype(np.float32)
        map_y = src_y.reshape(height, width).astype(np.float32)
        projected_depth = z.reshape(height, width).astype(np.float32)
        in_bounds = (
            np.isfinite(map_x) & np.isfinite(map_y) & np.isfinite(projected_depth)
            & (projected_depth > 0.0)
            & (map_x >= 0.0) & (map_x <= width - 1.0)
            & (map_y >= 0.0) & (map_y <= height - 1.0)
        )

        src_depth = read_dtu_depth(datapath, scan, src_view)
        if src_depth.shape != depth.shape:
            src_depth = cv2.resize(src_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        sampled_depth = cv2.remap(
            src_depth, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        src_mask = read_dtu_mask(datapath, scan, src_view, depth.shape)
        if src_mask is None:
            src_mask = np.isfinite(src_depth) & (src_depth > 0.0)
        sampled_mask = cv2.remap(
            src_mask.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)
        comparable = in_bounds & sampled_mask & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        tolerance = np.maximum(occ_abs_tol, occ_rel_tol * sampled_depth)
        occluded = comparable & (projected_depth > sampled_depth + tolerance)
        comparable_count += comparable.reshape(-1).astype(np.uint16)
        occluded_count += occluded.reshape(-1).astype(np.uint16)

    comparable_count = comparable_count.reshape(height, width)
    occluded_count = occluded_count.reshape(height, width)
    occlusion_ratio = np.zeros((height, width), dtype=np.float32)
    comparable = comparable_count > 0
    occlusion_ratio[comparable] = (
        occluded_count[comparable].astype(np.float32)
        / comparable_count[comparable].astype(np.float32)
    )
    occluded_any = comparable & (occluded_count > 0)
    occluded_majority = comparable & (2 * occluded_count >= comparable_count)
    return {
        "disparity": max_disparity.reshape(height, width),
        "occlusion_ratio": occlusion_ratio,
        "occluded_any": occluded_any,
        "occluded_majority": occluded_majority,
        "comparable_count": comparable_count,
    }


METRIC_NAMES = (
    "abs", "acc2", "acc4", "acc8",
    "stage1_in_range", "stage2_in_range", "stage3_in_range",
    "stage1_range_width", "stage2_range_width", "stage3_range_width",
)


def full_resolution_range_diagnostics(
        stage_outputs, depth_gt, original_depth_values):
    target_size = depth_gt.shape[-2:]
    range_masks = []
    range_widths = []
    base_interval = (
        original_depth_values[:, 1] - original_depth_values[:, 0]
    ).view(-1, 1, 1)
    for stage_output in stage_outputs:
        if len(stage_output) < 3:
            raise ValueError("model output does not contain stage depth hypotheses")
        hypotheses = stage_output[2]
        if hypotheses.dim() == 2:
            depth_min = hypotheses[:, 0].view(-1, 1, 1).expand(-1, *target_size)
            depth_max = hypotheses[:, -1].view(-1, 1, 1).expand(-1, *target_size)
        elif hypotheses.dim() == 4:
            depth_min = F.interpolate(
                hypotheses[:, 0:1], size=target_size,
                mode="bilinear", align_corners=False).squeeze(1)
            depth_max = F.interpolate(
                hypotheses[:, -1:], size=target_size,
                mode="bilinear", align_corners=False).squeeze(1)
        else:
            raise ValueError("unexpected depth hypothesis shape: {}".format(
                tuple(hypotheses.shape)))
        range_masks.append((depth_gt >= depth_min) & (depth_gt <= depth_max))
        range_widths.append((depth_max - depth_min) / base_interval)
    return range_masks, range_widths


def metrics(error, mask, range_masks, range_widths):
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return {"pixels": 0, **{name: np.nan for name in METRIC_NAMES}}
    values = error[mask]
    return {
        "pixels": pixel_count,
        "abs": float(values.mean()),
        "acc2": float((values < 2.0).mean()),
        "acc4": float((values < 4.0).mean()),
        "acc8": float((values < 8.0).mean()),
        "stage1_in_range": float(range_masks[0][mask].mean()),
        "stage2_in_range": float(range_masks[1][mask].mean()),
        "stage3_in_range": float(range_masks[2][mask].mean()),
        "stage1_range_width": float(range_widths[0][mask].mean()),
        "stage2_range_width": float(range_widths[1][mask].mean()),
        "stage3_range_width": float(range_widths[2][mask].mean()),
    }


def add_result(pixel_accumulator, image_accumulator, region, result):
    if result["pixels"] == 0:
        return
    if region not in pixel_accumulator:
        pixel_accumulator[region] = {
            "pixels": 0,
            "images": 0,
            **{name: 0.0 for name in METRIC_NAMES},
        }
        image_accumulator[region] = []
    count = result["pixels"]
    pixel_accumulator[region]["pixels"] += count
    pixel_accumulator[region]["images"] += 1
    for name in METRIC_NAMES:
        pixel_accumulator[region][name] += result[name] * count
    image_accumulator[region].append(result)


def summarize(pixel_accumulator, image_accumulator):
    rows = []
    for region, values in pixel_accumulator.items():
        count = values["pixels"]
        rows.append({
            "aggregation": "pixel_weighted",
            "region": region,
            "images": values["images"],
            "pixels": count,
            **{name: values[name] / count for name in METRIC_NAMES},
        })
        image_results = image_accumulator[region]
        rows.append({
            "aggregation": "image_mean",
            "region": region,
            "images": len(image_results),
            "pixels": sum(item["pixels"] for item in image_results),
            **{
                name: float(np.mean([item[name] for item in image_results]))
                for name in METRIC_NAMES
            },
        })
    return rows


def write_csv(filename, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(label, rows):
    print("\n{}".format(label))
    print("{:<15} {:<28} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "aggregation", "region", "pixels", "abs", "acc2", "acc4",
        "s2_range", "s3_range", "s2_width", "s3_width"))
    for row in rows:
        print("{:<15} {:<28} {:>10d} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.2f} {:>9.2f}".format(
            row["aggregation"], row["region"], row["pixels"], row["abs"],
            row["acc2"], row["acc4"], row["stage2_in_range"],
            row["stage3_in_range"], row["stage2_range_width"],
            row["stage3_range_width"]))


def main():
    args = parse_args()
    variant = get_model_variant(args.model_type)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.eval_nviews < 2 or args.region_nviews < 2:
        raise ValueError("evaluation and region masks require at least two views")
    if not 0.0 < args.large_disp_pct < 100.0:
        raise ValueError("--large_disp_pct must be in (0,100)")
    if args.occ_abs_tol < 0.0 or args.occ_rel_tol < 0.0:
        raise ValueError("occlusion tolerances must be non-negative")
    if args.hypothesis_residual_scale < 0.0:
        raise ValueError("--hypothesis_residual_scale must be non-negative")
    if not 0.0 <= args.visibility_fusion_beta <= 1.0:
        raise ValueError("--visibility_fusion_beta must be in [0,1]")
    if args.hybrid_sigma_scale < 0.0:
        raise ValueError("--hybrid_sigma_scale must be non-negative")
    if args.hybrid_max_scale < 1.0:
        raise ValueError("--hybrid_max_scale must be at least 1")
    if variant.hypothesis_fusion and args.vismode != "soft":
        raise ValueError("M1 currently requires --vismode soft")
    if variant.visibility_modeling and args.vismode != "soft":
        raise ValueError("M2 currently requires --vismode soft")
    if args.light != -1 and not 0 <= args.light <= 6:
        raise ValueError("--light must be -1 or in [0,6]")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    dataset_class = find_dataset_def("dtu_yao")
    dataset = dataset_class(
        args.testpath, args.testlist, "test", args.eval_nviews,
        args.numdepth, args.interval_scale)
    if args.light >= 0:
        dataset.metas = [meta for meta in dataset.metas if int(meta[1]) == args.light]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = nn.DataParallel(build_model(args)).cuda()
    load_checkpoint(model, args.loadckpt)
    model.eval()

    checkpoint_dir = os.path.basename(os.path.dirname(os.path.abspath(args.loadckpt)))
    label = args.label or checkpoint_dir or "model"
    checkpoint_path = os.path.abspath(args.loadckpt)
    testlist_path = os.path.abspath(args.testlist)
    output_dir = os.path.abspath(args.outdir)
    variant_metadata = {
        "ablation_code": variant.code,
        "factor_m1_hypothesis_fusion": int(variant.hypothesis_fusion),
        "factor_m2_visibility_modeling": int(variant.visibility_modeling),
        "factor_m3_hybrid_sampling": int(variant.hybrid_sampling),
    }

    per_image_rows = []
    pixel_accumulator = {}
    image_accumulator = {}
    evaluated_samples = 0
    previous_geometry_key = None
    cached_geometry = None

    print("Evaluating {} ({}): samples={}, eval_views={}, region_views={}, light={}".format(
        label, args.model_type, len(dataset), args.eval_nviews, args.region_nviews,
        "all" if args.light == -1 else args.light))
    print("M1/M2/M3 factors: {}".format(variant.code))

    with torch.no_grad():
        for batch_index, sample in enumerate(loader):
            first_sample_index = batch_index * args.batch_size
            if args.max_samples is not None and first_sample_index >= args.max_samples:
                break

            sample_cuda = to_cuda(sample)
            outputs, _, _ = model(
                sample_cuda["imgs"], sample_cuda["proj_matrices"],
                sample_cuda["depth_values"])
            depth_gt_tensor = sample_cuda["depth"]
            depth_est_full = F.interpolate(
                outputs[-1][0].unsqueeze(1), size=depth_gt_tensor.shape[-2:],
                mode="bilinear", align_corners=False).squeeze(1)
            range_tensors, range_width_tensors = full_resolution_range_diagnostics(
                outputs, depth_gt_tensor, sample_cuda["depth_values"])

            for batch_item in range(depth_gt_tensor.shape[0]):
                sample_index = first_sample_index + batch_item
                if args.max_samples is not None and sample_index >= args.max_samples:
                    break

                depth_gt = to_2d(depth_gt_tensor[batch_item].cpu().numpy())
                depth_est = to_2d(depth_est_full[batch_item].cpu().numpy())
                gt_mask = to_2d(sample_cuda["mask"][batch_item].cpu().numpy()) > 0.5
                valid = gt_mask & np.isfinite(depth_gt) & np.isfinite(depth_est)
                error = np.abs(depth_est - depth_gt)
                range_masks = [
                    item[batch_item].cpu().numpy().astype(bool)
                    for item in range_tensors
                ]
                range_widths = [
                    item[batch_item].cpu().numpy().astype(np.float32)
                    for item in range_width_tensors
                ]

                scan, light, ref_view, all_src_views = dataset.metas[sample_index]
                region_src_views = all_src_views[:args.region_nviews - 1]
                geometry_key = (scan, ref_view, tuple(region_src_views), depth_gt.shape)
                if geometry_key != previous_geometry_key:
                    cached_geometry = geometry_region_maps(
                        args.testpath, scan, ref_view, region_src_views, depth_gt,
                        occ_abs_tol=args.occ_abs_tol,
                        occ_rel_tol=args.occ_rel_tol)
                    previous_geometry_key = geometry_key

                boundary = boundary_mask(depth_gt, valid, args.boundary_pct)
                large_disparity = percentile_mask(
                    cached_geometry["disparity"], valid, args.large_disp_pct)
                occluded_any = valid & cached_geometry["occluded_any"]
                occluded_majority = valid & cached_geometry["occluded_majority"]
                region_masks = {
                    "full": valid,
                    "boundary": boundary,
                    "large_disparity": large_disparity,
                    "occluded_any": occluded_any,
                    "occluded_majority": occluded_majority,
                    "large_disp_and_occluded": large_disparity & occluded_any,
                    "boundary_and_occluded": boundary & occluded_any,
                }

                for region, region_mask in region_masks.items():
                    result = metrics(
                        error, region_mask, range_masks, range_widths)
                    if result["pixels"] == 0:
                        continue
                    row = {
                        "label": label,
                        "model_type": args.model_type,
                        **variant_metadata,
                        "checkpoint": checkpoint_path,
                        "testlist": testlist_path,
                        "eval_nviews": args.eval_nviews,
                        "region_nviews": args.region_nviews,
                        "hypothesis_residual_scale": args.hypothesis_residual_scale,
                        "visibility_fusion_beta": args.visibility_fusion_beta,
                        "hybrid_stage2_wide_num": args.hybrid_stage2_wide_num,
                        "hybrid_stage3_wide_num": args.hybrid_stage3_wide_num,
                        "hybrid_sigma_scale": args.hybrid_sigma_scale,
                        "hybrid_max_scale": args.hybrid_max_scale,
                        "scan": scan,
                        "view": int(ref_view),
                        "light": int(light),
                        "region": region,
                        **result,
                    }
                    per_image_rows.append(row)
                    add_result(
                        pixel_accumulator, image_accumulator,
                        region, result)
                evaluated_samples += 1

            if (batch_index + 1) % args.print_freq == 0:
                print("[batch {}/{}] evaluated {} samples".format(
                    batch_index + 1, len(loader), evaluated_samples))

    if not per_image_rows:
        raise RuntimeError("no valid samples were evaluated")

    summary_rows = summarize(pixel_accumulator, image_accumulator)
    for row in summary_rows:
        row.update({
            "label": label,
            "model_type": args.model_type,
            **variant_metadata,
            "checkpoint": checkpoint_path,
            "testlist": testlist_path,
            "eval_nviews": args.eval_nviews,
            "region_nviews": args.region_nviews,
            "hypothesis_residual_scale": args.hypothesis_residual_scale,
            "visibility_fusion_beta": args.visibility_fusion_beta,
            "hybrid_stage2_wide_num": args.hybrid_stage2_wide_num,
            "hybrid_stage3_wide_num": args.hybrid_stage3_wide_num,
            "hybrid_sigma_scale": args.hybrid_sigma_scale,
            "hybrid_max_scale": args.hybrid_max_scale,
            "light": "all" if args.light == -1 else args.light,
        })

    common_fields = [
        "label", "model_type", "ablation_code",
        "factor_m1_hypothesis_fusion", "factor_m2_visibility_modeling",
        "factor_m3_hybrid_sampling", "checkpoint", "testlist",
        "eval_nviews", "region_nviews", "hypothesis_residual_scale",
        "visibility_fusion_beta", "hybrid_stage2_wide_num",
        "hybrid_stage3_wide_num", "hybrid_sigma_scale", "hybrid_max_scale",
    ]
    per_image_fields = common_fields + [
        "scan", "view", "light", "region", "pixels", *METRIC_NAMES,
    ]
    summary_fields = common_fields + [
        "light", "aggregation", "region", "images", "pixels", *METRIC_NAMES,
    ]
    per_image_file = os.path.join(output_dir, "all_metrics.csv")
    summary_file = os.path.join(output_dir, "summary_metrics.csv")
    write_csv(per_image_file, per_image_fields, per_image_rows)
    write_csv(summary_file, summary_fields, summary_rows)

    print_summary(label, summary_rows)
    print("\nSaved per-image metrics:", per_image_file)
    print("Saved summary metrics:", summary_file)


if __name__ == "__main__":
    main()
