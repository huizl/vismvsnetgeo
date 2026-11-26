"""Compare existing checkpoints on official BlendedMVS inputs and GT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.blendedmvs import MVSDataset
from datasets.data_io import save_pfm
from tools.blendedmvs_metrics import (
    METRICS, PERCENT_METRICS, REGIONS, aggregate, describe_scene,
    geometry_maps, region_metrics, select_examples,
)
NAMES = ("Base", "Base+A+B+C")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--testpath", required=True)
    p.add_argument("--testlist", help="Scene names, one per line; default: all downloaded scenes")
    p.add_argument("--scene", help="Optional single-scene smoke check")
    p.add_argument("--series", choices=("View5", "View3"), default="View5",
                   help="Checkpoint training series, not inference view count")
    p.add_argument("--checkpoint_root", default="checkpoints/dtu")
    p.add_argument("--checkpoint_name", default="best_2mm.ckpt")
    p.add_argument("--base_checkpoint")
    p.add_argument("--full_checkpoint")
    # Preserve the historical evaluation pair independently of paper row labels.
    p.add_argument("--base_model_type", default="range")
    p.add_argument("--full_model_type", default="oa_hyp")
    p.add_argument("--outdir", default="outputs/cross_dataset/BlendedMVS")
    p.add_argument("--check_only", action="store_true", help="Check data without torch/GPU/checkpoints")
    p.add_argument("--check_samples", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=0, help="0=all; positive values are debug subsets")
    p.add_argument("--eval_nviews", type=int, default=5)
    p.add_argument("--max_width", type=int, default=768)
    p.add_argument("--max_height", type=int, default=576)
    p.add_argument("--normalize_depth_max", type=float, default=1000.0)
    p.add_argument("--numdepth", type=int, default=192)
    p.add_argument("--interval_scale", type=float, default=1.0)
    for stage, count, interval in ((1, 48, 4), (2, 32, 2), (3, 16, 1)):
        p.add_argument(f"--stage{stage}_dnum", type=int, default=count)
        p.add_argument(f"--stage{stage}_iscale", type=int, default=interval)
    p.add_argument("--range_sigma_scale", type=float, default=2.0)
    p.add_argument("--range_min_scale", type=float, default=1.0)
    p.add_argument("--range_max_scale", type=float, default=2.0)
    p.add_argument("--hypothesis_residual_scale", type=float, default=1.0)
    p.add_argument("--occ_abs_tol", type=float, default=2.0, help="Normalized scene units, NOT mm")
    p.add_argument("--occ_rel_tol", type=float, default=.01)
    p.add_argument("--large_disp_pct", type=float, default=80.0)
    p.add_argument("--examples_per_category", type=int, default=2)
    p.add_argument("--error_max", type=float, default=20.0)
    p.add_argument("--gain_max", type=float, default=10.0)
    p.add_argument("--overlay_alpha", type=float, default=.70)
    p.add_argument("--no_figures", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--print_freq", type=int, default=25)
    args = p.parse_args(argv)
    numeric = [args.normalize_depth_max, args.error_max, args.gain_max, args.interval_scale,
               args.occ_abs_tol, args.occ_rel_tol, args.range_sigma_scale,
               args.range_min_scale, args.range_max_scale,
               args.hypothesis_residual_scale, args.large_disp_pct, args.overlay_alpha]
    if not np.isfinite(numeric).all():
        p.error("Numeric settings must be finite")
    if min(args.check_samples, args.eval_nviews - 1, args.numdepth - 1,
           args.max_width // 64, args.max_height // 64, args.examples_per_category,
           args.print_freq) < 1 or args.max_samples < 0:
        p.error("Invalid sample count/view count/resize/candidate settings")
    if not 0 < args.large_disp_pct < 100 or not 0 <= args.overlay_alpha <= 1:
        p.error("Invalid disparity percentile or overlay alpha")
    if min(args.normalize_depth_max, args.error_max, args.gain_max, args.interval_scale) <= 0:
        p.error("Scale, interval and display limits must be positive")
    if min(args.occ_abs_tol, args.occ_rel_tol, args.range_sigma_scale,
           args.hypothesis_residual_scale) < 0:
        p.error("Tolerances and residual/sigma scales must be nonnegative")
    if not 0 < args.range_min_scale <= args.range_max_scale:
        p.error("Range scales require 0 < min <= max")
    if any(getattr(args, f"stage{s}_dnum") < 8 or getattr(args, f"stage{s}_dnum") % 8
           or getattr(args, f"stage{s}_iscale") <= 0 for s in (1, 2, 3)):
        p.error("Stage candidate counts must be multiples of 8; intervals positive")
    return args


def checkpoint_paths(args):
    types = (args.base_model_type, args.full_model_type)
    overrides = (args.base_checkpoint, args.full_checkpoint)
    paths = [Path(override) if override else Path(args.checkpoint_root) /
             f"{model_type}_{args.series.lower()}" / args.checkpoint_name
             for model_type, override in zip(types, overrides)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint absent: {path}. Set --base_checkpoint/"
                                    "--full_checkpoint if your checkpoint names differ.")
    return [path.resolve() for path in paths]


def comparison_rows(summaries):
    before = {(r["aggregation"], r["region"]): r for r in summaries[NAMES[0]]}
    after = {(r["aggregation"], r["region"]): r for r in summaries[NAMES[1]]}
    if before.keys() != after.keys():
        raise ValueError("Base/full summaries have different evaluation regions")
    results = []
    for key, a in before.items():
        b = after[key]
        if (a["pixels"], a["images"]) != (b["pixels"], b["images"]):
            raise ValueError("Base/full models evaluated different GT pixel sets")
        row = dict(aggregation=key[0], region=key[1], pixels=a["pixels"], images=a["images"],
                   abs_reduction_pct=100*(a["abs"]-b["abs"])/a["abs"] if a["abs"] else float("nan"),
                   abs_rel_reduction_pct=100*(a["abs_rel"]-b["abs_rel"])/a["abs_rel"]
                   if a["abs_rel"] else float("nan"))
        for metric in METRICS:
            row[f"base_{metric}"] = a[metric]
            row[f"full_{metric}"] = b[metric]
            if metric in PERCENT_METRICS:
                row[metric + "_delta_pp"] = 100*(b[metric]-a[metric])
        results.append(row)
    return results


def infer(model, sample, device):
    import torch
    import torch.nn.functional as F
    from tools.eval_region_metrics_dtu_yao import full_resolution_range_diagnostics
    inputs = [torch.from_numpy(np.ascontiguousarray(sample[key]))[None].to(device)
              for key in ("imgs", "proj_matrices", "depth_values")]
    gt = torch.from_numpy(sample["depth"].copy())[None].to(device)
    with torch.inference_mode():
        outputs, _, confidence = model(*inputs)
        pred = F.interpolate(outputs[-1][0][:, None], size=gt.shape[-2:],
                             mode="bilinear", align_corners=False)[0, 0]
        coverage, widths = full_resolution_range_diagnostics(outputs, gt, inputs[2])
        conf = F.interpolate(confidence[-1].reshape(1, 1, *confidence[-1].shape[-2:]),
                             size=gt.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        result = dict(pred=pred.cpu().numpy(), confidence=conf.cpu().numpy(),
                      coverage=[v[0].cpu().numpy() for v in coverage],
                      widths=[v[0].cpu().numpy() for v in widths])
    if not np.isfinite(result["pred"][sample["mask"]]).all():
        raise ValueError(f"{sample['scene']}/{sample['view']}: non-finite prediction on GT")
    return result


def main(argv=None):
    args = parse_args(argv)
    dataset = MVSDataset(args.testpath, args.testlist, "test", args.eval_nviews,
                         args.numdepth, args.interval_scale, scene=args.scene,
                         max_width=args.max_width, max_height=args.max_height,
                         normalize_depth_max=args.normalize_depth_max)
    out = Path(args.outdir) / args.series
    out.mkdir(parents=True, exist_ok=True)
    total = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    protocol = dict(dataset="BlendedMVS", data_root=str(dataset.root),
                    scene_list_source=dataset.split_source, scenes=dataset.scenes,
                    eligible_samples=len(dataset), requested_samples=total,
                    debug_subset=bool(args.scene or args.max_samples),
                    settings=vars(args).copy(), scene_scales=dataset.scene_scales,
                    normalization="target / median(valid camera depth_max); GT-independent",
                    depth_unit="scene-normalized units, not mm; native PFM = normalized / scene_scale",
                    source_selection="first eval_nviews-1 pair.txt entries, same for both models",
                    image_variant="unmasked blended_images/VIEW_ID.jpg",
                    network_camera_scale="quarter-resolution projection; full input K for GT geometry",
                    gt_sampling="nearest neighbour at resized input resolution; invalid GT excluded",
                    source_depth_sampling="nearest neighbour; finite positive source GT only",
                    majority_definition="at least half of GT-comparable source views",
                    ge2_definition="at least two occluded source views",
                    disparity_definition="max 2D displacement among positive-Z source projections",
                    accuracy_definition="strict < threshold; rel_acc1/2/5 use 1/2/5 percent of GT",
                    metric_pixels="fixed reference valid GT; non-finite model predictions fail",
                    training="existing checkpoints only, no training or fine-tuning",
                    visualization_selection="GT category, then descending full-image Abs reduction")
    write_csv(out / "skipped_samples.csv", dataset.skipped)
    if args.check_only:
        checks = []
        for index in range(total):
            sample = dataset[index]
            if not sample["mask"].any():
                dataset.skipped.append(dict(scene=sample["scene"], view=sample["view"],
                                            reason="reference GT has no finite positive depth"))
                continue
            geometry = geometry_maps(sample, args.occ_abs_tol, args.occ_rel_tol, args.large_disp_pct)
            checks.append(dict(scene=sample["scene"], view=sample["view"],
                               shape=list(sample["depth"].shape), scale=sample["scene_scale"],
                               **describe_scene(sample, geometry)))
            if len(checks) >= args.check_samples:
                break
        if not checks:
            raise ValueError("No valid reference GT in requested data-check samples")
        write_csv(out / "skipped_samples.csv", dataset.skipped)
        write_json(out / "data_check.json", dict(protocol, checked_samples=checks,
                                                 skipped_samples=dataset.skipped))
        print(f"Data check OK: {len(dataset.scenes)} scenes, {len(dataset)} eligible samples; "
              f"{len(checks)} image/GT pairs read. No GPU or checkpoints used.")
        print(f"Saved: {out / 'data_check.json'}")
        return

    import torch
    from models.model_variants import get_model_variant
    from tools.eval_region_metrics_dtu_yao import build_model, load_checkpoint
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use the server mvs environment, or --check_only.")
    if not args.no_figures:
        from tools.blendedmvs_visualization import render_comparison
    paths = checkpoint_paths(args)
    types = (args.base_model_type, args.full_model_type)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    models, identity = [], []
    args.vismode = "soft"
    for name, model_type, path in zip(NAMES, types, paths):
        args.model_type = model_type
        variant = get_model_variant(model_type)
        model = build_model(args)
        load_checkpoint(model, str(path))
        models.append(model.to(args.device).eval())
        identity.append(dict(analysis_config=name, raw_model_type=model_type,
                             raw_abc_code=variant.code, checkpoint=str(path), sha256=sha256(path)))
        print(f"{args.series}/{name}: raw model={model_type}, ABC={variant.code}, checkpoint={path}")
    protocol["checkpoints"] = identity
    write_json(out / "protocol.json", protocol)
    rows = {name: [] for name in NAMES}
    candidates, usable_indices = [], {}
    for index in range(total):
        sample = dataset[index]
        if not sample["mask"].any():
            dataset.skipped.append(dict(scene=sample["scene"], view=sample["view"],
                                        reason="reference GT has no finite positive depth"))
            continue
        geometry = geometry_maps(sample, args.occ_abs_tol, args.occ_rel_tol, args.large_disp_pct)
        whole = []
        for name, model, model_identity in zip(NAMES, models, identity):
            result = infer(model, sample, args.device)
            for region in REGIONS:
                metric = region_metrics(result["pred"], sample["depth"], geometry["masks"][region],
                                        result["coverage"], result["widths"])
                row = dict(analysis_config=name, raw_model_type=model_identity["raw_model_type"],
                           raw_abc_code=model_identity["raw_abc_code"], scene=sample["scene"],
                           view=sample["view"], scene_scale=sample["scene_scale"],
                           depth_unit="scene_normalized", region=region, **metric)
                row["abs_native"] = metric["abs"] / sample["scene_scale"]
                rows[name].append(row)
                if region == "full":
                    whole.append(metric["abs"])
            del result
        base, full = whole
        candidates.append(dict(scene=sample["scene"], view=sample["view"],
                               **describe_scene(sample, geometry), base_abs=base, full_abs=full,
                               abs_reduction_pct=100*(base-full)/base if base else 0.0))
        usable_indices[(sample["scene"], sample["view"])] = index
        if (index + 1) % args.print_freq == 0 or index + 1 == total:
            print(f"{args.series}: {index+1}/{total}, {sample['scene']}/view{sample['view']:08d}, "
                  f"Abs {base:.5f} -> {full:.5f} normalized units", flush=True)
    if not candidates:
        raise ValueError("No usable GT samples; no comparison produced")
    summaries = {}
    for name in NAMES:
        summaries[name] = aggregate(rows[name], scene_mean=True)
        folder = out / name / "exported_metrics"
        write_csv(folder / "all_metrics.csv", rows[name])
        write_csv(folder / "summary_metrics.csv", summaries[name])
        scene_rows = []
        for scene in dataset.scenes:
            subset = [r for r in rows[name] if r["scene"] == scene]
            for row in aggregate(subset):
                row.update(scene=scene, scene_scale=dataset.scene_scales[scene],
                           abs_native=row["abs"] / dataset.scene_scales[scene])
                scene_rows.append(row)
        write_csv(folder / "per_scene_metrics.csv", scene_rows)
    comparison = comparison_rows(summaries)
    write_csv(out / "comparison" / "summary_comparison.csv", comparison)
    scene_comparison = []
    for scene in dataset.scenes:
        scene_summaries = {
            name: aggregate([r for r in rows[name] if r["scene"] == scene])
            for name in NAMES}
        for row in comparison_rows(scene_summaries):
            row.update(scene=scene, scene_scale=dataset.scene_scales[scene],
                       base_abs_native=row["base_abs"] / dataset.scene_scales[scene],
                       full_abs_native=row["full_abs"] / dataset.scene_scales[scene])
            scene_comparison.append(row)
    write_csv(out / "comparison" / "per_scene_comparison.csv", scene_comparison)
    write_csv(out / "comparison" / "scene_candidates.csv", candidates)
    write_csv(out / "skipped_samples.csv", dataset.skipped)
    selections, missing = select_examples(candidates, args.examples_per_category)
    write_json(out / "scene_selection.json", dict(
        selections=selections, missing_categories=missing, evaluated_samples=len(candidates),
        selection_is_typical_improvement_not_dataset_estimate=True))
    for row in comparison:
        if row["aggregation"] == "pixel_weighted":
            print(f"{row['region']:<34} Abs {row['base_abs']:.5f} -> {row['full_abs']:.5f}; "
                  f"reduction {row['abs_reduction_pct']:.2f}%")
    for item in missing:
        print(f"WARNING: {item['category']}: selected {item['selected']}/{item['requested']} "
              "eligible improvement examples. No categories are fabricated.")
    if not args.no_figures:
        for selected in selections:
            sample = dataset[usable_indices[(selected["scene"], selected["view"])]]
            geometry = geometry_maps(sample, args.occ_abs_tol, args.occ_rel_tol, args.large_disp_pct)
            predictions = {}
            for name, model, model_identity in zip(NAMES, models, identity):
                result = infer(model, sample, args.device)
                metric = region_metrics(result["pred"], sample["depth"], sample["mask"],
                                        result["coverage"], result["widths"])
                expected = selected["base_abs" if name == NAMES[0] else "full_abs"]
                if not np.isclose(metric["abs"], expected, rtol=1e-4, atol=1e-5):
                    raise RuntimeError("Selected prediction differs from first-pass metrics; "
                                       "check CUDA determinism before using figures.")
                key = f"{sample['scene']}_view{sample['view']:08d}"
                folder = out / name / key
                folder.mkdir(parents=True, exist_ok=True)
                arrays = dict(rgb=sample["rgb"], gt=sample["depth"], pred=result["pred"],
                              valid=sample["mask"], disparity=geometry["disparity"],
                              occlusion_ratio=geometry["occlusion_ratio"],
                              occluded_count=geometry["occluded_count"],
                              comparable_count=geometry["comparable_count"],
                              **{"mask_" + k: v for k, v in geometry["masks"].items()})
                for stage in range(3):
                    arrays[f"stage{stage+1}_coverage"] = result["coverage"][stage]
                    arrays[f"stage{stage+1}_width"] = result["widths"][stage]
                np.savez_compressed(folder / "arrays.npz", **arrays)
                save_pfm(str(folder / "depth_native.pfm"),
                         (result["pred"] / sample["scene_scale"]).astype(np.float32))
                save_pfm(str(folder / "confidence.pfm"), result["confidence"].astype(np.float32))
                write_json(folder / "metadata.json", dict(
                    model_identity, scene=sample["scene"], view=sample["view"],
                    scene_scale=sample["scene_scale"], source_views=sample["source_views"],
                    category=selected["category"], array_depth_unit="scene_normalized",
                    pfm_depth_unit="native scene unit", original_metric_reproduced=True))
                predictions[name] = result
            render_comparison(args, out, sample, geometry, predictions, selected)
    protocol.update(completed=True, evaluated_samples=len(candidates),
                    excluded_samples=dataset.skipped, selected_examples=len(selections))
    write_json(out / "protocol.json", protocol)
    print(f"Saved metrics and comparison: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
