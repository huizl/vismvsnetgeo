"""Fifteen aligned paper panels; display units are never labelled as mm."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

from tools.blendedmvs_metrics import REGIONS, region_metrics
from tools.paper_qualitative import binary_rgba, overlay_rgb, rgba, yellow_contours

NAMES = ("Base", "Base+A+B+C")
CATEGORY_LABELS = {
    "large_disparity_dominant": "Large disparity dominant",
    "occlusion_dominant": "Complex occlusion dominant",
    "joint_difficulty": "Large disparity and occlusion",
}


def save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_comparison(args, out, sample, geometry, predictions, selected):
    from tools.eval_blendedmvs import write_csv, write_json
    valid, gt = sample["mask"], sample["depth"]
    masks = geometry["masks"]
    large, occ = masks["large_disparity"], masks["occluded_ge2"]
    errors = {name: np.abs(predictions[name]["pred"] - gt) for name in NAMES}
    gain = errors[NAMES[0]] - errors[NAMES[1]]
    lo, hi = float(gt[valid].min()), float(gt[valid].max())
    if lo == hi:
        lo, hi = lo - .5, hi + .5
    depth_norm = Normalize(lo, hi, clip=True)
    error_norm = Normalize(0, args.error_max, clip=True)
    gain_norm = TwoSlopeNorm(vmin=-args.gain_max, vcenter=0, vmax=args.gain_max)
    disp_max = max(1e-6, float(geometry["disparity"][valid].max()))
    colored = lambda values, cmap, norm: rgba(values, valid, cmap, norm)
    outlined = lambda raster: yellow_contours(raster, large, occ)
    panels = [
        ("reference_rgb", "Reference RGB", sample["rgb"]),
        ("gt_depth", "GT depth", colored(gt, "turbo", depth_norm)),
        ("large_disparity_map", "GT displacement", colored(
            geometry["disparity"], "viridis", Normalize(0, disp_max))),
        ("source_occlusion_ratio", "Source occlusion ratio", colored(
            geometry["occlusion_ratio"], "viridis", Normalize(0, 1))),
        ("difficulty_overlay", "Red: disparity; green: occlusion; yellow: both",
         overlay_rgb(sample["rgb"], large, occ, args.overlay_alpha)),
        ("base_depth", "Base depth", colored(predictions[NAMES[0]]["pred"], "turbo", depth_norm)),
        ("full_model_depth", "Base+A+B+C depth",
         colored(predictions[NAMES[1]]["pred"], "turbo", depth_norm)),
        ("base_abs_error", "Base absolute error",
         outlined(colored(errors[NAMES[0]], "magma", error_norm))),
        ("full_model_abs_error", "Base+A+B+C absolute error",
         outlined(colored(errors[NAMES[1]], "magma", error_norm))),
        ("error_gain", "Error reduction: red better / blue worse",
         outlined(colored(gain, "RdBu_r", gain_norm))),
        ("occluded_any_mask", "Occluded in >=1 source view", binary_rgba(masks["occluded_any"])),
        ("occluded_ge2_mask", "Occluded in >=2 source views", binary_rgba(occ)),
        ("large_disparity_mask", "Large disparity mask", binary_rgba(large)),
        ("large_disp_and_occluded_any_mask", "Large disparity & >=1 occluded view",
         binary_rgba(masks["large_disp_and_occluded"])),
        ("large_disp_and_occluded_ge2_mask", "Large disparity & >=2 occluded views",
         binary_rgba(masks["large_disp_and_occluded_ge2"])),
    ]
    key = f"{sample['scene']}_view{sample['view']:08d}"
    category = selected["category"]
    folder = Path(out) / "comparison" / category / key
    directory = folder / "panels"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for position, (name, title, raster) in enumerate(panels, 1):
        Image.fromarray(raster).save(directory / f"{name}.png")
        fig, ax = plt.subplots(figsize=(5.2, 3.9), layout="constrained")
        ax.imshow(raster, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        save_figure(fig, directory / "with_title" / name)
        manifest.append(dict(position=position, name=name, title=title,
                             png=f"panels/{name}.png"))
    for name, indices in ((NAMES[0], (5, 7)), (NAMES[1], (6, 8))):
        own = Path(out) / name / key / "figures"
        own.mkdir(parents=True, exist_ok=True)
        for index in indices:
            Image.fromarray(panels[index][2]).save(own / f"{panels[index][0]}.png")
    fig, axes = plt.subplots(3, 5, figsize=(22, 12), layout="constrained")
    for ax, (_, title, raster) in zip(axes.flat, panels):
        ax.imshow(raster, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{args.series} | {key} | {CATEGORY_LABELS[category]}", fontsize=12)
    fig.legend(handles=[
        Patch(color="#ff0000", label="Disparity only (RGB overlay)"),
        Patch(color="#00ff00", label=">=2 occluded sources only (RGB overlay)"),
        Patch(color="#ffff00", label="Intersection (RGB overlay)"),
        Patch(color="#ffff00", label="Fixed GT difficulty contours (error maps)")],
        loc="lower center", ncol=2, fontsize=9)
    fig.get_layout_engine().set(rect=(0, .06, 1, .94))
    save_figure(fig, folder / "overview_3x5")
    rows = []
    for name in NAMES:
        result = predictions[name]
        for region in REGIONS:
            mask = masks[region]
            metrics = region_metrics(result["pred"], gt, mask, result["coverage"], result["widths"])
            row = dict(analysis_config=name, scene=sample["scene"], view=sample["view"],
                       region=region, depth_unit="scene_normalized", **metrics)
            row["abs_native"] = row["abs"] / sample["scene_scale"]
            if row["pixels"]:
                row.update(improved_pixels_pct=float(100*(gain[mask] > 0).mean()),
                           worsened_pixels_pct=float(100*(gain[mask] < 0).mean()),
                           unchanged_pixels_pct=float(100*(gain[mask] == 0).mean()))
            rows.append(row)
    base_rows = {r["region"]: r for r in rows if r["analysis_config"] == NAMES[0]}
    for row in rows:
        base = base_rows[row["region"]]
        row["abs_reduction_pct"] = (100*(base["abs"]-row["abs"])/base["abs"]
                                    if base["abs"] > 0 else float("nan"))
    write_csv(folder / "region_metrics.csv", rows)
    write_json(folder / "display_settings.json", dict(
        dataset="BlendedMVS", sample=key, category=category,
        depth_unit="scene-normalized units, NOT mm", scene_scale=sample["scene_scale"],
        depth_min=lo, depth_max=hi, error_max=args.error_max, gain_max=args.gain_max,
        overlay_alpha=args.overlay_alpha, show_colorbars=False, show_axis_ticks=False,
        gain_definition="abs(Base-GT) - abs(Base+A+B+C-GT)",
        gain_colors=dict(positive="red", negative="blue", zero="white"),
        selection=selected, metric_source="same predictions as CSV, unclipped errors",
        panels=manifest))
    caption = (
        f"{args.series}, {key}: {CATEGORY_LABELS[category]}.\n"
        "The two models use the same images, cameras, depth candidates and GT masks.\n"
        f"Depth colors: {lo:.5f} to {hi:.5f}; absolute error: 0 to {args.error_max:g}; "
        f"error reduction: {-args.gain_max:g} to {args.gain_max:g}, in normalized scene units, not mm.\n"
        f"Native depth = normalized depth / {sample['scene_scale']:.9g}.\n"
        "Red means lower error for the full model, blue higher error, white zero difference.\n"
        "Yellow contours show fixed GT large-disparity and >=2-source-occlusion regions.\n"
        "A zero occlusion ratio without comparable source GT does not establish visibility.\n"
        "No colorbars or axis ticks. Display clipping never changes numeric metrics.\n"
        f"Selected by descending full-image Abs reduction: {selected['abs_reduction_pct']:.3f}%.\n"
        "This is a selected improvement example, not an estimate of dataset-wide performance.\n")
    (folder / "figure_caption.txt").write_text(caption, encoding="utf-8")
    print(f"Saved {folder}: overview + 15 separate panels", flush=True)
