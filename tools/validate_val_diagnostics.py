"""Run paired M2/M3 inference diagnostics exclusively on lists/dtu/val.txt.

No training or checkpoint selection is performed here. All interventions for
one module share its checkpoint. Run from the project root; --dry_run only
prints commands and needs neither torch, CUDA, DTU data nor checkpoints.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REGIONS = (
    "full", "boundary", "large_disparity", "occluded_any",
    "occluded_majority", "large_disp_and_occluded", "boundary_and_occluded",
)


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_experiments(args):
    checkpoint_root = project_path(args.checkpoint_root)
    outdir = project_path(args.outdir)
    # Checkpoint source is independent of the inference model for m3_vis.
    settings = [("vis", "vis", {}, "vis")]
    if args.suite in ("all", "m2"):
        for beta in ("0.0", "0.1", "0.2", "0.3"):
            settings.append(("m2_visibility", "m2_beta" + beta,
                             {"visibility_fusion_beta": beta}, "m2_visibility"))
    if args.suite in ("all", "m3"):
        for clipping in ("global", "none"):
            for scale in ("1.0", "2.0"):
                settings.append(("m3_hybrid", "m3_" + clipping + "_scale" + scale,
                                 {"hybrid_clip_mode": clipping,
                                  "hybrid_max_scale": scale}, "m3_hybrid"))
    if args.suite == "m3_vis":
        for scale in ("1.0", "2.0"):
            settings.append(("m3_hybrid", "vis_weights_m3_none_scale" + scale,
                             {"hybrid_clip_mode": "none", "hybrid_max_scale": scale}, "vis"))
    experiments = []
    for model_type, label, overrides, checkpoint_model in settings:
        checkpoint = checkpoint_root / (checkpoint_model + "_view" + str(args.train_nviews)) / args.checkpoint_name
        output = outdir / label
        options = {
            "model_type": model_type,
            "loadckpt": str(checkpoint),
            "label": label,
            "testpath": str(project_path(args.datapath)),
            "testlist": str(ROOT / "lists" / "dtu" / "val.txt"),
            "outdir": str(output),
            "eval_nviews": "5", "region_nviews": "5", "light": "3",
            "batch_size": str(args.batch_size), "num_workers": str(args.num_workers),
            "vismode": "soft", "numdepth": "192", "interval_scale": "1.06",
            "stage1_dnum": "48", "stage1_iscale": "4",
            "stage2_dnum": "32", "stage2_iscale": "2",
            "stage3_dnum": "16", "stage3_iscale": "1",
            "hypothesis_residual_scale": "1.0", "visibility_fusion_beta": "0.2",
            "hybrid_stage2_wide_num": "8", "hybrid_stage3_wide_num": "4",
            "hybrid_sigma_scale": "2.0", "hybrid_max_scale": "2.0",
            "hybrid_clip_mode": "global", "boundary_pct": "10",
            "large_disp_pct": "80", "occ_abs_tol": "2.0", "occ_rel_tol": "0.01",
            "seed": "1",
        }
        options.update(overrides)
        command = [sys.executable, str(ROOT / "tools" / "eval_region_metrics_dtu_yao.py")]
        for name, value in options.items():
            command.extend(["--" + name, value])
        experiments.append({"label": label, "checkpoint": str(checkpoint),
                            "output": str(output), "command": command})
    return experiments


def write_comparison(experiments, outdir):
    summaries = {}
    for experiment in experiments:
        with (Path(experiment["output"]) / "summary_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            summaries[experiment["label"]] = {
                r["region"]: r for r in csv.DictReader(stream)
                if r["aggregation"] == "pixel_weighted"
            }
    lines = ["# Val inference diagnostics", "",
             "Fixed val.txt, light=3, five inference/region views. Pixel-weighted metrics.",
             "These are inference diagnostics, not retrained ablations. Checkpoint sources are listed below.",
             "M2 beta=0 disables the inference gate; it does not undo visibility supervision during training.",
             "M3 scale=1 disables expansion; clipping is controlled independently.", "",
             "| Setting | Checkpoint |", "| --- | --- |"]
    for experiment in experiments:
        lines.append("| " + experiment["label"] + " | " + experiment["checkpoint"] + " |")
    lines.append("")
    for region in REGIONS:
        baseline = summaries["vis"][region]
        lines.extend(["## " + region, "",
                      "| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
        for label, summary in summaries.items():
            row = summary[region]
            if row["pixels"] != baseline["pixels"] or row["images"] != baseline["images"]:
                raise ValueError("Mismatched sample/pixel counts: " + label + "/" + region)
            numbers = [float(row["abs"]), float(row["abs"])-float(baseline["abs"]),
                       100*float(row["acc2"]), 100*(float(row["acc2"])-float(baseline["acc2"])),
                       *[100*float(row[k]) for k in ("acc4", "acc8", "stage2_in_range", "stage3_in_range")],
                       float(row["stage2_range_width"]), float(row["stage3_range_width"])]
            lines.append("| " + label + " | " + " | ".join(f"{x:.4f}" for x in numbers) + " |")
        lines.append("")
    (outdir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "m2", "m3", "m3_vis"), default="all",
                        help="m3_vis runs baseline and two M3 settings using the same vis checkpoint")
    parser.add_argument("--train_nviews", type=int, choices=(3, 5), default=5)
    parser.add_argument("--datapath", default="/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu")
    parser.add_argument("--checkpoint_root", default="checkpoints/dtu")
    parser.add_argument("--checkpoint_name", choices=("best_2mm.ckpt", "best_abs.ckpt"), default="best_2mm.ckpt")
    parser.add_argument("--outdir", default="eval/val_diagnostics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers non-negative")
    experiments = build_experiments(args)
    for experiment in experiments:
        print(experiment["label"] + ": " + shlex.join(experiment["command"]), flush=True)
    if args.dry_run:
        return
    # Validate every input before starting the first GPU evaluation.
    required = {Path(e["checkpoint"]) for e in experiments}
    required.add(ROOT / "lists" / "dtu" / "val.txt")
    missing = sorted(str(p) for p in required if not p.is_file())
    missing.extend(str(project_path(args.datapath) / name) for name in ("Rectified", "Depths", "Cameras")
                   if not (project_path(args.datapath) / name).is_dir())
    if missing:
        parser.error("Missing input(s):\n" + "\n".join(missing))
    outdir = project_path(args.outdir)
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        parser.error("Output must be a new or empty directory: " + str(outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    source_files = ("models/vismvsnet_oa.py", "tools/eval_region_metrics_dtu_yao.py",
                    "tools/validate_val_diagnostics.py", "lists/dtu/val.txt")
    manifest = {"arguments": vars(args), "experiments": experiments,
                "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index, experiment in enumerate(experiments, 1):
        logfile = outdir / (experiment["label"] + ".log")
        print(f"[{index}/{len(experiments)}] {experiment['label']} -> {logfile}", flush=True)
        with logfile.open("w", encoding="utf-8") as stream:
            command = [experiment["command"][0], "-u", *experiment["command"][1:]]
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    write_comparison(experiments, outdir)
    print("Saved comparison: " + str(outdir / "comparison.md"), flush=True)


if __name__ == "__main__":
    main()
