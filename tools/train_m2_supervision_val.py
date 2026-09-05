"""Train M2 with visibility supervision and beta=0, then evaluate only Val.

Uses a fresh initialization, the original 16-epoch training schedule, and
best_2mm.ckpt for the final light=3 region evaluation. --dry_run prints both
commands without creating files or starting training. CUDA_VISIBLE_DEVICES
is inherited from the caller.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build_commands(args):
    datapath = str(project_path(args.datapath))
    logdir = project_path(args.logdir)
    outdir = project_path(args.outdir)
    common = {
        "model_type": "m2_visibility", "testpath": datapath,
        "testlist": str(ROOT / "lists/dtu/val.txt"), "eval_nviews": 5,
        "batch_size": args.batch_size, "numdepth": 192, "interval_scale": 1.06,
        "vismode": "soft", "stage1_dnum": 48, "stage1_iscale": 4,
        "stage2_dnum": 32, "stage2_iscale": 2, "stage3_dnum": 16, "stage3_iscale": 1,
        "visibility_fusion_beta": 0.0, "occ_abs_tol": 2.0, "occ_rel_tol": 0.01,
        "seed": args.seed,
    }
    train = dict(common, mode="train", dataset="dtu_yao", trainpath=datapath,
                 trainlist=str(ROOT / "lists/dtu/train.txt"), logdir=str(logdir),
                 nviews=args.train_nviews, epochs=args.epochs, lr=0.001,
                 lrepochs="10,12,14:2", wd=0.0, visibility_gt_downsample=2,
                 pair_l1_weight=1.0, uncertainty_weight=1.0,
                 visibility_weight=0.2, visibility_focal_gamma=2.0,
                 hypothesis_visibility_weight=0.1, summary_freq=20, save_freq=1,
                 train_workers=args.train_workers, test_workers=args.eval_workers)
    evaluate = dict(common, loadckpt=str(logdir / "best_2mm.ckpt"),
                    label=f"m2_supervision_only_view{args.train_nviews}_seed{args.seed}",
                    outdir=str(outdir), region_nviews=5, light=3,
                    num_workers=args.eval_workers, boundary_pct=10, large_disp_pct=80)

    def command(script, options):
        result = [sys.executable, "-u", str(ROOT / script)]
        for name, value in options.items():
            result.extend(["--" + name, str(value)])
        return result

    return command("train.py", train), command("tools/eval_region_metrics_dtu_yao.py", evaluate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_nviews", type=int, choices=(3, 5), default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument("--eval_workers", type=int, default=4)
    parser.add_argument("--datapath", default="/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu")
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or min(args.train_workers, args.eval_workers, args.seed) < 0:
        parser.error("epochs/batch_size must be positive; workers/seed must be non-negative")
    label = f"m2_supervision_only_view{args.train_nviews}_seed{args.seed}"
    args.logdir = args.logdir or "checkpoints/val_round2/" + label
    args.outdir = args.outdir or "eval/val_round2/" + label
    train, evaluate = build_commands(args)
    print("TRAIN: " + shlex.join(train), flush=True)
    print("VAL:   " + shlex.join(evaluate), flush=True)
    if args.dry_run:
        return
    required = [ROOT / "lists/dtu/train.txt", ROOT / "lists/dtu/val.txt"]
    missing = [str(p) for p in required if not p.is_file()]
    missing.extend(str(project_path(args.datapath)/name) for name in ("Rectified", "Depths", "Cameras")
                   if not (project_path(args.datapath)/name).is_dir())
    if missing:
        parser.error("Missing input(s):\n" + "\n".join(missing))
    logdir, outdir = project_path(args.logdir), project_path(args.outdir)
    if logdir == outdir or logdir in outdir.parents or outdir in logdir.parents:
        parser.error("Training and validation outputs must be separate, non-nested directories")
    for directory in (logdir, outdir):
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            parser.error("Output must be new or empty: " + str(directory))
    logdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    sources = ("train.py", "models/vismvsnet_oa.py", "models/model_variants.py",
               "tools/train_m2_supervision_val.py", "tools/eval_region_metrics_dtu_yao.py",
               "lists/dtu/train.txt", "lists/dtu/val.txt")
    manifest = {"arguments": vars(args), "train_command": train, "validation_command": evaluate,
                "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}
    contents = json.dumps(manifest, indent=2)
    (logdir / "experiment.json").write_text(contents, encoding="utf-8")
    (outdir / "manifest.json").write_text(contents, encoding="utf-8")
    subprocess.run(train, cwd=ROOT, check=True)
    checkpoint = logdir / "best_2mm.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError("Training finished without " + str(checkpoint))
    with (outdir / "validation.log").open("w", encoding="utf-8") as stream:
        print("Training complete. Val evaluation log: " + str(outdir / "validation.log"), flush=True)
        subprocess.run(evaluate, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("Saved Val metrics: " + str(outdir / "summary_metrics.csv"), flush=True)


if __name__ == "__main__":
    main()
