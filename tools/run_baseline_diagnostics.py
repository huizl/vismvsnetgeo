"""Run one fixed v2_vis Val diagnostic; never train or evaluate the test split."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_command(args, output):
    return [sys.executable, "-u", str(ROOT / "tools/eval_region_metrics_dtu_yao.py"),
            "--model_type", "v2_vis", "--loadckpt", str(args.checkpoint.resolve()),
            "--testpath", str(args.datapath.resolve()),
            "--testlist", str(ROOT / "lists/dtu/val.txt"),
            "--outdir", str(output), "--eval_nviews", "5", "--region_nviews", "5",
            "--light", "3", "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--print_freq", "10",
            "--numdepth", "192", "--interval_scale", "1.06", "--vismode", "soft",
            "--stage1_dnum", "48", "--stage1_iscale", "4",
            "--stage2_dnum", "32", "--stage2_iscale", "2",
            "--stage3_dnum", "16", "--stage3_iscale", "1",
            "--hypothesis_residual_scale", "0.5", "--visibility_fusion_beta", "0.0",
            "--boundary_pct", "10", "--large_disp_pct", "80",
            "--occ_abs_tol", "2.0", "--occ_rel_tol", "0.01", "--seed", "1",
            "--failure_diagnostics",
            *(["--range_origin_diagnostics"] if getattr(args, "range_origin", False) else []),
            *(["--max_samples", str(args.max_samples)] if args.max_samples is not None else [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", type=Path,
                        default=Path("/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu"))
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints/v2/v2_vis_view3_seed1/best_2mm.ckpt")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--range_origin", action="store_true",
                        help="Also track initial coverage and later losses on the same GT grid.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("batch_size/max_samples must be positive; num_workers must be nonnegative")
    output = (args.outdir or ROOT / "eval/baseline_diagnostics" /
              datetime.now().strftime("v2_vis_%Y%m%d_%H%M%S_%f")).resolve()
    command = build_command(args, output)
    print(shlex.join(command), flush=True)
    print(f"Output: {output}", flush=True)
    if args.max_samples is not None:
        print("Limited samples: use this run only as a smoke check, not a full-Val conclusion.", flush=True)
    if args.dry_run:
        return
    for path in (args.checkpoint, args.datapath, ROOT / "lists/dtu/val.txt"):
        if not path.exists():
            parser.error(f"Missing input: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "train.py", *sorted((ROOT / "models").glob("*.py")),
               *sorted((ROOT / "datasets").glob("*.py")),
               ROOT / "tools/eval_region_metrics_dtu_yao.py",
               ROOT / "tools/failure_diagnostics.py", Path(__file__).resolve(),
               ROOT / "lists/dtu/val.txt"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "command": command, "max_samples": args.max_samples,
                "range_origin_diagnostics": args.range_origin,
                "scope": "full_val" if args.max_samples is None else "smoke_only",
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources}}
    manifest_path = output / "diagnostic_manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    try:
        with (output / "diagnostic.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        manifest.update(status="completed" if code == 0 else "failed", returncode=code)
        if code:
            raise RuntimeError(f"Diagnostic evaluation failed ({code}); inspect {output / 'diagnostic.log'}")
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print(f"Read: {output / 'failure_report.md'}", flush=True)
    if args.range_origin:
        print(f"Read: {output / 'range_origin_report.md'}", flush=True)


if __name__ == "__main__":
    main()
