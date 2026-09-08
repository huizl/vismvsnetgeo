"""Run one fixed v2_vis Val diagnostic; never train or evaluate the test split."""

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import time
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
            "--stage1_dnum", str(getattr(args, "stage1_dnum", 48)), "--stage1_iscale", "4",
            "--stage2_dnum", "32", "--stage2_iscale", "2",
            "--stage3_dnum", "16", "--stage3_iscale", "1",
            "--hypothesis_residual_scale", "0.5", "--visibility_fusion_beta", "0.0",
            "--boundary_pct", "10", "--large_disp_pct", "80",
            "--occ_abs_tol", "2.0", "--occ_rel_tol", "0.01", "--seed", "1",
            "--failure_diagnostics",
            *(["--range_origin_diagnostics"] if getattr(args, "range_origin", False) else []),
            *(["--max_samples", str(args.max_samples)] if args.max_samples is not None else [])]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_diagnostic(args, output):
    command = build_command(args, output)
    print(shlex.join(command), flush=True)
    print(f"Output: {output}", flush=True)
    if args.max_samples is not None:
        print("Limited samples: use this run only as a smoke check, not a full-Val conclusion.", flush=True)
    if args.dry_run:
        return
    for path in (args.checkpoint, args.datapath, ROOT / "lists/dtu/val.txt"):
        if not path.exists():
            raise ValueError(f"Missing input: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "train.py", *sorted((ROOT / "models").glob("*.py")),
               *sorted((ROOT / "datasets").glob("*.py")),
               ROOT / "tools/eval_region_metrics_dtu_yao.py",
               ROOT / "tools/failure_diagnostics.py", Path(__file__).resolve(),
               ROOT / "lists/dtu/val.txt"]
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "command": command, "max_samples": args.max_samples,
                "range_origin_diagnostics": args.range_origin,
                "stage1_dnum": args.stage1_dnum,
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "scope": "full_val" if args.max_samples is None else "smoke_only",
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources}}
    manifest_path = output / "diagnostic_manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    started = time.perf_counter()
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
        manifest["elapsed_seconds"] = time.perf_counter() - started
        save()
    print(f"Read: {output / 'failure_report.md'}", flush=True)
    if args.range_origin:
        print(f"Read: {output / 'range_origin_report.md'}", flush=True)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_control(output):
    runs = [output / "s1_d48", output / "s1_d64"]
    manifests = [json.loads((p / "diagnostic_manifest.json").read_text(encoding="utf-8")) for p in runs]
    if any(m["status"] != "completed" for m in manifests):
        raise ValueError("Both runs must complete before comparison")
    for field in ("checkpoint_sha256", "source_sha256", "scope", "max_samples"):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f"Unmatched runs: {field}")
    normalized_commands = []
    for expected, manifest in zip((48, 64), manifests):
        command = manifest["command"]
        if manifest["stage1_dnum"] != expected or command[command.index("--stage1_dnum") + 1] != str(expected):
            raise ValueError("Incorrect S1 configuration in control")
        remove = {index + offset for index, value in enumerate(command)
                  if value in ("--stage1_dnum", "--outdir") for offset in (0, 1)}
        normalized_commands.append([value for index, value in enumerate(command) if index not in remove])
    if normalized_commands[0] != normalized_commands[1]:
        raise ValueError("Control commands differ beyond S1 depth count and output path")
    sample_sets = []
    for p in runs:
        rows = read_csv(p / "all_metrics.csv")
        keys = {(r["scan"], r["view"], r["light"], r["region"]): int(r["pixels"]) for r in rows}
        if len(keys) != len(rows):
            raise ValueError("Duplicate image/region rows")
        sample_sets.append(keys)
    if sample_sets[0] != sample_sets[1]:
        raise ValueError("Image/region keys or pixel counts differ")
    metrics = [{(r["aggregation"], r["region"]): r for r in read_csv(p / "summary_metrics.csv")} for p in runs]
    if metrics[0].keys() != metrics[1].keys():
        raise ValueError("Unmatched summary regions")
    lines = ["# S1 远端覆盖：48 vs 64 固定权重对照", "",
             f"范围：{manifests[0]['scope']}。相同 checkpoint SHA256、源码和逐图区域像素计数已核对。",
             "仅 S1 深度数由 48 改为 64；起点/间隔4Δ不变，S2/S3仍为32/16。",
             "S1末端由start+188Δ变为start+252Δ，新增远端支持64Δ。",
             "此项增加了S1计算预算且改变推理分布，是覆盖干预，不是等预算方法或重训结论。",
             "ΔAbs<0、ΔAcc2>0表示改善；限定样本仅用于检查流程。", "",
             "| 聚合 | 区域 | Abs48 mm | Abs64 mm | ΔAbs mm | Acc2_48 % | Acc2_64 % | ΔAcc2 pp |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, a in metrics[0].items():
        b = metrics[1][key]
        lines.append(f"| {key[0]} | {key[1]} | {float(a['abs']):.4f} | {float(b['abs']):.4f} | {float(b['abs'])-float(a['abs']):+.4f} | {100*float(a['acc2']):.3f} | {100*float(b['acc2']):.3f} | {100*(float(b['acc2'])-float(a['acc2'])):+.3f} |")
    origins = [{(r["region"], r["comparison"], r["group"]): r
                for r in read_csv(p / "range_origin_summary.csv")} for p in runs]
    if origins[0].keys() != origins[1].keys():
        raise ValueError("Unmatched origin groups")
    lines += ["", "## 覆盖及严重误差", "",
              "越界是统一GT网格统计；Bad8使用原生S3网格、严格>8mm，分开列示。",
              "S1范围随设置改变，S1内/外条件子集也改变，不能将组间误差变化直接解释成同一像素的恢复。", "",
              "| 区域 | S1远端越界48 % | S1远端越界64 % | S3范围外48 % | S3范围外64 % | Bad8_48 % | Bad8_64 % |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    failures = [{(r["stage"], r["region"], r["group"]): r for r in read_csv(p / "failure_summary.csv")} for p in runs]
    for aggregation, region in metrics[0]:
        if aggregation != "pixel_weighted":
            continue
        tails = [100*float(o[region, "direction_s1", "above"]["pixel_fraction"]) for o in origins]
        outside = [100*sum(float(o[region, "s1_to_s3", group]["pixel_fraction"]) for group in ("in_out", "out_out")) for o in origins]
        bad8 = [100*float(f["3", region, "all"]["bad8"]) for f in failures]
        lines.append(f"| {region} | {tails[0]:.3f} | {tails[1]:.3f} | {outside[0]:.3f} | {outside[1]:.3f} | {bad8[0]:.3f} | {bad8[1]:.3f} |")
    lines += ["", "## 运行成本", "",
              "前向时间同步所有CUDA设备并排除首批；峰值显存包含评估/诊断缓冲，并非纯推理显存。",
              "以下是顺序单次测量，不作统计显著性或稳定吞吐承诺。", "",
              "| S1平面数 | batch | 前向ms/图 | 评估秒 | 各设备峰值allocated MiB |",
              "| --- | ---: | ---: | ---: | --- |"]
    performances = [json.loads((p / "performance.json").read_text(encoding="utf-8")) for p in runs]
    if performances[0]["batch_size"] != performances[1]["batch_size"]:
        raise ValueError("Control batch sizes differ")
    for n, perf in zip((48, 64), performances):
        timing = f"{perf['forward_ms_per_image']:.3f}" if perf['forward_ms_per_image'] is not None else "n/a"
        memory = ", ".join(f"GPU{d['id']}: {d['peak_allocated_bytes']/1024**2:.1f}" for d in perf["devices"])
        lines.append(f"| {n} | {perf['batch_size']} | {timing} | {perf['evaluation_seconds']:.2f} | {memory} |")
    lines += ["", "判定：先检查新增覆盖是否伴随交集/遮挡/边界Abs和Bad8改善；覆盖增加但误差不降不能判为有效。",
              "保留所有有效GT，不按每张Val真值动态设置范围。各scan明细保存在两组all_metrics.csv。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


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
    parser.add_argument("--range_origin", action="store_true")
    parser.add_argument("--stage1_dnum", type=int, choices=(48, 64), default=48)
    parser.add_argument("--s1_range_control", action="store_true",
                        help="Sequential 48/64 same-checkpoint control, including origin diagnostics and comparison.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("batch_size/max_samples must be positive; num_workers must be nonnegative")
    if args.s1_range_control and args.stage1_dnum != 48:
        parser.error("--s1_range_control already runs both 48 and 64; omit --stage1_dnum")
    parent = "eval/s1_range_control" if args.s1_range_control else "eval/baseline_diagnostics"
    prefix = "v2_vis_s1_48vs64" if args.s1_range_control else f"v2_vis_s1d{args.stage1_dnum}"
    output = (args.outdir or ROOT / parent /
              datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f")).resolve()
    if not args.s1_range_control:
        run_diagnostic(args, output)
        return
    if not args.dry_run and output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Control output must be new or empty: {output}")
    manifest = {"status": "running", "stage1_depth_nums": [48, 64],
                "scope": "full_val" if args.max_samples is None else "smoke_only",
                "started_utc": datetime.now(timezone.utc).isoformat()}
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / "control_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        for number in (48, 64):
            child = argparse.Namespace(**vars(args))
            child.stage1_dnum, child.range_origin = number, True
            run_diagnostic(child, output / f"s1_d{number}")
        if not args.dry_run:
            compare_control(output)
            manifest["status"] = "completed"
            print(f"Comparison: {output / 'comparison.md'}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        if not args.dry_run:
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (output / "control_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
