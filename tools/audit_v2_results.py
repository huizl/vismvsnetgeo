"""Audit the existing separate v2 runs without loading models or DTU data."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


MODELS = ("v2_vis", "v2_m1", "v2_m2", "v2_m3")
REGIONS = ("full", "boundary", "large_disparity", "occluded_any",
           "occluded_majority", "large_disp_and_occluded", "boundary_and_occluded")
METRICS = ("abs", "acc2", "acc4", "acc8", "stage1_in_range",
           "stage2_in_range", "stage3_in_range", "stage1_range_width",
           "stage2_range_width", "stage3_range_width")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def aggregate(rows, metric, pixel_weighted=True):
    rows = [row for row in rows if int(row["pixels"]) > 0]
    weights = [int(row["pixels"]) if pixel_weighted else 1 for row in rows]
    return sum(float(row[metric]) * weight for row, weight in zip(rows, weights)) / sum(weights)


def audit(root, train_nviews, seed):
    runs = {}
    manifests = {}
    for model in MODELS:
        run_path = root / f"{model}_view{train_nviews}_seed{seed}"
        manifests[model] = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        rows = read_rows(run_path / model / "all_metrics.csv")
        keyed = {(r["scan"], r["view"], r["light"], r["region"]): r for r in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"Duplicate image/region rows in {model}")
        for row in rows:
            if row["model_type"] != model or "test.txt" in row["testlist"]:
                raise ValueError(f"Unexpected model or test split in {model}")
            if int(row["pixels"]) > 0 and not all(math.isfinite(float(row[m])) for m in METRICS):
                raise ValueError(f"Nonfinite metrics in {model}")
        summaries = read_rows(run_path / model / "summary_metrics.csv")
        expected = {(a, r) for a in ("pixel_weighted", "image_mean") for r in REGIONS}
        if len(summaries) != len(expected) or {(r["aggregation"], r["region"]) for r in summaries} != expected:
            raise ValueError(f"Missing or duplicate summaries in {model}")
        for summary in summaries:
            selected = [r for r in rows if r["region"] == summary["region"] and int(r["pixels"]) > 0]
            if sum(int(r["pixels"]) for r in selected) != int(summary["pixels"]) or len(selected) != int(summary["images"]):
                raise ValueError(f"Summary counts mismatch in {model}")
            for metric in METRICS:
                value = aggregate(selected, metric, summary["aggregation"] == "pixel_weighted")
                if not math.isclose(value, float(summary[metric]), rel_tol=1e-7, abs_tol=1e-7):
                    raise ValueError(f"Summary mismatch: {model}/{summary['region']}/{metric}")
        runs[model] = keyed

    baseline = manifests["v2_vis"]
    for model in MODELS:
        manifest = manifests[model]
        if manifest["source_sha256"] != baseline["source_sha256"]:
            raise ValueError(f"Recorded source/list hashes differ in {model}")
        for key in baseline["arguments"]:
            if key not in ("models", "out_root") and manifest["arguments"].get(key) != baseline["arguments"][key]:
                raise ValueError(f"Run argument differs: {model}/{key}")
        if runs[model].keys() != runs["v2_vis"].keys():
            raise ValueError(f"Unmatched image/region keys in {model}")
        for key, row in runs[model].items():
            for field in ("pixels", "testlist", "eval_nviews", "region_nviews", "light"):
                if row[field] != runs["v2_vis"][key][field]:
                    raise ValueError(f"Unmatched {field}: {model}/{key}")

    settings = baseline["arguments"]
    example = next(iter(runs["v2_vis"].values()))
    lines = ["# v2 现有 Val 结果联合审计", "",
             f"由 tools/audit_v2_results.py 读取 eval/v2 下四组 view{train_nviews}/seed{seed} 记录生成。",
             f"训练 {settings['epochs']} epochs、batch={settings['batch_size']}；Val light={example['light']}，推理/区域定义视图数为 {example['eval_nviews']}/{example['region_nviews']}。",
             "每组权重选择以原 manifest 与 CSV 的 checkpoint 字段为准。当前四组均记录 best_2mm。",
             "记录中的公共运行参数及所记录源码/列表哈希一致；逐图区域键和像素数一致。",
             "两种聚合的全部 summary 指标均已从 all_metrics.csv 重新计算核对。",
             "这是对已保存 CSV/manifest 的审计，没有重跑模型；哈希清单不覆盖所有依赖，也不证明权重和运行环境完全一致。",
             "单 seed 不能证明稳定收益。ΔAbs < 0、ΔAcc2 > 0 表示改善。", ""]
    for weighted, title in ((True, "像素加权"), (False, "图像平均")):
        lines += [f"## {title}", "",
                  "| 区域 | 模型 | Abs mm | ΔAbs mm | Acc2 % | ΔAcc2 pp | S2 coverage % | S3 coverage % |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for region in REGIONS:
            base_rows = [r for r in runs["v2_vis"].values() if r["region"] == region]
            base_abs, base_acc = [aggregate(base_rows, m, weighted) for m in ("abs", "acc2")]
            for model in MODELS:
                rows = [r for r in runs[model].values() if r["region"] == region]
                ab, ac, s2, s3 = [aggregate(rows, m, weighted) for m in ("abs", "acc2", "stage2_in_range", "stage3_in_range")]
                lines.append(f"| {region} | {model} | {ab:.4f} | {ab-base_abs:+.4f} | {100*ac:.4f} | {100*(ac-base_acc):+.4f} | {100*s2:.4f} | {100*s3:.4f} |")
        lines.append("")
    lines += ["## 逐 scan 配对", "", "各 scan 内按像素加权；胜/平/负采用 1e-7 容差，未做显著性检验。", "",
              "| 区域 | 模型 | Abs 胜/平/负 | Acc2 胜/平/负 |", "| --- | --- | --- | --- |"]
    for region in REGIONS:
        groups = {}
        for model in MODELS:
            groups[model] = defaultdict(list)
            for row in runs[model].values():
                if row["region"] == region and int(row["pixels"]) > 0:
                    groups[model][row["scan"]].append(row)
        for model in MODELS[1:]:
            counts = []
            for metric, direction in (("abs", -1), ("acc2", 1)):
                deltas = [direction * (aggregate(rows, metric) - aggregate(groups["v2_vis"][scan], metric)) for scan, rows in groups[model].items()]
                counts.append(f"{sum(d > 1e-7 for d in deltas)}/{sum(abs(d) <= 1e-7 for d in deltas)}/{sum(d < -1e-7 for d in deltas)}")
            lines.append(f"| {region} | {model} | {counts[0]} | {counts[1]} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=project / "eval/v2")
    parser.add_argument("--train_nviews", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=project / "docs/V2_RESULT_AUDIT.md")
    args = parser.parse_args()
    report = audit(args.root, args.train_nviews, args.seed)
    args.output.write_text(report, encoding="utf-8")
    print(f"Verified four runs and both summary aggregations. Report: {args.output}")
