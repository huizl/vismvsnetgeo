#!/usr/bin/env python3
"""Generate Vis-MVSNet ablation reports from region summary CSV files."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "eval"
DOCS_ROOT = ROOT / "docs"

EXPERIMENTS = (
    ("View5", "ablation_test_light3", "五视图训练、五视图测试"),
    ("View3", "ablation_test_view3_light3", "三视图训练、五视图测试"),
)
MODELS = ("vis", "oa", "range", "hyp", "oa_range", "oa_hyp", "range_hyp", "oa_full")
MODEL_LABELS = {
    "vis": "vis（原始 Vis-MVSNet）",
    "oa": "oa（A）",
    "range": "range（B）",
    "hyp": "hyp（C）",
    "oa_range": "oa_range（A+B）",
    "oa_hyp": "oa_hyp（A+C）",
    "range_hyp": "range_hyp（B+C）",
    "oa_full": "oa_full（A+B+C）",
}
REGIONS = (
    ("full", "Full"),
    ("boundary", "Boundary"),
    ("large_disparity", "大视差"),
    ("occluded_any", "任一源遮挡"),
    ("occluded_majority", "多数源遮挡"),
    ("large_disp_and_occluded", "大视差+遮挡"),
    ("boundary_and_occluded", "边界+遮挡"),
)
METRICS = (
    ("abs", "Abs ↓", "relative"),
    ("acc2", "Acc2 ↑", "pp"),
    ("acc4", "Acc4 ↑", "pp"),
    ("acc8", "Acc8 ↑", "pp"),
    ("stage1_in_range", "S1 coverage ↑", "pp"),
    ("stage2_in_range", "S2 coverage ↑", "pp"),
    ("stage3_in_range", "S3 coverage ↑", "pp"),
    ("stage1_range_width", "S1 width", "relative"),
    ("stage2_range_width", "S2 width", "relative"),
    ("stage3_range_width", "S3 width", "relative"),
)


def load_experiment(directory):
    data = {}
    paths = sorted((EVAL_ROOT / directory).glob("*/summary_metrics.csv"))
    if len(paths) != 8:
        raise RuntimeError(f"expected 8 summaries in {directory}, found {len(paths)}")
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["aggregation"] == "pixel_weighted":
                    data[(row["model_type"], row["region"])] = row
    expected = {(model, region) for model in MODELS for region, _ in REGIONS}
    missing = expected.difference(data)
    if missing:
        raise RuntimeError(f"missing rows in {directory}: {sorted(missing)}")
    return data


def value(row, metric):
    return float(row[metric])


def change_text(current, reference, kind, metric):
    if kind == "pp":
        delta = (current - reference) * 100.0
        return f"{current:.4f} ({delta:+.2f} pp)"
    delta = (current / reference - 1.0) * 100.0
    if metric == "abs":
        if abs(delta) < 0.05:
            status = "持平"
        elif delta < 0:
            status = f"改善 {abs(delta):.1f}%"
        else:
            status = f"退化 {delta:.1f}%"
        return f"{current:.3f}（{status}）"
    return f"{current:.3f} ({delta:+.1f}%)"


def metric_table(data, region, reference_model):
    reference = data[(reference_model, region)]
    headers = ["模型"] + [label for _, label, _ in METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(METRICS)) + " |",
    ]
    for model in MODELS:
        row = data[(model, region)]
        cells = [MODEL_LABELS[model]]
        for metric, _, kind in METRICS:
            current = value(row, metric)
            ref = value(reference, metric)
            if model == reference_model:
                formatted = f"{current:.3f}" if kind == "relative" else f"{current:.4f}"
                cells.append(f"{formatted}（基准）")
            else:
                cells.append(change_text(current, ref, kind, metric))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def abs_summary_table(data, reference_model):
    headers = ["模型"] + [label for _, label in REGIONS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(REGIONS)) + " |",
    ]
    for model in MODELS:
        cells = [MODEL_LABELS[model]]
        for region, _ in REGIONS:
            current = value(data[(model, region)], "abs")
            reference = value(data[(reference_model, region)], "abs")
            if model == reference_model:
                cells.append(f"{current:.3f}（基准）")
            else:
                cells.append(change_text(current, reference, "relative", "abs"))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def report_intro(reference_model):
    reference_name = "原始 Vis-MVSNet `vis`" if reference_model == "vis" else "B-only `range`"
    purpose = (
        "用于正式判断各项改进相对原始 Vis-MVSNet 的效果。"
        if reference_model == "vis"
        else "用于内部判断从 B 出发加入 A/C 或移除 B 后的变化；它不替代原始 Vis-MVSNet baseline。"
    )
    return [
        f"# 以{reference_name}为参照的完整消融结果分析".replace("`为", "` 为"),
        "",
        "## 比较口径",
        "",
        f"本文以{reference_name}为参照，{purpose}".replace("`为", "` 为"),
        "",
        "| 模型 | A：逐源遮挡监督 | B：自适应范围 | C：假设级融合 |",
        "| --- | ---: | ---: | ---: |",
        "| `vis` | 0 | 0 | 0 |",
        "| `oa` | 1 | 0 | 0 |",
        "| `range` | 0 | 1 | 0 |",
        "| `hyp` | 0 | 0 | 1 |",
        "| `oa_range` | 1 | 1 | 0 |",
        "| `oa_hyp` | 1 | 0 | 1 |",
        "| `range_hyp` | 0 | 1 | 1 |",
        "| `oa_full` | 1 | 1 | 1 |",
        "",
        "Abs 使用相对变化 `(Abs_model / Abs_reference - 1) * 100%`，并直接标注改善或退化。",
        "Acc2/Acc4/Acc8 和 coverage 使用百分点变化 `(metric_model - metric_reference) * 100`。",
        "Range width 使用相对百分比变化。所有表格采用 DTU test、light 3、pixel-weighted 聚合。",
        "",
        "现有 `view3` checkpoint 是三视图训练，但 CSV 记录 `eval_nviews=5`、`region_nviews=5`，因此对应三视图训练、五视图测试。",
        "",
    ]


def interpretation(reference_model):
    if reference_model == "vis":
        return [
            "## 结果解释",
            "",
            "### A：逐源遮挡监督",
            "",
            "A 对边界和大视差遮挡区域有局部收益。View5 的边界+遮挡 Abs 改善 3.4%；View3 的大视差 Acc4 提高 1.39 pp，大视差+遮挡 Acc4 提高 1.76 pp。A 对严重离群误差的抑制不稳定，因此部分遮挡区域会出现 Acc 提高而 Abs 退化。",
            "",
            "### B：自适应范围",
            "",
            "B 当前系统性退化。View5 大视差 Abs 退化 45.7%，View3 大视差退化 34.2%。大视差 Stage 2 coverage 下降约 6.3 至 6.5 pp，大视差+遮挡下降约 8.2 至 8.4 pp。所有包含 B 的组合都被拖累。",
            "",
            "### C：假设级融合",
            "",
            "C 是当前最有效的模块。View5 的大视差、大视差+遮挡 Abs 分别改善 3.0% 和 5.0%；View3 分别改善 9.3% 和 8.1%。View3 七个区域的 Abs 全部改善，幅度为 5.5% 至 9.3%。",
            "",
            "### 组合效应",
            "",
            "A+C 通常优于 `vis`，但整体弱于 C-only，说明两个可见性监督存在重复或梯度竞争。包含 B 的 A+B、B+C、A+B+C 均明显差于 `vis`，完整模型失败的主要原因是 B。",
            "",
            "## 结论",
            "",
            "当前主模型应选择 `hyp`。A 可作为边界和阈值准确率的补充消融。B 应先修复边界裁剪、搜索覆盖率和固定假设数下的采样分辨率问题，再进入最终组合。正式结果还应补充多个随机种子及标准 DTU 点云 Accuracy、Completeness 和 Overall。",
        ]
    return [
        "## 条件比较与解释",
        "",
        "将 B-only 作为出发点时，`oa_range - range` 表示在 B 上加入 A，`range_hyp - range` 表示在 B 上加入 C，`oa_full - range` 表示在 B 上加入 A+C。其余不含 B 的配置同时包含了移除 B 所带来的恢复。",
        "",
        "View5 中，A+B 相对 B-only 的 Full Abs 改善 5.0%，大视差+遮挡改善 2.4%，边界+遮挡改善 4.2%；B+C 的对应改善为 3.2%、1.9% 和 1.6%。",
        "",
        "View3 中，A+B 相对 B-only 的 Full Abs 改善 2.7%，大视差改善 2.0%，边界+遮挡改善 4.6%；B+C 对大视差+遮挡改善 3.7%，对边界+遮挡改善 4.5%。A+C 同时加入 B 没有加法收益。",
        "",
        "不含 B 的模型相对 B-only 会出现很大的改善。例如 View3 大视差中，`range -> hyp` 改善 32.4%，其中 `range -> vis` 的 25.5% 是移除 B 的恢复，`vis -> hyp` 的 9.3% 才是 C 相对原始结构的独立收益。由于分母不同，两项百分比不能直接相加。",
        "",
        "## 结论",
        "",
        "以 B-only 为参照适合分析模块交互，但应称为 reference 或 anchor。结果表明 A/C 只能小幅缓解 B 的损害，移除 B 才是主要恢复来源；不含 B 时，C-only 的 `hyp` 是当前最强配置。",
    ]


def generate(reference_model, output_name):
    lines = report_intro(reference_model)
    for experiment_name, directory, description in EXPERIMENTS:
        data = load_experiment(directory)
        lines.extend([f"## {experiment_name}：{description}", "", "### 七区域 Abs 汇总", ""])
        lines.extend(abs_summary_table(data, reference_model))
        lines.append("")
        for region, region_label in REGIONS:
            lines.extend([f"### {region_label}：全部指标", ""])
            lines.extend(metric_table(data, region, reference_model))
            lines.append("")
    lines.extend(interpretation(reference_model))
    lines.append("")
    (DOCS_ROOT / output_name).write_text("\n".join(lines), encoding="utf-8")


def main():
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    generate("vis", "VIS_BASELINE_ANALYSIS.md")
    generate("range", "B_REFERENCED_ANALYSIS.md")
    print("Generated docs/VIS_BASELINE_ANALYSIS.md")
    print("Generated docs/B_REFERENCED_ANALYSIS.md")


if __name__ == "__main__":
    main()
