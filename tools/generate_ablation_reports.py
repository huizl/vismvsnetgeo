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
                cell = f"{current:.3f}（基准）"
            else:
                cell = change_text(current, reference, "relative", "abs")
            if current == min(value(data[(m, region)], "abs") for m in MODELS):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def accuracy_summary_table(data, reference_model):
    """Compact paper-facing table for Acc4 changes in the target regions."""
    selected = (
        ("full", "Full"),
        ("large_disparity", "大视差"),
        ("occluded_any", "任一遮挡"),
        ("large_disp_and_occluded", "大视差+遮挡"),
        ("boundary_and_occluded", "边界+遮挡"),
    )
    lines = [
        "| 模型 | " + " | ".join(label for _, label in selected) + " |",
        "| --- | " + " | ".join("---:" for _ in selected) + " |",
    ]
    for model in MODELS:
        cells = [MODEL_LABELS[model]]
        for region, _ in selected:
            current = value(data[(model, region)], "acc4")
            reference = value(data[(reference_model, region)], "acc4")
            if model == reference_model:
                cell = f"{current:.4f}（基准）"
            else:
                cell = f"{current:.4f} ({(current-reference)*100:+.2f} pp)"
            if current == max(value(data[(m, region)], "acc4") for m in MODELS):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def effect_decomposition_table(data):
    """Controlled Abs contrasts; positive values mean error reduction."""
    contrasts = (
        ("A 单独加入", "oa", "vis", "A 的单因素效应"),
        ("C 单独加入", "hyp", "vis", "C 的单因素效应"),
        ("在 C 上加入 A", "oa_hyp", "hyp", "A 在已有 C 时的条件效应"),
        ("在 A 上加入 C", "oa_hyp", "oa", "C 在已有 A 时的条件效应"),
        ("在 B 上加入 A", "oa_range", "range", "A 在已有 B 时的条件效应"),
        ("在 B 上加入 C", "range_hyp", "range", "C 在已有 B 时的条件效应"),
        ("在 B 上加入 A+C", "oa_full", "range", "A+C 在已有 B 时的联合效应"),
    )
    selected = (
        ("full", "Full"),
        ("large_disparity", "大视差"),
        ("large_disp_and_occluded", "大视差+遮挡"),
        ("boundary_and_occluded", "边界+遮挡"),
    )
    lines = [
        "| 对照 | 模型差分 | " + " | ".join(label for _, label in selected) + " | 可归因内容 |",
        "| --- | --- | " + " | ".join("---:" for _ in selected) + " | --- |",
    ]
    for label, current_model, base_model, meaning in contrasts:
        cells = [label, f"`{current_model} - {base_model}`"]
        for region, _ in selected:
            current = value(data[(current_model, region)], "abs")
            base = value(data[(base_model, region)], "abs")
            improvement = (base - current) / base * 100.0
            cells.append(f"{improvement:+.1f}%")
        cells.append(meaning)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def paper_ready_text(all_data, reference_model):
    if reference_model == "vis":
        return [
            "## 可直接用于论文的结果表述（草稿）",
            "",
            "在 DTU test、light 3、五视图测试条件下，假设级融合模块 C 表现出最稳定的收益。采用五视图训练时，C-only 相对原始 Vis-MVSNet 将 Full、Large-disparity、Large-disparity-and-occluded 区域的 Abs Error 分别降低 1.8%、3.0% 和 5.0%；采用三视图训练时，对应降幅扩大到 7.7%、9.3% 和 8.1%。这说明 C 的收益在训练视图减少时更明显，并集中体现在大视差及其与遮挡重叠的困难区域。",
            "",
            "逐源遮挡监督 A 的收益较为局部。五视图训练时，它将 Boundary-and-occluded 区域的 Abs Error 降低 3.4%，但 Full 指标退化 0.4%；三视图训练时，Full 和 Boundary-and-occluded 分别改善 1.4% 和 2.4%。因此，现有结果支持将 A 描述为边界和部分遮挡场景的辅助模块，不支持其在全部区域稳定优于基线的结论。",
            "",
            "自适应范围模块 B 在当前实现下明显降低性能。五视图训练时，大视差区域 Abs Error 增加 45.7%，Stage 2 和 Stage 3 coverage 分别下降 6.45 和 7.07 个百分点；三视图训练时，大视差 Abs Error 增加 34.2%。所有包含 B 的组合均受到明显拖累，说明搜索范围覆盖损失超过了范围自适应可能带来的收益。",
            "",
            "> 写入论文前请把模块全称、表号和数据集协议替换成正文中的正式命名。以上结论基于单次训练，不应使用“显著提升”等统计显著性措辞。",
            "",
        ]
    return [
        "## 可直接用于机制分析的结果表述（草稿）",
        "",
        "以 B-only 为内部参照时，在 B 上加入 A 或 C 只能获得小幅恢复。五视图训练下，`oa_range` 相对 `range` 的 Full、Large-disparity-and-occluded 和 Boundary-and-occluded Abs Error 分别降低 5.0%、2.4% 和 4.2%；`range_hyp` 的对应降幅为 3.2%、1.9% 和 1.6%。这些数字只能解释 A/C 在 B 存在时的条件效应。",
        "",
        "不含 B 的配置相对 B-only 呈现较大改善，但该差值混合了“移除 B”和“加入其他模块”两种变化。例如三视图训练的大视差区域中，`hyp` 相对 `range` 改善 32.4%，而 C 相对原始结构的受控对照 `hyp - vis` 为改善 9.3%。因此，32.4% 不应表述为 C 的独立贡献。",
        "",
    ]


def range_diagnostic_table(data, reference_model):
    """Show why factor B succeeds or fails without mixing every metric."""
    selected_regions = (
        ("large_disparity", "大视差"),
        ("large_disp_and_occluded", "大视差+遮挡"),
        ("boundary_and_occluded", "边界+遮挡"),
    )
    selected_models = (reference_model, "range", "oa_range", "range_hyp", "oa_full")
    # Preserve order and avoid printing range twice in the B-referenced report.
    selected_models = tuple(dict.fromkeys(selected_models))
    lines = [
        "| 区域 | 模型 | Abs | S2 coverage | S3 coverage | S2 width | S3 width |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region, label in selected_regions:
        reference = data[(reference_model, region)]
        for model in selected_models:
            row = data[(model, region)]
            cells = [label, MODEL_LABELS[model]]
            for metric, kind in (
                ("abs", "relative"),
                ("stage2_in_range", "pp"),
                ("stage3_in_range", "pp"),
                ("stage2_range_width", "relative"),
                ("stage3_range_width", "relative"),
            ):
                current = value(row, metric)
                ref = value(reference, metric)
                if model == reference_model:
                    cells.append(f"{current:.3f}（基准）")
                else:
                    cells.append(change_text(current, ref, kind, metric))
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def headline_table(all_data, reference_model):
    """Small first-page table containing the claims most useful in a paper."""
    lines = [
        "| 训练设置 | 模块/模型 | Full Abs | 大视差 Abs | 大视差+遮挡 Abs | 边界+遮挡 Abs |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    focus_models = ("oa", "range", "hyp", "oa_hyp", "oa_full")
    for experiment_name, _, _ in EXPERIMENTS:
        data = all_data[experiment_name]
        for model in focus_models:
            cells = [experiment_name, MODEL_LABELS[model]]
            for region in ("full", "large_disparity", "large_disp_and_occluded", "boundary_and_occluded"):
                current = value(data[(model, region)], "abs")
                ref = value(data[(reference_model, region)], "abs")
                cells.append(change_text(current, ref, "relative", "abs"))
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
        "表中粗体表示同一训练设置、同一区域、八组模型中的最优值。Abs 越低越好；Acc 与 coverage 越高越好。",
        "",
        "现有 `view3` checkpoint 是三视图训练，但 CSV 记录 `eval_nviews=5`、`region_nviews=5`，因此对应三视图训练、五视图测试。",
        "",
    ]


def paper_notes(reference_model):
    if reference_model == "vis":
        return [
            "## 论文写作提示",
            "",
            "- 主文优先报告 `vis`、`oa`、`range`、`hyp` 的单因素比较，再报告组合模型。",
            "- C 的核心证据是困难区域的 Abs 改善，并且 View3 中 Acc4 与 coverage 同步提高。",
            "- A 的结果应描述为阈值准确率和边界区域的补充收益，不能概括为所有遮挡区域都降低平均误差。",
            "- B 应作为失败消融或待改进模块分析；当前数据不支持将包含 B 的完整模型作为最终方法。",
            "- `view3` 必须写成三视图训练、五视图测试，除非重新以三个输入视图评测。",
            "- 单次训练结果不能用于宣称统计显著；正式论文应补多个随机种子和标准 DTU 点云指标。",
            "",
        ]
    return [
        "## 论文写作提示",
        "",
        "- 本报告是机制分析，B-only 应称为 reference、anchor 或出发点，不应称为原始 Vis-MVSNet baseline。",
        "- `range -> hyp` 的变化同时包含移除 B 与加入 C，不能全部归因于 C。",
        "- 判断 A/C 在保留 B 时的作用，应只使用 `oa_range vs range`、`range_hyp vs range` 和 `oa_full vs range`。",
        "- 正式方法提升仍应引用以 `vis` 为基线的报告。",
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
    all_data = {
        experiment_name: load_experiment(directory)
        for experiment_name, directory, _ in EXPERIMENTS
    }
    lines = report_intro(reference_model)
    lines.extend(["## 一页核心结果", ""])
    lines.extend(headline_table(all_data, reference_model))
    lines.extend(["", *paper_ready_text(all_data, reference_model), *paper_notes(reference_model)])

    for experiment_name, _, description in EXPERIMENTS:
        data = all_data[experiment_name]
        lines.extend([f"## {experiment_name}：{description}", "", "### Abs Error：七区域主结果", ""])
        lines.extend(abs_summary_table(data, reference_model))
        lines.extend([
            "",
            "### Acc4：论文目标区域摘要",
            "",
            "括号中为相对参照模型的百分点变化。",
            "",
        ])
        lines.extend(accuracy_summary_table(data, reference_model))
        lines.extend([
            "",
            "### 模块效应拆分（Abs Error）",
            "",
            "正值表示 Abs Error 降低，负值表示退化。只有每行给出的成对差分可以用于该行所述的模块归因。",
            "",
        ])
        lines.extend(effect_decomposition_table(data))
        lines.extend(["", "### B 与搜索范围诊断", ""])
        lines.extend(range_diagnostic_table(data, reference_model))
        lines.extend(["", "### 完整指标附录", ""])
        lines.append("以下折叠表保留八组模型的全部十项指标，供论文复核和补充材料使用。")
        lines.append("")
        for region, region_label in REGIONS:
            lines.extend(["<details>", f"<summary>{experiment_name} · {region_label} · 八组完整指标</summary>", ""])
            lines.extend(metric_table(data, region, reference_model))
            lines.extend(["", "</details>", ""])
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
