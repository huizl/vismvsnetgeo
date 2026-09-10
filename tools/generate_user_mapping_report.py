"""Read-only analysis of source CSVs under an explicitly hypothetical mapping."""

import csv
import math
from datetime import datetime

from generate_ablation_reports import ROOT, EXPERIMENTS, REGIONS, METRICS


CONFIGS = (
    ("range", "Base", "000", "010"),
    ("oa_range", "Base+A", "100", "110"),
    ("vis", "Base+B", "010", "000"),
    ("range_hyp", "Base+C", "001", "011"),
    ("oa", "Base+A+B", "110", "100"),
    ("oa_full", "Base+A+C", "101", "111"),
    ("hyp", "Base+B+C", "011", "001"),
    ("oa_hyp", "Base+A+B+C", "111", "101"),
)
LABELS = {m: label for m, label, _, _ in CONFIGS}
TARGETS = tuple((r, label) for r, label in REGIONS if r in (
    "full", "large_disparity", "large_disp_and_occluded", "boundary_and_occluded"))


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows], ""]


def read_series(directory):
    data, sources = {}, []
    expected_codes = {m: raw for m, _, _, raw in CONFIGS}
    for path in sorted((ROOT / "eval" / directory).glob("*/summary_metrics.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = [r for r in csv.DictReader(handle) if r["aggregation"] == "pixel_weighted"]
        if not records:
            raise ValueError(f"No pixel_weighted rows: {path}")
        models = {r["model_type"] for r in records}
        if len(models) != 1:
            raise ValueError(f"Mixed model identities: {path}")
        model = records[0]["model_type"]
        for row in records:
            key = (model, row["region"])
            if key in data:
                raise ValueError(f"Duplicate row: {key}")
            if row["ablation_code"] != expected_codes[model]:
                raise ValueError(f"Source ablation code changed: {path}")
            flags = ''.join(row[k] for k in (
                "factor_a_occlusion_supervision", "factor_b_adaptive_range", "factor_c_hypothesis_fusion"))
            if flags != expected_codes[model]:
                raise ValueError(f"Source factor flags changed: {path}")
            if (row["eval_nviews"], row["region_nviews"], row["light"]) != ("5", "5", "3"):
                raise ValueError(f"Evaluation protocol changed: {path}")
            for metric, _, _ in METRICS:
                v = float(row[metric])
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"Invalid metric: {key} {metric}")
                if (metric.startswith("acc") or metric.endswith("in_range")) and v > 1:
                    raise ValueError(f"Invalid probability: {key} {metric}")
            data[key] = row
        sources.append((model, path))
    expected = {(m, r) for m, _, _, _ in CONFIGS for r, _ in REGIONS}
    if set(data) != expected or len(sources) != 8:
        raise ValueError(f"Incomplete series: {directory}")
    for region, _ in REGIONS:
        for field in ("images", "pixels", "testlist"):
            if len({data[(m, region)][field] for m, _, _, _ in CONFIGS}) != 1:
                raise ValueError(f"Population/protocol mismatch: {directory} {region} {field}")
    return data, sources


def val(data, model, region, metric="abs"):
    return float(data[(model, region)][metric])


def delta(data, model, base, region, metric="abs"):
    current, reference = val(data, model, region, metric), val(data, base, region, metric)
    if metric == "abs":
        return (1 - current / reference) * 100
    if metric.startswith("acc") or metric.endswith("in_range"):
        return (current - reference) * 100
    return (current / reference - 1) * 100


def signed(number, places=2):
    number = round(number, places)
    return f"{0.0 if number == 0 else number:+.{places}f}"


def raw(number, kind):
    return f"{number * 100:.2f}%" if kind == "pp" else f"{number:.3f}"


def contrasts():
    by_code = {code: m for m, _, code, _ in CONFIGS}
    for i, factor in enumerate("ABC"):
        for base, _, code, _ in CONFIGS:
            if code[i] == "0":
                target_code = code[:i] + "1" + code[i + 1:]
                yield factor, base, by_code[target_code]


def narrative(data, series):
    lines = [f"### {series}：结论与数值依据", ""]
    for factor, model in (("A", "oa_range"), ("B", "vis"), ("C", "range_hyp")):
        gains = [delta(data, model, "range", r) for r, _ in TARGETS]
        lines.append(f"- 按指定映射，{factor} 单独加入 Base 后，Full、大视差、大视差+遮挡、边界+遮挡的 Abs 降幅依次为 "
                     + "、".join(f"{signed(g)}%" for g in gains) + "。")
    lines.extend(["", "完整模型与 B+C 的对照（加入 A）：", ""])
    for region, label in TARGETS:
        g = delta(data, "oa_hyp", "hyp", region)
        pp = delta(data, "oa_hyp", "hyp", region, "acc4")
        lines.append(f"- {label}：Abs 降幅 {signed(g)}%；Acc4 变化 {signed(pp)} pp。")
    lines.extend(["", "这些是已观察到的指标变化。Abs 与 Acc4 方向相反时，应表述为指标取舍；"
                  "误差尾部、梯度竞争或监督冲突均需额外实验，不能由汇总指标直接确定。", ""])
    return lines


def main():
    lines = [
        "# 以 range 为 Base 的八组条件分析：用户指定映射", "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}。数据来自本地既有测试 CSV，未重新训练或评测。", "",
        "## 分析前提与来源", "",
        "> 本文按用户指定的假设映射讨论：A 为遮挡监督，B 称为“开启自适应深度范围”，C 为假设级融合。"
        "该映射把原始 CSV 的 B 位取反，与 CSV 的运行开关记录不同。因此，文中的模块归因是条件分析，"
        "不能据此声称已验证“开启自适应范围”造成这些提升。数值来自真实 CSV；配置名称来自分析设定。", "",
        "训练视图数按目录名称与用户说明标识；CSV 的 eval_nviews=5、region_nviews=5 表示两套结果均为五视图评测。"
        "统一使用 test、light 3、pixel_weighted。原始 vis 的身份保留在来源映射中；Base 是本文内部参照。", "",
    ]
    lines += table(["原始 model_type", "分析配置", "指定 ABC 编码", "CSV 原始 ABC 编码"],
                   [(f"`{m}`", label, code, original) for m, label, code, original in CONFIGS])
    lines += ["## 计算与阅读方式", "",
              "- Abs 降幅 = (参照值 − 当前值) / 参照值 × 100%；正值改善，负值退化。",
              "- Acc2/4/8 与 coverage 原始比例转成百分数展示；变化 = (当前比例 − 参照比例) × 100，单位 pp（百分点）。",
              "- Width 变化 = (当前值 / 参照值 − 1) × 100%；宽度无统一优劣方向，应结合 coverage 与 Abs。",
              "- 所有变化先按 CSV 完整精度计算再舍入。显示 0.00 不代表数学上完全相同。",
              "- Abs 主表括号内为相对 range 的降幅。粗体为同一训练设置、同一区域的最低 Abs。",
              "- 各区域可能重叠，不能将七区域当作互斥分组相加或直接平均。", ""]
    for series, directory, description in EXPERIMENTS:
        data, sources = read_series(directory)
        lines += [f"## {series}：{description}", "", "### 八组 Abs 主表", ""]
        rows = []
        for model, label, _, _ in CONFIGS:
            cells = [f"{label} / `{model}`"]
            for region, _ in REGIONS:
                v = val(data, model, region)
                cell = f"{v:.3f}（{signed(delta(data, model, 'range', region))}%）"
                if v == min(val(data, m, region) for m, _, _, _ in CONFIGS):
                    cell = f"**{cell}**"
                cells.append(cell)
            rows.append(cells)
        lines += table(["配置 / 原始结果标识"] + [s for _, s in REGIONS], rows)
        lines += ["### 十二组受控比较", "", "每行仅改变指定映射中的一个因素，参照是该行起始配置。", ""]
        for metric, title, unit in (("abs", "Abs 降幅", "%"), ("acc4", "Acc4 变化", "pp")):
            lines += [f"#### {title}", ""]
            rows = []
            for factor, base, target in contrasts():
                rows.append([f"加入 {factor}", f"{LABELS[base]} → {LABELS[target]}", f"`{base} → {target}`"] +
                            [f"{signed(delta(data, target, base, region, metric))} {unit}" for region, _ in TARGETS])
            lines += table(["变化", "配置对照", "原始标识对照"] + [s for _, s in TARGETS], rows)
        lines += narrative(data, series)
        lines += ["### 全部十项指标：原始值与相对 Base 变化", "",
                  "每个折叠块包含八组原始值和八组变化值。Abs 保留 CSV 单位，不额外推定物理单位。", ""]
        for region, region_label in REGIONS:
            population = data[("range", region)]
            lines += ["<details>", f"<summary>{series} · {region_label} · 八组十项指标</summary>", "",
                      f"评测记录数 images={population['images']}，统计像素数 pixels={population['pixels']}。", "",
                      "原始值：", ""]
            rows = [[f"{label} / `{m}`"] + [raw(val(data, m, region, metric), kind) for metric, _, kind in METRICS]
                    for m, label, _, _ in CONFIGS]
            lines += table(["配置 / 原始标识"] + [label for _, label, _ in METRICS], rows)
            lines += ["相对 Base `range` 的变化：", ""]
            rows = [[label] + [f"{signed(delta(data, m, 'range', region, metric))} {'pp' if kind == 'pp' else '%'}"
                              for metric, _, kind in METRICS] for m, label, _, _ in CONFIGS]
            lines += table(["配置"] + ["Abs 降幅" if metric == "abs" else label + " 变化" for metric, label, _ in METRICS], rows)
            lines += ["</details>", ""]
        lines += ["### 原始数据索引", ""]
        lines += table(["原始标识", "CSV", "CSV checkpoint 记录"],
                       [(m, f"[summary_metrics.csv]({path.as_posix()})", f"`{data[(m, 'full')]['checkpoint']}`") for m, path in sources])
    lines += [
        "## 论文论证能支持到哪里", "",
        "1. 八种指定配置齐全，可讨论完整的三因素对照。按该映射，三个因素分别加入内部 Base 时，两套训练设置的 Full Abs 都降低。",
        "2. 完整配置相对内部 Base 有改善，但核心区域 Abs 未优于 B+C。不能写成“三模块叠加获得最优平均误差”。",
        "3. 三视图训练时，加入 A 后 Full、大视差、大视差+遮挡 Acc4 提高，而 Abs 增大；边界+遮挡两项均退化。A 的作用需要按区域和指标限定。",
        "4. 各配置相对 Base 的改善率不能直接相加，也不能作为协同作用的证明。判断必要性优先看完整模型与三个两因素配置的受控对照。",
        "5. 机制解释可提出假设，但 coverage 的变化不足以单独证明因果机制。需要逐场景误差、分布和搜索诊断作支撑。",
        "6. 如后续以论文最终模型做结论，应先固定主要指标，再用验证集决定配置；汇总单次结果不提供统计显著性证据。", "",
        "### 三视图训练的逐步加入路径与交叉检查", "",
        "Base → Base+A → Base+A+B → Base+A+B+C 的 Full Abs 依次为 "
        "7.634 → 7.427 → 6.319 → 6.233，大视差 Abs 为 22.940 → 22.473 → 16.850 → 16.549。"
        "这条路径支持按指定顺序加入模块能逐步降低这些误差。",
        "",
        "同一套实验的 Base+B+C 更低（Full 5.919，大视差 15.499），说明上述路径不能单独证明三个模块都必要。"
        "应同时展示完整八组表与去掉 A 的对照。五视图训练中，A+B → 完整配置在大视差和大视差+遮挡区域也并非持续改善。", "",
        "## 结果段落草稿（仅在指定映射前提下）", "",
        "以 range 对应结果为内部 Base，在两种训练视图设置下，分别加入 A、B、C 均降低了 Full 区域平均绝对误差。"
        "完整配置也优于内部 Base，但 B+C 在 Full、大视差及大视差与遮挡重叠区域具有更低的平均绝对误差。"
        "三视图训练时，A 加入 B+C 后提高了部分区域的 Acc4，同时增加平均绝对误差，体现出阈值准确率与误差幅度之间的取舍。"
        "因此，本组实验支持对模块条件效应与交互关系的讨论，尚不支持完整配置在所有目标指标上占优的结论。", "",
        "用于正式方法归因前，应使配置定义与运行记录一致。本文未改写源 CSV 的开关、模型标识或数值。", "",
    ]
    output = ROOT / "docs" / "USER_MAPPING_ABLATION_ANALYSIS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output}")
    print("Validated 16 CSV files, 112 region rows, 1120 raw metric values; 24 single-factor contrasts.")


if __name__ == "__main__":
    main()
