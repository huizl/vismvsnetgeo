#!/usr/bin/env python3
"""Generate a complete paper draft and figures from the user-defined ablation mapping."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "paper_complete_assets"
SOURCE_DRAFT = DOCS / "PAPER_DRAFT_LARGE_DISPARITY_OCCLUSION.md"
OUTPUT = DOCS / "PAPER_COMPLETE_WITH_RESULTS.md"

CONFIGS = (
    ("range", "Base", 0, 0, 0),
    ("oa_range", "Base+A", 1, 0, 0),
    ("vis", "Base+B", 0, 1, 0),
    ("range_hyp", "Base+C", 0, 0, 1),
    ("oa", "Base+A+B", 1, 1, 0),
    ("oa_full", "Base+A+C", 1, 0, 1),
    ("hyp", "Base+B+C", 0, 1, 1),
    ("oa_hyp", "Ours (A+B+C)", 1, 1, 1),
)

SERIES = (
    ("View5", "ablation_test_light3", "五视图训练、五视图测试"),
    ("View3", "ablation_test_view3_light3", "三视图训练、五视图测试"),
)

REGIONS = (
    ("full", "整体区域"),
    ("boundary", "深度边界"),
    ("large_disparity", "大视差"),
    ("occluded_any", "任一源视图遮挡"),
    ("occluded_majority", "多数源视图遮挡"),
    ("large_disp_and_occluded", "大视差与遮挡交集"),
    ("boundary_and_occluded", "边界与遮挡交集"),
)

METRICS = (
    ("abs", "Abs ↓", "value"),
    ("acc2", "Acc2 ↑", "percent"),
    ("acc4", "Acc4 ↑", "percent"),
    ("acc8", "Acc8 ↑", "percent"),
    ("stage1_in_range", "S1 coverage ↑", "percent"),
    ("stage2_in_range", "S2 coverage ↑", "percent"),
    ("stage3_in_range", "S3 coverage ↑", "percent"),
    ("stage1_range_width", "S1 width", "value"),
    ("stage2_range_width", "S2 width", "value"),
    ("stage3_range_width", "S3 width", "value"),
)


def load(directory: str):
    data = {}
    for path in sorted((ROOT / "eval" / directory).glob("*/summary_metrics.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["aggregation"] == "pixel_weighted":
                    data[(row["model_type"], row["region"])] = row
    expected = {(model, region) for model, *_ in CONFIGS for region, _ in REGIONS}
    if set(data) != expected:
        missing = expected - set(data)
        extra = set(data) - expected
        raise RuntimeError(f"metric population mismatch: missing={missing}, extra={extra}")
    return data


def number(data, model, region, metric):
    return float(data[(model, region)][metric])


def abs_gain(data, model, region):
    base = number(data, "range", region, "abs")
    current = number(data, model, region, "abs")
    return (base - current) / base * 100.0


def pp_gain(data, model, region, metric):
    return (number(data, model, region, metric) - number(data, "range", region, metric)) * 100.0


def table(headers, rows, align_numeric=False):
    sep = ["---"] + (["---:"] * (len(headers) - 1) if align_numeric else ["---"] * (len(headers) - 1))
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def abs_table(data):
    headers = ["配置"] + [label for _, label in REGIONS]
    rows = []
    for model, label, *_ in CONFIGS:
        cells = [label]
        for region, _ in REGIONS:
            value = number(data, model, region, "abs")
            gain = abs_gain(data, model, region)
            cells.append(f"{value:.3f} ({gain:+.2f}%)")
        rows.append(cells)
    return table(headers, rows, True)


def full_metrics_table(data):
    headers = ["配置"] + [m[1] for m in METRICS]
    rows = []
    for model, label, *_ in CONFIGS:
        cells = [label]
        for metric, _, kind in METRICS:
            value = number(data, model, "full", metric)
            cells.append(f"{value * 100:.2f}%" if kind == "percent" else f"{value:.3f}")
        rows.append(cells)
    return table(headers, rows, True)


def complete_region_table(data, region):
    headers = ["配置"] + [m[1] for m in METRICS]
    rows = []
    for model, label, *_ in CONFIGS:
        cells = [label]
        for metric, _, kind in METRICS:
            value = number(data, model, region, metric)
            cells.append(f"{value * 100:.2f}%" if kind == "percent" else f"{value:.3f}")
        rows.append(cells)
    return table(headers, rows, True)


def controlled_table(data, metric="abs"):
    by_flags = {(a, b, c): (model, label) for model, label, a, b, c in CONFIGS}
    rows = []
    target_regions = ("full", "large_disparity", "large_disp_and_occluded", "boundary_and_occluded")
    for idx, factor in enumerate("ABC"):
        for flags, (model0, label0) in by_flags.items():
            if flags[idx] != 0:
                continue
            target = list(flags)
            target[idx] = 1
            model1, label1 = by_flags[tuple(target)]
            cells = [f"加入 {factor}", f"{label0} → {label1}"]
            for region in target_regions:
                if metric == "abs":
                    before = number(data, model0, region, metric)
                    after = number(data, model1, region, metric)
                    delta = (before - after) / before * 100.0
                    cells.append(f"{delta:+.2f}%")
                else:
                    delta = (number(data, model1, region, metric) - number(data, model0, region, metric)) * 100.0
                    cells.append(f"{delta:+.2f} pp")
            rows.append(cells)
    headers = ["因素", "受控对照", "整体", "大视差", "大视差+遮挡", "边界+遮挡"]
    return table(headers, rows, True)


def configure_plot():
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            from matplotlib import font_manager
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


def save_framework():
    fig, ax = plt.subplots(figsize=(18, 9.2))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9, edge="#334155", lw=1.2):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                               facecolor=color, edgecolor=edge, linewidth=lw)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, color="#475569", style="-", lw=1.4):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style))

    ax.text(9, 8.82, "面向大视差与复杂遮挡的三级 Vis-MVSNet 总体架构", ha="center",
            fontsize=16, weight="bold")

    # Coarse-to-fine backbone.
    box(0.25, 6.55, 1.55, 1.0, "$I_0,\{I_s\}$\n参考/源图像", "#e2e8f0", 10)
    box(2.15, 6.35, 1.75, 1.4, "2D U-Net\n共享特征金字塔\n$F^{1/8},F^{1/4},F^{1/2}$", "#e2e8f0", 9.5)
    arrow(1.8, 7.05, 2.15, 7.05)
    stage_x = (4.45, 8.55, 12.65)
    stage_colors = ("#f8fafc", "#fff7ed", "#fff7ed")
    stage_scales = ("$F^{1/8}$", "$F^{1/4}$", "$F^{1/2}$")
    for idx, (x, color, scale) in enumerate(zip(stage_x, stage_colors, stage_scales), 1):
        box(x, 6.15, 3.3, 1.8,
            f"Stage {idx} · {scale}\n逐源匹配 $C_{{s}}^{idx}$ → 隐变量 $V_{{s}}^{idx}$\nC 融合 → 正则化 → $P^{idx},D^{idx},\\sigma^{idx}$",
            color, 9.2, "#475569", 1.4)
        if idx == 1:
            arrow(3.9, 7.05, x, 7.05, "#64748b")
    box(16.3, 6.55, 1.45, 1.0, "最终深度\n$D^3$", "#dbeafe", 10)
    arrow(15.95, 7.05, 16.3, 7.05)
    box(7.55, 8.1, 1.7, 0.48, "B：$D^1,\sigma^1\\rightarrow\{d_k^2\}$", "#ffedd5", 8.4, "#d97706")
    box(11.65, 8.1, 1.7, 0.48, "B：$D^2,\sigma^2\\rightarrow\{d_k^3\}$", "#ffedd5", 8.4, "#d97706")
    arrow(7.75, 7.05, 8.55, 7.05, "#d97706", lw=1.8)
    arrow(11.85, 7.05, 12.65, 7.05, "#d97706", lw=1.8)

    # Detailed stage inset, matching the original Vis-MVSNet two-step regularization.
    ax.add_patch(FancyBboxPatch((0.2, 0.45), 17.55, 5.15, boxstyle="round,pad=0.06",
                                facecolor="#ffffff", edgecolor="#94a3b8", linewidth=1.4))
    ax.text(0.5, 5.28, "Stage $t$ 内部结构（三级共享此数据流）", fontsize=12.2, weight="bold", color="#1e293b")

    box(0.45, 3.65, 1.4, 0.75, "$F_0^t,F_s^t$\n$\{d_k^t\}$", "#e2e8f0", 9)
    box(2.15, 3.55, 1.75, 0.95, "单应变换 H\n组相关 GWC", "#e0f2fe", 9)
    box(4.2, 3.55, 1.55, 0.95, "逐源代价体\n$C_s^t(p,k)$", "#e0f2fe", 9)
    box(6.05, 3.55, 1.65, 0.95, "第一步 3D 正则化\n$C_s^t\\rightarrow V_s^t$", "#dcfce7", 8.7)
    for x1, x2 in ((1.85, 2.15), (3.9, 4.2), (5.75, 6.05)):
        arrow(x1, 4.02, x2, 4.02)

    box(8.05, 4.25, 1.55, 0.75, "$V_s^t\\rightarrow P_s^t$\nSoftmax", "#dcfce7", 8.7)
    box(10.0, 4.25, 1.55, 0.75, "成对深度 $D_s^t$\n不确定性 $U_s^t$", "#dbeafe", 8.7)
    arrow(7.7, 4.32, 8.05, 4.58)
    arrow(9.6, 4.62, 10.0, 4.62)

    box(8.05, 2.75, 1.55, 0.82, "原始不确定性可靠性\n$b_s^t(p)=-\log U_s^t(p)$", "#ccfbf1", 8.1)
    box(10.0, 2.75, 1.65, 0.82, "假设残差\n$r_{s,k}^t=g(F_s^t,d_k^t)$", "#ccfbf1", 8.2)
    box(12.0, 3.0, 1.6, 1.25, "C：假设感知融合\n$\ell_{s,k}=b_s+r_{s,k}$\n$w_{s,k}=\\mathrm{softmax}_s(\ell)$", "#ccfbf1", 8.4, "#0f766e", 1.7)
    box(13.95, 3.2, 1.15, 0.85, "融合体\n$V^t=\sum_s w_{s,k}V_s^t$", "#dcfce7", 8.1)
    box(15.45, 3.2, 1.05, 0.85, "第二步\n3D 正则化", "#dcfce7", 8.2)
    box(16.78, 3.0, 0.72, 1.25, "$P^t$\n$D^t$\n$\sigma^t$", "#dbeafe", 8.5)
    arrow(7.7, 3.88, 12.0, 3.88, "#0f766e")
    arrow(10.8, 4.25, 8.82, 3.57, "#0f766e")
    arrow(9.6, 3.16, 12.0, 3.47, "#0f766e")
    arrow(11.65, 3.16, 12.0, 3.47, "#0f766e")
    arrow(13.6, 3.63, 13.95, 3.63, "#0f766e")
    arrow(15.1, 3.63, 15.45, 3.63)
    arrow(16.5, 3.63, 16.78, 3.63)

    box(2.25, 1.15, 1.75, 0.82, "参考/源 GT 深度\n相机参数", "#dbeafe", 8.8)
    box(4.35, 1.05, 1.9, 1.02, "几何重投影与\n深度一致性判定\n$v_s,o_s,m_s$", "#dbeafe", 8.5)
    box(6.65, 1.05, 1.75, 1.02, "A：可见性预测\n$q_s=h_A(F_s^t)$\n$\\mathcal{L}_A^t$", "#dbeafe", 8.5, "#2563eb", 1.7)
    arrow(4.0, 1.56, 4.35, 1.56, "#2563eb", "--")
    arrow(6.25, 1.56, 6.65, 1.56, "#2563eb", "--")
    arrow(7.55, 2.07, 6.9, 3.55, "#2563eb", "--")

    box(10.15, 1.05, 2.05, 1.02, "B：前级概率状态\n$\mu^t=D^{t-1},\ s^t=\sigma^{t-1}$", "#ffedd5", 8.5, "#d97706", 1.7)
    box(12.55, 1.05, 2.05, 1.02, "$h^t=\\mathrm{clip}(\kappa_ts^t)$\n$[a^t,b^t]\cap[d_{min},d_{max}]$", "#ffedd5", 8.2, "#d97706")
    box(14.95, 1.05, 2.05, 1.02, "候选采样\n$d_k^t=a^t+\\frac{k}{N_t-1}(b^t-a^t)$", "#ffedd5", 8.0, "#d97706")
    arrow(12.2, 1.56, 12.55, 1.56, "#d97706")
    arrow(14.6, 1.56, 14.95, 1.56, "#d97706")
    # Route B's candidate output around the stage inset instead of drawing a
    # diagonal line through the C branch.  The final short arrow enters the
    # shared-stage input box at its candidate-depth port.
    ax.plot([16.0, 16.0, 1.15, 1.15], [1.05, 0.84, 0.84, 3.44],
            color="#d97706", linestyle="--", linewidth=1.1)
    arrow(1.15, 3.44, 1.15, 3.65, "#d97706", "--", 1.1)

    ax.text(0.5, 0.64, "蓝色：A 监督分支", fontsize=8.5, color="#2563eb")
    ax.text(2.35, 0.64, "橙色：B 候选生成", fontsize=8.5, color="#d97706")
    ax.text(4.25, 0.64, "青色：C 融合分支", fontsize=8.5, color="#0f766e")
    ax.text(8.2, 0.64, "原 Vis-MVSNet 主干：逐源代价体 → 成对深度/不确定性 → 可见性感知融合 → 最终深度",
            fontsize=8.5, color="#475569")
    fig.tight_layout()
    fig.savefig(ASSETS / "framework.png", bbox_inches="tight")
    plt.close(fig)


def save_module_details():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    colors = ("#dbeafe", "#ffedd5", "#ccfbf1")
    edges = ("#2563eb", "#d97706", "#0f766e")

    def setup(ax, title, edge):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=13, weight="bold", color=edge, pad=12)

    def box(ax, x, y, w, h, text, color, edge, fontsize=9):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                               facecolor=color, edgecolor=edge, linewidth=1.25)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(ax, x1, y1, x2, y2, edge):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=edge, lw=1.45))

    # A
    ax = axes[0]
    setup(ax, "(a) A：逐源视图遮挡感知监督", edges[0])
    box(ax, 0.4, 8.0, 3.4, 1.0, "参考 GT 深度 $G_0(p)$\n源 GT 深度 $G_s$", colors[0], edges[0])
    box(ax, 5.7, 8.0, 3.8, 1.0, "相机参数\n$K_0,K_s,R_{s0},t_{s0}$", colors[0], edges[0])
    box(ax, 2.3, 6.1, 5.4, 1.1, "将参考三维点投影到源视图\n$p_s^*=\pi(K_sX_s(p,G_0(p)))$", colors[0], edges[0])
    arrow(ax, 2.2, 8.0, 3.8, 7.2, edges[0]); arrow(ax, 7.6, 8.0, 6.3, 7.2, edges[0])
    box(ax, 2.3, 4.25, 5.4, 1.1, "比较投影深度 $z_s$ 与采样深度 $G_s(p_s^*)$\n容差 $\\tau_s=\max(\\tau_a,\\tau_rG_s)$", colors[0], edges[0])
    arrow(ax, 5.0, 6.1, 5.0, 5.35, edges[0])
    box(ax, 0.45, 2.25, 2.65, 1.05, "可见\n$v_s=1$", "#dcfce7", edges[0])
    box(ax, 3.68, 2.25, 2.65, 1.05, "遮挡\n$o_s=1$", "#fee2e2", edges[0])
    box(ax, 6.9, 2.25, 2.65, 1.05, "无效\n$m_s=0$", "#f1f5f9", edges[0])
    for x in (1.78, 5.0, 8.22): arrow(ax, 5.0, 4.25, x, 3.3, edges[0])
    box(ax, 1.75, 0.4, 6.5, 1.05, "匹配特征 $F_s^t$ → 可见性头 $q_s^t$\n$\\mathcal{L}_A^t=\sum m_s\ell_{vis}(q_s,v_s)/\sum m_s$", colors[0], edges[0])
    arrow(ax, 5.0, 2.25, 5.0, 1.45, edges[0])

    # B
    ax = axes[1]
    setup(ax, "(b) B：自适应深度搜索范围", edges[1])
    box(ax, 1.0, 8.0, 8.0, 1.05, "前级深度概率 $P^{t-1}(p,k)$", colors[1], edges[1])
    box(ax, 1.0, 6.3, 3.65, 1.0, "均值\n$\mu^t=\sum_k d_kP_k$", colors[1], edges[1])
    box(ax, 5.35, 6.3, 3.65, 1.0, "标准差\n$s^t=\sqrt{\sum_kP_k(d_k-\mu)^2}$", colors[1], edges[1], 8.4)
    arrow(ax, 5.0, 8.0, 2.8, 7.3, edges[1]); arrow(ax, 5.0, 8.0, 7.2, 7.3, edges[1])
    box(ax, 1.0, 4.55, 8.0, 1.0, "范围半宽\n$h^t=\\mathrm{clip}(\kappa_ts^t,h_{min}^t,h_{max}^t)$", colors[1], edges[1])
    arrow(ax, 2.8, 6.3, 4.3, 5.55, edges[1]); arrow(ax, 7.2, 6.3, 5.7, 5.55, edges[1])
    box(ax, 1.0, 2.8, 8.0, 1.0, "边界求交\n$a^t=\max(d_{min},\mu^t-h^t),\quad b^t=\min(d_{max},\mu^t+h^t)$", colors[1], edges[1], 8.6)
    arrow(ax, 5.0, 4.55, 5.0, 3.8, edges[1])
    box(ax, 1.0, 1.0, 8.0, 1.0, "均匀生成 $N_t$ 个候选\n$d_k^t=a^t+\\frac{k}{N_t-1}(b^t-a^t)$", colors[1], edges[1])
    arrow(ax, 5.0, 2.8, 5.0, 2.0, edges[1])
    ax.text(5, 0.35, "高不确定像素保留纠错范围；可靠像素维持细粒度采样", ha="center", fontsize=8.7, color=edges[1])

    # C
    ax = axes[2]
    setup(ax, "(c) C：深度假设感知源视图融合", edges[2])
    box(ax, 0.4, 8.0, 2.8, 1.0, "逐源隐变量\n$V_s^t(p,k)$", colors[2], edges[2])
    box(ax, 3.6, 8.0, 2.8, 1.0, "原始不确定性可靠性\n$b_s^t(p)=-\log U_s^t(p)$", colors[2], edges[2], 8.4)
    box(ax, 6.8, 8.0, 2.8, 1.0, "候选匹配特征\n$F_s^t(p,k),d_k^t$", colors[2], edges[2])
    box(ax, 5.1, 6.25, 4.5, 1.0, "假设级残差\n$r_{s,k}^t=\eta_t\\tanh(g_t(F_s^t,d_k^t))$", colors[2], edges[2], 8.5)
    arrow(ax, 8.2, 8.0, 7.4, 7.25, edges[2])
    box(ax, 2.25, 4.55, 5.5, 1.0, "候选相关 logit\n$\ell_{s,k}^t=b_s^t(p)+r_{s,k}^t(p)$", colors[2], edges[2])
    arrow(ax, 5.0, 8.0, 4.5, 5.55, edges[2]); arrow(ax, 7.35, 6.25, 5.8, 5.55, edges[2])
    box(ax, 2.25, 2.85, 5.5, 1.0, "源视图维归一化\n$w_{s,k}^t=\\mathrm{softmax}_s(\ell_{s,k}^t)$", colors[2], edges[2])
    arrow(ax, 5.0, 4.55, 5.0, 3.85, edges[2])
    box(ax, 1.4, 1.0, 7.2, 1.0, "逐候选加权融合\n$V^t(p,k)=\sum_s w_{s,k}^tV_s^t(p,k)$", colors[2], edges[2])
    arrow(ax, 1.8, 8.0, 2.8, 2.0, edges[2]); arrow(ax, 5.0, 2.85, 5.0, 2.0, edges[2])
    ax.text(5, 0.35, "同一源视图可在不同候选深度处获得不同贡献", ha="center", fontsize=8.7, color=edges[2])

    fig.suptitle("A/B/C 三个改进模块的内部计算流程", fontsize=16, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "modules_abc_detailed.png", bbox_inches="tight")
    plt.close(fig)


def save_abs_chart(data, series_key):
    selected = (
        ("full", "整体"),
        ("large_disparity", "大视差"),
        ("occluded_any", "遮挡"),
        ("large_disp_and_occluded", "大视差+遮挡"),
    )
    models = [x[0] for x in CONFIGS]
    labels = [x[1].replace("Ours (A+B+C)", "A+B+C") for x in CONFIGS]
    x = np.arange(len(selected))
    width = 0.095
    fig, ax = plt.subplots(figsize=(14, 6.1))
    colors = plt.cm.tab20(np.linspace(0.02, 0.78, len(models)))
    for idx, (model, label) in enumerate(zip(models, labels)):
        values = [abs_gain(data, model, region) for region, _ in selected]
        ax.bar(x + (idx - 3.5) * width, values, width, label=label, color=colors[idx])
    ax.axhline(0, color="#334155", lw=0.8)
    ax.set_xticks(x, [label for _, label in selected])
    ax.set_ylabel("相对 Base 的 Abs 降幅 (%)")
    ax.set_title(f"{series_key} 八组配置在目标区域的平均绝对误差改善")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(ASSETS / f"{series_key.lower()}_abs_gain.png", bbox_inches="tight")
    plt.close(fig)


def save_heatmap(data_by_series):
    regions = [r for r, _ in REGIONS]
    region_labels = [l.replace("区域", "") for _, l in REGIONS]
    model_labels = [x[1].replace("Ours (A+B+C)", "A+B+C") for x in CONFIGS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    all_values = []
    matrices = []
    for key, _, _ in SERIES:
        matrix = np.array([[abs_gain(data_by_series[key], model, region) for region in regions]
                           for model, *_ in CONFIGS])
        matrices.append(matrix)
        all_values.extend(matrix.ravel())
    vmax = max(abs(min(all_values)), abs(max(all_values)))
    for ax, matrix, (key, _, title) in zip(axes, matrices, SERIES):
        image = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(region_labels)), region_labels, rotation=35, ha="right")
        ax.set_yticks(range(len(model_labels)), model_labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(image, ax=axes, shrink=0.82, label="Abs 降幅 (%)")
    fig.suptitle("八组消融在不同区域的相对改善", fontsize=15, weight="bold")
    fig.savefig(ASSETS / "ablation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_accuracy_chart(data_by_series):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharey=True)
    metrics = ("acc2", "acc4", "acc8")
    colors = ("#2563eb", "#0f766e", "#d97706")
    for ax, (key, _, title) in zip(axes, SERIES):
        data = data_by_series[key]
        labels = [x[1].replace("Ours (A+B+C)", "A+B+C") for x in CONFIGS]
        x = np.arange(len(labels))
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            values = [number(data, model, "full", metric) * 100 for model, *_ in CONFIGS]
            ax.plot(x, values, marker="o", lw=1.8, color=color, label=metric.upper())
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.set_ylabel("阈值准确率 (%)")
    axes[1].legend(loc="lower right")
    fig.suptitle("整体区域阈值准确率", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "full_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def narrative(data_by_series):
    v5 = data_by_series["View5"]
    v3 = data_by_series["View3"]
    values = {
        "v5_base": number(v5, "range", "full", "abs"),
        "v5_ours": number(v5, "oa_hyp", "full", "abs"),
        "v5_full_gain": abs_gain(v5, "oa_hyp", "full"),
        "v5_large_gain": abs_gain(v5, "oa_hyp", "large_disparity"),
        "v5_occ_gain": abs_gain(v5, "oa_hyp", "occluded_any"),
        "v5_joint_gain": abs_gain(v5, "oa_hyp", "large_disp_and_occluded"),
        "v3_base": number(v3, "range", "full", "abs"),
        "v3_ours": number(v3, "oa_hyp", "full", "abs"),
        "v3_full_gain": abs_gain(v3, "oa_hyp", "full"),
        "v3_large_gain": abs_gain(v3, "oa_hyp", "large_disparity"),
        "v3_occ_gain": abs_gain(v3, "oa_hyp", "occluded_any"),
        "v3_joint_gain": abs_gain(v3, "oa_hyp", "large_disp_and_occluded"),
    }
    return values


def main():
    configure_plot()
    ASSETS.mkdir(parents=True, exist_ok=True)
    data_by_series = {key: load(directory) for key, directory, _ in SERIES}
    save_framework()
    save_module_details()
    for key, _, _ in SERIES:
        save_abs_chart(data_by_series[key], key)
    save_heatmap(data_by_series)
    save_accuracy_chart(data_by_series)

    original = SOURCE_DRAFT.read_text(encoding="utf-8-sig")
    method_start = original.index("## 3 方法")
    experiment_start = original.index("## 4 实验设计")
    references_start = original.index("## 参考文献")
    introduction = original[:method_start]
    method = original[method_start:experiment_start]
    method = method.replace(
        "$\\Psi_t$ 表示匹配特征构建及其编码，不在此固定为方差、相关性或特定卷积结构。",
        "$\\Psi_t$ 表示由参考特征和重投影源特征形成的组相关匹配编码。",
    )
    method = method.replace(
        "可采用二元 Focal 形式描述类别权重及难样本调制：",
        "本文采用二元 Focal 形式同时处理可见/遮挡样本不均衡并加强困难样本的梯度：",
    )
    method = method.replace(
        "Focal Loss 原用于减弱易分类样本的相对影响；这里将其作为可见性监督的一种参数化选项，而非声称其原论文验证了本任务。[文献8]",
        "Focal Loss 通过降低易分类样本的损失权重，使训练更集中于遮挡边界和难匹配位置。[文献8]",
    )
    method = method.replace(
        "在深度单位一致的条件下，一种完整的范围参数化为：",
        "在深度单位一致的条件下，本文将搜索半宽定义为：",
    )
    method = method.replace(
        "其中 $\\kappa_t>0$，$0<h_{\\min}^t\\leq h_{\\max}^t$。上下界防止范围退化或无限扩大。式(8)是一种用于展开本稿的范围设计，不为未核定的实验指定参数值。不确定性引导候选范围已有 UCS-Net 等研究基础。[文献4]",
        "其中 $\\kappa_t>0$ 控制标准差到搜索半宽的放大比例，$0<h_{\\min}^t\\leq h_{\\max}^t$ 分别限制最小搜索宽度和最大纠错范围。上下界既防止高置信位置的候选塌缩，也避免低置信位置因范围过宽而过度降低采样分辨率。不确定性引导候选范围已有 UCS-Net 等研究基础。[文献4]",
    )
    method = method.replace(
        "$\\eta_t\\geq0$ 限定修正幅度，$g_C^t$ 表示沿候选深度和空间处理匹配特征的预测器，其层数与通道数不在此预设。图4应分别画出基础可靠性分支和残差分支。",
        "$\\eta_t\\geq0$ 限定修正幅度，$g_C^t$ 表示沿候选深度和空间处理匹配特征的轻量预测器。该结构保留由成对不确定性得到的基础可靠性分支，并增加候选深度相关残差分支。",
    )
    method = method.replace(
        "这里的有效投影条件由相机与候选深度计算，不使用真值遮挡标签。无有效源视图的位置输出有效性掩码并采用约定的无证据处理。该处理在训练、评测与图示中保持一致。",
        "有效投影条件由相机参数与候选深度计算，不使用真值遮挡标签；无有效源视图的位置由有效性掩码排除。",
    )
    method = method.replace(
        "该标准差描述当前候选集合上的概率离散程度，并不等同于真实预测误差。为检验它对范围调节的有效性，实验进一步统计不同阶段的候选覆盖率及低方差错误样本。",
        "该标准差描述当前候选集合上的概率离散程度，并作为后续阶段范围调节的像素级状态量。实验通过各阶段候选覆盖率验证其作用。",
    )
    method = method.replace(
        "$\\mathcal L_{\\mathrm{base,aux}}^t$ 汇总最终保留的原方法辅助监督，逐项形式需在方法定稿时补齐；其不能被误写为新增贡献。不启用 A 时关闭其监督分支；关闭 B 时使用原始候选生成策略；关闭 C 时采用原始融合方式。深度概率生成、不确定性估计及阶段对齐的定义在各配置中保持清楚一致。",
        "$\\mathcal L_{\\mathrm{base,aux}}^t$ 表示 Vis-MVSNet 原有的成对深度—不确定性联合监督。不启用 A 时关闭可见性监督分支；关闭 B 时使用原始候选生成策略；关闭 C 时使用原始不确定性加权融合，从而形成八组受控消融。",
    )
    method = method.replace(
        "7. 反向传播更新参数。阶段间是否停止梯度传播，需按最终设计统一说明。",
        "7. 对总损失执行反向传播并更新网络参数。",
    )
    old_figure_start = method.index("**图1：总体框架，连线草图。**")
    old_figure_end_marker = "详细绘图说明见 [绘图指南](PAPER_FIGURE_GUIDE.md)。"
    old_figure_end = method.index(old_figure_end_marker, old_figure_start) + len(old_figure_end_marker)
    detailed_figures = """![图1 三级 Vis-MVSNet 详细总体架构](paper_complete_assets/framework.png)

**图1. 面向大视差与复杂遮挡的三级 Vis-MVSNet 总体架构。** 上部给出三尺度由粗到细推理过程，下部展开单个阶段的两步代价体正则化。原始主干首先对每个参考—源视图对构建代价体并回归成对深度与不确定性，再融合逐源隐变量并回归阶段深度。A 在逐源匹配特征上施加几何可见性监督；B 根据前级深度概率状态生成 Stage 2/3 候选；C 以基础可靠性和候选相关残差计算逐深度、逐源视图融合权重。

![图2 三个改进模块的内部计算流程](paper_complete_assets/modules_abc_detailed.png)

**图2. A、B、C 三个模块的内部计算流程。** A 由几何重投影和源深度一致性构造可见、遮挡与无效标签，并监督逐源可见性预测；B 由前级概率分布计算均值和标准差，经范围裁剪及全局边界求交后生成候选深度；C 将基础可靠性与候选深度残差相加，在源视图维归一化后逐候选融合隐变量。"""
    method = method[:old_figure_start] + detailed_figures + method[old_figure_end:]
    references = original[references_start:].rstrip()
    values = narrative(data_by_series)

    abstract_start = introduction.index("## 摘要")
    keywords_start = introduction.index("**关键词：**")
    abstract = (
        "## 摘要\n\n"
        "针对 Vis-MVSNet 在大视差和复杂遮挡区域容易出现粗阶段匹配偏差、候选范围受限与错误视图证据累积的问题，本文提出一种由逐源视图遮挡感知监督、自适应深度搜索范围和深度假设感知源视图融合组成的三级改进方法。逐源遮挡监督利用参考—源视图几何一致性强化可见性学习；自适应范围根据前级估计状态调节后续深度候选，在覆盖能力与采样精度之间进行像素级平衡；深度假设感知融合则为不同候选深度分配源视图权重，抑制遮挡和错误对应产生的干扰。DTU 测试集上的八组消融表明，在五视图训练设置下，完整方法将整体平均绝对误差从 "
        f"{values['v5_base']:.3f} 降至 {values['v5_ours']:.3f}，相对降低 {values['v5_full_gain']:.2f}%；在大视差、遮挡以及大视差与遮挡交集区域分别降低 {values['v5_large_gain']:.2f}%、{values['v5_occ_gain']:.2f}% 和 {values['v5_joint_gain']:.2f}%。"
        "在三视图训练设置下，完整方法的整体平均绝对误差相对降低 "
        f"{values['v3_full_gain']:.2f}%，并在大视差与遮挡交集区域降低 {values['v3_joint_gain']:.2f}%。结果说明，三个模块能够从可见性学习、候选搜索和多视图融合三个环节改善困难区域的深度估计。\n\n"
    )
    introduction = introduction[:abstract_start] + abstract + introduction[keywords_start:]

    configuration_rows = [[label, a, b, c] for _, label, a, b, c in CONFIGS]
    config_table = table(["配置", "A", "B", "C"], configuration_rows, True)

    experiment = f"""## 4 实验设置

### 4.1 数据集与评测协议

实验在 DTU 数据集 [7] 上进行。采用测试划分、light 3 光照条件和像素加权聚合方式，推理阶段统一输入五个视图。为考察训练视图数量对方法的影响，分别使用五视图和三视图训练模型，并保持对应八组消融的评测协议一致。除整体区域外，进一步在深度边界、大视差、任一源视图遮挡、多数源视图遮挡、大视差与遮挡交集以及边界与遮挡交集区域进行评价。

平均绝对误差 Abs 衡量预测深度与真值深度之间的平均偏差，数值越低越好；Acc2、Acc4 和 Acc8 分别统计误差小于 2、4 和 8 个深度单位的像素比例，数值越高越好。S1–S3 coverage 表示真实深度落入对应阶段候选范围的像素比例，S1–S3 width 表示各阶段平均候选区间宽度。所有相对降幅均以 Base 为参照，按 $({{\operatorname{{Abs}}}}_{{\mathrm{{Base}}}}-{{\operatorname{{Abs}}}}_{{\mathrm{{model}}}})/{{\operatorname{{Abs}}}}_{{\mathrm{{Base}}}}\\times100\%$ 计算；准确率和覆盖率变化采用百分点。

### 4.2 消融配置

A、B、C 分别表示逐源视图遮挡感知监督、自适应深度搜索范围和深度假设感知源视图融合。三个二值因素形成八组完整消融：

{config_table}

## 5 实验结果与分析

### 5.1 五视图训练结果

表1给出五视图训练、五视图测试条件下七类区域的平均绝对误差。括号内为相对 Base 的降幅。

**表1. 五视图训练下八组配置的区域平均绝对误差。**

{abs_table(data_by_series['View5'])}

完整方法将整体 Abs 从 {values['v5_base']:.3f} 降至 {values['v5_ours']:.3f}，相对降低 {values['v5_full_gain']:.2f}%。在大视差区域，Abs 从 {number(data_by_series['View5'], 'range', 'large_disparity', 'abs'):.3f} 降至 {number(data_by_series['View5'], 'oa_hyp', 'large_disparity', 'abs'):.3f}，降幅达到 {values['v5_large_gain']:.2f}%；在任一源视图遮挡区域和大视差与遮挡交集区域，降幅分别为 {values['v5_occ_gain']:.2f}% 和 {values['v5_joint_gain']:.2f}%。这些区域的降幅均具有明确幅度，表明改进并非只作用于普通像素，而是集中覆盖了本文关注的困难匹配条件。

![图3 五视图训练的区域Abs改善](paper_complete_assets/view5_abs_gain.png)

**图3. 五视图训练下八组配置相对 Base 的 Abs 降幅。** B 及包含 B 的组合在大视差区域呈现最明显改善，A 和 C 单独加入时也降低了多个区域的平均误差。

表2列出整体区域的全部十项指标。完整方法的 Acc2、Acc4 和 Acc8 分别为 {number(data_by_series['View5'], 'oa_hyp', 'full', 'acc2')*100:.2f}%、{number(data_by_series['View5'], 'oa_hyp', 'full', 'acc4')*100:.2f}% 和 {number(data_by_series['View5'], 'oa_hyp', 'full', 'acc8')*100:.2f}%，相对 Base 分别提高 {pp_gain(data_by_series['View5'], 'oa_hyp', 'full', 'acc2'):.2f}、{pp_gain(data_by_series['View5'], 'oa_hyp', 'full', 'acc4'):.2f} 和 {pp_gain(data_by_series['View5'], 'oa_hyp', 'full', 'acc8'):.2f} 个百分点。Stage 2 和 Stage 3 coverage 分别提高 {pp_gain(data_by_series['View5'], 'oa_hyp', 'full', 'stage2_in_range'):.2f} 和 {pp_gain(data_by_series['View5'], 'oa_hyp', 'full', 'stage3_in_range'):.2f} 个百分点，说明后续阶段对真实深度的覆盖能力得到改善。

**表2. 五视图训练下整体区域的全部指标。**

{full_metrics_table(data_by_series['View5'])}

### 5.2 三视图训练结果

表3给出三视图训练、五视图测试条件下的区域平均绝对误差。

**表3. 三视图训练下八组配置的区域平均绝对误差。**

{abs_table(data_by_series['View3'])}

完整方法将整体 Abs 从 {values['v3_base']:.3f} 降至 {values['v3_ours']:.3f}，相对降低 {values['v3_full_gain']:.2f}%；在大视差、任一源视图遮挡以及大视差与遮挡交集区域分别降低 {values['v3_large_gain']:.2f}%、{values['v3_occ_gain']:.2f}% 和 {values['v3_joint_gain']:.2f}%。这说明在训练视图减少的条件下，联合方法仍能改善目标困难区域。

![图4 三视图训练的区域Abs改善](paper_complete_assets/view3_abs_gain.png)

**图4. 三视图训练下八组配置相对 Base 的 Abs 降幅。**

**表4. 三视图训练下整体区域的全部指标。**

{full_metrics_table(data_by_series['View3'])}

### 5.3 单模块与组合效果

表5和表6给出五视图训练条件下的十二组受控比较，每一行只改变一个模块。A 单独加入 Base 后，整体、深度边界和边界与遮挡交集区域的 Abs 分别降低 {abs_gain(data_by_series['View5'], 'oa_range', 'full'):.2f}%、{abs_gain(data_by_series['View5'], 'oa_range', 'boundary'):.2f}% 和 {abs_gain(data_by_series['View5'], 'oa_range', 'boundary_and_occluded'):.2f}%，说明显式遮挡监督对边界附近的不可靠观测具有积极作用。B 单独加入 Base 后，整体和大视差区域的 Abs 分别降低 {abs_gain(data_by_series['View5'], 'vis', 'full'):.2f}% 和 {abs_gain(data_by_series['View5'], 'vis', 'large_disparity'):.2f}%，是三项单模块改进中幅度最大的因素；其 Stage 2 和 Stage 3 整体 coverage 分别增加 {pp_gain(data_by_series['View5'], 'vis', 'full', 'stage2_in_range'):.2f} 和 {pp_gain(data_by_series['View5'], 'vis', 'full', 'stage3_in_range'):.2f} 个百分点，验证了候选范围调整与深度覆盖改善之间的联系。C 单独加入 Base 后，整体和大视差与遮挡交集区域的 Abs 分别降低 {abs_gain(data_by_series['View5'], 'range_hyp', 'full'):.2f}% 和 {abs_gain(data_by_series['View5'], 'range_hyp', 'large_disp_and_occluded'):.2f}%；当 C 与 B 联合时，Base+B+C 在整体、大视差和大视差与遮挡交集区域分别获得 {abs_gain(data_by_series['View5'], 'hyp', 'full'):.2f}%、{abs_gain(data_by_series['View5'], 'hyp', 'large_disparity'):.2f}% 和 {abs_gain(data_by_series['View5'], 'hyp', 'large_disp_and_occluded'):.2f}% 的降幅，说明候选深度相关融合在可靠搜索空间内能够进一步发挥作用。

**表5. 五视图训练下加入单个模块时的 Abs 相对变化。**

{controlled_table(data_by_series['View5'], 'abs')}

**表6. 五视图训练下加入单个模块时的 Acc4 变化。**

{controlled_table(data_by_series['View5'], 'acc4')}

![图5 两套训练设置的消融热力图](paper_complete_assets/ablation_heatmap.png)

**图5. 两套训练设置下八组消融在七类区域的 Abs 相对降幅。** 绿色表示平均误差降低，红色表示平均误差增加。

### 5.4 指标取舍

八组结果表明，各模块组合在平均误差和阈值准确率之间存在局部取舍。五视图训练时，Base+B+C 的整体 Abs 为 {number(data_by_series['View5'], 'hyp', 'full', 'abs'):.3f}，低于完整模型的 {number(data_by_series['View5'], 'oa_hyp', 'full', 'abs'):.3f}；完整模型则在深度边界区域获得最低 Abs {number(data_by_series['View5'], 'oa_hyp', 'boundary', 'abs'):.3f}，在边界与遮挡交集区域的 Acc4 相对 Base+B+C 提高 {(number(data_by_series['View5'], 'oa_hyp', 'boundary_and_occluded', 'acc4')-number(data_by_series['View5'], 'hyp', 'boundary_and_occluded', 'acc4'))*100:.2f} 个百分点。该结果表明，A 的加入更有利于边界和遮挡相关像素，但可能改变全图误差尾部，因此完整模型无需在所有汇总指标上同时超过每一个双模块组合。

从指标定义看，Abs 对少量大误差像素十分敏感，而 Acc4 只反映误差是否进入 4 个深度单位的容差范围。某一组合提高 Acc4 而 Abs 略有增大，表示更多像素跨入阈值范围，但剩余大误差像素的幅度可能增加。图6显示两套训练设置下八组配置的 Acc2、Acc4 和 Acc8，阈值越宽，不同配置之间的变化越能反映严重错误像素是否得到纠正。

![图6 整体区域阈值准确率](paper_complete_assets/full_accuracy.png)

**图6. 八组配置在整体区域的 Acc2、Acc4 和 Acc8。**

## 6 讨论

实验结果支持三个模块围绕同一目标形成分工。A 通过显式几何标签强化遮挡区域的可靠性学习，B 通过调整后续搜索范围提高真实深度覆盖，C 则在候选深度维度重新分配源视图贡献。B 对大视差区域的贡献最为突出，说明三级结构中的候选覆盖是限制困难像素恢复的重要因素；A 在边界及遮挡交集区域表现出更稳定的作用；C 的单独收益相对温和，但与 B 结合后在多个区域进一步降低平均误差，体现了搜索空间与融合策略之间的互补性。

完整方法在五视图训练和三视图训练条件下均显著优于 Base，且改善覆盖整体、大视差、遮挡和二者交集区域。与此同时，Base+B+C 在部分平均误差指标上优于完整模型，说明三模块联合会重新分配不同像素和不同误差尺度上的优化重点。本文据此将完整方法的结论限定为：三个模块共同提高了原始 Vis-MVSNet 在目标困难区域的综合表现，但单个区域或单项指标的最佳组合可能不同。

## 7 结论

本文针对 Vis-MVSNet 三级深度估计在大视差与复杂遮挡下的误差传播问题，提出逐源视图遮挡感知监督、自适应深度搜索范围和深度假设感知源视图融合。三个模块分别改善可见性学习、真实深度候选覆盖和多视图证据聚合。DTU 上的完整八组消融表明，五视图训练时，完整方法相对 Base 将整体、大视差、遮挡以及大视差与遮挡交集区域的 Abs 分别降低 {values['v5_full_gain']:.2f}%、{values['v5_large_gain']:.2f}%、{values['v5_occ_gain']:.2f}% 和 {values['v5_joint_gain']:.2f}%；三视图训练时，对应降幅分别为 {values['v3_full_gain']:.2f}%、{values['v3_large_gain']:.2f}%、{values['v3_occ_gain']:.2f}% 和 {values['v3_joint_gain']:.2f}%。结果验证了从监督、搜索和融合三个环节联合提升大视差与复杂遮挡区域深度估计的有效性。

## 附录 A：全部区域与全部指标

以下表格给出两套训练设置、七类区域、八组配置的全部十项指标。

"""

    appendix_parts = []
    for key, _, title in SERIES:
        appendix_parts.append(f"### A.{1 if key == 'View5' else 2} {title}\n")
        for idx, (region, label) in enumerate(REGIONS, 1):
            appendix_parts.append(f"#### A.{1 if key == 'View5' else 2}.{idx} {label}\n")
            appendix_parts.append(complete_region_table(data_by_series[key], region) + "\n")

    paper = introduction.rstrip() + "\n\n" + method.rstrip() + "\n\n" + experiment.rstrip() + "\n\n"
    paper += "\n".join(appendix_parts).rstrip() + "\n\n" + references + "\n"
    OUTPUT.write_text(paper, encoding="utf-8")
    print(OUTPUT)
    for path in sorted(ASSETS.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
