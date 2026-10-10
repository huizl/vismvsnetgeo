"""Assemble a separate depth-estimation manuscript from existing results and real exports.

The original manuscript and visualization exports are read-only inputs.  This
script checks paired pixel metrics before adding figures and case statistics.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SOURCE = DOCS / 'PAPER_COMPLETE_WITH_RESULTS.md'
OUTPUT = DOCS / 'PAPER_DEPTH_ESTIMATION_WITH_VISUALIZATIONS.md'
ASSETS = DOCS / 'paper_depth_visual_assets'
EXPORTS = ROOT / 'outputs' / 'paper_visualizations_ranked'
CONFIGS = [('range', 'Base'), ('oa_range', 'Base+A'), ('vis', 'Base+B'),
           ('range_hyp', 'Base+C'), ('oa', 'Base+A+B'), ('oa_full', 'Base+A+C'),
           ('hyp', 'Base+B+C'), ('oa_hyp', 'Base+A+B+C')]
SERIES = {'View5': 'ablation_test_light3', 'View3': 'ablation_test_view3_light3'}
CATEGORIES = ['large_disparity_dominant', 'occlusion_dominant', 'joint_difficulty']
LABELS = {'large_disparity_dominant': '大视差为主',
          'occlusion_dominant': '复杂遮挡为主', 'joint_difficulty': '大视差与遮挡交集'}
REGIONS = {'full': '整体', 'large_disparity': '大视差', 'occluded_any': '任一源遮挡',
           'occluded_ge2': '至少两源遮挡', 'large_disp_and_occluded': '大视差∩任一源遮挡',
           'large_disp_and_occluded_ge2': '大视差∩至少两源遮挡'}
METRICS = ['abs', 'acc2', 'acc4', 'acc8'] + [
    f'stage{s}_{m}' for m in ('in_range', 'range_width') for s in (1, 2, 3)]


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def check_manuscript_metric_cells(text):
    appendix = text.split('## 附录 A：')[1].split('## 附录 B：')[0]
    regions = ['full', 'boundary', 'large_disparity', 'occluded_any',
               'occluded_majority', 'large_disp_and_occluded', 'boundary_and_occluded']
    fields = ['abs', 'acc2', 'acc4', 'acc8'] + [f'stage{s}_in_range' for s in (1, 2, 3)] + [
        f'stage{s}_range_width' for s in (1, 2, 3)]
    raw_by_label = {label: raw for raw, label in CONFIGS}
    count = 0
    for section, series in [('A.1', 'View5'), ('A.2', 'View3')]:
        part = appendix.split('### '+section+' ')[1]
        if section == 'A.1':
            part = part.split('### A.2 ')[0]
        blocks = re.split(r'#### '+re.escape(section)+r'\.\d+ ', part)[1:]
        assert len(blocks) == len(regions)
        for region, block in zip(regions, blocks):
            lines = [line for line in block.splitlines() if line.startswith('| Base')]
            assert len(lines) == len(CONFIGS)
            for line in lines:
                cells = [x.strip() for x in line.strip('|').split('|')]
                raw = raw_by_label[cells[0]]
                path = ROOT/'eval'/SERIES[series]/f'{raw}_{series.lower()}'/'summary_metrics.csv'
                row = next(x for x in read_csv(path) if x['aggregation'] == 'pixel_weighted' and x['region'] == region)
                for field, shown in zip(fields, cells[1:]):
                    expected = (f"{float(row[field])*100:.2f}%" if field.startswith('acc') or field.endswith('in_range')
                                else f"{float(row[field]):.3f}")
                    if shown != expected:
                        raise ValueError(f'Manuscript table mismatch: {series}/{region}/{raw}/{field}')
                    count += 1
    return count


def audit_population():
    audits = []
    for series, directory in SERIES.items():
        reference = None
        for raw, label in CONFIGS:
            folder = ROOT / 'eval' / directory / f'{raw}_{series.lower()}'
            rows = read_csv(folder / 'all_metrics.csv')
            keys = {(r['scan'], r['view'], r['light'], r['region']): int(r['pixels']) for r in rows}
            if len(keys) != len(rows):
                raise ValueError(f'Duplicate rows: {folder}')
            if reference is None:
                reference = keys
            if keys != reference:
                raise ValueError(f'Unpaired evaluation population: {folder}')
            summary = [r for r in read_csv(folder / 'summary_metrics.csv')
                       if r['aggregation'] == 'pixel_weighted']
            for r in summary:
                selected = [x for x in rows if x['region'] == r['region'] and int(x['pixels']) > 0]
                pixels = sum(int(x['pixels']) for x in selected)
                assert pixels == int(r['pixels'])
                assert len(selected) == int(r['images'])
                for metric in METRICS:
                    actual = sum(float(x[metric]) * int(x['pixels']) for x in selected) / pixels
                    if not np.isclose(actual, float(r[metric]), rtol=1e-8, atol=1e-8):
                        raise ValueError(f'Weighted statistic mismatch: {folder}, {metric}')
            full = next(r for r in summary if r['region'] == 'full')
            audits.append({'series': series, 'analysis_config': label, 'raw_model_type': raw,
                           'images': int(full['images']), 'pixels': int(full['pixels']),
                           'eval_nviews': int(full['eval_nviews']),
                           'region_nviews': int(full['region_nviews']),
                           'abs': float(full['abs']), 'weighted_metrics_checked': True})
    return audits


def load_cases():
    cases = []
    details = []
    tails = []
    for series in SERIES:
        selection = json.loads((EXPORTS / series / 'scene_selection.json').read_text(encoding='utf-8'))
        for category in CATEGORIES:
            targets = [x for x in selection if category in x['scene_categories']]
            if len(targets) != 2:
                raise ValueError(f'Expected two cases: {series}/{category}')
            for rank, target in enumerate(targets, 1):
                key = f"{target['scan']}_view{target['view']:02d}_light{target['light']}"
                path = EXPORTS / series / 'scenes' / category / key
                rows = read_csv(path / 'region_metrics.csv')
                pairs = {(r['analysis_config'], r['region']): r for r in rows}
                if len(pairs) != 18:
                    raise ValueError(f'Unexpected case metric rows: {path}')
                settings = json.loads((path / 'display_settings.json').read_text(encoding='utf-8'))
                with np.load(path / 'comparison_arrays.npz') as z:
                    arrays = {k: z[k] for k in z.files}
                model_arrays = {}
                for label in ('Base', 'Base+A+B+C'):
                    with np.load(EXPORTS / series / label / key / 'arrays.npz') as z:
                        model_arrays[label] = {k: z[k] for k in z.files}
                    np.testing.assert_array_equal(model_arrays[label]['valid'], arrays['valid'])
                    np.testing.assert_allclose(
                        np.abs(model_arrays[label]['pred']-model_arrays[label]['gt'])[arrays['valid']],
                        arrays['base_error' if label == 'Base' else 'full_model_error'][arrays['valid']],
                        rtol=0, atol=0)
                details.extend({'series': series, 'category': category, 'rank': rank, **r} for r in rows)
                valid = arrays['valid']
                large = arrays['large_disparity_mask'] & valid
                occ = arrays['occluded_ge2_mask'] & valid
                masks = {'full': valid, 'large_disparity': large,
                         'occluded_any': arrays['occluded_any_mask'] & valid,
                         'occluded_ge2': occ,
                         'large_disp_and_occluded': large & arrays['occluded_any_mask'],
                         'large_disp_and_occluded_ge2': large & occ}
                for region in ('boundary', 'occluded_majority', 'boundary_and_occluded'):
                    b_mask = model_arrays['Base'][f'mask_{region}']
                    np.testing.assert_array_equal(b_mask, model_arrays['Base+A+B+C'][f'mask_{region}'])
                    masks[region] = b_mask & valid
                np.testing.assert_allclose(arrays['error_gain'][valid],
                                           (arrays['base_error']-arrays['full_model_error'])[valid],
                                           rtol=0, atol=0)
                for region, mask in masks.items():
                    b, f = pairs['Base', region], pairs['Base+A+B+C', region]
                    assert int(mask.sum()) == int(b['pixels']) == int(f['pixels'])
                    for label, error in [('Base', arrays['base_error']),
                                         ('Base+A+B+C', arrays['full_model_error'])]:
                        row = pairs[label, region]
                        e = error[mask]
                        for m, value in [('abs', float(e.mean()))] + [
                                (f'acc{t}', float((e < t).mean())) for t in (2, 4, 8)]:
                            if not np.isclose(value, float(row[m]), rtol=1e-6, atol=1e-6):
                                raise ValueError(f'Array/CSV mismatch: {path}/{region}/{label}/{m}')
                        for stage in (1, 2, 3):
                            for suffix, field in [('coverage', 'in_range'), ('width', 'range_width')]:
                                value = float(model_arrays[label][f'stage{stage}_{suffix}'][mask].mean())
                                if not np.isclose(value, float(row[f'stage{stage}_{field}']), rtol=1e-6, atol=1e-6):
                                    raise ValueError(f'Stage array/CSV mismatch: {path}/{region}/{label}/{field}')
                    reduction = 100 * (float(b['abs']) - float(f['abs'])) / float(b['abs'])
                    np.testing.assert_allclose(reduction, float(f['abs_reduction_pct']), atol=1e-6)
                    for label, error in [('Base', arrays['base_error']),
                                         ('Base+A+B+C', arrays['full_model_error'])]:
                        e = error[mask]
                        tails.append({'series': series, 'sample': key, 'region': region,
                                      'analysis_config': label, 'pixels': len(e),
                                      'abs': float(e.mean()), 'acc2_pct': float((e < 2).mean()*100),
                                      'q95_mm': float(np.percentile(e, 95)),
                                      'error_ge20_pct': float((e >= 20).mean()*100),
                                      'base_under2_to_full_ge2': int(((arrays['base_error'] < 2) &
                                                                      (arrays['full_model_error'] >= 2) & mask).sum()),
                                      'base_ge2_to_full_under2': int(((arrays['base_error'] >= 2) &
                                                                      (arrays['full_model_error'] < 2) & mask).sum())})
                dest = ASSETS / series / category / key
                dest.mkdir(parents=True, exist_ok=True)
                for name in ('overview_3x5.png', 'overview_3x5.pdf',
                             'region_metrics.csv', 'display_settings.json', 'figure_caption.txt'):
                    shutil.copy2(path / name, dest / name)
                cases.append({'series': series, 'category': category, 'rank': rank,
                              'key': key, 'pairs': pairs, 'settings': settings,
                              'image': (dest/'overview_3x5.png').relative_to(DOCS).as_posix(),
                              'path': path})
    return cases, details, tails


def row_pair(case, region):
    b, f = case['pairs']['Base', region], case['pairs']['Base+A+B+C', region]
    return [case['series'], case['key'], REGIONS[region], b['pixels'],
            f"{float(b['abs']):.3f}", f"{float(f['abs']):.3f}",
            f"{float(f['abs_reduction_pct']):+.2f}%",
            f"{float(b['acc2'])*100:.2f}%", f"{float(f['acc2'])*100:.2f}%",
            f"{float(f['acc2_delta_pp']):+.2f}"]


def figure(case, number):
    s = case['settings']
    return (f"![图{number} {case['series']} {case['key']} {LABELS[case['category']]}]({case['image']})\n\n"
            f"**图{number}. {case['series']} 的{LABELS[case['category']]}案例 {case['key']}。** "
            f"子图顺序与表7一致。GT 与两组预测的深度色域统一为 {s['depth_min_mm']:.1f}–{s['depth_max_mm']:.1f} mm；"
            f"绝对误差显示为 0–{s['error_max_mm']:g} mm，误差差值显示为 ±{s['gain_max_mm']:g} mm。"
            "误差差值中的红色为改善，蓝色为退化，黄色轮廓标记固定 GT 困难区域。超过显示范围的数值只在着色时截断，仍完整参与指标计算。\n\n")


EXPERIMENTS = r'''## 4 实验设置

### 4.1 数据集与评测协议

实验采用 DTU 数据集 [19] 的测试划分，在 light 3 光照条件下评价参考视图深度。两套模型分别使用五视图和三视图训练，评测时均输入一个参考视图和四个源视图。每套设置包含 1078 个参考图像样本，整体有效深度像素为 16,254,332 个；同一设置内的八组配置使用相同样本、参考有效掩码及困难区域掩码。源视图按数据集 pair 文件给出的顺序选择，不因某个模型的结果重新选择。本文只评价深度估计，不由深度误差推导点云质量。

推理采用 192 个输入基础深度值及 1.06 的间距缩放，三个阶段的候选数量分别为 48、32、16，名义间距倍率分别为 4、2、1。可视化导出采用 batch size 1，输出深度与 GT 对齐后的案例数组分辨率为 128×160。训练视图数与推理视图数分开记述：三视图训练结果用于分析训练观测数量变化，不作为三视图推理的实验。

全测试集表格采用原逐图评测记录的像素加权汇总，个例表采用绘图时重新导出的预测数组。两类结果分别统计，案例指标不替代全测试集指标。重新导出与原评测记录间存在小幅数值差异，复核结果在附录 C 给出。

### 4.2 固定困难区域的构造

以参考 GT 有效像素集合 $\Omega_i$ 为评价域。根据式(1)，计算真值点在四个源视图中的投影，并定义最大二维重投影位移：

$$
\nu_i(\mathbf p)=\max_{s=1,\ldots,S}\|\mathbf p_s(G_0(\mathbf p))-\mathbf p\|_2,
\quad \mathcal D_i=\{\mathbf p\in\Omega_i:\nu_i(\mathbf p)\ge Q_{0.8}(\nu_i|_{\Omega_i})\}.
\tag{22}
$$

$Q_{0.8}$ 表示逐图第 80 百分位，$\nu$ 的单位为当前评价网格的像素。本文“大视差”采用上述重投影位移定义，包含水平和垂直分量，并非校正双目中的纯水平视差。位移统计沿用现有评测协议，未按源投影是否位于图像内进一步筛除；遮挡判断则严格检查源投影和源 GT 的有效性。百分位规则使每幅图约 20% 的有效像素进入大视差区域，因此该掩码用于区分图内相对位移较大的像素，不能单凭掩码面积比较不同场景的绝对位移难度。

以式(4)的逐源遮挡标签 $o_s$ 累加得到 $n_{\mathrm{occ}}$，以参考 GT 有效、投影在源图像内、源深度有效等条件确定的几何可比较视图数记为 $n_{\mathrm{cmp}}$。遮挡容差采用 $\tau_a=2$ mm、$\tau_r=0.01$。定义：

$$
\mathcal O_{1,i}=\{\mathbf p\in\Omega_i:n_{\mathrm{occ}}\ge1\},\quad
\mathcal O_{2,i}=\{\mathbf p\in\Omega_i:n_{\mathrm{occ}}\ge2\},\quad
\mathcal O_{\mathrm{half},i}=\{\mathbf p\in\Omega_i:n_{\mathrm{cmp}}>0,\ 2n_{\mathrm{occ}}\ge n_{\mathrm{cmp}}\}.
\tag{23}
$$

全测试集原表中的“多数源视图遮挡”对应 $\mathcal O_{\mathrm{half}}$，其准确含义为至少一半可比较源视图遮挡；“至少两个源视图遮挡”对应 $\mathcal O_2$。两者的分母或计数条件不同，本文分别报告，不将已有多数比例结果改名为两视图遮挡结果。遮挡比例图显示 $n_{\mathrm{occ}}/n_{\mathrm{cmp}}$，无可比较源视图时显示为零，零值本身不证明在全部源图像中可见。越界投影及源 GT 缺失不直接计为遮挡。

全测试集交集采用 $\mathcal D_i\cap\mathcal O_{1,i}$，定性分析额外统计 $\mathcal D_i\cap\mathcal O_{2,i}$。深度边界按参考 GT 深度梯度幅度的较大 10% 有效像素构造。所有困难区域只由 GT 与相机定义，同一像素集合用于比较所有模型。

### 4.3 评价指标

对于区域 $\mathcal R_i\subseteq\Omega_i$，令 $e_i(\mathbf p)=|\hat d_i(\mathbf p)-G_i(\mathbf p)|$。跨图像像素加权指标定义为：

$$
\operatorname{Abs}_{\mathcal R}=\frac{\sum_i\sum_{\mathbf p\in\mathcal R_i}e_i(\mathbf p)}{\sum_i|\mathcal R_i|},
\quad \operatorname{Acc}_{\tau,\mathcal R}=\frac{\sum_i\sum_{\mathbf p\in\mathcal R_i}\mathbb1[e_i(\mathbf p)<\tau]}{\sum_i|\mathcal R_i|},\quad \tau\in\{2,4,8\}\ \mathrm{mm}.
\tag{24}
$$

Abs 越低越好，Acc2/4/8 越高越好，阈值采用严格小于。Stage $t$ 的区间上下界对齐到评价网格后，覆盖率和归一化宽度定义为：

$$
\operatorname{Cov}_{t,\mathcal R}=\frac{\sum_i\sum_{\mathbf p\in\mathcal R_i}\mathbb1[d_{-,i}^t(\mathbf p)\le G_i(\mathbf p)\le d_{+,i}^t(\mathbf p)]}{\sum_i|\mathcal R_i|},
\quad \operatorname{Width}_{t,\mathcal R}=\frac{\sum_i\sum_{\mathbf p\in\mathcal R_i}(d_{+,i}^t(\mathbf p)-d_{-,i}^t(\mathbf p))/\delta_{0,i}}{\sum_i|\mathcal R_i|}.
\tag{25}
$$

Width 的单位为基础深度间距倍数，不是 mm，也不是越大或越小越好。覆盖率与最终误差共同评价候选搜索。相对改善与配对像素误差差值定义为：

$$
r_{\mathrm{Abs}}=100\frac{\operatorname{Abs}_{\mathrm{Base}}-\operatorname{Abs}_{\mathrm{model}}}{\operatorname{Abs}_{\mathrm{Base}}},
\qquad \Delta e(\mathbf p)=e_{\mathrm{Base}}(\mathbf p)-e_{\mathrm{Base+A+B+C}}(\mathbf p).
\tag{26}
$$

$r_{\mathrm{Abs}}>0$ 表示平均误差下降；准确率和覆盖率的变化使用百分点。差值图中 $\Delta e>0$ 为红色改善，$\Delta e<0$ 为蓝色退化，白色接近零。两种模型相同的大误差也可能显示为白色，需与绝对误差图联合阅读。

### 4.4 消融配置与定性选例

A、B、C 分别表示逐源视图遮挡感知监督、自适应深度搜索范围和深度假设感知源视图融合。本文采用如下八个实验配置标识，Base 为数值比较的参照配置：

| 配置 | A | B | C |
| --- | ---: | ---: | ---: |
| Base | 0 | 0 | 0 |
| Base+A | 1 | 0 | 0 |
| Base+B | 0 | 1 | 0 |
| Base+C | 0 | 0 | 1 |
| Base+A+B | 1 | 1 | 0 |
| Base+A+C | 1 | 0 | 1 |
| Base+B+C | 0 | 1 | 1 |
| Base+A+B+C | 1 | 1 | 1 |

定性分析按 GT 几何区分三类：大视差为主要求 $|\mathcal D|\ge30$、$|\mathcal O_2|/|\Omega|\le10\%$ 且 $|\mathcal D\cap\mathcal O_2|/|\mathcal D|\le25\%$；复杂遮挡为主要求 $|\mathcal O_2|\ge30$、$|\mathcal O_2|/|\Omega|\ge10\%$ 且交集占遮挡区域不超过 35%；交集类要求交集至少 30 像素、占有效像素至少 5%、占大视差区域至少 25%。满足交集条件时优先归入交集类。这些条件用于组织示例，不是通用的困难场景标准。

View3、View5 分别在各类合格候选中按原逐图记录的全图 Abs 相对降幅排序，仅选择正降幅，每类保留两个不同 scan 的案例。每类排名第一的案例放入正文，第二个案例放入附录 B，全部十二例的指标均报告。选例旨在解释改善发生的位置，属于收益导向的案例展示，不是随机抽样；整体效果由全部 1078 个样本的统计支持。选例后不修改困难掩码，也不隐藏所选图像中的退化像素。

'''


def build_qualitative(cases, tails):
    out = ['### 5.5 大视差与复杂遮挡的定性分析\n\n',
           '图7—图12对比 Base 与完整方法。每幅图由三行五列组成（表7）。参考图困难叠加中的红色、绿色、黄色分别表示仅大视差、仅至少两源遮挡及两者交集；其颜色含义不同于误差差值图。黄色轮廓依据固定 GT 困难掩码绘制。黑色深度及误差区域表示 GT 无效，不将其解释为模型漏估。\n\n',
           '**表7. 定性图子图顺序。**\n\n',
           table(['行', '第一列', '第二列', '第三列', '第四列', '第五列'], [
               ['第一行', '参考 RGB', 'GT 深度', 'GT 重投影位移', '源视图遮挡比例', '困难区域 RGB 叠加'],
               ['第二行', 'Base 深度', '完整方法深度', 'Base 绝对误差', '完整方法绝对误差', '误差差值：红改善、蓝退化'],
               ['第三行', '至少一源遮挡', '至少两源遮挡', '大视差掩码', '大视差∩至少一源遮挡', '大视差∩至少两源遮挡']])+'\n\n']
    fig = 7
    titles = ['5.5.1 大视差为主的场景', '5.5.2 复杂遮挡为主的场景', '5.5.3 大视差与遮挡交集明显的场景']
    descriptions = [
        '图7的纸盒场景中，大视差主要分布在图像侧部，至少两源遮挡区域较小。Base 在盒体上部出现明显的异常深度块，完整方法缩小了这一大误差区域；同时，红蓝差值图保留了边缘处的局部退化。该例全图 Abs 降幅为 85.44%，但大视差区域降幅为 30.20%，说明按全图收益选出的案例，其主要收益不一定全部发生在大视差掩码内。图8从另一参考方向观察同一 scan，较大位移覆盖盒体上沿，其大视差 Abs 从 5.286 降至 2.031 mm，Acc2 从 65.74% 提高至 84.22%。附录中的 scan34 案例进一步提供了不同对象的对照。',
        '图9与图10展示多物体形成的遮挡带。遮挡比例图与至少两源遮挡掩码定位到前景轮廓附近的带状区域，而物体内部的可比较遮挡比例较低。完整方法纠正了部分背景深度偏差，并改变了轮廓附近的过渡。View5 的 scan12_view13 至少两源遮挡区域 Abs 从 36.076 降至 20.317 mm，Acc2 从 8.01% 提高至 15.56%；View3 的 scan12_view14 对应 Abs 从 32.352 降至 22.974 mm，Acc2 从 7.98% 提高至 11.29%。两例仍存在误差较高和蓝色退化的轮廓像素，因此结论是该遮挡区域的统计误差降低，而非所有遮挡像素均恢复准确。',
        '图11中黄色叠加位于前景轮廓与后方表面的连接处，交集掩码占有效像素的 8.54%。完整方法纠正了较大的背景偏差，但部分遮挡过渡带出现蓝色退化。至少两源遮挡交集 Abs 从 19.168 降至 14.197 mm，Acc2 却从 9.37% 降至 1.25%，需要结合误差分布解释。图12的玩偶场景中，交集区域为 763 像素，占有效像素的 5.23%；对应 Abs 从 89.511 降至 45.495 mm，Acc2 从 10.09% 提高至 12.58%。该案例 75.88% 的交集像素误差下降，但改进后平均误差仍较高，说明改善幅度与达到严格精度要求是两个不同问题。']
    headers = ['设置', '案例', '区域', '像素数', 'Base Abs/mm', '完整 Abs/mm', 'Abs 降幅', 'Base Acc2', '完整 Acc2', 'Acc2变化/pp']
    for idx, category in enumerate(CATEGORIES):
        out.append(f'#### {titles[idx]}\n\n')
        subset = [c for c in cases if c['category'] == category]
        for series in SERIES:
            case = next(c for c in subset if c['series'] == series and c['rank'] == 1)
            out.append(figure(case, fig)); fig += 1
        region = {'large_disparity_dominant': 'large_disparity',
                  'occlusion_dominant': 'occluded_ge2',
                  'joint_difficulty': 'large_disp_and_occluded_ge2'}[category]
        rows = [row_pair(c, r) for c in subset for r in ('full', region)]
        out.append(f'**表{8+idx}. {LABELS[category]}的四个案例及其固定目标区域指标。** Abs 使用未截断误差，降幅按完整精度计算。\n\n')
        out.append(table(headers, rows)+'\n\n'+descriptions[idx]+'\n\n')
    out.append('### 5.6 误差幅度与阈值准确率的配对分析\n\n')
    out.append('为了区分平均误差下降与严格精度改善，对图11同一 GT 像素集合直接计算误差分布。表11给出全图、大视差和至少两源遮挡交集的 Abs、Acc2、第95百分位误差及误差不小于20 mm的比例。这些统计均来自本次实际预测，不由误差图颜色估计。\n\n')
    selected = [x for x in tails if x['series'] == 'View5' and x['sample'] == 'scan12_view11_light3'
                and x['region'] in ('full', 'large_disparity', 'large_disp_and_occluded_ge2')]
    out.append('**表11. 图11案例的误差分布。**\n\n'+table(
        ['区域', '模型', '像素数', 'Abs/mm', 'Acc2', 'P95/mm', '误差≥20mm'],
        [[REGIONS[x['region']], x['analysis_config'], x['pixels'], f"{x['abs']:.3f}",
          f"{x['acc2_pct']:.2f}%", f"{x['q95_mm']:.3f}", f"{x['error_ge20_pct']:.2f}%"] for x in selected])+'\n\n')
    out.append('Abs 是误差分布的均值，Acc2 只统计是否进入 2 mm 容差。部分严重错误即使大幅减小，仍可能位于该阈值外；与此同时，原本低于阈值的像素可能因轮廓过渡变化跨到阈值外。因此，均值下降与 Acc2 下降可以同时发生。表11和图11说明该取舍确实出现在实际案例中，但该配对结果不意味着所有配置变化都由相同机制引起。\n\n')
    paired = next(x for x in selected if x['region'] == 'large_disp_and_occluded_ge2' and x['analysis_config'] == 'Base')
    out.append(f"具体而言，交集区域误差不小于20 mm的比例由32.92%降至7.68%，第95百分位误差由48.262降至23.054 mm，体现了严重错误幅度的降低。对1355个相同像素进行2 mm阈值配对，有{paired['base_under2_to_full_ge2']}个像素由阈值内变为阈值外，另有{paired['base_ge2_to_full_under2']}个由阈值外进入阈值内；两者的净变化解释了该区域Acc2下降。该结果同时呈现了大误差修正和严格阈值精度退化，不能简写为全面提高准确率。\n\n")
    return ''.join(out)


DISCUSSION = '''## 6 讨论

### 6.1 面向困难区域的效果与解释

两套训练设置下，完整配置在全测试集大视差、任一源遮挡和二者交集的 Abs 均低于 Base；固定 GT 掩码和配对个例进一步定位了部分收益。因此，现有证据支持在当前 DTU 深度评测协议下改善目标困难区域的平均估计误差。全图与区域统计、绝对误差图和配对差值图相互补充：前者评价总体幅度，后者显示改善与退化的位置。

从设计看，A 补充几何可见性监督，B 调节后级候选覆盖，C 细化候选相关视图权重。但最终误差和 coverage 的变化只提供与设计动机相关的观察，不能单独识别某个内部机制。特别是概率标准差不等于真实误差；低标准差的错误预测仍可能产生过窄的后续范围。具体机制需通过成对可靠性、候选覆盖及逐候选权重的联合诊断检验。

### 6.2 模块组合与局部退化

单模块相对 Base 的收益与模块加入已有组合后的收益不同。五视图训练时，Base+B+C 的整体、大视差和交集 Abs 低于完整配置，完整配置则在深度边界 Abs 上更低；三视图训练时，完整配置的整体 Acc2/4/8 高于 Base+B+C，而整体 Abs 较高。这些结果说明不同组合可能改变误差分布，不能把八组消融写成每增加一个模块都必然提高的单调路径。

对于完整方法，其实验结论是相对 Base 的目标区域改善。现有结果尚不支持完整组合在所有评价维度上均优于所有子组合，也不支持每个模块在每种组合中的边际收益均为正。定性图保留的蓝色区域以及图11的 Acc2 下降，与这一边界一致。

### 6.3 评价范围与局限

本文结果限定于 DTU 测试划分、light 3 和五视图推理。三视图训练实验考察训练观测数量，不证明更少推理视图下的性能。大视差使用逐图重投影位移百分位，“至少两源遮挡”仅在当前十二个可视化案例中补充评价，不能将其个例降幅写成全测试集的复杂遮挡降幅。案例按全图收益排序，适于解释成功现象，但不能用于估计成功率。

目前未通过重复训练给出方差或统计显著性，也未通过统一协议的外部方法比较建立领先性。因此本文不使用统计显著或最优方法的结论。进一步工作包括扩大光照和场景评价、报告重复训练及运行开销，并通过可靠性与融合权重诊断检验模块作用。本文关注参考视图深度估计，结论不延伸到未经评价的三维重建质量。

## 7 结论

本文围绕 Vis-MVSNet 三级结构在大视差与复杂遮挡下的匹配困难，构建逐源视图遮挡感知监督、自适应深度搜索范围和深度假设感知源视图融合方法。三个环节分别从监督、候选生成与源视图聚合角度处理困难匹配。按本文八组实验配置，五视图训练时完整方法相对 Base 的整体、大视差、任一源遮挡及交集 Abs 分别降低 23.43%、31.34%、16.17% 和 20.75%；三视图训练时分别降低 18.35%、27.86%、12.99% 和 18.10%。

三类场景的真实预测对照显示，部分大误差背景、遮挡轮廓和交集区域得到改善，同时仍存在局部退化和平均误差与严格阈值准确率之间的取舍。本文据此将结论限定为当前评测条件下目标困难区域的深度误差改善，保留不同模块组合和不同误差尺度下的差异。

'''


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    ASSETS.mkdir(parents=True, exist_ok=True)
    population = audit_population()
    cases, details, tails = load_cases()
    write_csv(ASSETS/'case_region_metrics.csv', details)
    write_csv(ASSETS/'case_error_distribution.csv', tails)
    for path in (DOCS/'paper_complete_assets').glob('*.png'):
        shutil.copy2(path, ASSETS/path.name)
    original = SOURCE.read_text(encoding='utf-8')
    body, bibliography = original.split('## 参考文献', 1)
    appendix = body.split('## 附录 A：全部区域与全部指标', 1)[1]
    introduction = body.split('## 4 实验设置', 1)[0]
    introduction = introduction.replace('# 面向大视差与遮挡的 Vis-MVSNet 深度估计改进方法',
                                        '# 面向大视差与复杂遮挡的 Vis-MVSNet 深度估计改进方法', 1)
    introduction = introduction.replace('本文以原始 Vis-MVSNet 为基线，', '本文以 Vis-MVSNet 三级深度估计框架为研究基础，')
    introduction = introduction.replace('结果说明，三个模块能够从可见性学习、候选搜索和多视图融合三个环节改善困难区域的深度估计。',
        '实际预测的三类困难场景对照进一步显示了误差降低与局部退化的空间分布。实验支持完整配置相对 Base 的困难区域平均误差改善，同时揭示不同组合在平均误差与阈值准确率之间的取舍。')
    introduction = introduction.replace('结果说明，三个模块', '结果表明，完整方法')
    introduction = introduction.replace('构成统一方法。A 所学习的几何可见性表征改善匹配特征，B 为前级偏差保留修正空间，C 在该空间中进一步选择与每个深度假设一致的源视图证据。通过这种设计，三级结构的计算优势得以保留，而困难区域中的误差传播受到针对性约束。',
        '构成统一设计。A 旨在改善可靠性学习，B 旨在为前级偏差保留修正空间，C 在候选空间中学习逐深度假设的源视图贡献。保留三级结构使方法能够沿用由粗到细的估计流程，具体收益由后文实验检验。')
    results = '## 5 实验结果与分析'+body.split('## 5 实验结果与分析', 1)[1].split('## 6 讨论', 1)[0]
    results = results.replace('这些区域的降幅均具有明确幅度，表明改进并非只作用于普通像素，而是集中覆盖了本文关注的困难匹配条件。',
        '这表明改善覆盖了本文关注的困难匹配区域。该比较评价区域均值，不要求每个困难像素均改善。')
    results = results.replace('验证了候选范围调整与深度覆盖改善之间的联系。',
        '提供了该配置变化与深度覆盖改善同时出现的观察。覆盖率本身不证明自适应范围的内部机制。')
    results = results.replace('说明显式遮挡监督对边界附近的不可靠观测具有积极作用。',
        '表明 A 对应配置在这些区域取得了平均误差收益。')
    results = results.replace('该结果表明，A 的加入更有利于边界和遮挡相关像素，但可能改变全图误差尾部，因此完整模型无需在所有汇总指标上同时超过每一个双模块组合。',
        '这只支持该对照中边界 Abs 和边界遮挡 Acc4 的局部收益，不能据此推断 A 对所有遮挡区域均有利。完整组合在整体 Abs 上并未超过该双模块组合，应保留这一结果。')
    results = results.replace('某一组合提高 Acc4 而 Abs 略有增大，表示更多像素跨入阈值范围，但剩余大误差像素的幅度可能增加。',
        '某一组合提高 Acc4 而 Abs 增大，意味着进入阈值范围的像素净比例提高，但阈值内外的误差幅度变化仍需配对分布验证。')
    manuscript = introduction + EXPERIMENTS + results + build_qualitative(cases, tails) + DISCUSSION
    manuscript += '## 附录 A：全测试集全部区域与全部指标\n\n'+appendix.strip()+'\n\n'
    manuscript += '## 附录 B：其余六个真实预测案例\n\n每类第二个 scan 的案例使用与正文相同的布局、固定困难掩码和颜色规则，指标已包含于表8—表10。\n\n'
    number = 13
    for category in CATEGORIES:
        for series in SERIES:
            c = next(c for c in cases if c['category'] == category and c['series'] == series and c['rank'] == 2)
            manuscript += figure(c, number); number += 1
    manuscript += '## 附录 C：可视化重新导出的数值复核\n\n'
    manuscript += '全测试集表来自原始完整评测；图7—图18及表8—表11来自本次十二例的重新导出。每个设置和配置核对六例、七个原评测区域、十项指标，共420项。采用绝对容差0.001及相对容差0.0001，结果如下；未用原 CSV 数值替换本次图像对应指标。\n\n'
    reproduction = []
    for series in SERIES:
        for label in ('Base', 'Base+A+B+C'):
            path = EXPORTS/series/label/'exported_metrics/reproduction_status.json'
            status = json.loads(path.read_text(encoding='utf-8'))
            worst = {x['metric']: x for x in status['worst_differences']}
            reproduction.append({'series': series, 'config': label,
                                 'failed_checks': status['failed_checks'], 'total_checks': status['total_checks'],
                                 'max_abs_delta_mm': worst['abs']['max_abs_delta'],
                                 'max_acc2_delta_pp': worst['acc2']['max_abs_delta']*100})
    manuscript += '**表12. 原评测与可视化重新导出的复核差异。**\n\n'+table(
        ['设置', '配置', '超容差项/核对项', 'Abs最大差/mm', 'Acc2最大差/pp'],
        [[x['series'], x['config'], f"{x['failed_checks']}/{x['total_checks']}",
          f"{x['max_abs_delta_mm']:.6f}", f"{x['max_acc2_delta_pp']:.4f}"] for x in reproduction])+'\n\n'
    manuscript += '上述差异没有被视为严格复现成功。案例表已从所示预测数组重新核算 Abs、Acc2/4/8 与固定区域像素数；可视化导出的区域统计不与原全测试集汇总混合。\n\n'
    manuscript += '## 参考文献'+bibliography
    manuscript = manuscript.replace(
        '**Large Scale Multi-view Stereopsis Evaluation**. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.',
        '**Large Scale Multi-view Stereopsis Evaluation**. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, 406–413.')
    manuscript = manuscript.replace('](paper_complete_assets/', '](paper_depth_visual_assets/')
    manuscript = manuscript.replace('图1 三级 Vis-MVSNet 详细总体架构', '图1 三级深度估计总体架构')
    OUTPUT.write_text(manuscript, encoding='utf-8')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    from check_paper_integrity import check
    equations, references, figures = check(OUTPUT)
    report = {'source_manuscript_sha256': source_hash, 'source_manuscript_unchanged': True,
              'output': OUTPUT.relative_to(ROOT).as_posix(), 'equations': equations,
              'references': references, 'figures': figures, 'real_cases': len(cases),
              'case_arrays_vs_csv_checked': True, 'case_region_rows': len(details),
              'manuscript_appendix_metric_cells_checked': check_manuscript_metric_cells(manuscript),
              'population_checks': population, 'reproduction': reproduction,
              'baseline_identity_verified': False,
              'identity_note': 'Analysis labels retained. Raw B switch is inverted relative to analysis B; no original-baseline identity or causal switch validation is implied.'}
    (ASSETS/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = ['# 本版论文数据与实验身份核对记录\n',
             '该文件独立于论文正文，保留数据出处和投稿前必须核定的事实，不改写原始 CSV。\n',
             '新稿只讨论深度估计，Base 是当前数值参照配置；不将当前映射写成已经核定的原始 Vis-MVSNet 复现。\n',
             '当前模型/CSV 的原始 B 开关与论文分析 B 位取反。显示名不会改变检查点的网络。因而表格可作为现有实验的配置比较，但模块启用与原始基线的因果身份仍需核对训练命令、Git commit 和检查点。该问题不可用图像效果或配置改名解决。\n',
             '方法章节继承现有稿中已核对的数学定义；损失权重和训练超参数未由推理元数据推断。完整方法设计不等于已证明各模块在每个组合中均有效。\n',
             '全测试集来源：eval/ablation_test_light3 与 eval/ablation_test_view3_light3。两套各八组、1078样本，已核对配对键/区域像素和逐图像素加权的十项指标。\n',
             '定性来源：outputs/paper_visualizations_ranked。View3、View5各六例；按原全图 Abs 收益排序的案例选择是收益导向，不代表随机或无偏抽样。案例新指标从数组核算，不替换旧表。\n',
             '原稿 PAPER_COMPLETE_WITH_RESULTS.md 保持不变。新稿图片包复制现有框架/统计图和十二例真实导出 PNG/PDF，未修改预测或误差颜色。\n',
             '全部区域案例指标：paper_depth_visual_assets/case_region_metrics.csv；原始误差分布：case_error_distribution.csv；逐项核对摘要：validation.json。\n',
             '投稿前需统一实验身份、确认训练超参数，增加统一协议相关方法对比和运行开销；若只主张深度估计，不要求以点云指标补足该任务。\n']
    (DOCS/'PAPER_DEPTH_ESTIMATION_EVIDENCE_AUDIT.md').write_text('\n'.join(audit), encoding='utf-8')
    print(f'Written: {OUTPUT}\nCopied real-case figures: {len(cases)}\nVerified paired populations: {len(population)}')


if __name__ == '__main__':
    main()
