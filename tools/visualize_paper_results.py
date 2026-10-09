#!/usr/bin/env python3
"""Paper figures from existing CSVs and actual checkpoint outputs.

stats: plots and neutral sample selection from all eight CSV configurations.
export: invoke the existing evaluator with actual checkpoints on a server.
render: shared-scale depth/error/coverage panels from exported NPZs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.paper_configs import CONFIGS, BY_NAME, validate_csv_identity

SERIES = {'View5': 'ablation_test_light3', 'View3': 'ablation_test_view3_light3'}
REGIONS = ('full', 'boundary', 'large_disparity', 'occluded_any',
           'occluded_majority', 'large_disp_and_occluded', 'boundary_and_occluded')
LABELS = ('All', 'Boundary', 'Large disparity', 'Any occlusion',
          'Majority occlusion', 'Disparity + occlusion', 'Boundary + occlusion')
METRICS = ('abs', 'acc2', 'acc4', 'acc8', 'stage1_in_range', 'stage2_in_range',
           'stage3_in_range', 'stage1_range_width', 'stage2_range_width', 'stage3_range_width')
PERCENT = {'acc2', 'acc4', 'acc8', 'stage1_in_range', 'stage2_in_range', 'stage3_in_range'}
PAIRS = (('Base', 'Base+A', 'A'), ('Base+B', 'Base+A+B', 'A'),
         ('Base+C', 'Base+A+C', 'A'), ('Base+B+C', 'Base+A+B+C', 'A'),
         ('Base', 'Base+B', 'B'), ('Base+A', 'Base+A+B', 'B'),
         ('Base+C', 'Base+B+C', 'B'), ('Base+A+C', 'Base+A+B+C', 'B'),
         ('Base', 'Base+C', 'C'), ('Base+A', 'Base+A+C', 'C'),
         ('Base+B', 'Base+B+C', 'C'), ('Base+A+B', 'Base+A+B+C', 'C'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_key(row):
    return row['scan'], int(row['view']), int(row['light'])


def sample_id(key):
    scan, view, light = key
    return f'{scan}_view{view:02d}_light{light}'


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def load_series(eval_root, series):
    folder = Path(eval_root) / SERIES[series]
    per_image, summaries = {}, {}
    for config in CONFIGS:
        matches = []
        for path in folder.glob('*/summary_metrics.csv'):
            rows = read_csv(path)
            if rows and rows[0]['model_type'] == config.model_type:
                matches.append((path, rows))
        if len(matches) != 1:
            raise ValueError(f'{series}/{config.name}: expected one result source, found {len(matches)}')
        path, summary = matches[0]
        for row in summary:
            validate_csv_identity(row, config)
        summaries[config.name] = {r['region']: r for r in summary if r['aggregation'] == 'pixel_weighted'}
        if set(summaries[config.name]) != set(REGIONS):
            raise ValueError(f'{config.name}: incomplete regions')
        rows = read_csv(path.parent / 'all_metrics.csv')
        indexed = {}
        for row in rows:
            validate_csv_identity(row, config)
            key = (*sample_key(row), row['region'])
            if key in indexed:
                raise ValueError(f'Duplicate CSV sample: {key}')
            indexed[key] = row
        per_image[config.name] = indexed
        # Prove that sample plots and global tables use the same population.
        for region in REGIONS:
            rr = [r for r in rows if r['region'] == region]
            total = sum(int(r['pixels']) for r in rr)
            sr = summaries[config.name][region]
            if total != int(sr['pixels']) or len(rr) != int(sr['images']):
                raise ValueError(f'{config.name}/{region}: sample population differs from summary')
            for metric in METRICS:
                weighted = sum(float(r[metric])*int(r['pixels']) for r in rr)/total
                if not np.isclose(weighted, float(sr[metric]), atol=1e-5, rtol=1e-6):
                    raise ValueError(f'{config.name}/{region}/{metric}: summary mismatch')
    baseline = per_image['Base']
    for name, indexed in per_image.items():
        if set(indexed) != set(baseline):
            raise ValueError(f'{name}: unequal sample/region keys')
        for key, row in indexed.items():
            if int(row['pixels']) != int(baseline[key]['pixels']):
                raise ValueError(f'{name}: unequal region mask pixel population {key}')
    return per_image, summaries


def stat_figures(args, series):
    per_image, summary = load_series(args.eval_root, series)
    folder = Path(args.outdir) / series
    names = [c.name for c in CONFIGS]
    display_rows = []
    for config in CONFIGS:
        for region in REGIONS:
            row = summary[config.name][region]
            base = float(summary['Base'][region]['abs'])
            display_rows.append({'analysis_config': config.name, 'analysis_code': config.analysis_code,
                                 'raw_model_type': config.model_type, 'raw_csv_code': config.csv_code,
                                 'region': region, 'pixels': row['pixels'], 'images': row['images'],
                                 **{m: float(row[m]) for m in METRICS},
                                 'abs_reduction_pct': 100*(base-float(row['abs']))/base,
                                 **{m+'_delta_pp': 100*(float(row[m])-float(summary['Base'][region][m])) for m in PERCENT}})
        values = np.array([float(summary[config.name][r]['abs']) for r in REGIONS])
        bases = np.array([float(summary['Base'][r]['abs']) for r in REGIONS])
        fig, ax = plt.subplots(figsize=(10, 3.7))
        x = np.arange(len(REGIONS))
        ax.bar(x-.18, bases, .36, label='Base', color='#94a3b8')
        ax.bar(x+.18, values, .36, label=config.name, color='#0f766e')
        ax.set_xticks(x, LABELS, rotation=18, ha='right')
        ax.set_ylabel('Pixel-weighted Abs (mm)'); ax.legend(); ax.grid(axis='y', alpha=.2)
        ax.set_title(series + ' | ' + config.name)
        save_figure(fig, folder/config.name/'region_abs')
    write_csv(folder/'comparison'/'global_metrics.csv', display_rows)
    gain = np.array([[100*(float(summary['Base'][r]['abs'])-float(summary[n][r]['abs']))/
                       float(summary['Base'][r]['abs']) for r in REGIONS] for n in names])
    fig, ax = plt.subplots(figsize=(13, 5))
    limit = max(1, np.abs(gain).max())
    img = ax.imshow(gain, cmap='RdYlGn', aspect='auto',
                    norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(7), ('All','Boundary','Large\ndisparity','Any\nocclusion',
                           'Majority\nocclusion','Disparity +\nocclusion','Boundary +\nocclusion'))
    ax.set_yticks(range(8), names, fontsize=8)
    for i in range(8):
        for j in range(7):
            ax.text(j, i, f'{gain[i,j]:+.2f}%', ha='center', va='center', fontsize=8)
    fig.colorbar(img, ax=ax, label='Abs reduction versus Base (%)')
    ax.set_title(series + ' | all eight analysis configurations')
    save_figure(fig, folder/'comparison'/'region_gain')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ('acc2', 'acc4', 'acc8')):
        values = [100*float(summary[n]['full'][metric]) for n in names]
        ax.bar(range(8), values, color='#2563eb'); ax.set_ylim(0,100)
        ax.set_xticks(range(8), names, rotation=55, ha='right', fontsize=8)
        ax.set_title(metric.upper()); ax.set_ylabel('Accuracy (%)')
    save_figure(fig, folder/'comparison'/'threshold_accuracy')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    colors = plt.get_cmap('tab10').colors
    markers = ('o','s','^','D','v','P','X','*')
    for ax, stage in zip(axes, (2,3)):
        xx = [float(summary[n]['full'][f'stage{stage}_range_width']) for n in names]
        yy = [100*float(summary[n]['full'][f'stage{stage}_in_range']) for n in names]
        for i,(x,y,n) in enumerate(zip(xx, yy, names)):
            ax.scatter(x,y,s=55,color=colors[i],marker=markers[i],label=n,edgecolors='white',linewidths=.5)
        ax.set_xlabel('Width / original depth interval'); ax.set_ylabel('GT coverage (%)')
        ax.set_title(f'Stage {stage}: coverage and width'); ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.subplots_adjust(bottom=.24,wspace=.22)
    save_figure(fig, folder/'comparison'/'coverage_width')
    # Reaggregate by scan, preserving pixel weights within each scan.
    scans = sorted({key[0] for key in per_image['Base']}, key=lambda s: int(s.replace('scan','')))
    paired = []
    for scan in scans:
        for region in REGIONS:
            subset = {n: [r for k,r in per_image[n].items() if k[0] == scan and k[3] == region] for n in names}
            if not subset['Base']:
                continue
            means = {n: sum(float(r['abs'])*int(r['pixels']) for r in rr)/sum(int(r['pixels']) for r in rr)
                     for n, rr in subset.items()}
            paired.append({'scan': scan, 'region': region, **means,
                           'ours_reduction_pct': 100*(means['Base']-means['Base+A+B+C'])/means['Base']})
    write_csv(folder/'comparison'/'per_scan_abs.csv', paired)
    fig, ax = plt.subplots(figsize=(12, 4))
    rr = [r for r in paired if r['region']=='large_disp_and_occluded']
    ax.bar(range(len(rr)), [r['ours_reduction_pct'] for r in rr],
           color=['#0f766e' if r['ours_reduction_pct'] >= 0 else '#b91c1c' for r in rr])
    ax.set_xticks(range(len(rr)), [r['scan'] for r in rr], rotation=55)
    ax.axhline(0, color='black', lw=.8); ax.set_ylabel('Abs reduction versus Base (%)')
    ax.set_title('Per-scan disparity + occlusion: improvements and regressions')
    save_figure(fig, folder/'comparison'/'scan_improvements_regressions')
    selections, ranking = select_examples(per_image)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'selection.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')
    write_csv(folder/'comparison'/'sample_ranking.csv', ranking)
    (folder/'mapping.json').write_text(json.dumps([vars(c) for c in CONFIGS], indent=2), encoding='utf-8')
    print(f'{series}: eight configurations verified; figures and selection -> {folder}')


def select_examples(per_image):
    ranking = []
    for key, base in per_image['Base'].items():
        ours = per_image['Base+A+B+C'][key]
        ranking.append({'scan': key[0], 'view': key[1], 'light': key[2], 'region': key[3],
                        'pixels': int(base['pixels']), 'base_abs': float(base['abs']),
                        'ours_abs': float(ours['abs']),
                        'abs_change_mm': float(ours['abs'])-float(base['abs'])})
    chosen = []
    for region in ('large_disparity', 'occluded_any', 'large_disp_and_occluded', 'boundary_and_occluded'):
        candidates = sorted([r for r in ranking if r['region'] == region and r['pixels'] >= 100],
                            key=lambda r: (r['base_abs'],r['scan'],r['view']))
        if not candidates:
            continue
        chosen.append((candidates[len(candidates)//2], f'median Base Abs in {region}; not selected by gain'))
        if region == 'large_disp_and_occluded':
            chosen.append((candidates[min(len(candidates)-1,int(.9*len(candidates)))], '90th percentile Base joint-region Abs'))
            chosen.append((max(candidates, key=lambda r:r['abs_change_mm']), 'largest joint-region regression; failure analysis'))
    selections = []
    seen = set()
    for row, reason in chosen:
        key = row['scan'],row['view'],row['light']
        if key in seen:
            continue
        seen.add(key)
        selections.append({'scan':key[0], 'view':key[1], 'light':key[2], 'reason':reason,
                           'roi':None, 'probe':None})
    return selections, ranking


def export(args):
    series = args.series
    selection = Path(args.selection or Path(args.outdir)/series/'selection.json').resolve()
    if not selection.is_file():
        raise FileNotFoundError('Run stats first or provide --selection JSON')
    selected = json.loads(selection.read_text(encoding='utf-8'))
    if not selected:
        raise ValueError('Empty sample selection')
    if len({int(x['light']) for x in selected}) != 1:
        raise ValueError('One export must use a single light, matching the original CSV protocol')
    overrides = json.loads(Path(args.eval_args_json).read_text()) if args.eval_args_json else {}
    manifest = json.loads(Path(args.checkpoint_manifest).read_text()) if args.checkpoint_manifest else {}
    _, summaries = load_series(args.eval_root, series)
    configs = [BY_NAME[n] for n in args.configs] if args.configs else CONFIGS
    for config in configs:
        source = summaries[config.name]['full']
        checkpoint = manifest.get(config.name)
        if checkpoint is None:
            # Checkpoint identity comes from CSV. Only relocate its directory.
            csv_path = Path(source['checkpoint'])
            checkpoint = (Path(args.checkpoint_root)/csv_path.parent.name/csv_path.name
                          if args.checkpoint_root else csv_path)
        checkpoint = Path(checkpoint).resolve()
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(f'{config.name}: checkpoint absent: {checkpoint}')
        folder = Path(args.outdir)/series
        metrics_dir = folder/config.name/'exported_metrics'
        cmd = [sys.executable, str(ROOT/'tools'/'eval_region_metrics_dtu_yao.py'),
               '--model_type', config.model_type, '--loadckpt', str(checkpoint),
               '--label', config.name, '--testpath', str(Path(args.testpath).resolve()),
               '--testlist', str(Path(args.testlist).resolve()), '--outdir', str(metrics_dir.resolve()),
               '--paper_dumpdir', str(folder.resolve()), '--paper_sample_keys', str(selection),
               '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers),
               '--eval_nviews', source['eval_nviews'], '--region_nviews', source['region_nviews'],
               '--light', str(selected[0]['light']), '--vismode', 'soft']
        for name in ('range_sigma_scale','range_min_scale','range_max_scale','hypothesis_residual_scale'):
            cmd.extend(['--'+name, source[name]])
        forbidden = {'model_type','loadckpt','outdir','paper_dumpdir','paper_sample_keys','label','vismode',
                     'eval_nviews','region_nviews','light'}
        if forbidden & set(overrides):
            raise ValueError('eval_args_json cannot override identity/output/protocol: '+str(forbidden & set(overrides)))
        for key, value in overrides.items():
            cmd.extend(['--'+key, str(value)])
        print(' '.join(json.dumps(x) if ' ' in x else x for x in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)
            verify_export(args, series, config, metrics_dir)


def verify_export(args, series, config, folder):
    original, _ = load_series(args.eval_root, series)
    exported = read_csv(folder/'all_metrics.csv')
    if not exported:
        raise RuntimeError('Export produced no metrics')
    checks = []
    for row in exported:
        key = (*sample_key(row), row['region'])
        if key not in original[config.name]:
            raise ValueError(f'Exported sample absent from original evaluation: {key}')
        reference = original[config.name][key]
        if int(row['pixels']) != int(reference['pixels']):
            raise ValueError('Exported region population differs from original evaluation')
        for metric in METRICS:
            delta = float(row[metric])-float(reference[metric])
            good = np.isclose(float(row[metric]), float(reference[metric]), atol=args.verify_atol, rtol=1e-4)
            checks.append({'sample': sample_id(sample_key(row)), 'region': row['region'],
                           'metric': metric, 'original':reference[metric], 'exported':row[metric],
                           'delta':delta, 'within_tolerance':bool(good)})
    write_csv(folder/'reproduction_check.csv', checks)
    if not all(r['within_tolerance'] for r in checks):
        raise RuntimeError('New predictions differ from CSV; see reproduction_check.csv. '
                           'Check checkpoint, input, stage candidate counts, intervals and tolerances.')


def crop_bounds(meta, arrays):
    h,w = arrays['gt'].shape
    roi = meta.get('roi')
    if roi is None:
        mask = arrays['mask_large_disp_and_occluded'].astype(bool)
        if not mask.any():
            mask = arrays['valid'].astype(bool)
        yy,xx = np.where(mask)
        if not len(yy):
            raise ValueError('No valid ROI')
        x,y = int(np.median(xx)),int(np.median(yy))
        cw,ch = min(160,w),min(120,h)
        x0,y0 = max(0,min(w-cw,x-cw//2)),max(0,min(h-ch,y-ch//2))
        roi = [x0,y0,x0+cw,y0+ch]
    x0,y0,x1,y1 = map(int,roi)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI must be [x0,y0,x1,y1] inside the image')
    return x0,y0,x1,y1


def map_axis(ax, values, mask, title, cmap='magma', vmin=0, vmax=None):
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap,str) else cmap
    cm.set_bad('#d1d5db')
    im = ax.imshow(np.ma.masked_where(~mask, values), cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    ax.set_title(title, fontsize=9); ax.axis('off')
    return im


def region_rows(arrays, config, key, roi_mask):
    error = np.abs(arrays['pred']-arrays['gt'])
    rows = []
    for region in REGIONS:
        for label, extra in (('full_image', np.ones_like(roi_mask)), ('shared_roi',roi_mask)):
            mask = arrays['mask_'+region].astype(bool) & extra
            count = int(mask.sum())
            row = {'analysis_config':config, 'sample':key, 'scope':label,'region':region,'pixels':count}
            row.update({m:float('nan') for m in METRICS})
            if count:
                values = error[mask]
                row.update(abs=float(values.mean()), acc2=float((values<2).mean()),
                           acc4=float((values<4).mean()), acc8=float((values<8).mean()))
                for stage in (1,2,3):
                    row[f'stage{stage}_in_range'] = float(arrays[f'stage{stage}_coverage'][mask].mean())
                    row[f'stage{stage}_range_width'] = float(arrays[f'stage{stage}_width'][mask].mean())
            rows.append(row)
    return rows


def render_sample(args, series, key, files):
    names = [c.name for c in CONFIGS if c.name in files]
    arrays = {}
    for n in names:
        with np.load(files[n]/'arrays.npz', allow_pickle=False) as loaded:
            arrays[n] = dict(loaded)
    metadata = {n:json.loads((files[n]/'metadata.json').read_text(encoding='utf-8')) for n in names}
    reference = arrays['Base']
    for name in names:
        if metadata[name]['analysis_config'] != name:
            raise ValueError('Folder and analysis name differ')
        for field in ('gt','rgb','valid',*('mask_'+r for r in REGIONS)):
            if not np.array_equal(reference[field], arrays[name][field]):
                raise ValueError(f'{key}/{name}: unaligned input/GT/region masks ({field})')
        if metadata[name]['probe'] != metadata['Base']['probe']:
            raise ValueError(f'{key}/{name}: probe coordinates differ')
    x0,y0,x1,y1 = crop_bounds(metadata['Base'], reference)
    sl = np.s_[y0:y1,x0:x1]
    roi_mask = np.zeros_like(reference['valid'],dtype=bool); roi_mask[sl]=True
    valid = reference['valid'].astype(bool)
    gt = reference['gt']
    if not valid.any():
        raise ValueError('No valid GT pixels')
    dmin,dmax = float(gt[valid].min()),float(gt[valid].max())
    errmax = args.error_max
    out = Path(args.outdir)/series/'comparison'/key
    fig, axes = plt.subplots(2, 4, figsize=(14,7))
    axes.flat[0].imshow(reference['rgb']); axes.flat[0].axis('off'); axes.flat[0].set_title('Reference RGB')
    im = map_axis(axes.flat[1],gt,valid,'GT depth (mm)','viridis',dmin,dmax)
    fig.colorbar(im,ax=axes.flat[1],shrink=.65)
    map_axis(axes.flat[2],reference['disparity'],valid,'GT reprojection displacement (pixels)',vmax=float(reference['disparity'][valid].max()))
    map_axis(axes.flat[3],reference['occlusion_ratio'],valid,'GT source occlusion ratio',vmax=1)
    for ax, region, title in zip(axes.flat[4:],REGIONS[2:6],LABELS[2:6]):
        map_axis(ax,reference['mask_'+region],valid,title,'Greys',0,1)
    save_figure(fig,out/'region_definition')
    # Same RGB, GT, crops, scales and masks in all eight panels.
    fig, axes = plt.subplots(len(names)+1, 4, figsize=(13, 2.1*(len(names)+1)))
    axes[0,0].imshow(reference['rgb']); axes[0,0].set_title('Reference RGB')
    axes[0,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#22c55e',lw=1.5))
    map_axis(axes[0,1],gt,valid,'GT depth','viridis',dmin,dmax)
    axes[0,2].imshow(reference['rgb'][sl]); axes[0,2].set_title('Shared ROI RGB')
    map_axis(axes[0,3],gt[sl],valid[sl],'Shared ROI GT','viridis',dmin,dmax)
    for ax in axes[0]: ax.axis('off')
    allrows = []
    for i,name in enumerate(names,1):
        arr = arrays[name]; error = np.abs(arr['pred']-gt)
        rows = region_rows(arr,name,key,roi_mask); allrows.extend(rows)
        metric = next(r for r in rows if r['scope']=='shared_roi' and r['region']=='full')
        map_axis(axes[i,0],arr['pred'],valid,name+' | depth','viridis',dmin,dmax)
        ei = map_axis(axes[i,1],error,valid,'Absolute error (mm)','magma',0,errmax)
        map_axis(axes[i,2],arr['pred'][sl],valid[sl],'ROI depth','viridis',dmin,dmax)
        map_axis(axes[i,3],error[sl],valid[sl],f"ROI Abs {metric['abs']:.3f} mm | Acc2 {metric['acc2']*100:.1f}%",'magma',0,errmax)
        cfg_dir = files[name]/'figures'
        fig1, ax1 = plt.subplots(1,3,figsize=(11,3))
        map_axis(ax1[0],arr['pred'],valid,name+' depth','viridis',dmin,dmax)
        map_axis(ax1[1],error,valid,'Absolute error (mm)','magma',0,errmax)
        im2 = map_axis(ax1[2],error[sl],valid[sl],'Shared ROI error','magma',0,errmax)
        fig1.colorbar(im2,ax=list(ax1),shrink=.65,label='Absolute error (mm)')
        save_figure(fig1,cfg_dir/'depth_error')
        fig2,ax2=plt.subplots(2,3,figsize=(12,6))
        for col,stage in enumerate((1,2,3)):
            widthmax=max(float(arrays[n][f'stage{stage}_width'][valid].max()) for n in names)
            map_axis(ax2[0,col],arr[f'stage{stage}_coverage'],valid,f'S{stage} coverage',
                     ListedColormap(['#ef4444','#22c55e']),0,1)
            wi=map_axis(ax2[1,col],arr[f'stage{stage}_width'],valid,f'S{stage} width / base interval',
                        'viridis',0,widthmax)
            fig2.colorbar(wi,ax=ax2[1,col],shrink=.65)
        fig2.suptitle(name+' | shared width scales across configurations')
        save_figure(fig2,cfg_dir/'stage_coverage_width')
        write_csv(files[name]/'region_metrics.csv',rows)
    fig.colorbar(ei,ax=list(axes[:,3]),shrink=.3,label=f'Absolute error (mm); values above {errmax:g} clipped for display')
    save_figure(fig,out/'eight_config_depth_error')
    base_error = np.abs(reference['pred']-gt)
    fig, axes = plt.subplots(2,4,figsize=(13,7))
    for ax,name in zip(axes.flat,names):
        gain = base_error-np.abs(arrays[name]['pred']-gt)
        im = map_axis(ax,gain,valid,name,'RdBu',-args.gain_max,args.gain_max)
    for ax in list(axes.flat)[len(names):]:
        ax.axis('off')
    fig.colorbar(im,ax=list(axes.flat),shrink=.65,label='Base error minus configuration error (mm); blue = improvement')
    save_figure(fig,out/'error_gain')
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    compare_names = ['Base','Base+A+B+C'] if 'Base+A+B+C' in names else [names[0],names[-1]]
    for row,name in enumerate(compare_names):
        arr=arrays[name]
        for col,stage in enumerate((1,2,3)):
            current=arr[f'stage{stage}_coverage'].astype(bool)
            cover = 100*float(current[valid].mean())
            map_axis(axes[row,col],current,valid,f'{name} | S{stage} coverage {cover:.2f}%',ListedColormap(['#ef4444','#22c55e']),0,1)
    save_figure(fig,out/'stage_coverage')
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    px,py=metadata['Base']['probe']; probe_gt=float(gt[py,px])
    for ax,stage in zip(axes,(1,2,3)):
        for i,name in enumerate(names):
            candidates=arrays[name][f'stage{stage}_probe_candidates']
            ax.plot([candidates[0],candidates[-1]],[i,i],lw=2)
            ax.scatter(candidates,np.full(len(candidates),i),s=8)
        ax.set_yticks(range(len(names)),names,fontsize=7)
        if valid[py,px]: ax.axvline(probe_gt,color='black',ls='--',label='GT')
        ax.set_xlabel('Candidate depth (mm)'); ax.set_title(f'Stage {stage} | probe ({px},{py})')
    save_figure(fig,out/'probe_search_intervals')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax,region in zip(axes,('full','large_disp_and_occluded')):
        mask=reference['mask_'+region].astype(bool)
        for name in names:
            values=np.sort(np.abs(arrays[name]['pred']-gt)[mask])
            if len(values): ax.plot(values,100*np.arange(1,len(values)+1)/len(values),label=name,lw=1)
        ax.set_xlim(0,args.cdf_max); ax.set_ylim(0,100); ax.set_xlabel('Absolute depth error (mm)')
        ax.set_ylabel('Pixels with error below threshold (%)'); ax.set_title(region)
        for threshold in (2,4,8): ax.axvline(threshold,color='gray',lw=.5,ls='--')
    axes[1].legend(fontsize=7); save_figure(fig,out/'error_cdf')
    add_deltas(allrows)
    write_csv(out/'metrics_full_and_roi.csv',allrows)
    write_csv(out/'single_factor_comparisons.csv',factor_rows(allrows))
    (out/'display_settings.json').write_text(json.dumps({'roi':[x0,y0,x1,y1],'depth_min_mm':dmin,
        'depth_max_mm':dmax,'error_max_mm':errmax,'gain_max_mm':args.gain_max,'configs':names},indent=2),encoding='utf-8')
    for name in names:
        render_fusion(args,files[name],arrays[name],metadata[name],valid)
    print(f'{series}/{key}: aligned panels, ROI metrics, CDF and module diagnostics saved')


def add_deltas(rows):
    index={(r['analysis_config'],r['scope'],r['region']):r for r in rows}
    for r in rows:
        base=index[('Base',r['scope'],r['region'])]
        r['abs_reduction_pct']=100*(base['abs']-r['abs'])/base['abs'] if base['abs']>0 else float('nan')
        for m in PERCENT: r[m+'_delta_pp']=100*(r[m]-base[m])


def factor_rows(rows):
    index={(r['analysis_config'],r['scope'],r['region']):r for r in rows}
    result=[]
    for before,after,factor in PAIRS:
        for scope in ('full_image','shared_roi'):
            for region in REGIONS:
                a=index.get((before,scope,region)); b=index.get((after,scope,region))
                if a is None or b is None: continue
                result.append({'added_factor':factor,'before':before,'after':after,'scope':scope,'region':region,
                    'before_abs':a['abs'],'after_abs':b['abs'],
                    'abs_reduction_pct':100*(a['abs']-b['abs'])/a['abs'] if a['abs']>0 else float('nan'),
                    **{m+'_delta_pp':100*(b[m]-a[m]) for m in PERCENT}})
    return result


def render_fusion(args, folder, arr, meta, valid):
    sources=meta['source_views']; stage=3
    # Only trained visibility heads with raw A enabled are evidence for A.
    raw_a=meta['raw_csv_code'][0]=='1'
    rows=[]
    fig,axes=plt.subplots(len(sources),3,figsize=(10,2.4*len(sources)),squeeze=False)
    for i,source in enumerate(sources):
        supervised=arr['source_supervised'][i].astype(bool)&valid
        visible=arr['source_visible'][i].astype(bool)
        map_axis(axes[i,0],visible,supervised,f'Source {source} GT visible','Greys',0,1)
        weights=arr[f'stage{stage}_weights_at_gt'][i]
        map_axis(axes[i,1],weights,valid,'Fusion weight at nearest GT candidate','viridis',0,1)
        prediction=arr[f'stage{stage}_visibility_pred'][i]
        predicted_visible=prediction>=.5
        tp=int((predicted_visible&visible&supervised).sum())
        fp=int((predicted_visible&(~visible)&supervised).sum())
        fn=int(((~predicted_visible)&visible&supervised).sum())
        tn=int(((~predicted_visible)&(~visible)&supervised).sum())
        if raw_a:
            map_axis(axes[i,2],prediction,supervised,'Trained visibility probability','viridis',0,1)
        else:
            axes[i,2].axis('off'); axes[i,2].text(.5,.5,'Visibility head has no A supervision',ha='center',va='center',fontsize=9)
        for state,mask in (('visible',visible&supervised),('occluded',(~visible)&supervised)):
            count=int(mask.sum())
            rows.append({'source_view':source,'state':state,'pixels':count,
                         'mean_fusion_weight':float(weights[mask].mean()) if count else float('nan'),
                         'mean_visibility_probability':float(prediction[mask].mean()) if count and raw_a else float('nan'),
                         'visibility_accuracy':(tp+tn)/(tp+fp+fn+tn) if raw_a and tp+fp+fn+tn else float('nan'),
                         'visible_precision':tp/(tp+fp) if raw_a and tp+fp else float('nan'),
                         'visible_recall':tp/(tp+fn) if raw_a and tp+fn else float('nan'),
                         'occluded_recall':tn/(tn+fp) if raw_a and tn+fp else float('nan')})
    fig.suptitle(meta['analysis_config']+' | GT-based diagnostics; GT is not an inference input',fontsize=10)
    save_figure(fig,folder/'figures'/'source_fusion_diagnostics')
    write_csv(folder/'source_visibility_metrics.csv',rows)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    px,py=meta['probe']; gt=float(arr['gt'][py,px])
    for ax,stage in zip(axes,(1,2,3)):
        candidates=arr[f'stage{stage}_probe_candidates']; weights=arr[f'stage{stage}_probe_weights']
        for source,weight in zip(sources,weights): ax.plot(candidates,weight,label=f'src {source}')
        if meta['gt_valid_at_probe']: ax.axvline(gt,color='black',ls='--',lw=1,label='GT depth')
        ax.set_title(f'S{stage} probe ({px},{py})'); ax.set_xlabel('Candidate depth (mm)'); ax.set_ylim(0,1)
    axes[0].set_ylabel('Normalized source weight'); axes[-1].legend(fontsize=7)
    save_figure(fig,folder/'figures'/'candidate_source_weights')


def render(args,series):
    folder=Path(args.outdir)/series
    groups={}
    for config in CONFIGS:
        for path in (folder/config.name).glob('*/arrays.npz'):
            groups.setdefault(path.parent.name,{})[config.name]=path.parent
    if not groups: raise FileNotFoundError('No exported arrays; run export on actual checkpoints first')
    for key,files in sorted(groups.items()):
        if 'Base' not in files: raise ValueError(f'{key}: Base missing')
        if len(files)!=8 and not args.allow_partial: raise ValueError(f'{key}: missing configurations; use --allow_partial for diagnostics only')
        render_sample(args,series,key,files)


def main():
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',choices=['stats','export','render'])
    parser.add_argument('--eval_root',default=str(ROOT/'eval'))
    parser.add_argument('--outdir',default=str(ROOT/'outputs'/'paper_visualizations'))
    parser.add_argument('--series',choices=['all','View5','View3'],default='all')
    parser.add_argument('--selection',help='JSON with scan/view/light, optional roi/probe')
    parser.add_argument('--checkpoint_root',help='Relocate CSV checkpoint folder under this root')
    parser.add_argument('--checkpoint_manifest',help='JSON keyed by Base, Base+A, ..., paths to actual checkpoints')
    parser.add_argument('--eval_args_json',help='Explicit evaluator parameters used in original CSV run')
    parser.add_argument('--testpath'); parser.add_argument('--testlist',default=str(ROOT/'lists'/'dtu'/'test.txt'))
    parser.add_argument('--configs',nargs='+',choices=list(BY_NAME))
    parser.add_argument('--batch_size',type=int,default=1); parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--verify_atol',type=float,default=1e-3)
    parser.add_argument('--dry_run',action='store_true')
    parser.add_argument('--error_max',type=float,default=20); parser.add_argument('--gain_max',type=float,default=10)
    parser.add_argument('--cdf_max',type=float,default=30); parser.add_argument('--allow_partial',action='store_true')
    args=parser.parse_args()
    if args.command=='export' and (not args.testpath or args.series=='all'):
        parser.error('export requires --testpath and one explicit --series View5 or View3')
    if min(args.error_max,args.gain_max,args.cdf_max)<=0: parser.error('Plot bounds must be positive')
    series=list(SERIES) if args.series=='all' else [args.series]
    for s in series:
        if args.command=='stats': stat_figures(args,s)
        elif args.command=='render': render(args,s)
        else: export(args)


if __name__=='__main__':
    main()
