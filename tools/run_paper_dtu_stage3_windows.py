"""Run and summarize a stage-three-only window intervention on existing DTU weights."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from paper_tables import ROOT, SERIES, write_csv
from paper_search_windows import search_bounds


RULES = ('matched_fixed', 'bounded_kappa_sigma')
REGIONS = ('full', 'occluded_any', 'large_disp_and_occluded')


def run_series(args, series, half_width):
    options = json.loads(args.eval_args_json.read_text(encoding='utf-8'))
    for rule in RULES:
        folder = args.outdir / rule
        folder.mkdir(parents=True, exist_ok=True)
        options_path = folder / f'{series}_eval_args.json'
        rule_options = {**options, 'paper_stage3_window': rule}
        if rule == 'matched_fixed':
            rule_options['paper_stage3_half_width'] = half_width
        options_path.write_text(json.dumps(rule_options, indent=2) + '\n', encoding='utf-8')
        subprocess.run([
            sys.executable, str(ROOT / 'tools/visualize_paper_results.py'), 'export',
            '--series', series, '--configs', 'Base+B',
            '--testpath', str(args.dtu_root), '--checkpoint_root', str(args.checkpoint_root),
            '--selection', str(ROOT / 'tools/paper_dtu_window_test.json'),
            '--eval_args_json', str(options_path), '--outdir', str(folder), '--verify_mode', 'warn',
        ], check=True, cwd=ROOT)


def compare_sample(folders, sample, half_width):
    key = f"{sample['scan']}_view{sample['view']:02d}_light{sample['light']}"
    paths = [folder / key for folder in folders]
    metadata = [json.loads((path / 'metadata.json').read_text(encoding='utf-8')) for path in paths]
    for rule, meta in zip(RULES, metadata):
        args = meta['eval_args']
        if meta['raw_model_type'] != 'vis' or meta['raw_csv_code'] != '000' or args['paper_stage3_window'] != rule:
            raise ValueError(f'{key}: wrong checkpoint variant or window rule')
        if rule == 'matched_fixed' and args['paper_stage3_half_width'] != half_width:
            raise ValueError(f'{key}: fixed-window half-width differs from calibration')
    for field in ('checkpoint', 'source_views', 'region_source_views', 'scan', 'view', 'light'):
        if metadata[0][field] != metadata[1][field]:
            raise ValueError(f'{key}: different {field}')
    intervention_fields = {'paper_stage3_window', 'paper_stage3_half_width', 'outdir', 'paper_dumpdir'}
    protocols = [{k: v for k, v in meta['eval_args'].items() if k not in intervention_fields}
                 for meta in metadata]
    if protocols[0] != protocols[1]:
        raise ValueError(f'{key}: paired evaluation parameters differ')

    with np.load(paths[0] / 'arrays.npz', allow_pickle=False) as fixed, \
         np.load(paths[1] / 'arrays.npz', allow_pickle=False) as bounded:
        for field in ('gt', 'valid', 'stage1_pred', 'stage2_pred', 'stage2_min', 'stage2_max',
                      'stage3_search_center', 'stage3_search_std', 'stage3_search_gt',
                      'stage3_search_global_min', 'stage3_search_global_max',
                      'stage3_search_interval', 'stage3_search_count',
                      *('mask_' + region for region in REGIONS)):
            if not np.array_equal(fixed[field], bounded[field]):
                raise ValueError(f'{key}: paired inference differs before stage three ({field})')
        prefix = 'stage3_search_'
        center, std, interval, lo, hi = (fixed[prefix + field] for field in
                                        ('center', 'std', 'interval', 'global_min', 'global_max'))
        fixed_bounds = (np.clip(center - half_width, lo, hi), np.clip(center + half_width, lo, hi))
        windows, _ = search_bounds(center, std, interval, lo, hi, int(fixed[prefix + 'count']), protocols[0])
        for export, bounds in ((fixed, fixed_bounds), (bounded, windows['bounded_kappa_sigma'])):
            for field, expected in zip(('actual_min', 'actual_max'), bounds):
                if not np.allclose(export[prefix + field], expected, atol=1e-3, rtol=0):
                    raise ValueError(f'{key}: stage-three candidates differ from requested bounds')
        valid = fixed['valid'].astype(bool)
        if not all(np.isfinite(a['pred'][valid]).all() for a in (fixed, bounded)):
            raise ValueError(f'{key}: nonfinite prediction on valid pixels')
        gt = fixed['gt']
        errors = tuple(np.abs(a['pred'] - gt) for a in (fixed, bounded))
        widths = tuple(a['stage3_max'] - a['stage3_min'] for a in (fixed, bounded))
        a, b = (export['stage3_coverage'].astype(bool) for export in (fixed, bounded))
        groups = {'all': valid, 'both': a & b, 'bounded_only': ~a & b,
                  'fixed_only': a & ~b, 'neither': ~a & ~b}
        for region in REGIONS:
            for group, allowed in groups.items():
                mask = valid & fixed['mask_' + region].astype(bool) & allowed
                if mask.any():
                    yield region, group, tuple(x[mask] for x in (*errors, *widths, a, b))


def summarize(results):
    fixed, bounded, fixed_width, bounded_width, fixed_cover, bounded_cover = results
    return {
        'pixels': len(fixed),
        'fixed_width_mm': float(fixed_width.mean()),
        'bounded_width_mm': float(bounded_width.mean()),
        'fixed_coverage': float(fixed_cover.mean()),
        'bounded_coverage': float(bounded_cover.mean()),
        'fixed_mae_mm': float(fixed.mean()),
        'bounded_mae_mm': float(bounded.mean()),
        'mae_change_mm': float((bounded - fixed).mean()),
        'fixed_p95_mm': float(np.percentile(fixed, 95)),
        'bounded_p95_mm': float(np.percentile(bounded, 95)),
        'fixed_over_8mm': float((fixed > 8).mean()),
        'bounded_over_8mm': float((bounded > 8).mean()),
        'corrected_over_8mm': int(((fixed > 8) & (bounded <= 8)).sum()),
        'new_over_8mm': int(((fixed <= 8) & (bounded > 8)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dtu_root', type=Path)
    parser.add_argument('checkpoint_root', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--series', choices=('all', *SERIES), default='all')
    parser.add_argument('--eval_args_json', type=Path, default=ROOT / 'tools/paper_eval_args.example.json')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/first_batch_existing_weights/dtu_stage3_windows')
    parser.add_argument('--summarize_only', action='store_true', help='use existing exports in outdir')
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    args.dtu_root = args.dtu_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    samples = json.loads((ROOT / 'tools/paper_dtu_window_test.json').read_text(encoding='utf-8'))
    per_scan, summary = [], []
    for series in SERIES if args.series == 'all' else (args.series,):
        entry = calibration[series]
        if set(entry['validation_scans']) & {x['scan'] for x in samples}:
            raise ValueError(f'{series}: calibration and test scans overlap')
        half = entry['stages']['3']['half_width_mm']
        if not np.isfinite(half) or half <= 0:
            raise ValueError(f'{series}: invalid calibrated half-width')
        if not args.summarize_only:
            run_series(args, series, half)
        folders = [args.outdir / rule / series / 'Base+B' for rule in RULES]
        grouped = {}
        for sample in samples:
            for region, group, results in compare_sample(folders, sample, half):
                grouped.setdefault((region, group), []).append(results)
                per_scan.append(dict(series=series, **sample, region=region, group=group, **summarize(results)))
        for (region, group), results in grouped.items():
            merged = tuple(np.concatenate(x) for x in zip(*results))
            summary.append(dict(series=series, region=region, group=group, scans=len(results), **summarize(merged)))
    write_csv(args.outdir / 'stage3_window_per_scan.csv', per_scan)
    write_csv(args.outdir / 'stage3_window_summary.csv', summary)
    print(f'Paired stage-three metrics written to {args.outdir}')


if __name__ == '__main__':
    main()
