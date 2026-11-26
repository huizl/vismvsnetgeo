"""Measure the effect of the third-stage fusion residual with the same checkpoint."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from paper_tables import ROOT, SERIES, write_csv


MODES = ('off', 'on')
REGIONS = ('full', 'occluded_any', 'large_disp_and_occluded')
OUTPUT = ROOT / 'outputs/first_batch_existing_weights/dtu_stage3_residual'


def export_pair(args, series):
    options = json.loads(args.eval_args_json.read_text(encoding='utf-8'))
    for mode in MODES:
        folder = args.outdir / mode
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'{series}_eval_args.json'
        path.write_text(json.dumps({**options, 'paper_stage3_residual': mode}, indent=2) + '\n',
                        encoding='utf-8')
        subprocess.run([
            sys.executable, str(ROOT / 'tools/visualize_paper_results.py'), 'export',
            '--series', series, '--configs', 'Base+A+B+C', '--testpath', str(args.dtu_root.resolve()),
            '--checkpoint_root', str(args.checkpoint_root.resolve()),
            '--selection', str(ROOT / 'tools/paper_dtu_window_test.json'),
            '--eval_args_json', str(path), '--outdir', str(folder), '--verify_mode', 'warn',
        ], check=True, cwd=ROOT)


def sample_pairs(folders, sample):
    key = f"{sample['scan']}_view{sample['view']:02d}_light{sample['light']}"
    paths = [folder / key for folder in folders]
    metadata = [json.loads((path / 'metadata.json').read_text(encoding='utf-8')) for path in paths]
    for mode, meta in zip(MODES, metadata):
        if (meta['raw_model_type'], meta['raw_csv_code'], meta['eval_args']['paper_stage3_residual']) != (
                'hyp', '001', mode):
            raise ValueError(f'{key}: expected hyp checkpoint with stage-three residual {mode}')
    for field in ('checkpoint', 'scan', 'view', 'light', 'source_views', 'region_source_views', 'probe'):
        if metadata[0][field] != metadata[1][field]:
            raise ValueError(f'{key}: different {field}')
    ignored = {'paper_stage3_residual', 'outdir', 'paper_dumpdir'}
    protocol = [{k: v for k, v in meta['eval_args'].items() if k not in ignored} for meta in metadata]
    if protocol[0] != protocol[1]:
        raise ValueError(f'{key}: different inference protocols')

    with np.load(paths[0] / 'arrays.npz', allow_pickle=False) as off, \
         np.load(paths[1] / 'arrays.npz', allow_pickle=False) as on:
        shared = (
            'gt', 'valid', 'source_visible', 'source_occluded', 'source_supervised',
            'stage1_pred', 'stage2_pred', 'stage2_min', 'stage2_max', 'stage2_weights_at_gt',
            'stage3_search_center', 'stage3_search_std', 'stage3_search_count',
            'stage3_search_actual_min', 'stage3_search_actual_max',
            'stage3_coverage', 'stage3_probe_candidates',
            *('mask_' + region for region in REGIONS),
        )
        for field in shared:
            if not np.array_equal(off[field], on[field]):
                raise ValueError(f'{key}: paired export differs in {field}')
        probe_weights = off['stage3_probe_weights']
        if not np.allclose(probe_weights, probe_weights[:, :1], atol=1e-6, rtol=0):
            raise ValueError(f'{key}: disabled weights still vary by candidate')
        valid = off['valid'].astype(bool)
        visible, occluded, supervised = (off['source_' + label].astype(bool)
                                         for label in ('visible', 'occluded', 'supervised'))
        eligible = off['stage3_coverage'].astype(bool) & visible.any(axis=0) & occluded.any(axis=0)
        errors, occluded_weights, unknown_weights = [], [], []
        for data in (off, on):
            weights = data['stage3_weights_at_gt']
            if not np.allclose(weights[:, valid].sum(axis=0), 1, atol=1e-5, rtol=0):
                raise ValueError(f'{key}: invalid source weight normalization')
            error = np.abs(data['pred'] - off['gt'])
            if not np.isfinite(error[valid]).all():
                raise ValueError(f'{key}: nonfinite error on valid pixels')
            errors.append(error)
            occluded_weights.append((weights * occluded).sum(axis=0))
            unknown_weights.append((weights * ~supervised).sum(axis=0))
        for region in REGIONS:
            region_mask = valid & off['mask_' + region].astype(bool)
            for group, allowed in (('all', valid), ('covered_visible_and_occluded', eligible)):
                mask = region_mask & allowed
                if mask.any():
                    yield region, group, tuple(x[mask] for x in (*errors, *occluded_weights, *unknown_weights))


def summarize(results, group):
    off, on, occ_off, occ_on, unknown_off, unknown_on = results
    row = dict(
        pixels=len(off), off_mae_mm=float(off.mean()), on_mae_mm=float(on.mean()),
        mae_change_mm=float((on - off).mean()),
        off_p95_mm=float(np.percentile(off, 95)), on_p95_mm=float(np.percentile(on, 95)),
        off_over_8mm=float((off > 8).mean()), on_over_8mm=float((on > 8).mean()),
        corrected_over_8mm=int(((off > 8) & (on <= 8)).sum()),
        new_over_8mm=int(((off <= 8) & (on > 8)).sum()),
    )
    weights = dict(off_occluded_weight=occ_off, on_occluded_weight=occ_on,
                   occluded_weight_change=occ_on - occ_off,
                   off_unknown_weight=unknown_off, on_unknown_weight=unknown_on)
    row.update({key: float(value.mean()) if group != 'all' else '' for key, value in weights.items()})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dtu_root', type=Path)
    parser.add_argument('checkpoint_root', type=Path)
    parser.add_argument('--series', choices=('all', *SERIES), default='all')
    parser.add_argument('--outdir', type=Path, default=OUTPUT)
    parser.add_argument('--eval_args_json', type=Path, default=ROOT / 'tools/paper_eval_args.example.json')
    parser.add_argument('--summarize_only', action='store_true')
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    samples = json.loads((ROOT / 'tools/paper_dtu_window_test.json').read_text(encoding='utf-8'))
    per_scan, summary = [], []
    for series in SERIES if args.series == 'all' else (args.series,):
        if not args.summarize_only:
            export_pair(args, series)
        folders = [args.outdir / mode / series / 'Base+A+B+C' for mode in MODES]
        grouped = {}
        for sample in samples:
            for region, group, data in sample_pairs(folders, sample):
                grouped.setdefault((region, group), []).append(data)
                per_scan.append(dict(series=series, **sample, region=region, group=group,
                                     **summarize(data, group)))
        for (region, group), records in grouped.items():
            combined = tuple(np.concatenate(field) for field in zip(*records))
            summary.append(dict(series=series, region=region, group=group, scans=len(records),
                                **summarize(combined, group)))
    write_csv(args.outdir / 'stage3_residual_per_scan.csv', per_scan)
    write_csv(args.outdir / 'stage3_residual_summary.csv', summary)
    print(f'Paired stage-three residual results written to {args.outdir}')


if __name__ == '__main__':
    main()
