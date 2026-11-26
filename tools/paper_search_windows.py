"""Compare search windows on the same saved previous-stage prediction."""

import argparse
import json
from pathlib import Path

import numpy as np

from paper_tables import write_csv


REGIONS = ('full', 'occluded_any', 'large_disp_and_occluded')


def search_bounds(center, std, interval, minimum, maximum, count, settings):
    # The existing fixed grid has one fewer step above the center for even N.
    fixed_start = center - (count // 2) * interval
    fixed_end = fixed_start + (count - 1) * interval
    nominal = (count // 2) * interval
    adaptive = settings['range_sigma_scale'] * std
    bounded = np.clip(adaptive, settings['range_min_scale'] * nominal,
                      settings['range_max_scale'] * nominal)
    raw = {
        'fixed': (fixed_start, fixed_end),
        'kappa_sigma': (center - adaptive, center + adaptive),
        'bounded_kappa_sigma': (center - bounded, center + bounded),
    }
    # All offline rules use the same physical bounds and candidate count.
    clipped = {name: (np.clip(lo, minimum, maximum), np.clip(hi, minimum, maximum))
               for name, (lo, hi) in raw.items()}
    return clipped, raw


def nearest_mask(mask, shape):
    y = np.arange(shape[0]) * mask.shape[0] // shape[0]
    x = np.arange(shape[1]) * mask.shape[1] // shape[1]
    return mask[np.ix_(y, x)].astype(bool)


def window_rows(arrays, meta, series, stage):
    prefix = f'stage{stage}_search_'
    if prefix + 'std' not in arrays:
        raise ValueError('Export has no previous-stage standard deviation. Rerun paper inference.')
    args = meta['eval_args']
    center, std, interval, minimum, maximum = (
        arrays[prefix + key] for key in ('center', 'std', 'interval', 'global_min', 'global_max'))
    gt = arrays[prefix + 'gt']
    count = int(arrays[prefix + 'count'])
    windows, raw = search_bounds(center, std, interval, minimum, maximum, count, args)
    if meta['raw_csv_code'][1] == '1':
        lo, hi = raw['bounded_kappa_sigma']
        lo, hi = np.maximum(lo, minimum), np.minimum(hi, maximum)
        expected = lo, np.maximum(hi, lo + 1e-6)
    else:
        expected = raw['fixed']
    for key, bound in zip(('actual_min', 'actual_max'), expected):
        if not np.allclose(arrays[prefix + key], bound, rtol=0, atol=1e-3):
            raise ValueError(f'Stage {stage}: reconstructed search bound differs from saved {key}')

    fixed_lo, fixed_hi = windows['fixed']
    fixed_covers = (gt >= fixed_lo) & (gt <= fixed_hi)
    valid = nearest_mask(arrays['valid'], gt.shape) & np.isfinite(gt)
    for region in REGIONS:
        mask = valid & nearest_mask(arrays['mask_' + region], gt.shape)
        pixels = int(mask.sum())
        if not pixels:
            continue
        for rule, (lo, hi) in windows.items():
            width = hi - lo
            step = width / (count - 1)
            # A zero-width interval repeats its single depth at every candidate.
            index = np.rint(np.clip(gt - lo, 0, width) / np.where(step > 0, step, 1))
            nearest_error = np.abs(gt - (lo + step * index))
            covers = (gt >= lo) & (gt <= hi)
            yield dict(series=series, configuration=meta['analysis_config'],
                       raw_model_type=meta['raw_model_type'], scan=meta['scan'], view=meta['view'],
                       light=meta['light'], stage=stage, region=region, rule=rule,
                       candidates=count, pixels=pixels, covered=int((mask & covers).sum()),
                       coverage=float(covers[mask].mean()), mean_width_mm=float(width[mask].mean()),
                       nearest_abs_mm=float(nearest_error[mask].mean()),
                       gained_vs_fixed=int((mask & covers & ~fixed_covers).sum()),
                       lost_vs_fixed=int((mask & ~covers & fixed_covers).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_dir', type=Path)
    parser.add_argument('--series', nargs='+', choices=('View5', 'View3'), default=['View5', 'View3'])
    args = parser.parse_args()
    rows = []
    for series in args.series:
        for metadata in sorted((args.export_dir / series).glob('*/*/metadata.json')):
            meta = json.loads(metadata.read_text(encoding='utf-8'))
            with np.load(metadata.with_name('arrays.npz'), allow_pickle=False) as arrays:
                for stage in (2, 3):
                    rows.extend(window_rows(arrays, meta, series, stage))
    if not rows:
        parser.error('No usable DTU paper exports found in the supplied directory')
    target = args.export_dir / 'search_window_comparison.csv'
    write_csv(target, rows)
    print(f'{len(rows)} diagnostic rows written to {target}')


if __name__ == '__main__':
    main()
