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


def match_half_width(center, minimum, maximum, target):
    """Solve for the fixed half-width after accounting for boundary clipping."""
    lower = 0.0
    upper = float(max(np.max(abs(center - minimum)), np.max(abs(center - maximum))))
    for _ in range(25):
        half = (lower + upper) / 2
        width = np.clip(center + half, minimum, maximum) - np.clip(center - half, minimum, maximum)
        if width.mean(dtype=np.float64) < target:
            lower = half
        else:
            upper = half
    return (lower + upper) / 2


def calibrate_windows(metadata_paths):
    stages = {2: [], 3: []}
    validation_scans = set()
    for metadata in metadata_paths:
        meta = json.loads(metadata.read_text(encoding='utf-8'))
        if meta['raw_csv_code'] != '000' or meta['raw_model_type'] != 'vis':
            raise ValueError(f'Expected the vis anchor for calibration: {metadata}')
        validation_scans.add(meta['scan'])
        with np.load(metadata.with_name('arrays.npz'), allow_pickle=False) as arrays:
            for stage, records in stages.items():
                prefix = f'stage{stage}_search_'
                center, std, interval, minimum, maximum = (
                    arrays[prefix + k] for k in ('center', 'std', 'interval', 'global_min', 'global_max'))
                count = int(arrays[prefix + 'count'])
                (lo, hi) = search_bounds(center, std, interval, minimum, maximum, count,
                                         meta['eval_args'])[0]['bounded_kappa_sigma']
                mask = nearest_mask(arrays['valid'], center.shape) & np.isfinite(arrays[prefix + 'gt'])
                records.append((center[mask], np.broadcast_to(minimum, center.shape)[mask],
                                np.broadcast_to(maximum, center.shape)[mask], (hi - lo)[mask]))
    calibration = {}
    for stage, records in stages.items():
        if not records or not any(len(record[0]) for record in records):
            raise ValueError(f'No valid validation pixels for stage {stage}')
        center, minimum, maximum, width = (np.concatenate(field) for field in zip(*records))
        target = float(width.mean(dtype=np.float64))
        half = match_half_width(center, minimum, maximum, target)
        matched = np.clip(center + half, minimum, maximum) - np.clip(center - half, minimum, maximum)
        calibration[stage] = {'half_width_mm': half, 'target_validation_width_mm': target,
                              'matched_validation_width_mm': float(matched.mean(dtype=np.float64)),
                              'validation_pixels': len(center)}
    return calibration, validation_scans


def window_rows(arrays, meta, series, stage, matched_half_width=None):
    prefix = f'stage{stage}_search_'
    if prefix + 'std' not in arrays:
        raise ValueError('Export has no previous-stage standard deviation. Rerun paper inference.')
    args = meta['eval_args']
    center, std, interval, minimum, maximum = (
        arrays[prefix + key] for key in ('center', 'std', 'interval', 'global_min', 'global_max'))
    gt = arrays[prefix + 'gt']
    count = int(arrays[prefix + 'count'])
    windows, raw = search_bounds(center, std, interval, minimum, maximum, count, args)
    if matched_half_width is not None:
        windows['matched_fixed'] = (np.clip(center - matched_half_width, minimum, maximum),
                                    np.clip(center + matched_half_width, minimum, maximum))
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
    matched_covers = fixed_covers
    if matched_half_width is not None:
        lo, hi = windows['matched_fixed']
        matched_covers = (gt >= lo) & (gt <= hi)
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
                       lost_vs_fixed=int((mask & ~covers & fixed_covers).sum()),
                       gained_vs_matched=int((mask & covers & ~matched_covers).sum())
                       if matched_half_width is not None else '',
                       lost_vs_matched=int((mask & ~covers & matched_covers).sum())
                       if matched_half_width is not None else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_dir', type=Path)
    parser.add_argument('--validation_dir', type=Path, help='calibrate matched fixed widths from vis exports')
    parser.add_argument('--series', nargs='+', choices=('View5', 'View3'), default=['View5', 'View3'])
    args = parser.parse_args()
    rows = []
    calibrations = {}
    for series in args.series:
        if args.validation_dir is not None:
            calibration, validation_scans = calibrate_windows(
                sorted((args.validation_dir / series).glob('*/*/metadata.json')))
            calibrations[series] = {'stages': calibration, 'validation_scans': sorted(validation_scans)}
        for metadata in sorted((args.export_dir / series).glob('*/*/metadata.json')):
            meta = json.loads(metadata.read_text(encoding='utf-8'))
            if args.validation_dir is not None:
                if meta['scan'] in validation_scans:
                    raise ValueError(f'Validation scan also present in test exports: {metadata}')
                if meta['raw_csv_code'] != '000' or meta['raw_model_type'] != 'vis':
                    raise ValueError(f'Expected same vis anchor on validation and test: {metadata}')
            with np.load(metadata.with_name('arrays.npz'), allow_pickle=False) as arrays:
                for stage in (2, 3):
                    half = calibration[stage]['half_width_mm'] if args.validation_dir is not None else None
                    rows.extend(window_rows(arrays, meta, series, stage, half))
    if not rows:
        parser.error('No usable DTU paper exports found in the supplied directory')
    target = args.export_dir / 'search_window_comparison.csv'
    write_csv(target, rows)
    if calibrations:
        (args.export_dir / 'matched_window_calibration.json').write_text(
            json.dumps(calibrations, indent=2) + '\n', encoding='utf-8')
    print(f'{len(rows)} diagnostic rows written to {target}')


if __name__ == '__main__':
    main()
