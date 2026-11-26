"""Choose a common DTU fusion protocol on validation scans, then freeze it for test."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

from paper_tables import ROOT, SERIES, read_csv, write_csv
from run_paper_dtu_fusion import NAMES, PILOT_SCANS


# Change one parameter at a time. Pixel distances refer to the native stage-three grid.
PRESETS = ((1.0, 0.01, 2), (1.0, 0.01, 3), (1.5, 0.01, 2))
METRICS = ('accuracy_mm', 'completeness_mm', 'overall_mm', 'retention')


def preset_name(preset):
    px, rel, votes = preset
    return f'px{px:g}_rel{rel:g}_min{votes}'


def threshold_options(preset):
    px, rel, votes = preset
    return ['--pixel_threshold', str(px), '--relative_threshold', str(rel), '--min_consistent', str(votes)]


def run_fusion(args, split, scans, preset, *flags):
    subprocess.run([
        sys.executable, str(ROOT / 'tools/run_paper_dtu_fusion.py'),
        str(args.dtu_root), str(args.checkpoint_root), '--split', split,
        '--series', args.series, '--outdir', str(args.outdir), '--scans', *scans,
        *threshold_options(preset), *flags,
    ], check=True, cwd=ROOT)


def evaluator_options(args):
    return ['--evaluator', str(args.evaluator), '--dtu_eval_root', str(args.dtu_eval_root)]


def summarize_scores(rows, scans, series):
    expected = {(config, scan) for config in NAMES for scan in scans}
    actual = {(r['configuration'], r['scan']) for r in rows}
    if actual != expected or len(rows) != len(expected) or any(r['series'] != series for r in rows):
        raise ValueError('Scores must contain exactly one row for each requested configuration and scan')
    if not all(math.isfinite(float(r[key])) for r in rows for key in METRICS):
        raise ValueError('Point-cloud scores contain nonfinite values')
    summaries = []
    for config in (*NAMES, 'all_configurations'):
        selected = [r for r in rows if config == 'all_configurations' or r['configuration'] == config]
        summaries.append(dict(configuration=config, scores=len(selected),
                              **{key: mean(float(r[key]) for r in selected) for key in METRICS}))
    return summaries


def tune(args, protocol_path, evaluator_hash):
    scans = PILOT_SCANS['val']
    summaries = []
    for preset in PRESETS:
        name = preset_name(preset)
        folder = args.outdir / args.series / 'val' / name
        if not (folder / 'pointcloud_per_scan.csv').is_file():
            run_fusion(args, 'val', scans, preset, '--fusion_only')
        scores = folder / 'pointcloud_scores.csv'
        score_protocol_path = folder / 'score_protocol.json'
        cached = json.loads(score_protocol_path.read_text(encoding='utf-8')) if score_protocol_path.is_file() else {}
        fusion_protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
        if not (scores.is_file() and cached.get('evaluator_sha256') == evaluator_hash
                and cached.get('dtu_eval_root') == str(args.dtu_eval_root)
                and set(cached.get('scans', ())) == set(scans)
                and cached.get('fusion_protocol') == fusion_protocol):
            run_fusion(args, 'val', scans, preset, '--score_only', *evaluator_options(args))
        summaries.extend(dict(preset=name, **row) for row in summarize_scores(read_csv(scores), scans, args.series))

    best = min((row for row in summaries if row['configuration'] == 'all_configurations'),
               key=lambda row: row['overall_mm'])
    preset = next(preset for preset in PRESETS if preset_name(preset) == best['preset'])
    write_csv(protocol_path.with_name('fusion_threshold_comparison.csv'), summaries)
    selected = dict(series=args.series, validation_scans=list(scans), preset=list(preset),
                    validation_mean_overall_mm=best['overall_mm'],
                    evaluator_sha256=evaluator_hash, dtu_eval_root=str(args.dtu_eval_root))
    protocol_path.write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
    print(f"Selected {best['preset']} at {best['overall_mm']:.4f} mm validation Overall")


def test(args, protocol_path, evaluator_hash):
    selected = json.loads(protocol_path.read_text(encoding='utf-8'))
    preset = tuple(selected['preset'])
    if (selected['series'] != args.series or preset not in PRESETS
            or selected['evaluator_sha256'] != evaluator_hash
            or selected['dtu_eval_root'] != str(args.dtu_eval_root)):
        raise ValueError('Test evaluator or protocol differs from the frozen validation run')
    scans = (ROOT / 'lists/dtu/test.txt').read_text().split()
    if set(scans) & set(selected['validation_scans']):
        raise ValueError('Validation and test scans overlap')
    run_fusion(args, 'test', scans, preset, *evaluator_options(args))
    folder = args.outdir / args.series / 'test' / preset_name(preset)
    rows = read_csv(folder / 'pointcloud_per_scan.csv')
    write_csv(folder / 'test_pointcloud_summary.csv', summarize_scores(rows, scans, args.series))
    print(f'Test summary written to {folder / "test_pointcloud_summary.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dtu_root', type=Path)
    parser.add_argument('checkpoint_root', type=Path)
    parser.add_argument('--evaluator', type=Path, required=True)
    parser.add_argument('--dtu_eval_root', type=Path, required=True)
    parser.add_argument('--series', choices=tuple(SERIES), default='View5')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/first_batch_existing_weights/dtu_fusion')
    parser.add_argument('--test', action='store_true', help='evaluate the 22 test scans using the frozen choice')
    args = parser.parse_args()
    for name in ('dtu_root', 'checkpoint_root', 'evaluator', 'dtu_eval_root', 'outdir'):
        setattr(args, name, getattr(args, name).resolve())
    evaluator_hash = sha256(args.evaluator.read_bytes()).hexdigest()
    path = args.outdir / args.series / 'val' / 'frozen_fusion_thresholds.json'
    if args.test:
        test(args, path, evaluator_hash)
    else:
        tune(args, path, evaluator_hash)


if __name__ == '__main__':
    main()
