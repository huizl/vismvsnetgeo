"""Train isolated v2 candidates on Train and evaluate each best checkpoint on Val.

Default queue: baseline + three single modules. Combinations are explicit.
CUDA_VISIBLE_DEVICES is inherited; --dry_run requires no data or CUDA.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_variants import MODEL_VARIANTS
from tools.train_m2_supervision_val import build_commands, project_path

V2_MODELS = tuple(name for name in MODEL_VARIANTS if name.startswith('v2_'))


def build_plans(args):
    plans = []
    for model_type in args.models:
        label = f'{model_type}_view{args.train_nviews}_seed{args.seed}'
        logdir = project_path(args.log_root) / label
        outdir = project_path(args.out_root) / model_type
        local = argparse.Namespace(**vars(args), model_type=model_type, label=label,
                                   logdir=str(logdir), outdir=str(outdir))
        train, evaluate = build_commands(local)
        # A conservative initial M1 residual bound. M2 is supervision only;
        # M3 guided centers use the unchanged uniform depth grid.
        for command in (train, evaluate):
            command.extend(['--hypothesis_residual_scale', '0.5'])
        plans.append(dict(model_type=model_type, label=label, logdir=str(logdir),
                          outdir=str(outdir), train=train, evaluate=evaluate))
    return plans


def write_comparison(plans, out_root):
    summaries = {}
    for plan in plans:
        with (Path(plan['outdir'])/'summary_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
            summaries[plan['model_type']] = {r['region']: r for r in csv.DictReader(stream)
                                               if r['aggregation'] == 'pixel_weighted'}
    lines = ['# v2 Val trained candidates', '',
             'Each row uses its own best_2mm checkpoint. Val light=3, five inference/region views.',
             'Candidate results; a positive effect must be established from the metrics and repeated training.', '',
             '| Model | Region | Abs mm | Delta Abs vs v2_vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for model_type, rows in summaries.items():
        for region, row in rows.items():
            baseline = summaries.get('v2_vis', {}).get(region)
            if baseline and (row['pixels'], row['images']) != (baseline['pixels'], baseline['images']):
                raise ValueError('Sample/pixel count mismatch: '+model_type+'/'+region)
            da = f"{float(row['abs'])-float(baseline['abs']):+.4f}" if baseline else '-'
            d2 = f"{100*(float(row['acc2'])-float(baseline['acc2'])):+.4f}" if baseline else '-'
            cells = [model_type, region, f"{float(row['abs']):.4f}", da,
                     f"{100*float(row['acc2']):.4f}", d2,
                     *[f'{100*float(row[k]):.4f}' for k in ('acc4','acc8','stage2_in_range','stage3_in_range')]]
            lines.append('| '+' | '.join(cells)+' |')
    (out_root/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=V2_MODELS,
                        default=['v2_vis','v2_m1','v2_m2','v2_m3'])
    parser.add_argument('--train_nviews', type=int, choices=(3,5), default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_workers', type=int, default=8)
    parser.add_argument('--eval_workers', type=int, default=4)
    parser.add_argument('--datapath', default='/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu')
    parser.add_argument('--log_root', default='checkpoints/v2')
    parser.add_argument('--out_root', default=None)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('Each model must occur once')
    if min(args.epochs,args.batch_size)<1 or min(args.seed,args.train_workers,args.eval_workers)<0:
        parser.error('epochs/batch_size must be positive; seed/workers must be non-negative')
    args.out_root = args.out_root or f'eval/v2_val_view{args.train_nviews}_seed{args.seed}'
    plans = build_plans(args)
    for plan in plans:
        print(plan['model_type']+' TRAIN: '+shlex.join(plan['train']), flush=True)
        print(plan['model_type']+' VAL: '+shlex.join(plan['evaluate']), flush=True)
    if args.dry_run:
        return
    datapath = project_path(args.datapath)
    missing = [str(datapath/name) for name in ('Rectified','Depths','Cameras') if not (datapath/name).is_dir()]
    missing.extend(str(ROOT/'lists/dtu'/name) for name in ('train.txt','val.txt')
                   if not (ROOT/'lists/dtu'/name).is_file())
    if missing:
        parser.error('Missing input(s):\n'+'\n'.join(missing))
    out_root = project_path(args.out_root)
    for plan in plans:
        logdir = Path(plan['logdir'])
        if logdir == out_root or logdir in out_root.parents or out_root in logdir.parents:
            parser.error('Checkpoint and evaluation directories must not overlap')
        if logdir.exists() and (not logdir.is_dir() or any(logdir.iterdir())):
            parser.error('Checkpoint directory must be new or empty: '+str(logdir))
    if out_root.exists() and (not out_root.is_dir() or any(out_root.iterdir())):
        parser.error('Evaluation root must be new or empty: '+str(out_root))
    out_root.mkdir(parents=True, exist_ok=True)
    sources = ['train.py','models/model_variants.py','models/module.py','models/vismvsnet_oa.py',
               'models/guided_centers.py','tools/train_m2_supervision_val.py','tools/train_v2_val.py',
               'tools/eval_region_metrics_dtu_yao.py','lists/dtu/train.txt','lists/dtu/val.txt']
    manifest = dict(arguments=vars(args), plans=plans,
                    source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (out_root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for plan in plans:
        logdir, outdir = Path(plan['logdir']), Path(plan['outdir'])
        logdir.mkdir(parents=True,exist_ok=True)
        outdir.mkdir(parents=True,exist_ok=True)
        (logdir/'experiment.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        print('Training '+plan['label'],flush=True)
        subprocess.run(plan['train'],cwd=ROOT,check=True)
        if not (logdir/'best_2mm.ckpt').is_file():
            raise FileNotFoundError(logdir/'best_2mm.ckpt')
        print('Val evaluation '+plan['label'],flush=True)
        with (outdir/'validation.log').open('w',encoding='utf-8') as stream:
            subprocess.run(plan['evaluate'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    write_comparison(plans,out_root)
    print('Saved comparison: '+str(out_root/'comparison.md'),flush=True)


if __name__ == '__main__':
    main()
