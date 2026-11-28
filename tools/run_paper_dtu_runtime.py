"""Benchmark the paper's DTU configurations without changing their weights."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
from types import SimpleNamespace

import numpy as np

from paper_tables import ROOT, SERIES, read_csv, write_csv
from run_paper_dtu_fusion import NAMES, historical_protocol


DEFAULT_OUTPUT = ROOT / 'outputs/first_batch_existing_weights/dtu_runtime'
SELECTION = ROOT / 'tools/paper_dtu_window_test.json'


def summarize_runtime(rows):
    summary = []
    for config in NAMES:
        group = [row for row in rows if row['configuration'] == config]
        times = [row['forward_ms'] for row in group]
        if not times or not all(math.isfinite(t) and t > 0 for t in times):
            raise ValueError(f'{config}: missing or invalid forward measurements')
        summary.append(dict(
            configuration=config, raw_model_type=group[0]['raw_model_type'],
            scans=len({row['scan'] for row in group}), forwards=len(times),
            median_forward_ms=median(times), p95_forward_ms=float(np.percentile(times, 95)),
            mean_forward_ms=mean(times),
            peak_allocated_mib=max(row['peak_allocated_mib'] for row in group),
            peak_extra_forward_mib=max(row['extra_forward_mib'] for row in group),
        ))
    return summary


def summarize_fusion(path, series, scans, protocols):
    rows = [row for row in read_csv(path) if row['series'] == series and row['scan'] in scans]
    keys = {(row['configuration'], row['scan']) for row in rows}
    if len(rows) != len(keys) or keys != {(name, scan) for name in NAMES for scan in scans}:
        raise ValueError('Fusion CSV must contain one row per configuration and selected scan')
    summary = []
    for config in NAMES:
        group = [row for row in rows if row['configuration'] == config]
        expected = protocols[config]
        if any((row['raw_model_type'], row['raw_csv_code']) != (expected['model_type'], expected['csv_code'])
               for row in group):
            raise ValueError(f'{config}: fusion CSV used a different model variant')
        times = [float(row['fusion_seconds']) for row in group]
        if not all(math.isfinite(t) and t > 0 for t in times):
            raise ValueError(f'{config}: invalid fusion durations')
        summary.append(dict(configuration=config, raw_model_type=expected['model_type'], scans=len(times),
                            median_fusion_s=median(times), p95_fusion_s=float(np.percentile(times, 95)),
                            mean_fusion_s=mean(times)))
    return summary


def load_inputs(dtu_root, samples, eval_nviews, options):
    from datasets.dtu_yao import MVSDataset
    import torch

    data = MVSDataset(str(dtu_root), str(ROOT / 'lists/dtu/test.txt'), 'test', eval_nviews,
                      options['numdepth'], options['interval_scale'])
    by_key = {(scan, int(ref), int(light)): idx for idx, (scan, light, ref, _) in enumerate(data.metas)}
    inputs, manifest = [], []
    for spec in samples:
        key = (spec['scan'], spec['view'], spec['light'])
        idx = by_key[key]
        sample = data[idx]
        inputs.append(tuple(torch.as_tensor(sample[name], dtype=torch.float32).unsqueeze(0)
                            for name in ('imgs', 'proj_matrices', 'depth_values')))
        manifest.append({**spec, 'sources': data.metas[idx][3][:eval_nviews-1]})
    shapes = {tuple(t.shape) for t, _, _ in inputs}
    if len(shapes) != 1 or next(iter(shapes))[1] != eval_nviews:
        raise ValueError('Benchmark inputs must share one image shape and view count')
    return inputs, manifest


def benchmark(args, protocol, samples, inputs):
    import torch
    from tools.eval_region_metrics_dtu_yao import build_model, load_checkpoint

    model_params = ('model_type', 'range_sigma_scale', 'range_min_scale', 'range_max_scale',
                    'hypothesis_residual_scale')
    settings = SimpleNamespace(**protocol['options'], **{k: protocol[k] for k in model_params}, vismode='soft')
    model = build_model(settings).cuda().eval()
    load_checkpoint(model, protocol['checkpoint'])
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    rows = []
    mib = 1024**2
    with torch.inference_mode():
        for i in range(args.warmup):
            warm = tuple(t.cuda() for t in inputs[i % len(inputs)])
            model(*warm)
            del warm
        start.record()
        stop.record()
        stop.synchronize()
        for repeat in range(1, args.repeats+1):
            for sample, cpu_inputs in zip(samples, inputs):
                gpu_inputs = tuple(t.cuda() for t in cpu_inputs)
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start.record()
                prediction = model(*gpu_inputs)
                stop.record()
                stop.synchronize()
                peak = torch.cuda.max_memory_allocated()
                rows.append(dict(configuration=protocol['name'], raw_model_type=protocol['model_type'],
                                 raw_csv_code=protocol['csv_code'], **sample, repeat=repeat,
                                 forward_ms=float(start.elapsed_time(stop)),
                                 peak_allocated_mib=peak/mib, extra_forward_mib=(peak-baseline)/mib))
                del prediction, gpu_inputs
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dtu_root', type=Path)
    parser.add_argument('checkpoint_root', type=Path)
    parser.add_argument('--series', choices=tuple(SERIES), default='View5')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--fusion_csv', type=Path, help='Optional previously recorded DTU fusion timings')
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error('warmup and repeats must be positive')

    import torch

    if not torch.cuda.is_available():
        parser.error('CUDA is required for the runtime benchmark')
    if torch.cuda.device_count() != 1:
        parser.error('Expose one GPU with CUDA_VISIBLE_DEVICES for this benchmark')
    torch.manual_seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    samples = json.loads(SELECTION.read_text(encoding='utf-8'))
    options = json.loads((ROOT / 'tools/paper_eval_args.example.json').read_text(encoding='utf-8'))
    args.split = 'test'
    protocols = historical_protocol(args, options)
    input_protocol = {(p['light'], p['eval_nviews']) for p in protocols.values()}
    if len(input_protocol) != 1:
        raise ValueError('Historical configurations use different lights or view counts')
    (light, nviews), = input_protocol
    if any(spec['light'] != light for spec in samples):
        raise ValueError('Fixed benchmark samples use a different light than the checkpoints')
    for checkpoint in (Path(p['checkpoint']) for p in protocols.values()):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    output = args.outdir / args.series
    output.mkdir(parents=True, exist_ok=True)
    inputs, manifest = load_inputs(args.dtu_root, samples, nviews, options)
    scans = {sample['scan'] for sample in samples}
    fusion = summarize_fusion(args.fusion_csv, args.series, scans, protocols) if args.fusion_csv else None
    records = []
    for name, protocol in protocols.items():
        print(f'Benchmarking {name}: {protocol["model_type"]}', flush=True)
        records.extend(benchmark(args, protocol, samples, inputs))
        torch.cuda.empty_cache()
    write_csv(output / 'runtime_per_view.csv', records)
    write_csv(output / 'runtime_summary.csv', summarize_runtime(records))
    if fusion is not None:
        write_csv(output / 'fusion_runtime_summary.csv', fusion)

    device = torch.cuda.get_device_properties(0)
    details = dict(series=args.series, gpu=device.name, gpu_memory_mib=device.total_memory/1024**2,
                   gpu_compute_capability=[device.major, device.minor], pytorch=torch.__version__,
                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), host=platform.platform(),
                   dtype='float32', autocast=False, tf32=False, cudnn_benchmark=False,
                   image_tensor_shape=list(inputs[0][0].shape), warmup=args.warmup, repeats=args.repeats,
                   samples=manifest, configurations=protocols, dtu_root=str(args.dtu_root.resolve()),
                   measurement='CUDA events around model forward; input transfer and data loading excluded',
                   memory_measurement='PyTorch allocated peak includes model, CUDA inputs and forward outputs',
                   fusion_source=str(args.fusion_csv.resolve()) if args.fusion_csv else None,
                   fusion_sha256=sha256(args.fusion_csv.read_bytes()).hexdigest() if args.fusion_csv else None,
                   fusion_measurement='historical per-scan depth read, geometric fusion and PLY write')
    (output / 'runtime_protocol.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    print(f'Runtime results written to {output}')


if __name__ == '__main__':
    main()
