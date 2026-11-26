"""Export DTU depth, fuse it with common geometric filters, and optionally score PLYs."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from paper_tables import ROOT, SERIES, load_series, write_csv
from tools.paper_configs import BY_NAME


NAMES = ('Base', 'Base+B', 'Base+A+B+C')
PILOT_SCANS = {'val': ('scan5', 'scan17'), 'test': ('scan4', 'scan12')}


def project(points, intrinsics):
    projected = intrinsics @ points
    z = projected[2]
    safe_z = np.where(z > 0, z, 1)
    return projected[0] / safe_z, projected[1] / safe_z, z


def sample_depth(depth, x, y):
    height, width = depth.shape
    inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x = np.where(inside, x, 0)
    y = np.where(inside, y, 0)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    values = np.stack((depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]))
    wx, wy = x - x0, y - y0
    weights = np.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy))
    valid = np.isfinite(values) & (values > 0)
    inside &= (valid | (weights == 0)).all(axis=0)
    return (np.where(valid, values, 0) * weights).sum(axis=0), inside


def fuse_view(reference, sources, pixel_threshold, relative_threshold, min_consistent):
    depth, k, extrinsics = (reference[key] for key in ('depth', 'intrinsics', 'extrinsics'))
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    depth = depth[y, x].astype(np.float64)
    pixels = np.stack((x, y, np.ones_like(x)))
    rays = np.linalg.solve(k, pixels)
    camera_to_world = np.linalg.inv(extrinsics)
    reference_points = rays * depth
    votes = np.zeros(len(depth), dtype=np.int16)
    depth_sum = depth.copy()
    for source in sources:
        src_k, src_ext = source['intrinsics'], source['extrinsics']
        ref_to_src = src_ext @ camera_to_world
        src_points = ref_to_src[:3, :3] @ reference_points + ref_to_src[:3, 3:4]
        sx, sy, sz = project(src_points, src_k)
        src_depth, inside = sample_depth(source['depth'], sx, sy)
        src_rays = np.linalg.solve(src_k, np.stack((sx, sy, np.ones_like(sx))))
        src_to_ref = np.linalg.inv(ref_to_src)
        back = src_to_ref[:3, :3] @ (src_rays * src_depth) + src_to_ref[:3, 3:4]
        bx, by, bz = project(back, k)
        match = (
            inside & (sz > 0) & (bz > 0) &
            (np.hypot(bx - x, by - y) < pixel_threshold) &
            (np.abs(bz - depth) < relative_threshold * depth)
        )
        votes += match
        depth_sum += np.where(match, bz, 0)
    keep = votes >= min_consistent
    averaged_depth = depth_sum[keep] / (1 + votes[keep])
    xyz = camera_to_world[:3, :3] @ (rays[:, keep] * averaged_depth) + camera_to_world[:3, 3:4]
    return xyz.T.astype('<f4'), len(depth)


def write_ply(path, points):
    header = (
        f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
        'property float x\nproperty float y\nproperty float z\nend_header\n'
    )
    with path.open('wb') as handle:
        handle.write(header.encode('ascii'))
        points.astype('<f4', copy=False).tofile(handle)


def historical_protocol(args, options):
    _, summaries = load_series(ROOT / 'eval', args.series, validation=args.split == 'val')
    result = {}
    for name in NAMES:
        row = summaries[name]['full']
        checkpoint = Path(row['checkpoint'])
        values = {key: float(row[key]) for key in
                  ('range_sigma_scale', 'range_min_scale', 'range_max_scale', 'hypothesis_residual_scale')}
        result[name] = dict(vars(BY_NAME[name]), **values, options=options,
                            checkpoint=str((args.checkpoint_root / checkpoint.parent.name / checkpoint.name).resolve()),
                            eval_nviews=int(row['eval_nviews']), light=int(row['light']))
    return result


def export_depths(args, folder, protocol, scanlist):
    import torch
    from torch.utils.data import DataLoader
    from datasets.dtu_yao import MVSDataset
    from tools.eval_region_metrics_dtu_yao import build_model, load_checkpoint, read_train_camera, to_cuda

    settings = SimpleNamespace(**protocol['options'], **{k: protocol[k] for k in
                               ('range_sigma_scale', 'range_min_scale', 'range_max_scale',
                                'hypothesis_residual_scale', 'model_type')}, vismode='soft')
    dataset = MVSDataset(str(args.dtu_root), str(scanlist), 'test', protocol['eval_nviews'],
                         settings.numdepth, settings.interval_scale)
    dataset.metas = [m for m in dataset.metas if m[1] == protocol['light']]
    if not dataset.metas:
        raise ValueError('No reference views found')
    model = build_model(settings).cuda().eval()
    load_checkpoint(model, protocol['checkpoint'])
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    records = {scan: [] for scan in args.scans}
    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            scan, light, ref, sources = dataset.metas[idx]
            sources = sources[:protocol['eval_nviews'] - 1]
            if len(set(sources)) != protocol['eval_nviews'] - 1 or ref in sources:
                raise ValueError(f'{scan}/{ref}: unexpected source view list')
            sample = to_cuda(batch)
            _, prediction, _ = model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
            depth = prediction[0, 0].cpu().numpy()
            if tuple(2 * size for size in depth.shape) != tuple(sample['imgs'].shape[-2:]):
                raise ValueError(f'{scan}/{ref}: unexpected native third-stage resolution')
            k, extrinsics = read_train_camera(str(args.dtu_root), ref)
            k[:2] *= 2  # Camera files are at quarter resolution; stage three is at half resolution.
            scan_dir = folder / scan
            scan_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(scan_dir / f'{ref:08d}.npz', depth=depth.astype(np.float32),
                                intrinsics=k, extrinsics=extrinsics)
            records[scan].append(dict(view=ref, sources=sources))
            if (idx + 1) % 25 == 0:
                print(f"{protocol['model_type']}: exported {idx + 1}/{len(dataset)} depth maps", flush=True)
    for scan, views in records.items():
        (folder / scan / 'views.json').write_text(json.dumps(views, indent=2) + '\n', encoding='utf-8')


def fuse_scan(depth_dir, ply_path, args):
    start = perf_counter()
    views = json.loads((depth_dir / 'views.json').read_text(encoding='utf-8'))
    if not views or len({view['view'] for view in views}) != len(views):
        raise ValueError(f'{depth_dir}: empty or duplicate reference views')
    depths = {}
    for view in views:
        with np.load(depth_dir / f"{view['view']:08d}.npz", allow_pickle=False) as data:
            depths[view['view']] = dict(data)
    clouds, valid = [], 0
    for view in views:
        missing = set(view['sources']) - depths.keys()
        if missing:
            raise ValueError(f'{depth_dir}: missing source depth maps {sorted(missing)}')
        points, count = fuse_view(depths[view['view']], [depths[src] for src in view['sources']],
                                  args.pixel_threshold, args.relative_threshold, args.min_consistent)
        valid += count
        clouds.append(points)
    points = np.concatenate(clouds)
    if not len(points):
        raise ValueError(f'{depth_dir}: geometric filtering retained no points')
    write_ply(ply_path, points)
    return dict(reference_views=len(views), valid_depth_pixels=valid, retained_points=len(points),
                retention=len(points) / valid, fusion_seconds=perf_counter() - start)


def score_cloud(args, scan, ply_path):
    evaluation_dir = ply_path.parent / f'{scan}_evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    result = subprocess.run([
        sys.executable, str(args.evaluator), '--data', str(ply_path), '--scan', scan[4:],
        '--mode', 'pcd', '--dataset_dir', str(args.dtu_eval_root), '--vis_out_dir', str(evaluation_dir),
    ], capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = evaluation_dir / 'evaluator.log'
    log.write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(f'DTU evaluator failed: {log}')
    accuracy, completeness, overall = map(float, result.stdout.strip().splitlines()[-1].split())
    if not all(math.isfinite(value) for value in (accuracy, completeness, overall)):
        raise ValueError(f'Nonfinite DTU metrics: {log}')
    return dict(accuracy_mm=accuracy, completeness_mm=completeness, overall_mm=overall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dtu_root', type=Path)
    parser.add_argument('checkpoint_root', type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--scans', nargs='+')
    parser.add_argument('--series', choices=tuple(SERIES), default='View5')
    parser.add_argument('--evaluator', type=Path, help='Path to DTUeval-python eval.py')
    parser.add_argument('--dtu_eval_root', type=Path, help='Directory containing ObsMask/ and Points/')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/first_batch_existing_weights/dtu_fusion')
    parser.add_argument('--pixel_threshold', type=float, default=1.0)
    parser.add_argument('--relative_threshold', type=float, default=0.01)
    parser.add_argument('--min_consistent', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--fusion_only', action='store_true', help='Reuse already exported depth maps')
    args = parser.parse_args()
    args.scans = args.scans or PILOT_SCANS[args.split]
    allowed = (ROOT / 'lists/dtu' / f'{args.split}.txt').read_text().split()
    if len(set(args.scans)) != len(args.scans) or not set(args.scans) <= set(allowed):
        parser.error('scans must be unique members of the selected split')
    if bool(args.evaluator) != bool(args.dtu_eval_root):
        parser.error('provide both --evaluator and --dtu_eval_root to evaluate point clouds')
    if not all(math.isfinite(x) and x > 0 for x in (args.pixel_threshold, args.relative_threshold)):
        parser.error('thresholds must be finite and positive')
    if not 1 <= args.min_consistent <= 4:
        parser.error('--min_consistent must be between 1 and 4')
    if args.evaluator:
        args.evaluator = args.evaluator.resolve()
        args.dtu_eval_root = args.dtu_eval_root.resolve()
        required = [args.evaluator]
        for scan in args.scans:
            number = int(scan[4:])
            required.extend(args.dtu_eval_root / p for p in (
                f'ObsMask/ObsMask{number}_10.mat', f'ObsMask/Plane{number}.mat',
                f'Points/stl/stl{number:03d}_total.ply'))
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            parser.error('missing evaluator or reference files: ' + ', '.join(missing))

    out = args.outdir.resolve() / args.series / args.split
    out.mkdir(parents=True, exist_ok=True)
    scanlist = out / 'scans.txt'
    scanlist.write_text('\n'.join(args.scans) + '\n', encoding='ascii')
    options = json.loads((ROOT / 'tools/paper_eval_args.example.json').read_text(encoding='utf-8'))
    protocols = historical_protocol(args, options)
    if not args.fusion_only:
        missing = [p['checkpoint'] for p in protocols.values() if not Path(p['checkpoint']).is_file()]
        if missing:
            parser.error('missing checkpoint files: ' + ', '.join(missing))
    settings = dict(split=args.split, series=args.series, scans=args.scans, depth_unit='mm',
                    pixel_threshold=args.pixel_threshold, relative_threshold=args.relative_threshold,
                    min_consistent_source_views=args.min_consistent, configurations=protocols,
                    evaluator=str(args.evaluator) if args.evaluator else None,
                    evaluator_sha256=sha256(args.evaluator.read_bytes()).hexdigest() if args.evaluator else None)
    report = out / f'px{args.pixel_threshold:g}_rel{args.relative_threshold:g}_min{args.min_consistent}'
    report.mkdir(exist_ok=True)
    (report / 'protocol.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    rows = []
    for name, protocol in protocols.items():
        depth_dir = out / 'depth' / name
        manifest = depth_dir / 'protocol.json'
        depth_protocol = dict(dtu_root=str(args.dtu_root.resolve()), **protocol)
        if args.fusion_only:
            if json.loads(manifest.read_text(encoding='utf-8')) != depth_protocol:
                raise ValueError(f'{name}: cached depth protocol differs from requested protocol')
        else:
            export_depths(args, depth_dir, protocol, scanlist)
            manifest.write_text(json.dumps(depth_protocol, indent=2) + '\n', encoding='utf-8')
        cloud_dir = report / name
        cloud_dir.mkdir(exist_ok=True)
        for scan in args.scans:
            ply = cloud_dir / f'{scan}.ply'
            metrics = fuse_scan(depth_dir / scan, ply, args)
            if args.evaluator:
                metrics.update(score_cloud(args, scan, ply))
            rows.append(dict(series=args.series, scan=scan, configuration=name,
                             raw_model_type=protocol['model_type'], raw_csv_code=protocol['csv_code'], **metrics))
            print(f'{name} {scan}: retained {metrics["retention"]:.2%}', flush=True)
    write_csv(report / 'pointcloud_per_scan.csv', rows)
    print(f'Point-cloud results written to {report}')


if __name__ == '__main__':
    main()
