"""Export actual predictions/diagnostics without changing evaluation math."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.paper_configs import BY_MODEL


def resize_tensor(item, size, mode='bilinear'):
    item = item.reshape(1, 1, *item.shape[-2:])
    kwargs = {'align_corners': False} if mode == 'bilinear' else {}
    return F.interpolate(item.float(), size=size, mode=mode, **kwargs)[0, 0].detach().cpu().numpy()


def dump_sample(args, sample, batch_item, outputs, depth_gt, depth_est,
                valid, geometry, region_masks, range_masks, range_widths,
                meta, confidences):
    config = BY_MODEL[args.model_type]
    scan, light, ref, src = meta
    folder = Path(args.paper_dumpdir) / config.name / f'{scan}_view{ref:02d}_light{light}'
    folder.mkdir(parents=True, exist_ok=True)
    h, w = depth_gt.shape
    image_path = Path(args.testpath) / 'Rectified' / f'{scan}_train' / f'rect_{ref+1:03d}_{light}_r5000.png'
    rgb = np.array(Image.open(image_path).convert('RGB'))
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    arrays = dict(rgb=rgb, gt=depth_gt, pred=depth_est, valid=valid,
                  disparity=geometry['disparity'], occlusion_ratio=geometry['occlusion_ratio'])
    # Region geometry may use a different source count from inference.
    arrays['occluded_count'] = np.where(valid, geometry['source_occluded'].sum(axis=0), 0).astype(np.uint16)
    if 'comparable_count' in geometry:
        arrays['comparable_count'] = geometry['comparable_count']
    arrays.update({'mask_' + name: value for name, value in region_masks.items()})
    # Ground-truth labels are diagnostic only and never influence inference.
    from tools.eval_region_metrics_dtu_yao import geometry_region_maps
    source_geometry = geometry_region_maps(args.testpath, scan, ref, src[:args.eval_nviews-1],
                                          depth_gt, args.occ_abs_tol, args.occ_rel_tol)
    for name in ('source_visible', 'source_occluded', 'source_supervised'):
        arrays[name] = source_geometry[name] & valid[None]
    target = next((x for x in getattr(args, '_paper_selections', [])
                   if (x['scan'], int(x['view']), int(x.get('light', args.light))) == (scan, ref, light)), {})
    probe = target.get('probe')
    if probe is None:
        yy, xx = np.where(region_masks['large_disp_and_occluded'])
        if not len(yy):
            yy, xx = np.where(valid)
        i = len(yy) // 2
        probe = [int(xx[i]), int(yy[i])] if len(yy) else [w//2, h//2]
    if len(probe) != 2 or not (0 <= probe[0] < w and 0 <= probe[1] < h):
        raise ValueError('probe must be an in-image [x,y] coordinate')

    for i, stage in enumerate(outputs, 1):
        depth, pairs, hypotheses = stage[:3]
        sh, sw = depth.shape[-2:]
        hyp = hypotheses[batch_item]
        if hyp.ndim == 1:
            hyp = hyp[:, None, None].expand(-1, sh, sw)
        arrays[f'stage{i}_pred'] = resize_tensor(depth[batch_item], (h, w))
        arrays[f'stage{i}_coverage'] = range_masks[i-1]
        arrays[f'stage{i}_width'] = range_widths[i-1]
        arrays[f'stage{i}_min'] = resize_tensor(hyp[0], (h, w))
        arrays[f'stage{i}_max'] = resize_tensor(hyp[-1], (h, w))
        gy = min(sh-1, max(0, int((probe[1]+0.5)*sh/h)))
        gx = min(sw-1, max(0, int((probe[0]+0.5)*sw/w)))
        gt_small = torch.as_tensor(cv2.resize(depth_gt, (sw, sh), interpolation=cv2.INTER_NEAREST),
                                   device=hyp.device)
        nearest = (hyp - gt_small[None]).abs().argmin(dim=0)
        logits_gt, logits_probe, predictions = [], [], []
        for pair in pairs:
            u = pair[1][batch_item, 0]
            q = torch.sigmoid(pair[2][batch_item, 0])
            predictions.append(resize_tensor(q, (h, w)))
            if len(pair) == 4:
                z = pair[3][batch_item, 0]
                residual_gt = args.hypothesis_residual_scale * torch.tanh(z.gather(0, nearest[None])[0])
                residual_probe = args.hypothesis_residual_scale * torch.tanh(z[:, gy, gx])
            else:
                residual_gt = torch.zeros_like(u)
                residual_probe = torch.zeros(hyp.shape[0], device=u.device)
            logits_gt.append(-u + residual_gt)
            logits_probe.append(-u[gy, gx] + residual_probe)
        # Diagnostic values from the actual fusion expression. A's probability
        # is saved separately; it is not used as a replacement fusion weight.
        weights = torch.softmax(torch.stack(logits_gt), dim=0)
        arrays[f'stage{i}_weights_at_gt'] = np.stack([resize_tensor(x, (h, w)) for x in weights])
        arrays[f'stage{i}_visibility_pred'] = np.stack(predictions)
        arrays[f'stage{i}_probe_candidates'] = hyp[:, gy, gx].detach().cpu().numpy()
        arrays[f'stage{i}_probe_weights'] = torch.softmax(torch.stack(logits_probe), dim=0).detach().cpu().numpy()
    arrays['confidence'] = resize_tensor(confidences[-1][batch_item], (h, w))
    np.savez_compressed(folder / 'arrays.npz', **arrays)
    provenance = {
        'schema_version': 1, 'analysis_config': config.name,
        'analysis_code': config.analysis_code, 'raw_model_type': config.model_type,
        'raw_csv_code': config.csv_code, 'checkpoint': str(Path(args.loadckpt).resolve()),
        'scan': scan, 'view': ref, 'light': light, 'source_views': src[:args.eval_nviews-1],
        'region_source_views': src[:args.region_nviews-1], 'shape': [h, w],
        'probe': probe, 'roi': target.get('roi'), 'rois': target.get('rois'),
        'scene_categories': target.get('scene_categories'),
        'selection_reason': target.get('reason', 'manual'),
        'gt_valid_at_probe': bool(valid[probe[1], probe[0]]),
        'depth_unit': 'mm', 'width_unit': 'original_depth_interval',
        'nearest_gt_weights_are_diagnostic': True,
        'eval_args': {k: v for k, v in vars(args).items() if not k.startswith('_')},
    }
    (folder / 'metadata.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
