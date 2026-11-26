"""Audit the direct depth gradient and the entropy-to-uncertainty route of factor A."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from paper_tables import ROOT, SERIES, write_csv
from datasets.dtu_yao import MVSDataset
from models.dynamic_visibility import ground_truth_pair_visibility
from models.vismvsnet_oa import VisMVSLoss
from tools.eval_region_metrics_dtu_yao import build_model, load_checkpoint, to_cuda


def gradient_norms(loss, depth, uncertainty, stage, *, retain_graph):
    groups = {
        'pair_depth_tensor': (depth,),
        'uncertainty_tensor': (uncertainty,),
        'pair_score_head': tuple(stage.reg_pair.parameters()),
        'uncertainty_head': tuple(stage.uncert_net.uncert_head.parameters()),
        'shared_pair_regularizer': tuple(stage.reg.parameters()),
    }
    flat = tuple(t for tensors in groups.values() for t in tensors)
    grads = iter(torch.autograd.grad(loss, flat, retain_graph=retain_graph, allow_unused=True))
    norms = {}
    for name, tensors in groups.items():
        values = [next(grads) for _ in tensors]
        squared = sum(float(g.detach().double().square().sum()) for g in values if g is not None)
        norms[name + '_l2'] = squared ** 0.5
    return norms


def audit_sample(model, sample, meta, settings, number):
    stage = getattr(model, f'stage{number}')
    captured = []
    hook = stage.register_forward_pre_hook(lambda module, inputs: captured.append(inputs))
    try:
        with torch.no_grad():
            model(sample['imgs'], sample['proj_matrices'], sample['depth_values'])
    finally:
        hook.remove()
    (ref_feat, ref_proj, src_feats, src_projs, hypotheses), = captured
    height, width = ref_feat.shape[-2:]
    shape = (height, width)
    gt = F.interpolate(sample['depth'].unsqueeze(1), size=shape, mode='bilinear',
                       align_corners=False).squeeze(1)
    view_depths = F.interpolate(sample['visibility_depths'], size=shape, mode='nearest')
    view_masks = F.interpolate(sample['visibility_masks'], size=shape, mode='nearest') > 0.5
    interval = sample['depth_values'][:, 1] - sample['depth_values'][:, 0]
    scan, light, ref, sources = meta
    for index, (src_feat, src_proj) in enumerate(zip(src_feats, src_projs)):
        source = index + 1
        target, supervised = ground_truth_pair_visibility(
            view_depths[:, 0], view_depths[:, source], view_masks[:, 0], view_masks[:, source],
            ref_proj, src_proj, height, width,
            occ_abs_tol=settings.occ_abs_tol, occ_rel_tol=settings.occ_rel_tol)
        supervised = supervised.squeeze(1)
        visible = supervised & (target.squeeze(1) > 0.5)
        occluded = supervised & ~visible

        # Keep the stage inputs fixed and build an autograd graph for one source.
        _, _, pair_results, _ = stage(ref_feat, ref_proj, [src_feat], [src_proj], hypotheses, mode='soft')
        depth, uncertainty = pair_results[0][:2]
        error = (depth - gt).abs() / interval.view(-1, 1, 1)
        cases = [(group, mask, rule) for group, mask in (('visible', visible), ('occluded', occluded))
                 if mask.any() for rule in ('factor_a', 'no_detach')]
        if hypotheses.ndim == 2:
            limits = hypotheses[:, :1, None], hypotheses[:, -1:, None]
        else:
            limits = hypotheses[:, 0], hypotheses[:, -1]
        outside = (gt < limits[0]) | (gt > limits[1])
        for step, (group, mask, rule) in enumerate(cases):
            nll = VisMVSLoss._pair_uncertainty_nll(error, uncertainty.squeeze(1),
                                                    visible if rule == 'factor_a' else None)
            loss = nll[mask].mean()
            norms = gradient_norms(loss, depth, uncertainty, stage, retain_graph=step + 1 < len(cases))
            if rule == 'factor_a' and group == 'occluded' and norms['pair_depth_tensor_l2'] != 0:
                raise ValueError('Occluded pair depth received a direct gradient under factor A')
            yield dict(scan=scan, reference_view=ref, source_view=sources[index], light=light,
                       stage=number, group=group, loss_rule=rule, pixels=int(mask.sum()),
                       outside_search_range=int((mask & outside).sum()), nll=float(loss.detach()), **norms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dtu_root', type=Path)
    parser.add_argument('checkpoint_root', type=Path)
    parser.add_argument('--series', choices=('all', *SERIES), default='all')
    parser.add_argument('--stage', choices=(1, 2, 3), type=int, default=3)
    parser.add_argument('--visibility_downsample', type=int, default=2)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/first_batch_existing_weights/dtu_gradient_audit')
    args = parser.parse_args()
    options = json.loads((ROOT / 'tools/paper_eval_args.example.json').read_text(encoding='utf-8'))
    settings = SimpleNamespace(**options, model_type='oa_hyp', vismode='soft', range_sigma_scale=2.0,
                               range_min_scale=1.0, range_max_scale=2.0, hypothesis_residual_scale=1.0)
    dataset = MVSDataset(str(args.dtu_root), str(ROOT / 'lists/dtu/test.txt'), 'test', 5,
                         settings.numdepth, settings.interval_scale, return_visibility_gt=True,
                         visibility_downsample=args.visibility_downsample)
    selected = {(x['scan'], x['light'], x['view']) for x in
                json.loads((ROOT / 'tools/paper_dtu_smoke_samples.json').read_text(encoding='utf-8'))}
    indices = [i for i, meta in enumerate(dataset.metas) if tuple(meta[:3]) in selected]
    if len(indices) != len(selected):
        raise ValueError('The DTU dataset is missing fixed audit samples')
    rows = []
    for series in SERIES if args.series == 'all' else (args.series,):
        checkpoint = (args.checkpoint_root / f'oa_hyp_{series.lower()}' / 'best_2mm.ckpt').resolve()
        model = build_model(settings).cuda().eval()
        load_checkpoint(model, checkpoint)
        for index in indices:
            sample = to_cuda(default_collate([dataset[index]]))
            for row in audit_sample(model, sample, dataset.metas[index], settings, args.stage):
                rows.append(dict(series=series, raw_model_type='oa_hyp', checkpoint=str(checkpoint),
                                 visibility_downsample=args.visibility_downsample, **row))
        del model
    if {row['group'] for row in rows} != {'visible', 'occluded'}:
        raise ValueError('The selected samples did not provide both visibility classes')
    path = args.outdir / f'stage{args.stage}_gradient_audit.csv'
    write_csv(path, rows)
    print(f'{len(rows)} gradient measurements written to {path}')


if __name__ == '__main__':
    main()
