"""Full-scene Base/ABC comparison with GT-defined difficulty masks."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
import cv2
import numpy as np
from PIL import Image

NAMES = ('Base', 'Base+A+B+C')
CATEGORIES = ('large_disparity_dominant', 'occlusion_dominant', 'joint_difficulty')
CATEGORY_LABELS = {'large_disparity_dominant': 'Large disparity dominant',
                   'occlusion_dominant': 'Complex occlusion dominant',
                   'joint_difficulty': 'Large disparity and occlusion',
                   'mixed_other': 'Other cached scene'}


def rgba(values, valid, cmap, norm):
    colors = plt.get_cmap(cmap)(norm(values), bytes=True)
    colors[~valid] = (0, 0, 0, 255)
    return colors


def binary_rgba(mask):
    result = np.zeros((*mask.shape, 4), dtype=np.uint8)
    result[..., 3] = 255
    result[mask] = (255, 255, 255, 255)
    return result


def occlusion_count(arrays, metadata):
    """Count exactly the region sources used by the evaluator."""
    if 'occluded_count' in arrays:
        count = arrays['occluded_count']
    else:
        if 'source_occluded' not in arrays:
            raise ValueError('Cache lacks source occlusion labels; export predictions again')
        labels = arrays['source_occluded'].astype(bool)
        source_ids = metadata.get('source_views')
        region_ids = metadata.get('region_source_views', source_ids)
        if source_ids is not None and region_ids is not None:
            if not set(region_ids) <= set(source_ids):
                raise ValueError('Region sources absent from old cache; export again')
            labels = labels[[source_ids.index(i) for i in region_ids]]
        count = labels.sum(axis=0)
    if count.shape != arrays['gt'].shape or not np.isfinite(count).all() or (count < 0).any():
        raise ValueError('Invalid source-occlusion counts')
    if not np.equal(count, np.floor(count)).all():
        raise ValueError('Occlusion counts must be integers')
    return np.where(arrays['valid'].astype(bool), count, 0).astype(np.uint16)


def scene_descriptor(arrays, count):
    valid = arrays['valid'].astype(bool)
    large = arrays['mask_large_disparity'].astype(bool) & valid
    complex_occ = (count >= 2) & valid
    joint = large & complex_occ
    n = int(valid.sum())
    l, c, j = (int(x.sum()) for x in (large, complex_occ, joint))
    disparity = arrays['disparity'][valid]
    if not n or not np.isfinite(disparity).all():
        raise ValueError('Invalid GT disparity or no valid scene pixels')
    return {'valid_pixels': n, 'large_disparity_pixels': l, 'occluded_ge2_pixels': c,
            'joint_ge2_pixels': j, 'large_disparity_fraction': l/n,
            'occluded_ge2_fraction': c/n, 'joint_ge2_fraction': j/n,
            'joint_fraction_of_large': j/l if l else 0.,
            'joint_fraction_of_occlusion': j/c if c else 0.,
            'disparity_p80_px': float(np.percentile(disparity, 80)),
            'disparity_max_px': float(np.max(disparity))}


def category_scores(record):
    """GT-only eligibility and severity, never ranked by model improvement."""
    result = {}
    if (record['large_disparity_pixels'] >= 30 and record['occluded_ge2_fraction'] <= .10
            and record['joint_fraction_of_large'] <= .25):
        result['large_disparity_dominant'] = record['disparity_p80_px']
    if (record['occluded_ge2_pixels'] >= 30 and record['occluded_ge2_fraction'] >= .10
            and record['joint_fraction_of_occlusion'] <= .35):
        result['occlusion_dominant'] = record['occluded_ge2_fraction']
    if (record['joint_ge2_pixels'] >= 30 and record['joint_ge2_fraction'] >= .05
            and record['joint_fraction_of_large'] >= .25):
        result['joint_difficulty'] = record['joint_ge2_fraction']
        # Keep the displayed categories distinct when both rules could apply.
        result.pop('occlusion_dominant', None)
        result.pop('large_disparity_dominant', None)
    return result


def select_scene_examples(records, per_category=2):
    from tools.visualize_paper_results import sample_key
    chosen = {}
    for category in CATEGORIES:
        candidates = [r for r in records if category in category_scores(r)]
        candidates.sort(key=lambda r: (-category_scores(r)[category], r['scan'], r['view'], r['light']))
        used_scans = set()
        for record in candidates:
            if record['scan'] in used_scans:
                continue
            key = sample_key(record)
            if key not in chosen:
                chosen[key] = {'scan': key[0], 'view': key[1], 'light': key[2],
                               'scene_categories': [], 'reason': 'Fixed GT geometry; not ranked by improvement',
                               'scene_geometry': record}
            chosen[key]['scene_categories'].append(category)
            used_scans.add(record['scan'])
            if len(used_scans) >= per_category:
                break
        if not used_scans:
            raise ValueError(f'No eligible scenes for {category}; inspect scene_candidates.csv')
    return list(chosen.values())


def select_from_dtu(args, series):
    """Screen the test set using GT/cameras on CPU before model inference."""
    from datasets.dtu_yao import MVSDataset
    from tools.eval_region_metrics_dtu_yao import geometry_region_maps, read_dtu_depth, percentile_mask
    from tools.visualize_paper_results import load_series, sample_key, write_csv
    per_image, summaries = load_series(args.eval_root, series)
    protocol = summaries['Base']['full']
    overrides = json.loads(Path(args.eval_args_json).read_text()) if args.eval_args_json else {}
    dataset = MVSDataset(args.testpath, args.testlist, 'test', int(protocol['eval_nviews']))
    allowed = {k[:3] for k in per_image['Base'] if k[3] == 'full'}
    records = []
    for scan, light, ref, sources in dataset.metas:
        key = (scan, ref, light)
        if key not in allowed:
            continue
        gt = read_dtu_depth(args.testpath, scan, ref)
        mask_path = Path(args.testpath)/'Depths'/f'{scan}_train'/f'depth_visual_{ref:04d}.png'
        raw = np.asarray(Image.open(mask_path), dtype=np.float32)/255.
        if raw.ndim == 3:
            raw = raw.mean(axis=2)
        valid = (raw > .5) & np.isfinite(gt)
        if int(valid.sum()) != int(per_image['Base'][(*key, 'full')]['pixels']):
            raise ValueError(f'{key}: GT validity differs from original CSV')
        geom = geometry_region_maps(args.testpath, scan, ref,
                                    sources[:int(protocol['region_nviews'])-1], gt,
                                    float(overrides.get('occ_abs_tol', 2.)), float(overrides.get('occ_rel_tol', .01)))
        large = percentile_mask(geom['disparity'], valid, float(overrides.get('large_disp_pct', 80.)))
        if int(large.sum()) != int(per_image['Base'][(*key, 'large_disparity')]['pixels']):
            raise ValueError(f'{key}: large-disparity mask differs from original CSV')
        arrays = {'gt': gt, 'valid': valid, 'mask_large_disparity': large, 'disparity': geom['disparity']}
        records.append(dict(scan=scan, view=ref, light=light,
                            **scene_descriptor(arrays, geom['source_occluded'].sum(axis=0))))
        if len(records) % 100 == 0:
            print(f'{series}: GT geometry screened {len(records)}/{len(allowed)} samples', flush=True)
    if {sample_key(r) for r in records} != allowed:
        raise ValueError('GT selection does not cover the original CSV test samples')
    out = Path(args.outdir)/series
    write_csv(out/'scenes'/'scene_candidates.csv', records)
    selections = select_scene_examples(records, args.examples_per_category)
    (out/'scene_selection.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')
    print(f'{series}: selected {len(selections)} scenes across three categories -> {out / "scene_selection.json"}', flush=True)


def yellow_contours(raster, large, complex_occ):
    result = raster.copy()
    for mask in (large, complex_occ):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (255, 255, 0, 255) if result.shape[-1] == 4 else (255, 255, 0)
        cv2.drawContours(result, contours, -1, color, 1, lineType=cv2.LINE_8)
    return result


def overlay_rgb(rgb, large, complex_occ):
    result = rgb.astype(float).copy()
    for mask, color in ((large & ~complex_occ, (255, 40, 40)),
                        (complex_occ & ~large, (0, 230, 100)),
                        (large & complex_occ, (255, 220, 0))):
        result[mask] = .65*result[mask] + .35*np.asarray(color)
    return np.uint8(np.clip(result, 0, 255))


def scene_metric_rows(arrays, name, key, masks, gain):
    from tools.visualize_paper_results import METRICS
    error = np.abs(arrays['pred']-arrays['gt'])
    result = []
    for region, mask in masks.items():
        n = int(mask.sum())
        row = dict(analysis_config=name, sample=key, scope='full_image', region=region, pixels=n,
                   **{m: float('nan') for m in METRICS})
        row.update(improved_pixels_pct=float('nan'), worsened_pixels_pct=float('nan'), unchanged_pixels_pct=float('nan'))
        if n:
            values = error[mask]
            row.update(abs=float(values.mean()), **{f'acc{t}': float((values < t).mean()) for t in (2,4,8)},
                       improved_pixels_pct=float(100*(gain[mask] > 0).mean()),
                       worsened_pixels_pct=float(100*(gain[mask] < 0).mean()),
                       unchanged_pixels_pct=float(100*(gain[mask] == 0).mean()))
            for stage in (1,2,3):
                row[f'stage{stage}_in_range'] = float(arrays[f'stage{stage}_coverage'][mask].mean())
                row[f'stage{stage}_range_width'] = float(arrays[f'stage{stage}_width'][mask].mean())
        result.append(row)
    return result


def render_paper_sample(args, series, key, files, selection=None):
    from tools.visualize_paper_results import REGIONS, add_deltas, write_csv, save_figure
    arrays, metadata = {}, {}
    for name in NAMES:
        with np.load(files[name]/'arrays.npz', allow_pickle=False) as loaded:
            arrays[name] = dict(loaded)
        metadata[name] = json.loads((files[name]/'metadata.json').read_text(encoding='utf-8'))
        if metadata[name]['analysis_config'] != name:
            raise ValueError(f'{key}: folder and analysis configuration differ')
    base, full = (arrays[n] for n in NAMES)
    for field in ('rgb', 'gt', 'valid', 'disparity', 'occlusion_ratio', *('mask_'+r for r in REGIONS)):
        if not np.array_equal(base[field], full[field], equal_nan=True):
            raise ValueError(f'{key}: unaligned Base/full model {field}')
    valid = base['valid'].astype(bool)
    if not valid.any() or not np.isfinite(base['gt'][valid]).all():
        raise ValueError('No finite valid GT')
    for name in NAMES:
        if arrays[name]['pred'].shape != valid.shape or not np.isfinite(arrays[name]['pred'][valid]).all():
            raise ValueError(f'{key}/{name}: invalid prediction')
        for region in REGIONS:
            if (arrays[name]['mask_'+region].astype(bool) & ~valid).any():
                raise ValueError('Region contains invalid GT pixels')
    count = occlusion_count(base, metadata['Base'])
    if not np.array_equal(count, occlusion_count(full, metadata[NAMES[1]])):
        raise ValueError('Base/full model source-occlusion counts differ')
    large = base['mask_large_disparity'].astype(bool) & valid
    any_occ, complex_occ = (count >= 1) & valid, (count >= 2) & valid
    if not np.array_equal(any_occ, base['mask_occluded_any'].astype(bool)):
        raise ValueError('Source counts disagree with fixed any-occlusion mask')
    description = scene_descriptor(base, count)
    categories = (selection or {}).get('scene_categories') or metadata['Base'].get('scene_categories')
    if not categories:
        scores = category_scores(description)
        categories = [next((c for c in reversed(CATEGORIES) if c in scores), 'mixed_other')]
    if not isinstance(categories, list) or any(c not in CATEGORY_LABELS for c in categories):
        raise ValueError('Unknown scene category')
    eligible = category_scores(description)
    if any(c != 'mixed_other' and c not in eligible for c in categories):
        raise ValueError(f'{key}: category does not meet the recorded GT criteria')
    errors = {n: np.abs(arrays[n]['pred']-base['gt']) for n in NAMES}
    gain = errors['Base']-errors[NAMES[1]]
    lo, hi = float(base['gt'][valid].min()), float(base['gt'][valid].max())
    if lo == hi:
        lo, hi = lo-.5, hi+.5
    depth_norm, error_norm = Normalize(lo, hi, clip=True), Normalize(0, args.error_max, clip=True)
    gain_norm = TwoSlopeNorm(vmin=-args.gain_max, vcenter=0, vmax=args.gain_max)
    disp_max = max(1e-6, description['disparity_max_px'])
    panels = [
        ('reference_rgb', 'Reference RGB', base['rgb'], None, None, None),
        ('gt_depth', 'GT depth', rgba(base['gt'],valid,'turbo',depth_norm), 'turbo',depth_norm,'Depth (mm)'),
        ('large_disparity_map', 'GT displacement', rgba(base['disparity'],valid,'viridis',Normalize(0,disp_max)), 'viridis',Normalize(0,disp_max),'Displacement (px)'),
        ('source_occlusion_ratio', 'Source occlusion ratio', rgba(base['occlusion_ratio'],valid,'viridis',Normalize(0,1)), 'viridis',Normalize(0,1),'Occluded / comparable sources'),
        ('difficulty_overlay', 'Red: disparity; green: occlusion; yellow: both', overlay_rgb(base['rgb'],large,complex_occ),None,None,None),
        ('base_depth', 'Base depth', rgba(base['pred'],valid,'turbo',depth_norm),'turbo',depth_norm,'Depth (mm)'),
        ('full_model_depth', 'Base+A+B+C depth', rgba(full['pred'],valid,'turbo',depth_norm),'turbo',depth_norm,'Depth (mm)'),
        ('base_abs_error', 'Base absolute error', yellow_contours(rgba(errors['Base'],valid,'magma',error_norm),large,complex_occ),'magma',error_norm,'Absolute error (mm)'),
        ('full_model_abs_error', 'Base+A+B+C absolute error', yellow_contours(rgba(errors[NAMES[1]],valid,'magma',error_norm),large,complex_occ),'magma',error_norm,'Absolute error (mm)'),
        ('error_gain', 'Error reduction: red better / blue worse', yellow_contours(rgba(gain,valid,'RdBu_r',gain_norm),large,complex_occ),'RdBu_r',gain_norm,'Base error - ABC error (mm)'),
        ('occluded_any_mask', 'Occluded in >=1 source view', binary_rgba(any_occ),None,None,None),
        ('occluded_ge2_mask', 'Occluded in >=2 source views', binary_rgba(complex_occ),None,None,None),
        ('large_disparity_mask', 'Large disparity mask', binary_rgba(large),None,None,None),
        ('large_disp_and_occluded_any_mask', 'Large disparity & >=1 occluded view', binary_rgba(large&any_occ),None,None,None),
        ('large_disp_and_occluded_ge2_mask', 'Large disparity & >=2 occluded views', binary_rgba(large&complex_occ),None,None,None)]
    masks = {r: base['mask_'+r].astype(bool)&valid for r in REGIONS}
    masks.update(occluded_ge2=complex_occ, large_disp_and_occluded_ge2=large&complex_occ)
    rows = [r for n in NAMES for r in scene_metric_rows(arrays[n],n,key,masks,gain)]
    add_deltas(rows)
    reproduction = {}
    for name in NAMES:
        status = files[name].parent/'exported_metrics'/'reproduction_status.json'
        reproduction[name] = json.loads(status.read_text()) if status.is_file() else {'status':'not checked'}

    def draw_panel(ax, panel):
        _,title,raster,_,_,_ = panel
        ax.imshow(raster,interpolation='nearest'); ax.set_title(title,fontsize=9); ax.axis('off')

    for name, depth_index, error_index in ((NAMES[0],5,7),(NAMES[1],6,8)):
        folder = Path(args.outdir)/series/name/key/'figures'/'scenes'
        folder.mkdir(parents=True,exist_ok=True)
        for index in (depth_index,error_index):
            Image.fromarray(panels[index][2]).save(folder/(panels[index][0]+'.png'))

    for category in categories:
        out = Path(args.outdir)/series/'scenes'/category/key
        directory = out/'panels'; directory.mkdir(parents=True,exist_ok=True)
        manifest = []
        for i,panel in enumerate(panels,1):
            Image.fromarray(panel[2]).save(directory/(panel[0]+'.png'))
            fig,ax = plt.subplots(figsize=(4.8,3.8),layout='constrained'); draw_panel(ax,panel)
            save_figure(fig,directory/'with_title'/panel[0])
            manifest.append({'position':i,'title':panel[1],'png':'panels/'+panel[0]+'.png',
                             'title_png':'panels/with_title/'+panel[0]+'.png',
                             'title_pdf':'panels/with_title/'+panel[0]+'.pdf'})
        fig,axes = plt.subplots(3,5,figsize=(22,12),layout='constrained')
        for ax,panel in zip(axes.flat,panels): draw_panel(ax,panel)
        fig.suptitle(f'{key} | {CATEGORY_LABELS[category]} | Base vs Base+A+B+C',fontsize=13)
        fig.legend(handles=[Patch(color=c,label=t) for c,t in
                    (('#ff2828','Disparity only (RGB overlay)'),('#00e664','>=2 occluded views only (RGB overlay)'),
                     ('#ffdc00','Intersection (RGB overlay)'),('#ffff00','GT difficulty contours (error maps)'))],
                   loc='lower center',ncol=2,fontsize=9)
        fig.get_layout_engine().set(rect=(0,.06,1,.94))
        save_figure(fig,out/'overview_3x5')
        write_csv(out/'region_metrics.csv',rows)
        settings = {'schema_version':3,'sample':key,'analysis_configs':list(NAMES),'scene_category':category,
                    'scene_geometry':description,'depth_min_mm':lo,'depth_max_mm':hi,
                    'error_max_mm':args.error_max,'gain_max_mm':args.gain_max,'disparity_max_px':disp_max,
                    'gain_definition':'abs(Base-GT) - abs(Base+A+B+C-GT)',
                    'gain_colors':{'positive':'red','negative':'blue','zero':'white'},
                    'yellow_contours':'fixed GT large disparity and >=2 occluded source views',
                    'occluded_ge2_definition':'source occlusion count >= 2, not majority fraction',
                    'occlusion_ratio_definition':'occluded source count / GT-comparable source count',
                    'metric_source':'current exported arrays; unclipped errors',
                    'csv_reproduction':reproduction,'panels':manifest,'show_colorbars':False,
                    'show_axis_ticks':False}
        (out/'display_settings.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
        caption = (f'{key}: {CATEGORY_LABELS[category]}. Base and Base+A+B+C use the same GT and source views.\n'
                   f'Depth colors: {lo:.3f}-{hi:.3f} mm; absolute-error colors: 0-{args.error_max:g} mm; '
                   f'error-reduction colors: {-args.gain_max:g} to {args.gain_max:g} mm.\n'
                   'Error reduction = abs(Base-GT) - abs(Base+A+B+C-GT): red means improvement, '
                   'blue means degradation, white means zero. Yellow error-map contours denote fixed GT '
                   'large-disparity and >=2-source occlusion regions.\n'
                   'No colorbars or axis ticks are drawn. Display clipping never changes numeric metrics.\n')
        (out/'figure_caption.txt').write_text(caption,encoding='utf-8')
        np.savez_compressed(out/'comparison_arrays.npz',valid=valid,base_error=errors['Base'],
                            full_model_error=errors[NAMES[1]],error_gain=gain,occluded_count=count,
                            large_disparity_mask=large,occluded_any_mask=any_occ,occluded_ge2_mask=complex_occ)
        print(f'{series}/{category}/{key}: full-scene overview + 15 separate panels + regional metrics -> {out}',flush=True)
    return {'sample':key,'scene_categories':categories,**description}
