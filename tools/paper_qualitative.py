"""Shared-scale Base/Ours figures and individual, native-resolution panels."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw

NAMES = ('Base', 'Base+A+B+C')
ROI_COLORS = ('#00ff88', '#ff88ff', '#00bfff')


def rgba(values, valid, cmap, norm):
    """Use the identical mapping for the overview, crops and standalone PNGs."""
    colors = plt.get_cmap(cmap)(norm(values), bytes=True)
    colors[~valid] = (0, 0, 0, 255)
    return colors


def resolve_rois(meta, arrays):
    h, w = arrays['gt'].shape
    requested = meta.get('rois')
    if requested is None and meta.get('roi') is not None:
        requested = [meta['roi']]
    if requested is None:
        requested = []
        for region in ('large_disparity', 'occluded_majority', 'large_disp_and_occluded'):
            mask = arrays['mask_' + region].astype(bool) & arrays['valid'].astype(bool)
            yy, xx = np.where(mask)
            if not len(yy):
                continue
            # Choose an actual difficult pixel nearest the region's center.
            i = np.argmin((xx - np.median(xx))**2 + (yy - np.median(yy))**2)
            cw, ch = min(160, w), min(120, h)
            x0 = max(0, min(w-cw, int(xx[i])-cw//2))
            y0 = max(0, min(h-ch, int(yy[i])-ch//2))
            bounds = [x0, y0, x0+cw, y0+ch]
            if not any(r['bounds'] == bounds for r in requested):
                requested.append({'name': region, 'bounds': bounds})
        if not requested:
            requested = [{'name': 'valid_region', 'bounds': [0, 0, w, h]}]
    if not isinstance(requested, list) or not requested:
        raise ValueError('rois must be a nonempty list of bounds or {name, bounds} objects')
    result = []
    for i, item in enumerate(requested, 1):
        bounds = item.get('bounds') if isinstance(item, dict) else item
        name = item.get('name', f'ROI {i}') if isinstance(item, dict) else f'ROI {i}'
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise ValueError('Each ROI needs four integer coordinates')
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bounds):
            raise ValueError('ROI coordinates must be integers')
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('ROI must be [x0,y0,x1,y1] inside exported image dimensions')
        result.append({'id': f'roi_{i:02d}', 'name': str(name), 'bounds': list(bounds),
                       'color': ROI_COLORS[(i-1) % len(ROI_COLORS)]})
    return result


def overlay_rgb(rgb, large, occluded, rois):
    out = rgb.astype(float).copy()
    # Red: large disparity only; yellow: majority occluded only; orange: both.
    for mask, color in ((large & ~occluded, (255, 40, 40)),
                        (occluded & ~large, (255, 230, 0)),
                        (large & occluded, (255, 130, 0))):
        out[mask] = .65*out[mask] + .35*np.asarray(color)
    img = Image.fromarray(np.uint8(np.clip(out, 0, 255)))
    draw = ImageDraw.Draw(img)
    for i, roi in enumerate(rois, 1):
        x0, y0, x1, y1 = roi['bounds']
        draw.rectangle((x0, y0, x1-1, y1-1), outline=roi['color'], width=2)
        draw.text((x0+3, y0+3), f'R{i}', fill=roi['color'])
    return np.asarray(img)


def render_paper_sample(args, series, key, files, selection=None):
    from tools.visualize_paper_results import (REGIONS, region_rows, add_deltas,
                                               write_csv, save_figure)
    arrays, metadata = {}, {}
    for name in NAMES:
        with np.load(files[name]/'arrays.npz', allow_pickle=False) as loaded:
            arrays[name] = dict(loaded)
        metadata[name] = json.loads((files[name]/'metadata.json').read_text(encoding='utf-8'))
        if metadata[name]['analysis_config'] != name:
            raise ValueError(f'{key}: folder and analysis configuration differ')
    base, ours = (arrays[n] for n in NAMES)
    for field in ('rgb', 'gt', 'valid', 'disparity', 'occlusion_ratio',
                  *('mask_' + r for r in REGIONS)):
        if not np.array_equal(base[field], ours[field], equal_nan=True):
            raise ValueError(f'{key}: unaligned Base/Ours {field}')
    valid = base['valid'].astype(bool)
    if not valid.any() or not np.isfinite(base['gt'][valid]).all():
        raise ValueError(f'{key}: no finite valid GT')
    for name in NAMES:
        if arrays[name]['pred'].shape != valid.shape or not np.isfinite(arrays[name]['pred'][valid]).all():
            raise ValueError(f'{key}/{name}: invalid prediction shape or nonfinite valid prediction')
        for region in REGIONS:
            if (arrays[name]['mask_' + region].astype(bool) & ~valid).any():
                raise ValueError(f'{key}/{name}: region contains invalid GT pixels')
    meta = dict(metadata['Base'])
    if selection is not None:
        # Rendering can adjust crops without another GPU inference pass.
        meta.update({k: selection[k] for k in ('roi', 'rois') if k in selection})
        if 'roi' in selection and 'rois' not in selection:
            meta.pop('rois', None)
    rois = resolve_rois(meta, base)
    errors = {n: np.abs(arrays[n]['pred'] - base['gt']) for n in NAMES}
    gain = errors['Base'] - errors['Base+A+B+C']
    lo, hi = float(base['gt'][valid].min()), float(base['gt'][valid].max())
    if hi == lo:
        lo, hi = lo-.5, hi+.5
    depth_norm = Normalize(lo, hi, clip=True)
    error_norm = Normalize(0, args.error_max, clip=True)
    gain_norm = TwoSlopeNorm(vmin=-args.gain_max, vcenter=0, vmax=args.gain_max)
    large = base['mask_large_disparity'].astype(bool) & valid
    occ = base['mask_occluded_majority'].astype(bool) & valid
    binary = np.zeros((*valid.shape, 4), dtype=np.uint8)
    binary[..., 3] = 255
    binary[large] = (255, 255, 255, 255)
    # Each panel is (filename, title, native raster, scalar colormap, norm, unit).
    panels = [
        ('reference_rgb', 'Reference RGB', base['rgb'], None, None, None),
        ('gt_depth', 'GT depth', rgba(base['gt'], valid, 'turbo', depth_norm), 'turbo', depth_norm, 'Depth (mm)'),
        ('large_disparity_mask', 'Large disparity (GT mask)', binary, None, None, None),
        ('source_occlusion_ratio', 'Source occlusion ratio', rgba(base['occlusion_ratio'], valid, 'viridis', Normalize(0, 1)), 'viridis', Normalize(0, 1), 'Occluded source fraction'),
        ('difficulty_overlay', 'Difficult regions / ROIs', overlay_rgb(base['rgb'], large, occ, rois), None, None, None),
        ('base_depth', 'Base depth', rgba(base['pred'], valid, 'turbo', depth_norm), 'turbo', depth_norm, 'Depth (mm)'),
        ('ours_depth', 'Ours depth', rgba(ours['pred'], valid, 'turbo', depth_norm), 'turbo', depth_norm, 'Depth (mm)'),
        ('base_abs_error', 'Base absolute error', rgba(errors['Base'], valid, 'magma', error_norm), 'magma', error_norm, 'Absolute error (mm)'),
        ('ours_abs_error', 'Ours absolute error', rgba(errors['Base+A+B+C'], valid, 'magma', error_norm), 'magma', error_norm, 'Absolute error (mm)'),
        ('error_gain', 'Error reduction (blue = better)', rgba(gain, valid, 'RdBu', gain_norm), 'RdBu', gain_norm, 'Base error - Ours error (mm)'),
    ]
    out = Path(args.outdir)/series/'comparison'/key
    manifest = []

    def draw_panel(ax, panel, bounds=None):
        filename, title, raster, cmap, norm, unit = panel
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            raster = raster[y0:y1, x0:x1]
        ax.imshow(raster, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        if cmap:
            fig = ax.figure
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                               fraction=.04, pad=.02, shrink=.8)
            bar.set_label(unit, fontsize=7)
            bar.ax.tick_params(labelsize=6)

    def save_panel(panel, directory, bounds=None):
        filename, _, raster, _, _, _ = panel
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            raster = raster[y0:y1, x0:x1]
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(raster).save(directory/(filename+'.png'))
        fig, ax = plt.subplots(figsize=(4.8, 3.8), layout='constrained')
        draw_panel(ax, panel, bounds)
        save_figure(fig, directory/'with_legend'/filename)

    for index, panel in enumerate(panels):
        save_panel(panel, out/'panels')
        manifest.append({'position': index+1, 'title': panel[1], 'png': 'panels/'+panel[0]+'.png',
                         'legend_png': 'panels/with_legend/'+panel[0]+'.png',
                         'legend_pdf': 'panels/with_legend/'+panel[0]+'.pdf'})
    # Also group model predictions using the requested analysis names.
    for name, depth_index, error_index in (('Base', 5, 7), ('Base+A+B+C', 6, 8)):
        directory = Path(args.outdir)/series/name/key/'figures'/'paper'
        save_panel(panels[depth_index], directory)
        save_panel(panels[error_index], directory)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), layout='constrained')
    for ax, panel in zip(axes.flat, panels):
        draw_panel(ax, panel)
    fig.suptitle(f'{key} | Base vs Ours', fontsize=13)
    fig.legend(handles=[Patch(color=c, label=t) for c, t in
               (('#ff2828', 'Large disparity only'), ('#ffe600', 'Majority occlusion only'),
                ('#ff8200', 'Both'))], loc='lower center', ncol=3, fontsize=9)
    fig.get_layout_engine().set(rect=(0, .05, 1, .95))
    save_figure(fig, out/'overview_2x5')
    all_rows = []
    for name in NAMES:
        for i, roi in enumerate(rois):
            x0, y0, x1, y1 = roi['bounds']
            roi_mask = np.zeros_like(valid)
            roi_mask[y0:y1, x0:x1] = True
            rows = region_rows(arrays[name], name, key, roi_mask)
            for row in rows:
                if row['scope'] == 'full_image':
                    if i:
                        continue
                    row.update(roi_name='', roi_bounds='')
                else:
                    row.update(scope=roi['id'], roi_name=roi['name'], roi_bounds=json.dumps(roi['bounds']))
                all_rows.append(row)
    add_deltas(all_rows)
    write_csv(out/'metrics_full_and_roi.csv', all_rows)
    for i, roi in enumerate(rois, 1):
        directory = out/roi['id']
        for panel in panels:
            save_panel(panel, directory/'panels', roi['bounds'])
        fig, axes = plt.subplots(2, 5, figsize=(22, 9), layout='constrained')
        for ax, panel in zip(axes.flat, panels):
            draw_panel(ax, panel, roi['bounds'])
        scope = [r for r in all_rows if r['scope'] == roi['id'] and r['region'] == 'full']
        b, o = scope
        title = (f'{key} | R{i}: {roi["name"]} | '
                 f'Abs {b["abs"]:.3f} -> {o["abs"]:.3f} mm; '
                 f'Acc2 {100*b["acc2"]:.2f} -> {100*o["acc2"]:.2f}%')
        fig.suptitle(title, fontsize=12)
        save_figure(fig, directory/'zoom_2x5')
        write_csv(directory/'metrics.csv', [r for r in all_rows if r['scope'] == roi['id']])
    settings = {'schema_version': 1, 'sample': key, 'analysis_configs': list(NAMES),
                'display_labels': {'Base': 'Base', 'Base+A+B+C': 'Ours'},
                'depth_min_mm': lo, 'depth_max_mm': hi, 'error_max_mm': args.error_max,
                'gain_max_mm': args.gain_max, 'gain_definition': 'abs(Base-GT) - abs(Ours-GT)',
                'gain_colors': {'positive': 'blue', 'negative': 'red', 'zero': 'white'},
                'invalid_gt_color': 'black', 'rois': rois, 'panels': manifest,
                'metrics_use_unclipped_errors': True,
                'roi_selection': 'manual' if meta.get('roi') is not None or meta.get('rois') is not None else 'GT region center'}
    (out/'display_settings.json').write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{key}: overview + 10 standalone panels + {len(rois)} ROI sets -> {out}', flush=True)
