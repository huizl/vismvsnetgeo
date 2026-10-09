"""Numeric/provenance checks for publication figure generation."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from tools.paper_configs import CONFIGS, BY_MODEL, validate_csv_identity
from tools.paper_dump import dump_sample
from tools.paper_method_revision import renumber_references
from tools.check_paper_integrity import check
from tools.visualize_paper_results import add_deltas, factor_rows, region_rows, render_sample, verify_export
from tools.paper_qualitative import (render_paper_sample, occlusion_count, scene_descriptor,
                                     category_scores, select_scene_examples, select_from_dtu, yellow_contours, overlay_rgb)


class PaperVisualizationTest(unittest.TestCase):
    def test_stronger_overlay_preserves_gt_masks_and_unmarked_rgb(self):
        rgb = np.full((40,60,3),255,dtype=np.uint8)
        large = np.zeros((40,60),dtype=bool); large[3:24,3:27]=True
        occ = np.zeros_like(large); occ[14:36,18:52]=True
        marked = overlay_rgb(rgb,large,occ)
        self.assertEqual(tuple(marked[8,8]),(255,76,76))
        self.assertEqual(tuple(marked[29,43]),(76,255,76))
        self.assertEqual(tuple(marked[18,22]),(255,255,76))
        self.assertEqual(tuple(marked[3,3]),(255,0,0))
        self.assertEqual(tuple(marked[35,51]),(0,255,0))
        outside = ~(large|occ)
        np.testing.assert_array_equal(marked[outside],rgb[outside])
        self.assertTrue((rgb==255).all())
        self.assertEqual(int(large.sum()),21*24)
        self.assertEqual(int(occ.sum()),22*34)
        with self.assertRaises(ValueError): overlay_rgb(rgb,large,occ,1.1)

    def test_cpu_scene_selection_checks_gt_and_covers_three_categories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = np.ones((20,50),dtype=bool)
            large = np.zeros_like(valid); large[:4]=True
            occlusions = []
            for start, end in ((980,1000),(500,850),(0,120)):
                mask = np.zeros_like(valid); mask.flat[start:end]=True
                occlusions.append(mask)
            occlusions[2].flat[500:680]=True
            metas = [(f'scan{i}',3,0,[1,2,3,4]) for i in range(1,4)]
            original = {}
            for scan,light,ref,sources in metas:
                folder = root/'data'/'Depths'/f'{scan}_train'; folder.mkdir(parents=True)
                Image.new('L',(50,20),255).save(folder/'depth_visual_0000.png')
                original[(scan,ref,light,'full')]={'pixels':1000,'abs':10.}
                original[(scan,ref,light,'large_disparity')]={'pixels':200}
            protocol = {'Base':{'full':{'eval_nviews':5,'region_nviews':5}}}
            args = SimpleNamespace(eval_root='unused',testpath=str(root/'data'),testlist='unused',
                                   eval_args_json=None,outdir=str(root/'out'),examples_per_category=1)
            def geometry(_path,scan,*_args):
                occ = occlusions[int(scan[4:])-1]
                return dict(disparity=np.ones_like(valid,dtype=float)*20,
                            source_occluded=np.stack([occ,occ,~valid,~valid]))
            full_index = {k:{**v, 'abs':5.} for k,v in original.items()}
            with patch('tools.visualize_paper_results.load_series',return_value=({'Base':original,'Base+A+B+C':full_index},protocol)), \
                 patch('datasets.dtu_yao.MVSDataset',return_value=SimpleNamespace(metas=metas)), \
                 patch('tools.eval_region_metrics_dtu_yao.read_dtu_depth',return_value=np.ones_like(valid,dtype=float)), \
                 patch('tools.eval_region_metrics_dtu_yao.geometry_region_maps',side_effect=geometry), \
                 patch('tools.eval_region_metrics_dtu_yao.percentile_mask',return_value=large):
                select_from_dtu(args,'View5')
                selected = json.loads((root/'out'/'View5'/'scene_selection.json').read_text())
                self.assertEqual(len(selected),3)
                self.assertEqual([r['scene_categories'][0] for r in selected],
                                 ['large_disparity_dominant','occlusion_dominant','joint_difficulty'])
                original[('scan1',0,3,'full')]['pixels']=999
                with self.assertRaisesRegex(ValueError,'GT validity'):
                    select_from_dtu(args,'View5')

    def test_count_ge2_is_not_majority_and_uses_region_source_ids(self):
        labels = np.array([[[1, 0]], [[0, 1]], [[0, 1]]], dtype=bool)
        arrays = dict(gt=np.ones((1,2)), valid=np.ones((1,2),dtype=bool), source_occluded=labels)
        count = occlusion_count(arrays, {'source_views':[1,2,3], 'region_source_views':[1,2]})
        np.testing.assert_array_equal(count, [[1,1]])
        count = occlusion_count(arrays, {'source_views':[1,2,3], 'region_source_views':[1,2,3]})
        np.testing.assert_array_equal(count, [[1,2]])
        # Pixel 0 could be 'majority' when only its one occluded view is comparable,
        # but cannot be in the mask requiring at least two occluded views.
        self.assertFalse((count>=2)[0,0])
        self.assertTrue((count>=2)[0,1])
        with self.assertRaises(ValueError):
            occlusion_count(arrays, {'source_views':[1,2,3], 'region_source_views':[4]})

    def test_three_scene_categories_ranked_by_full_image_improvement(self):
        records = []
        for i, (complex_fraction, joint_large, joint_occ, joint_fraction) in enumerate(
                ((.02,.02,.20,.004),(.35,.10,.05,.02),(.30,.60,.40,.12)), 1):
            records.append(dict(scan=f'scan{i}', view=0, light=3, large_disparity_pixels=200,
                                occluded_ge2_pixels=round(complex_fraction*1000),
                                joint_ge2_pixels=round(joint_fraction*1000),
                                occluded_ge2_fraction=complex_fraction, joint_fraction_of_large=joint_large,
                                joint_fraction_of_occlusion=joint_occ, joint_ge2_fraction=joint_fraction,
                                disparity_p80_px=20., base_abs=10., full_model_abs=8., abs_reduction_pct=20.))
        # Give each category a second, more improved case from another scan.
        records += [{**r, 'scan':r['scan']+'0', 'full_model_abs':5., 'abs_reduction_pct':50.}
                    for r in list(records)]
        selected = select_scene_examples(records,1)
        self.assertEqual([r['scene_categories'] for r in selected],
                         [['large_disparity_dominant'],['occlusion_dominant'],['joint_difficulty']])
        self.assertEqual([r['scan'] for r in selected],['scan10','scan20','scan30'])
        self.assertTrue(all(r['abs_reduction_pct']==50. for r in selected))
        # Regression cases cannot displace a positive-improvement example.
        records += [{**records[0], 'scan':'scan99','full_model_abs':15.,'abs_reduction_pct':-50.}]
        self.assertEqual([r['scan'] for r in select_scene_examples(records,1)],['scan10','scan20','scan30'])
        with self.assertRaises(ValueError): select_scene_examples(records[:1],1)
        ambiguous = {**records[1], 'joint_ge2_pixels':100,'joint_ge2_fraction':.1,
                     'joint_fraction_of_large':.5,'joint_fraction_of_occlusion':.25}
        self.assertEqual(set(category_scores(ambiguous)), {'joint_difficulty'})

    def test_yellow_contours_do_not_change_metric_arrays(self):
        image = np.full((12,14,3),100,dtype=np.uint8)
        large = np.zeros((12,14),dtype=bool); large[2:6,2:6]=True
        complex_occ = np.zeros_like(large); complex_occ[7:11,8:12]=True
        marked = yellow_contours(image,large,complex_occ)
        self.assertEqual(tuple(marked[2,2]),(255,255,0))
        self.assertEqual(tuple(marked[7,8]),(255,255,0))
        self.assertEqual(tuple(marked[0,0]),(100,100,100))
        self.assertTrue((image==100).all())

    def test_verification_reports_mismatch_and_warn_uses_current_metrics(self):
        from tools.visualize_paper_results import METRICS, read_csv, write_csv
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            reference = dict(model_type='range', ablation_code='010', scan='scan1', view=0, light=3,
                             region='full', pixels=100, eval_nviews=5, region_nviews=5,
                             range_sigma_scale=2., range_min_scale=1., range_max_scale=2.,
                             hypothesis_residual_scale=1., **{m: .8 for m in METRICS})
            reference['abs'] = 5.
            current = {**reference, 'abs': 7., 'acc2': .75}
            write_csv(folder/'all_metrics.csv', [current])
            args = SimpleNamespace(eval_root='unused', verify_atol=1e-3, verify_rtol=1e-4,
                                   verify_mode='strict')
            index = {'Base': {('scan1', 0, 3, 'full'): reference}}
            with patch('tools.visualize_paper_results.load_series', return_value=(index, {})):
                with self.assertRaisesRegex(RuntimeError, 'New predictions differ'):
                    verify_export(args, 'View5', BY_MODEL['range'], folder)
                report = json.loads((folder/'reproduction_status.json').read_text())
                self.assertEqual(report['failed_checks'], 2)
                self.assertEqual(len(read_csv(folder/'reproduction_summary.csv')), 10)
                self.assertEqual(len(read_csv(folder/'reproduction_failures.csv')), 2)
                args.verify_mode = 'warn'
                report = verify_export(args, 'View5', BY_MODEL['range'], folder)
                self.assertFalse(report['matches_original_csv'])
                self.assertEqual(float(read_csv(folder/'all_metrics.csv')[0]['abs']), 7.)
                self.assertEqual(reference['abs'], 5.)
                args.verify_mode = 'strict'
                write_csv(folder/'all_metrics.csv', [reference])
                self.assertTrue(verify_export(args, 'View5', BY_MODEL['range'], folder)['matches_original_csv'])
                self.assertFalse((folder/'reproduction_failures.csv').exists())

    def test_warn_does_not_accept_wrong_identity_protocol_or_region_population(self):
        from tools.visualize_paper_results import METRICS, write_csv
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            reference = dict(model_type='range', ablation_code='010', scan='scan1', view=0, light=3,
                             region='full', pixels=100, eval_nviews=5, region_nviews=5,
                             range_sigma_scale=2., range_min_scale=1., range_max_scale=2.,
                             hypothesis_residual_scale=1., **{m: .8 for m in METRICS})
            args = SimpleNamespace(eval_root='unused', verify_atol=1e-3, verify_mode='warn')
            index = {'Base': {('scan1', 0, 3, 'full'): reference}}
            with patch('tools.visualize_paper_results.load_series', return_value=(index, {})):
                for changes in ({'model_type': 'vis'}, {'eval_nviews': 3}, {'pixels': 99},
                                {'abs': float('nan')}):
                    write_csv(folder/'all_metrics.csv', [{**reference, **changes}])
                    with self.assertRaises(ValueError):
                        verify_export(args, 'View5', BY_MODEL['range'], folder)

    def test_paper_layout_separate_panels_and_unclipped_metrics(self):
        from tools.visualize_paper_results import REGIONS, read_csv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = np.ones((4, 6), dtype=bool)
            valid[0, 0] = False
            base = {'gt': np.ones((4, 6))*10, 'pred': np.ones((4, 6))*14,
                    'rgb': np.full((4, 6, 3), 100, dtype=np.uint8), 'valid': valid,
                    'disparity': np.ones((4, 6)), 'occlusion_ratio': np.full((4, 6), .5)}
            for region in REGIONS:
                base['mask_'+region] = valid.copy()
            count = np.ones((4,6),dtype=np.uint16); count[:,:3] = 2; count[~valid] = 0
            base['occluded_count'] = count
            for stage in (1, 2, 3):
                base[f'stage{stage}_coverage'] = valid.copy()
                base[f'stage{stage}_width'] = np.ones((4, 6))*3
            ours = {**base, 'pred': np.full((4, 6), 16.)}
            ours['pred'][:, :3] = 12.
            files = {}
            for name, data in (('Base', base), ('Base+A+B+C', ours)):
                folder = root/name/'sample'
                folder.mkdir(parents=True)
                np.savez_compressed(folder/'arrays.npz', **data)
                (folder/'metadata.json').write_text(json.dumps({'analysis_config': name}), encoding='utf-8')
                files[name] = folder
            args = SimpleNamespace(outdir=str(root/'figs'), error_max=1., gain_max=1.)
            with patch('matplotlib.figure.Figure.colorbar',side_effect=AssertionError('No colorbars requested')):
                render_paper_sample(args, 'View5', 'sample', files)
            out = root/'figs'/'View5'/'scenes'/'mixed_other'/'sample'
            self.assertEqual(len(list((out/'panels').glob('*.png'))), 15)
            self.assertTrue((out/'overview_3x5.pdf').is_file())
            self.assertFalse(list(out.glob('roi_*')))
            settings = json.loads((out/'display_settings.json').read_text())
            self.assertEqual(settings['gain_colors']['positive'], 'red')
            self.assertEqual(settings['gain_colors']['negative'], 'blue')
            self.assertFalse(settings['show_colorbars'])
            self.assertFalse(settings['show_axis_ticks'])
            self.assertTrue((out/'figure_caption.txt').is_file())
            with Image.open(out/'panels'/'error_gain.png') as img:
                pixels = np.array(img)
                self.assertEqual(img.size, (6, 4))
            self.assertEqual(tuple(pixels[0, 0]), (0, 0, 0, 255))
            self.assertGreater(pixels[1, 1, 0], pixels[1, 1, 2])  # Improvement is red.
            self.assertGreater(pixels[1, 4, 2], pixels[1, 4, 0])  # Regression is blue.
            rows = read_csv(out/'region_metrics.csv')
            self.assertEqual(len(rows), 2*9)
            left = next(r for r in rows if r['analysis_config']=='Base+A+B+C'
                        and r['region']=='occluded_ge2')
            right = next(r for r in rows if r['analysis_config']=='Base+A+B+C' and r['region']=='full')
            self.assertEqual(float(left['abs']), 2.)  # Plot bound is 1 mm; raw metric stays 2 mm.
            self.assertEqual(float(left['acc2']), 0.)  # Strict < 2 mm.
            self.assertEqual(float(left['abs_reduction_pct']), 50.)
            self.assertEqual(int(left['pixels']), 11)
            self.assertEqual(int(right['pixels']), 23)
            self.assertAlmostEqual(float(right['improved_pixels_pct']),100*11/23)
            bad = {**ours, 'rgb': base['rgb']+1}
            np.savez_compressed(files['Base+A+B+C']/'arrays.npz', **bad)
            with self.assertRaisesRegex(ValueError, 'unaligned'):
                render_paper_sample(args, 'View5', 'sample', files)

    def test_manuscript_equations_citations_and_images(self):
        self.assertEqual(check(Path(__file__).resolve().parents[1]/'docs'/'PAPER_COMPLETE_WITH_RESULTS.md'),(21,19,6))

    def test_reference_numbers_follow_first_appearance(self):
        text = 'Method [2], another [9], repeated [2]. Formula \\tag{9}.\n## 参考文献\n[9] R\n[2] M\n'
        actual, mapping = renumber_references(text)
        self.assertEqual(mapping, {2:1,9:2})
        self.assertIn('Method [1], another [2], repeated [1]', actual)
        self.assertIn('\\tag{9}', actual)
        self.assertIn('[1] M\n\n[2] R', actual)
        self.assertEqual(renumber_references(actual)[0], actual)

    def test_presentation_mapping_does_not_overwrite_csv_code(self):
        self.assertEqual(BY_MODEL['range'].name,'Base')
        self.assertEqual(BY_MODEL['range'].analysis_code,'000')
        self.assertEqual(BY_MODEL['range'].csv_code,'010')
        validate_csv_identity({'model_type':'range','ablation_code':'010'},BY_MODEL['range'])
        with self.assertRaises(ValueError):
            validate_csv_identity({'model_type':'range','ablation_code':'000'},BY_MODEL['range'])

    def test_roi_metrics_use_raw_errors_and_strict_thresholds(self):
        arr={'gt':np.ones((2,2))*10,'pred':np.array([[11.,12.],[14.,30.]])}
        for region in ('full','boundary','large_disparity','occluded_any','occluded_majority',
                       'large_disp_and_occluded','boundary_and_occluded'):
            arr['mask_'+region]=np.ones((2,2),dtype=bool)
        for stage in (1,2,3):
            arr[f'stage{stage}_coverage']=np.array([[True,False],[True,False]])
            arr[f'stage{stage}_width']=np.ones((2,2))*4
        rows=region_rows(arr,'Base','test',np.array([[True,True],[False,False]]))
        full=next(r for r in rows if r['region']=='full' and r['scope']=='full_image')
        roi=next(r for r in rows if r['region']=='full' and r['scope']=='shared_roi')
        self.assertEqual(full['abs'],6.75)  # Includes tail error 20, no plot clipping.
        self.assertEqual(full['acc2'],.25)  # Exactly 2 is not < 2.
        self.assertEqual(roi['abs'],1.5)
        self.assertEqual(roi['stage2_in_range'],.5)

    def test_percentage_and_percentage_point_changes(self):
        rows=[]
        for name,error,accuracy in [('Base',10.,.5),('Base+B',8.,.55)]:
            rows.append(dict(analysis_config=name,scope='full_image',region='full',abs=error,
                             acc2=accuracy,acc4=accuracy,acc8=accuracy,
                             stage1_in_range=accuracy,stage2_in_range=accuracy,stage3_in_range=accuracy))
        add_deltas(rows)
        self.assertAlmostEqual(rows[1]['abs_reduction_pct'],20.)
        self.assertAlmostEqual(rows[1]['acc2_delta_pp'],5.)
        result=factor_rows(rows)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['added_factor'],'B')

    def test_dump_reconstructs_real_candidate_weights_and_named_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            image_dir=root/'data'/'Rectified'/'scan1_train'; image_dir.mkdir(parents=True)
            Image.new('RGB',(2,2),'white').save(image_dir/'rect_001_3_r5000.png')
            args=SimpleNamespace(model_type='oa_hyp',paper_dumpdir=str(root/'out'),
                testpath=str(root/'data'),loadckpt='checkpoint.ckpt',eval_nviews=3,
                region_nviews=3,occ_abs_tol=2.,occ_rel_tol=.01,hypothesis_residual_scale=1.,light=3,
                _paper_selections=[{'scan':'scan1','view':0,'light':3,'probe':[0,0]}])
            gt=np.ones((2,2),dtype=np.float32)*5
            valid=np.ones((2,2),dtype=bool)
            regions={r:valid for r in ('full','boundary','large_disparity','occluded_any',
                                      'occluded_majority','large_disp_and_occluded','boundary_and_occluded')}
            geom={'disparity':gt,'occlusion_ratio':np.zeros_like(gt),
                  'source_visible':np.stack([valid,valid]),'source_occluded':np.stack([~valid,~valid]),
                  'source_supervised':np.stack([valid,valid])}
            pairs=[]
            for u,z in ((0.,0.),(1.,.5)):
                pairs.append((torch.ones(1,2,2)*5,torch.ones(1,1,2,2)*u,
                              torch.zeros(1,1,2,2),torch.ones(1,1,2,2,2)*z))
            stage=[torch.ones(1,2,2)*5,pairs,torch.tensor([[4.,6.]])]
            with patch('tools.eval_region_metrics_dtu_yao.geometry_region_maps',return_value=geom):
                dump_sample(args,{},0,[stage,stage,stage],gt,gt,valid,geom,regions,
                            [valid]*3,[np.ones_like(gt)]*3,('scan1',3,0,[1,2]),
                            [torch.ones(1,1,2,2)]*3)
            folder=root/'out'/'Base+A+B+C'/'scan1_view00_light3'
            self.assertTrue((folder/'arrays.npz').is_file())
            metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['analysis_code'],'111')
            self.assertEqual(metadata['raw_csv_code'],'101')
            expected=torch.softmax(torch.tensor([0.,-1.+np.tanh(.5)]),dim=0).numpy()
            with np.load(folder/'arrays.npz') as arr:
                np.testing.assert_allclose(arr['stage3_probe_weights'][:,0],expected,atol=1e-6)
                np.testing.assert_allclose(arr['stage3_weights_at_gt'].sum(axis=0),1.,atol=1e-6)
                copied=dict(arr)
            base_folder=root/'out'/'Base'/'scan1_view00_light3'
            base_folder.mkdir(parents=True)
            np.savez_compressed(base_folder/'arrays.npz',**copied)
            base_meta={**metadata,'analysis_config':'Base','analysis_code':'000',
                       'raw_model_type':'range','raw_csv_code':'010'}
            (base_folder/'metadata.json').write_text(json.dumps(base_meta),encoding='utf-8')
            render_args=SimpleNamespace(outdir=str(root/'figs'),error_max=20.,gain_max=10.,cdf_max=30.)
            render_sample(render_args,'View5','scan1_view00_light3',
                          {'Base':base_folder,'Base+A+B+C':folder})
            self.assertTrue((root/'figs'/'View5'/'comparison'/'scan1_view00_light3'/'metrics_full_and_roi.csv').is_file())


if __name__=='__main__':
    unittest.main()
