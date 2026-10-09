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
from tools.visualize_paper_results import add_deltas, factor_rows, region_rows, render_sample


class PaperVisualizationTest(unittest.TestCase):
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
