import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.guided_centers import GuidedCenterUpsampler
from models.model_variants import MODEL_VARIANTS, get_model_variant
from models.module import homo_warping
from models.vismvsnet_oa import VisMVSModel, VisMVSLoss
from tools.eval_region_metrics_dtu_yao import build_model, parse_args
from tools.train_v2_val import build_plans, main, ROOT


class V2ModuleTest(unittest.TestCase):
    def test_eight_v2_variants_are_independent(self):
        expected={'v2_vis':'000','v2_m1':'100','v2_m2':'010','v2_m3':'001',
                  'v2_m1_m2':'110','v2_m1_m3':'101','v2_m2_m3':'011','v2_full':'111'}
        self.assertEqual({n:v.code for n,v in MODEL_VARIANTS.items() if n.startswith('v2_')},expected)
        for name,code in expected.items():
            v=get_model_variant(name)
            self.assertEqual(v.needs_visibility_gt, code[1]=='1')
            self.assertEqual(v.projection_validity, code[0]=='1')
            self.assertEqual(v.guided_centers, code[2]=='1')
            self.assertFalse(v.hybrid_sampling)
            self.assertFalse(v.uses_visibility_gate)
            self.assertFalse(v.hypothesis_visibility_supervision)

    def test_guided_center_initializes_as_bilinear_at_borders_and_odd_sizes(self):
        torch.manual_seed(5)
        module=GuidedCenterUpsampler()
        for shape,target in [((2,3,4),(6,8)),((1,3,5),(7,9)),((1,1,1),(2,2))]:
            depth=torch.rand(*shape)*20
            features=torch.randn(shape[0],32,*target)
            out=module(depth,features,torch.ones(shape[0],1))
            expected=F.interpolate(depth[:,None],size=target,mode='bilinear',align_corners=False)[:,0]
            torch.testing.assert_close(out,expected,atol=3e-6,rtol=1e-6)

    def test_guided_center_is_convex_and_learns_from_depth_loss(self):
        module=GuidedCenterUpsampler()
        depth=torch.tensor([[[1.,10.],[3.,20.]]],requires_grad=True)
        features=torch.randn(1,32,4,4,requires_grad=True)
        out=module(depth,features,torch.ones(1,1))
        (out-5).square().mean().backward()
        self.assertGreater(float(module.logits.weight.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(depth.grad).all())
        with torch.no_grad():
            module.logits.weight.normal_()
            out=module(depth,features,torch.ones(1,1))
        neighbors,_=module.neighbors_and_prior(depth,(4,4))
        self.assertTrue((out>=neighbors.min(dim=1).values-1e-5).all())
        self.assertTrue((out<=neighbors.max(dim=1).values+1e-5).all())

    def test_projection_validity_rejects_out_of_view_and_behind_camera(self):
        features=torch.ones(1,4,8,8)
        ref=torch.eye(4)[None]
        depths=torch.tensor([[2.,4.]])
        old=homo_warping(features,ref,ref,depths)
        warped,valid=homo_warping(features,ref,ref,depths,return_valid=True)
        self.assertTrue(valid.all())
        torch.testing.assert_close(warped,old)
        for axis,translation in [(0,1000.),(2,-10.)]:
            src=ref.clone(); src[:,axis,3]=translation
            warped,valid=homo_warping(features,src,ref,depths,return_valid=True)
            self.assertFalse(valid.any())
            self.assertTrue(torch.isfinite(warped).all())
            self.assertEqual(float(warped.abs().sum()),0.)

    def test_v2_full_forward_and_backward(self):
        torch.manual_seed(12)
        model=VisMVSModel(stage1_depth_num=8,stage2_depth_num=8,stage3_depth_num=8,
                          hypothesis_fusion=True,projection_validity=True,
                          guided_centers=True,visibility_supervision_only=True).train()
        images=torch.randn(2,3,3,64,64)
        proj=torch.eye(4)[None,None].repeat(2,3,1,1)
        proj[:,1,0,3]=0.5; proj[:,2,1,3]=-0.5
        original=torch.linspace(1,32,32)[None].repeat(2,1)
        outputs,depth,_=model(images,proj,original)
        self.assertTrue(torch.isfinite(depth).all())
        for stage in (1,2):
            gaps=outputs[stage][2][:,1:]-outputs[stage][2][:,:-1]
            torch.testing.assert_close(gaps,torch.full_like(gaps,2. if stage==1 else 1.))
        gt=torch.full((2,1,16,16),10.)
        visibility=torch.full((2,3,16,16),10.)
        visibility[:,1,:,8:]=5.
        loss,stats=VisMVSLoss(visibility_supervision=True,hypothesis_visibility_weight=0.)(
            outputs,gt,torch.ones_like(gt),torch.ones(2),visibility,
            torch.ones_like(visibility),proj)
        loss.backward()
        self.assertIn('visibility_loss_stage1',stats)
        self.assertNotIn('hypothesis_visibility_loss_stage1',stats)
        for layer in [model.stage1.uncert_net.occ_head,model.stage1.hypothesis_weight_net.logit,
                      model.center_upsampler2.logits,model.center_upsampler3.logits]:
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()),0.)

    def test_v2_runner_wires_all_single_modules_and_val(self):
        args=argparse.Namespace(models=['v2_vis','v2_m1','v2_m2','v2_m3'],
                                log_root='checkpoints/test_v2',out_root='eval/test_v2',
                                train_nviews=5,seed=1,datapath='data',batch_size=4,epochs=16,
                                train_workers=8,eval_workers=4)
        plans=build_plans(args)
        self.assertEqual(len(plans),4)
        for plan in plans:
            command=plan['train']
            self.assertEqual(command[command.index('--model_type')+1],plan['model_type'])
            self.assertEqual(Path(command[command.index('--testlist')+1]),ROOT/'lists/dtu/val.txt')
            self.assertNotIn('--loadckpt',command)
            with patch('sys.argv',plan['evaluate'][2:]):
                settings=parse_args()
            model=build_model(settings)
            variant=get_model_variant(plan['model_type'])
            self.assertEqual(model.guided_centers,variant.guided_centers)
            self.assertEqual(model.stage1.projection_validity,variant.projection_validity)
            self.assertFalse(model.stage1.visibility_fusion)
        with tempfile.TemporaryDirectory() as folder:
            with patch('sys.argv',['runner','--dry_run','--out_root',str(Path(folder)/'out')]), \
                    patch('builtins.print'),patch('subprocess.run') as run:
                main()
            run.assert_not_called()
            self.assertFalse((Path(folder)/'out').exists())


if __name__=='__main__':
    unittest.main()
