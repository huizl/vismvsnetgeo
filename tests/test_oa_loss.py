import unittest

import torch

from models.vismvsnet import VisMVSModel as BaselineModel
from models.vismvsnet import VisMVSLoss as BaselineLoss
from models.vismvsnet_oa import VisMVSModel as OcclusionAwareModel
from models.vismvsnet_oa import VisMVSLoss


class OcclusionAwareLossTest(unittest.TestCase):
    def test_m1_and_m2_off_matches_original_loss(self):
        torch.manual_seed(7)
        baseline_outputs = []
        configurable_outputs = []
        for _ in range(3):
            fused_depth = torch.rand(1, 4, 4) + 5.0
            pair_depth = torch.rand(1, 4, 4) + 5.0
            uncertainty = torch.rand(1, 1, 4, 4) - 0.5
            visibility_logit = torch.rand(1, 1, 4, 4) - 0.5
            pairs = [(pair_depth, uncertainty, visibility_logit)]
            baseline_outputs.append([fused_depth, pairs])
            configurable_outputs.append([
                fused_depth,
                pairs,
                torch.linspace(1.0, 10.0, 8).view(1, 8),
            ])

        depth_gt = torch.rand(1, 1, 8, 8) + 5.0
        mask = torch.ones_like(depth_gt)
        interval = torch.ones(1)
        original_loss, _ = BaselineLoss(occ_guide=False)(
            baseline_outputs, depth_gt, mask, interval)
        configurable_loss, stats = VisMVSLoss(
            visibility_supervision=False,
            hypothesis_visibility_weight=0.0,
        )(configurable_outputs, depth_gt, mask, interval)

        self.assertTrue(torch.allclose(
            original_loss, configurable_loss, atol=1e-6, rtol=1e-6))
        self.assertNotIn("visibility_loss_stage1", stats)

    def test_checkpoint_parameters_match_baseline(self):
        baseline = BaselineModel().state_dict()
        method = OcclusionAwareModel().state_dict()
        self.assertEqual(list(baseline), list(method))
        for name in baseline:
            self.assertEqual(baseline[name].shape, method[name].shape)

    def test_m2_starts_from_a_neutral_visibility_gate(self):
        model = OcclusionAwareModel(
            visibility_fusion=True,
            visibility_fusion_betas=(0.2, 0.2, 0.2),
        )
        for stage in (model.stage1, model.stage2, model.stage3):
            self.assertEqual(
                float(stage.uncert_net.occ_head.weight.abs().sum()), 0.0)
            self.assertAlmostEqual(stage.visibility_fusion_beta, 0.2)

    def test_hybrid_sampling_preserves_local_density_and_expands_tails(self):
        model = OcclusionAwareModel(
            stage2_depth_num=8,
            hybrid_sampling=True,
            hybrid_stage2_wide_num=4,
            hybrid_sigma_scale=2.0,
            hybrid_max_scale=2.0,
        )
        center = torch.tensor([[[10.0, 10.0]]])
        standard_deviation = torch.tensor([[[0.25, 10.0]]])
        hypotheses = model._build_hybrid_depth_range(
            center,
            standard_deviation,
            depth_num=8,
            wide_num=4,
            interval=torch.ones(1, 1),
            global_min=torch.zeros(1, 1),
            global_max=torch.full((1, 1), 30.0),
        )
        width = hypotheses[:, -1] - hypotheses[:, 0]
        baseline = model._build_per_pixel_depth_range(
            center, 8, torch.ones(1, 1), 1, 1, 2,
            center.device, center.dtype)
        self.assertTrue(torch.allclose(
            hypotheses[0, :, 0, 0], baseline[0, :, 0, 0]))
        self.assertTrue(torch.allclose(
            hypotheses[0, 2:6, 0, 1], baseline[0, 2:6, 0, 1]))
        self.assertGreater(float(width[0, 0, 1]), float(width[0, 0, 0]))
        self.assertTrue(torch.all(hypotheses[:, 1:] >= hypotheses[:, :-1]))

    def test_unclipped_scale_one_matches_baseline_at_global_boundaries(self):
        model = OcclusionAwareModel(hybrid_sampling=True,
                                    hybrid_clip_mode='none', hybrid_max_scale=1.0)
        center = torch.tensor([[[2.0, 29.0]]], requires_grad=True)
        hypotheses = model._build_hybrid_depth_range(
            center, torch.full_like(center, 100.0), 8, 4,
            torch.ones(1, 1), torch.zeros(1, 1), torch.full((1, 1), 30.0))
        baseline = model._build_per_pixel_depth_range(
            center, 8, torch.ones(1, 1), 1, 1, 2, center.device, center.dtype)
        self.assertTrue(torch.allclose(hypotheses, baseline))
        self.assertLess(float(hypotheses.min()), 0.0)
        self.assertGreater(float(hypotheses.max()), 30.0)
        hypotheses.sum().backward()
        self.assertTrue(torch.isfinite(center.grad).all())
        self.assertTrue((center.grad > 0).all())

    def test_global_clipping_preserves_legacy_behavior(self):
        legacy = OcclusionAwareModel(hybrid_sampling=True)
        unclipped = OcclusionAwareModel(hybrid_sampling=True, hybrid_clip_mode='none')
        center = torch.tensor([[[2.0, 29.0]]])
        args = (center, torch.full_like(center, 10.0), 8, 4,
                torch.ones(1, 1), torch.zeros(1, 1), torch.full((1, 1), 30.0))
        raw = unclipped._build_hybrid_depth_range(*args)
        clipped = legacy._build_hybrid_depth_range(*args)
        self.assertTrue(torch.equal(clipped, raw.clamp(0.0, 30.0)))
        self.assertTrue((raw[:, 1:] > raw[:, :-1]).all())
        self.assertTrue((clipped[:, 1:] == clipped[:, :-1]).any())
        self.assertEqual(set(legacy.state_dict()), set(unclipped.state_dict()))

    def test_hybrid_rejects_unknown_clipping_policy(self):
        with self.assertRaisesRegex(ValueError, 'hybrid_clip_mode'):
            OcclusionAwareModel(hybrid_clip_mode='typo')

    def test_full_model_adds_only_zero_initialized_weight_heads(self):
        base = OcclusionAwareModel()
        full = OcclusionAwareModel(
            hypothesis_fusion=True,
            visibility_fusion=True,
            hybrid_sampling=True,
        )
        incompatible = full.load_state_dict(base.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            "hypothesis_weight_net" in key
            for key in incompatible.missing_keys
        ))
        for stage in (full.stage1, full.stage2, full.stage3):
            self.assertEqual(
                float(stage.hypothesis_weight_net.logit.weight.abs().sum()),
                0.0,
            )
            self.assertEqual(
                float(stage.hypothesis_weight_net.logit.bias.abs().sum()),
                0.0,
            )

    def test_m2_supervises_visibility_without_masking_pair_depth(self):
        batch, height, width, depth_num = 1, 4, 4, 16
        stage_outputs = []
        pair_depths = []

        for stage_idx in range(3):
            fused_depth = torch.full(
                (batch, height, width), 9.0, requires_grad=True)
            pair_depth = torch.full(
                (batch, height, width), 8.0, requires_grad=True)
            uncertainty = torch.zeros(
                batch, 1, height, width, requires_grad=True)
            visibility_logit = torch.zeros(
                batch, 1, height, width, requires_grad=True)
            pair_depths.append(pair_depth)

            hypotheses = torch.linspace(1.0, 16.0, depth_num).view(1, depth_num)
            if stage_idx > 0:
                hypotheses = hypotheses.view(1, depth_num, 1, 1).expand(
                    batch, depth_num, height, width)
            stage_outputs.append([
                fused_depth,
                [(pair_depth, uncertainty, visibility_logit)],
                hypotheses,
            ])

        depth_gt = torch.full((batch, 1, 8, 8), 10.0)
        mask = torch.ones_like(depth_gt)
        visibility_depths = torch.full((batch, 2, 8, 8), 10.0)
        # The right half contains a nearer source surface, so the projected
        # reference point is occluded there.
        visibility_depths[:, 1, :, 4:] = 5.0
        visibility_masks = torch.ones_like(visibility_depths)
        projections = torch.eye(4).view(1, 1, 4, 4).repeat(batch, 2, 1, 1)

        loss_fn = VisMVSLoss(
            visibility_supervision=True,
            visibility_weight=0.2,
        )
        loss, stats = loss_fn(
            stage_outputs,
            depth_gt,
            mask,
            torch.ones(batch),
            visibility_depths,
            visibility_masks,
            projections,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        visible_gradient = pair_depths[0].grad[:, :, :2].abs().sum()
        occluded_gradient = pair_depths[0].grad[:, :, 2:].abs().sum()
        self.assertGreater(float(visible_gradient), 0.0)
        self.assertGreater(float(occluded_gradient), 0.0)
        self.assertGreater(
            float(stage_outputs[0][1][0][2].grad.abs().sum()), 0.0)
        self.assertAlmostEqual(float(stats["pair_visible_ratio_stage1"]), 0.5)
        self.assertAlmostEqual(float(stats["pair_occluded_ratio_stage1"]), 0.5)
        self.assertAlmostEqual(float(stats["range_coverage_stage2"]), 1.0)

    def test_hypothesis_visibility_supervises_gt_depth_plane(self):
        batch, height, width, depth_num = 1, 4, 4, 16
        stage_outputs = []
        hypothesis_logits = []
        pair_depths = []

        for stage_idx in range(3):
            fused_depth = torch.full(
                (batch, height, width), 9.0, requires_grad=True)
            pair_depth = torch.full(
                (batch, height, width), 8.0, requires_grad=True)
            pair_depths.append(pair_depth)
            uncertainty = torch.zeros(
                batch, 1, height, width, requires_grad=True)
            visibility_logit = torch.zeros(
                batch, 1, height, width, requires_grad=True)
            hypothesis_logit = torch.zeros(
                batch, 1, depth_num, height, width, requires_grad=True)
            hypothesis_logits.append(hypothesis_logit)

            hypotheses = torch.linspace(1.0, 16.0, depth_num).view(1, depth_num)
            if stage_idx > 0:
                hypotheses = hypotheses.view(1, depth_num, 1, 1).expand(
                    batch, depth_num, height, width)
            stage_outputs.append([
                fused_depth,
                [(pair_depth, uncertainty, visibility_logit, hypothesis_logit)],
                hypotheses,
            ])

        depth_gt = torch.full((batch, 1, 8, 8), 10.0)
        visibility_depths = torch.full((batch, 2, 8, 8), 10.0)
        visibility_depths[:, 1, :, 4:] = 5.0
        visibility_masks = torch.ones_like(visibility_depths)
        projections = torch.eye(4).view(1, 1, 4, 4).repeat(batch, 2, 1, 1)

        loss_fn = VisMVSLoss(
            visibility_supervision=False,
            hypothesis_visibility_weight=0.1,
        )
        loss, stats = loss_fn(
            stage_outputs,
            depth_gt,
            torch.ones_like(depth_gt),
            torch.ones(batch),
            visibility_depths,
            visibility_masks,
            projections,
        )
        loss.backward()

        gradient_by_depth = hypothesis_logits[0].grad.abs().sum(
            dim=(0, 1, 3, 4))
        self.assertGreater(float(gradient_by_depth[9]), 0.0)
        self.assertEqual(float(gradient_by_depth[:9].sum()), 0.0)
        self.assertEqual(float(gradient_by_depth[10:].sum()), 0.0)
        # C receives its visibility label, but A remains disabled: baseline
        # pair-depth supervision still updates the occluded right half.
        self.assertGreater(float(pair_depths[0].grad[:, :, 2:].abs().sum()), 0.0)
        self.assertIn("hypothesis_visibility_loss_stage1", stats)
        self.assertIn("hypothesis_visibility_accuracy", stats)
        self.assertNotIn("visibility_loss_stage1", stats)


if __name__ == "__main__":
    unittest.main()
