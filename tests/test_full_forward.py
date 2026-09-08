import unittest

import torch

from models.vismvsnet_oa import VisMVSModel


class FullMethodForwardTest(unittest.TestCase):
    def test_s1_64_loads_48_weights_and_only_extends_coarse_hypothesis_support(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            baseline = VisMVSModel().eval()
            extended = VisMVSModel(stage1_depth_num=64).eval()
            extended.load_state_dict(baseline.state_dict(), strict=True)
            images = torch.randn(1, 2, 3, 64, 64)
            projections = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
            projections[:, 1, 0, 3] = .5
            original = 425. + 2.65 * torch.arange(192).view(1, -1)
            with torch.no_grad():
                a, _, _ = baseline(images, projections, original)
                b, depth, _ = extended(images, projections, original)
            self.assertEqual([x[2].shape[1] for x in b], [64, 32, 16])
            torch.testing.assert_close(a[0][2], b[0][2][:, :48], atol=0, rtol=0)
            step = original[:, 1] - original[:, 0]
            torch.testing.assert_close(b[0][2][:, -1] - a[0][2][:, -1], step * 64)
            for stage, scale in ((1, 2), (2, 1)):
                torch.testing.assert_close(b[stage][2][:, 1:] - b[stage][2][:, :-1],
                                           (step*scale).view(1, 1, 1, 1).expand_as(b[stage][2][:, 1:]),
                                           atol=1e-4, rtol=1e-5)
            self.assertTrue(torch.isfinite(depth).all())
        finally:
            torch.set_num_threads(old_threads)

    def test_m3_unclipped_scale_one_matches_baseline_with_identical_weights(self):
        torch.manual_seed(19)
        settings = dict(stage1_depth_num=8, stage2_depth_num=8, stage3_depth_num=8,
                        hybrid_stage2_wide_num=4, hybrid_stage3_wide_num=4)
        baseline = VisMVSModel(**settings).eval()
        hybrid = VisMVSModel(**settings, hybrid_sampling=True,
                             hybrid_clip_mode='none', hybrid_max_scale=1.0).eval()
        hybrid.load_state_dict(baseline.state_dict(), strict=True)
        images = torch.randn(1, 3, 3, 64, 64)
        projections = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
        projections[:, 1, 0, 3] = 0.5
        projections[:, 2, 1, 3] = -0.5
        original_depths = torch.linspace(1.0, 32.0, 32).view(1, 32)
        with torch.no_grad():
            outputs_a, depth_a, conf_a = baseline(images, projections, original_depths)
            outputs_b, depth_b, conf_b = hybrid(images, projections, original_depths)
        for a, b in zip(outputs_a, outputs_b):
            torch.testing.assert_close(a[0], b[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(a[2], b[2], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(depth_a, depth_b, atol=1e-5, rtol=1e-5)
        for a, b in zip(conf_a, conf_b):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_all_three_modules_forward(self):
        model = VisMVSModel(
            stage1_depth_num=8,
            stage2_depth_num=8,
            stage3_depth_num=8,
            hypothesis_fusion=True,
            visibility_fusion=True,
            hybrid_sampling=True,
            hybrid_stage2_wide_num=4,
            hybrid_stage3_wide_num=4,
        ).eval()
        images = torch.randn(1, 2, 3, 64, 64)
        projections = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
        original_depths = torch.linspace(1.0, 32.0, 32).view(1, 32)

        with torch.no_grad():
            outputs, final_depth, confidence = model(
                images, projections, original_depths)

        self.assertEqual(
            [tuple(item[0].shape) for item in outputs],
            [(1, 8, 8), (1, 16, 16), (1, 32, 32)],
        )
        self.assertEqual(
            [tuple(item[2].shape) for item in outputs],
            [(1, 8), (1, 8, 16, 16), (1, 8, 32, 32)],
        )
        self.assertEqual([len(item[1][0]) for item in outputs], [4, 4, 4])
        self.assertEqual(tuple(final_depth.shape), (1, 1, 32, 32))
        self.assertEqual(len(confidence), 3)
        self.assertTrue(torch.isfinite(final_depth).all())


if __name__ == "__main__":
    unittest.main()
