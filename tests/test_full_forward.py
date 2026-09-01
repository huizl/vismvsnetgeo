import unittest

import torch

from models.vismvsnet_oa import VisMVSModel


class FullMethodForwardTest(unittest.TestCase):
    def test_adaptive_range_and_hypothesis_fusion_forward(self):
        model = VisMVSModel(
            stage1_depth_num=8,
            stage2_depth_num=8,
            stage3_depth_num=8,
            adaptive_range=True,
            hypothesis_fusion=True,
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
