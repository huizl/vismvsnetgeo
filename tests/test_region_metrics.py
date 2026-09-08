import unittest

import numpy as np
import torch

from tools.eval_region_metrics_dtu_yao import (
    common_grid_intervals,
    full_resolution_range_diagnostics,
    metrics,
)


class RegionRangeDiagnosticsTest(unittest.TestCase):
    def test_common_grid_selects_matching_depth_and_interval_without_surface_blending(self):
        depth = torch.tensor([[[5., 105.], [205., 305.]]])
        local = torch.stack((depth - 1, depth + 1), dim=1)
        outputs = [[depth, None, torch.tensor([[0., 310.]])],
                   [depth, None, local], [depth, None, local]]
        prediction, intervals = common_grid_intervals(outputs, torch.tensor([[0., 320.]]), (1, 1))
        self.assertEqual(prediction.item(), 5.)
        self.assertEqual(intervals["s3"][0].item(), 4.)
        self.assertEqual(intervals["s3"][1].item(), 6.)
        self.assertEqual(intervals["original"][1].item(), 320.)
        self.assertEqual(intervals["s1"][1].item(), 310.)

    def test_range_widths_and_coverage_are_reported(self):
        depth_gt = torch.tensor([[[5.0, 9.0], [5.0, 9.0]]])
        original = torch.arange(1.0, 11.0).view(1, 10)
        stage1 = torch.tensor([[1.0, 5.0, 9.0]])
        stage2 = torch.tensor([[[[3.0, 3.0], [3.0, 3.0]],
                                [[5.0, 5.0], [5.0, 5.0]],
                                [[7.0, 7.0], [7.0, 7.0]]]])
        stage3 = torch.tensor([[[[4.0, 4.0], [4.0, 4.0]],
                                [[6.0, 6.0], [6.0, 6.0]]]])
        outputs = [
            [None, None, stage1],
            [None, None, stage2],
            [None, None, stage3],
        ]

        range_masks, range_widths = full_resolution_range_diagnostics(
            outputs, depth_gt, original)
        result = metrics(
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=bool),
            [item[0].numpy() for item in range_masks],
            [item[0].numpy() for item in range_widths],
        )

        self.assertAlmostEqual(result["stage1_in_range"], 1.0)
        self.assertAlmostEqual(result["stage2_in_range"], 0.5)
        self.assertAlmostEqual(result["stage3_in_range"], 0.5)
        self.assertAlmostEqual(result["stage2_range_width"], 4.0)
        self.assertAlmostEqual(result["stage3_range_width"], 2.0)
