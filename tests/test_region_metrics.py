import unittest

import numpy as np
import torch

from tools.eval_region_metrics_dtu_yao import (
    full_resolution_range_diagnostics,
    metrics,
)


class RegionRangeDiagnosticsTest(unittest.TestCase):
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
