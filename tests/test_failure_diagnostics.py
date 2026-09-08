import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from tools.failure_diagnostics import measure_stage, summarize, report
from tools.run_baseline_diagnostics import build_command


class FailureDiagnosticsTest(unittest.TestCase):
    def test_interval_exclusion_and_sampling_distance_are_different(self):
        # GT=3 lies between candidates [2,4] and can be regressed exactly.
        # GT=10 is outside that interval and forces at least 6 mm error.
        rows = measure_stage(np.array([[3., 4.]]), np.array([[3., 10.]]),
                             np.array([2., 4.]), {"full": np.ones((1, 2), bool)}, 3)
        all_row, inside, outside = rows
        self.assertEqual(inside["abs"], 0)
        self.assertEqual(inside["nearest_hyp_abs"], 1)
        self.assertEqual(inside["range_floor_abs"], 0)
        self.assertEqual(outside["pixel_fraction"], .5)
        self.assertEqual(outside["range_floor_abs"], 6)
        self.assertEqual(outside["error_share"], 1)
        self.assertEqual(all_row["range_floor_error_share"], 1)

    def test_empty_groups_have_nan_means_and_zero_counts(self):
        rows = measure_stage(np.array([[3.]]), np.array([[3.]]), np.array([2., 4.]),
                             {"full": np.ones((1, 1), bool)}, 1)
        self.assertEqual(rows[-1]["pixels"], 0)
        self.assertTrue(np.isnan(rows[-1]["abs"]))
        self.assertTrue(np.isnan(rows[0]["error_share"]))
        self.assertEqual(rows[-1]["pixel_fraction"], 0)

    def test_summary_uses_total_pixels_and_error_not_mean_of_ratios(self):
        a = measure_stage(np.array([[4.]]), np.array([[10.]]), np.array([2., 4.]),
                          {"full": np.ones((1, 1), bool)}, 2)
        b = measure_stage(np.full((1, 3), 3.), np.full((1, 3), 3.), np.array([2., 4.]),
                          {"full": np.ones((1, 3), bool)}, 2)
        result = {r["group"]: r for r in summarize(a + b)}
        self.assertEqual(result["all"]["abs"], 1.5)
        self.assertEqual(result["out_of_range"]["pixel_fraction"], .25)
        self.assertEqual(result["out_of_range"]["error_share"], 1)
        self.assertEqual(result["out_of_range"]["images"], 2)
        text = report(list(result.values()), {"model_type": "v2_vis", "checkpoint": "fixture"})
        self.assertIn("25.00", text)

    def test_per_pixel_unsorted_hypotheses_and_inclusive_boundaries(self):
        hypotheses = np.array([[[4., 12.]], [[2., 8.]]])
        rows = measure_stage(np.array([[2., 12.]]), np.array([[2., 12.]]), hypotheses,
                             {"full": np.ones((1, 2), bool)}, 3)
        self.assertEqual(rows[1]["pixels"], 2)
        self.assertEqual(rows[0]["nearest_hyp_abs"], 0)

    def test_nonfinite_in_valid_region_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            measure_stage(np.array([[np.nan]]), np.array([[3.]]), np.array([2., 4.]),
                          {"full": np.ones((1, 1), bool)}, 1)

    def test_runner_fixes_baseline_val_and_never_requests_training(self):
        args = SimpleNamespace(checkpoint=Path("baseline.ckpt"), datapath=Path("data"),
                               batch_size=4, num_workers=0, max_samples=8)
        command = build_command(args, Path("output"))
        self.assertEqual(command[command.index("--model_type") + 1], "v2_vis")
        self.assertEqual(Path(command[command.index("--testlist") + 1]).name, "val.txt")
        self.assertIn("--failure_diagnostics", command)
        self.assertEqual(command[-2:], ["--max_samples", "8"])
        self.assertNotIn("--mode", command)


if __name__ == "__main__":
    unittest.main()
