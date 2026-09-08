import unittest
import csv
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.failure_diagnostics import (
    measure_stage, summarize, report, measure_origins, summarize_origins, origin_report,
)
from tools.run_baseline_diagnostics import build_command, compare_control, run_diagnostic, main


class FailureDiagnosticsTest(unittest.TestCase):
    def test_control_stops_after_first_failure_and_records_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "control"
            with patch("sys.argv", ["runner", "--s1_range_control", "--outdir", str(output)]), \
                    patch("tools.run_baseline_diagnostics.run_diagnostic", side_effect=RuntimeError("fixture failure")) as run:
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    main()
            self.assertEqual(run.call_count, 1)
            saved = json.loads((output / "control_manifest.json").read_text())
            self.assertEqual(saved["status"], "failed")

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"fixture")
            output = root / "output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("existing")
            args = SimpleNamespace(checkpoint=checkpoint, datapath=root, batch_size=4,
                                   num_workers=0, max_samples=None, stage1_dnum=64,
                                   range_origin=True, dry_run=False)
            with self.assertRaisesRegex(ValueError, "new or empty"):
                run_diagnostic(args, output)
            self.assertEqual(marker.read_text(), "existing")

    def test_s1_control_changes_only_count_in_model_arguments(self):
        args = SimpleNamespace(checkpoint=Path("baseline.ckpt"), datapath=Path("data"),
                               batch_size=4, num_workers=0, max_samples=None, stage1_dnum=48)
        a = build_command(args, Path("output"))
        args.stage1_dnum = 64
        b = build_command(args, Path("output"))
        self.assertEqual([(x, y) for x, y in zip(a, b) if x != y], [("48", "64")])

    def test_control_comparison_uses_deltas_and_rejects_unmatched_pixels(self):
        def write_csv(path, rows):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number in (48, 64):
                p = root / f"s1_d{number}"
                p.mkdir()
                args = SimpleNamespace(checkpoint=Path("baseline.ckpt"), datapath=Path("data"),
                                       batch_size=4, num_workers=0, max_samples=None, stage1_dnum=number)
                manifest = dict(status="completed", checkpoint_sha256="same", source_sha256={},
                                scope="full_val", max_samples=None, stage1_dnum=number,
                                command=build_command(args, p))
                (p / "diagnostic_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                write_csv(p / "all_metrics.csv", [dict(scan="scan3", view=0, light=3, region="full", pixels=10)])
                write_csv(p / "summary_metrics.csv", [dict(aggregation="pixel_weighted", region="full", abs=5 if number==48 else 4, acc2=.8 if number==48 else .85)])
                write_csv(p / "range_origin_summary.csv", [dict(region="full", comparison=c, group=g, pixel_fraction=v)
                          for c,g,v in [("direction_s1","above",.1),("s1_to_s3","in_out",.05),("s1_to_s3","out_out",.02)]])
                write_csv(p / "failure_summary.csv", [dict(stage=3,region="full",group="all",bad8=.1)])
                (p / "performance.json").write_text(json.dumps(dict(batch_size=4,forward_ms_per_image=None,
                    evaluation_seconds=10,devices=[])), encoding="utf-8")
            compare_control(root)
            result = (root / "comparison.md").read_text(encoding="utf-8")
            self.assertIn("-1.0000", result)
            self.assertIn("+5.000", result)
            write_csv(root / "s1_d64/all_metrics.csv", [dict(scan="scan3",view=0,light=3,region="full",pixels=11)])
            with self.assertRaisesRegex(ValueError, "pixel counts"):
                compare_control(root)

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

    def test_origin_partitions_distinguish_persistent_loss_recovery_and_new_loss(self):
        gt = np.array([[5., 15., 15., 5.]])
        pred = np.array([[5., 6., 15., 6.]])
        intervals = {"original": (0., 20.), "s1": (0., 10.), "s2": (0., 10.),
                     "s3": (np.array([[4., 4., 14., 6.]]), np.array([[6., 6., 16., 8.]]))}
        rows = measure_origins(pred, gt, intervals, {"full": np.ones(gt.shape, bool)})
        summary = summarize_origins(rows)
        chosen = {r["group"]: r for r in summary if r["comparison"] == "s1_to_s3"}
        for name in ("in_in", "in_out", "out_in", "out_out"):
            self.assertEqual(chosen[name]["pixels"], 1)
        self.assertEqual(chosen["in_out"]["error_share"], .1)
        self.assertEqual(chosen["out_out"]["error_share"], .9)
        self.assertEqual(chosen["out_in"]["abs"], 0)
        self.assertEqual(sum(chosen[n]["abs_sum"] for n in ("in_in", "in_out", "out_in", "out_out")), chosen["all"]["abs_sum"])
        self.assertIn("10.000", origin_report(summary, {"model_type": "v2_vis", "checkpoint": "fixture"}))

    def test_original_coverage_is_distinct_from_actual_first_stage_coverage(self):
        gt = np.array([[-1., 10., 13., 15.]])
        intervals = {"original": (0., 14.), "s1": (0., 12.),
                     "s2": (0., 12.), "s3": (0., 12.)}
        rows = measure_origins(np.clip(gt, 0, 12), gt, intervals, {"full": np.ones(gt.shape, bool)})
        indexed = {(r["comparison"], r["group"]): r for r in rows}
        endpoint = indexed["original_to_s1", "in_out"]
        self.assertEqual(endpoint["pixels"], 1)
        self.assertEqual(endpoint["gap_mean"], 1)
        self.assertEqual(indexed["direction_original", "above"]["pixels"], 1)
        self.assertEqual(indexed["direction_s1", "above"]["pixels"], 2)
        self.assertEqual(indexed["direction_s1", "below"]["pixels"], 1)
        self.assertEqual(indexed["direction_s1", "above"]["gap_max"], 3)

    def test_origin_summary_preserves_empty_groups_and_weighted_totals(self):
        intervals = {key: (0., 10.) for key in ("original", "s1", "s2", "s3")}
        rows = measure_origins(np.array([[10.]]), np.array([[20.]]), intervals,
                               {"full": np.ones((1, 1), bool)})
        rows += measure_origins(np.zeros((1, 3)), np.zeros((1, 3)), intervals,
                                {"full": np.ones((1, 3), bool)})
        indexed = {(r["comparison"], r["group"]): r for r in summarize_origins(rows)}
        self.assertEqual(indexed["s1_to_s3", "all"]["abs"], 2.5)
        self.assertEqual(indexed["s1_to_s3", "out_out"]["pixel_fraction"], .25)
        self.assertEqual(indexed["direction_s1", "above"]["gap_max"], 10)
        self.assertTrue(np.isnan(indexed["s1_to_s3", "in_out"]["abs"]))

    def test_origin_runner_flag_is_opt_in(self):
        args = SimpleNamespace(checkpoint=Path("baseline.ckpt"), datapath=Path("data"),
                               batch_size=4, num_workers=0, max_samples=None, range_origin=True)
        self.assertIn("--range_origin_diagnostics", build_command(args, Path("output")))
        args.range_origin = False
        self.assertNotIn("--range_origin_diagnostics", build_command(args, Path("output")))


if __name__ == "__main__":
    unittest.main()
