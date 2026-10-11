"""CPU tests for cross-dataset camera/GT geometry and statistical correctness."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from datasets.blendedmvs import MVSDataset, read_camera, read_pairs
from datasets.data_io import save_pfm
from tools.blendedmvs_metrics import (
    METRICS, aggregate, geometry_maps, region_metrics, select_examples,
)
from tools.eval_blendedmvs import comparison_rows, main, checkpoint_paths, parse_args

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / "tmp"
TEST_TMP.mkdir(exist_ok=True)


def write_fixture(root, scene="scene1", size=128, far=10):
    folder = root / scene
    for subdir in ("blended_images", "cams", "rendered_depth_maps"):
        (folder / subdir).mkdir(parents=True)
    for view in (0, 1):
        Image.new("RGB", (size, size), (30, 60, 90)).save(
            folder / "blended_images" / f"{view:08d}.jpg")
        text = ("extrinsic\n1 0 0 " + str(.1 * view) +
                "\n0 1 0 0\n0 0 1 0\n0 0 0 1\n\nintrinsic\n"
                f"80 0 {size/2}\n0 80 {size/2}\n0 0 1\n\n1 0.1 192 {far}\n")
        (folder / "cams" / f"{view:08d}_cam.txt").write_text(text)
        depth = np.full((size, size), 2., np.float32)
        depth[0, 0], depth[0, 1] = np.inf, 0
        save_pfm(str(folder / "rendered_depth_maps" / f"{view:08d}.pfm"), depth)
    (folder / "cams" / "pair.txt").write_text("2\n0\n1 1 1.0\n1\n1 0 1.0\n")
    return folder


def geometry_sample(depths, masks=None, translate=None):
    h, w = depths[0].shape
    ext = np.stack([np.eye(4) for _ in depths])
    if translate is not None:
        ext[1, 0, 3] = translate
    return dict(depth=depths[0], mask=np.ones((h, w), bool),
                gt_depths=np.stack(depths),
                gt_masks=np.stack(masks if masks is not None
                                  else [np.ones((h, w), bool) for _ in depths]),
                intrinsics=np.stack([np.eye(3) for _ in depths]), extrinsics=ext)


class BlendedDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="blended-test-", dir=TEST_TMP)
        self.root = Path(self.tmp.name)
        self.folder = write_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resize_and_quarter_scale_network_camera(self):
        dataset = MVSDataset(self.root, nviews=2, max_width=64, max_height=64)
        sample = dataset[0]
        self.assertEqual(sample["imgs"].shape, (2, 3, 64, 64))
        self.assertAlmostEqual(sample["scene_scale"], 100)
        self.assertAlmostEqual(sample["depth"][10, 10], 200)
        self.assertAlmostEqual(sample["intrinsics"][0, 0, 0], 40)
        self.assertAlmostEqual(sample["intrinsics"][0, 0, 2], 32)
        self.assertAlmostEqual(sample["extrinsics"][1, 0, 3], 10)
        full_proj = sample["intrinsics"][1] @ sample["extrinsics"][1, :3]
        np.testing.assert_allclose(sample["proj_matrices"][1, :2], full_proj[:2]/4)
        # The model scales Stage 1 projection by .5: full K/8 is required.
        np.testing.assert_allclose(sample["proj_matrices"][1, :2]*.5, full_proj[:2]/8)
        self.assertEqual(len(sample["depth_values"]), 192)
        self.assertAlmostEqual(sample["depth_values"][0], 100)
        self.assertAlmostEqual(sample["depth_values"][-1], 1000)
        self.assertTrue(np.isfinite(sample["depth"]).all())

    def test_scale_is_camera_only_and_stable_for_sample_subset(self):
        first = MVSDataset(self.root, nviews=2)
        depth_path = self.folder / "rendered_depth_maps" / "00000000.pfm"
        save_pfm(str(depth_path), np.full((128, 128), 9, np.float32))
        second = MVSDataset(self.root, nviews=2)
        self.assertEqual(first.scene_scales, second.scene_scales)
        self.assertEqual(first[0]["depth_values"].tolist(), second[0]["depth_values"].tolist())

    def test_gt_alignment_and_missing_source_gt_fail(self):
        depth_path = self.folder / "rendered_depth_maps" / "00000001.pfm"
        save_pfm(str(depth_path), np.ones((64, 64), np.float32))
        dataset = MVSDataset(self.root, nviews=2)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            dataset[0]
        depth_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "lacks input/GT"):
            MVSDataset(self.root, nviews=2)

    def test_data_check_cli_requires_no_torch(self):
        main(["--testpath", str(self.root), "--eval_nviews", "2", "--check_only",
              "--outdir", str(self.root / "reports"), "--max_samples", "1"])
        report = self.root / "reports" / "View5" / "data_check.json"
        self.assertTrue(report.is_file())
        self.assertIn('"debug_subset": true', report.read_text())

    def test_camera_prefers_explicit_bounds_and_pair_count_validation(self):
        camera = read_camera(self.folder / "cams" / "00000000_cam.txt")
        self.assertEqual(camera.depth_max, 10)
        path = self.folder / "cams" / "pair.txt"
        path.write_text("2\n0\n1 1 1\n")
        with self.assertRaisesRegex(ValueError, "count differs"):
            read_pairs(path)

    def test_empty_reference_gt_is_explicitly_recorded_in_data_check(self):
        save_pfm(str(self.folder / "rendered_depth_maps" / "00000000.pfm"),
                 np.zeros((128, 128), np.float32))
        main(["--testpath", str(self.root), "--eval_nviews", "2", "--check_only",
              "--outdir", str(self.root / "reports")])
        report = self.root / "reports" / "View5" / "skipped_samples.csv"
        self.assertIn("no finite positive depth", report.read_text(encoding="utf-8-sig"))


class BlendedGeometryTest(unittest.TestCase):
    def test_half_comparable_is_not_ge_two(self):
        depth = np.full((8, 8), 4., np.float32)
        nearer = np.full((8, 8), 2., np.float32)
        empty = np.zeros((8, 8), bool)
        valid = np.ones((8, 8), bool)
        sample = geometry_sample([depth, nearer, depth], [valid, valid, empty])
        result = geometry_maps(sample, occ_abs_tol=.1)
        self.assertTrue(result["masks"]["occluded_majority"].all())
        self.assertFalse(result["masks"]["occluded_ge2"].any())
        np.testing.assert_array_equal(result["comparable_count"], 1)
        sample = geometry_sample([depth, nearer, depth], [valid, valid, valid])
        result = geometry_maps(sample, occ_abs_tol=.1)
        self.assertTrue(result["masks"]["occluded_majority"].all())  # exactly half
        np.testing.assert_allclose(result["occlusion_ratio"], .5)

    def test_unknown_source_gt_and_outside_projection_are_not_occlusion(self):
        depth = np.full((8, 8), 4., np.float32)
        empty = np.zeros((8, 8), bool)
        sample = geometry_sample([depth, depth], [~empty, empty])
        result = geometry_maps(sample)
        self.assertFalse(result["masks"]["occluded_any"].any())
        self.assertFalse(result["comparable_count"].any())
        sample = geometry_sample([depth, depth], translate=100)
        result = geometry_maps(sample)
        self.assertFalse(result["masks"]["occluded_any"].any())
        self.assertFalse(result["comparable_count"].any())
        self.assertGreater(float(result["disparity"].min()), 0)

    def test_scene_scale_preserves_projection_and_occlusion(self):
        depth = np.full((8, 8), 4., np.float32)
        sample = geometry_sample([depth, depth/2], translate=.25)
        raw = geometry_maps(sample, occ_abs_tol=.1)
        normalized = dict(sample, depth=sample["depth"]*100, gt_depths=sample["gt_depths"]*100,
                          extrinsics=sample["extrinsics"].copy())
        normalized["extrinsics"][:, :3, 3] *= 100
        scaled = geometry_maps(normalized, occ_abs_tol=10)
        np.testing.assert_allclose(raw["disparity"], scaled["disparity"])
        np.testing.assert_array_equal(raw["occluded_count"], scaled["occluded_count"])


class BlendedStatisticsTest(unittest.TestCase):
    def test_strict_thresholds_relative_metrics_and_invalid_predictions(self):
        gt = np.array([[100., 200.]])
        pred = np.array([[101., 210.]])
        mask = np.ones(gt.shape, bool)
        result = region_metrics(pred, gt, mask, [mask]*3, [gt*0+1]*3)
        self.assertAlmostEqual(result["abs"], 5.5)
        self.assertAlmostEqual(result["abs_rel"], .03)
        self.assertAlmostEqual(result["rmse"], np.sqrt(50.5))
        self.assertEqual(result["rel_acc1"], 0)
        self.assertEqual(result["rel_acc2"], .5)
        self.assertEqual(result["rel_acc5"], .5)
        pred[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            region_metrics(pred, gt, mask, [mask]*3, [gt]*3)

    def test_pixel_rmse_pools_squares_and_scene_mean_weights_scenes_equally(self):
        def row(scene, pixels, error):
            return dict(scene=scene, region="full", pixels=pixels,
                        **{m: float(error) for m in METRICS})
        rows = [row("a", 1, 0), row("b", 3, 4)]
        result = {r["aggregation"]: r for r in aggregate(rows, scene_mean=True)}
        self.assertEqual(result["pixel_weighted"]["abs"], 3)
        self.assertAlmostEqual(result["pixel_weighted"]["rmse"], np.sqrt(12))
        self.assertEqual(result["image_mean"]["rmse"], 2)
        self.assertEqual(result["scene_mean"]["abs"], 2)
        summary = {"Base": list(result.values()),
                   "Base+A+B+C": [dict(r, abs=r["abs"]/2) for r in result.values()]}
        compared = comparison_rows(summary)
        self.assertTrue(all(r["abs_reduction_pct"] == 50 for r in compared))

    def test_selection_ranks_gain_within_gt_category_and_distinct_scenes(self):
        def record(scene, view, gain):
            return dict(scene=scene, view=view, abs_reduction_pct=gain,
                        large_disparity_pixels=100, occluded_ge2_pixels=0,
                        occluded_ge2_fraction=0, joint_ge2_pixels=0,
                        joint_ge2_fraction=0, joint_fraction_of_large=0,
                        joint_fraction_of_occlusion=0, disparity_p80_px=20)
        selected, missing = select_examples([record("a", 0, 30), record("a", 1, 50),
                                             record("b", 0, 20), record("c", 0, -1)], 2)
        self.assertEqual([(r["scene"], r["view"]) for r in selected], [("a", 1), ("b", 0)])
        self.assertEqual(len(missing), 2)  # no fabricated occlusion or joint examples

    def test_checkpoint_mapping_uses_training_series_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory(prefix="blended-ckpt-", dir=TEST_TMP) as directory:
            root = Path(directory)
            for model in ("range", "oa_hyp"):
                folder = root / f"{model}_view3"
                folder.mkdir()
                (folder / "best_2mm.ckpt").write_bytes(b"fixture only")
            args = parse_args(["--testpath", str(root), "--checkpoint_root", str(root),
                               "--series", "View3"])
            paths = checkpoint_paths(args)
            self.assertEqual(paths[0].parent.name, "range_view3")
            self.assertEqual(paths[1].parent.name, "oa_hyp_view3")


if __name__ == "__main__":
    unittest.main()
