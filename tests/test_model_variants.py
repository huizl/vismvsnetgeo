import unittest

from models.model_variants import MODEL_VARIANTS, get_model_variant


class ModelVariantTest(unittest.TestCase):
    def test_complete_three_factor_table(self):
        expected = {
            "vis": "000",
            "m1_hyp": "100",
            "m2_visibility": "010",
            "m3_hybrid": "001",
            "m1_m2": "110",
            "m1_m3": "101",
            "m2_m3": "011",
            "full": "111",
        }
        self.assertEqual(set(MODEL_VARIANTS), set(expected))
        self.assertEqual(
            {name: variant.code for name, variant in MODEL_VARIANTS.items()},
            expected,
        )

    def test_only_m1_or_m2_requires_source_visibility_gt(self):
        self.assertFalse(get_model_variant("vis").needs_visibility_gt)
        self.assertFalse(get_model_variant("m3_hybrid").needs_visibility_gt)
        self.assertTrue(get_model_variant("m1_hyp").needs_visibility_gt)
        self.assertTrue(get_model_variant("m2_visibility").needs_visibility_gt)


if __name__ == "__main__":
    unittest.main()
