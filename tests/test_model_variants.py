import unittest

from models.model_variants import MODEL_VARIANTS, get_model_variant


class ModelVariantTest(unittest.TestCase):
    def test_complete_three_factor_table(self):
        expected = {
            "vis": "000",
            "oa": "100",
            "range": "010",
            "hyp": "001",
            "oa_range": "110",
            "oa_hyp": "101",
            "range_hyp": "011",
            "oa_full": "111",
        }
        self.assertEqual(set(MODEL_VARIANTS), set(expected))
        self.assertEqual(
            {name: variant.code for name, variant in MODEL_VARIANTS.items()},
            expected,
        )

    def test_only_a_or_c_requires_source_visibility_gt(self):
        self.assertFalse(get_model_variant("vis").needs_visibility_gt)
        self.assertFalse(get_model_variant("range").needs_visibility_gt)
        self.assertTrue(get_model_variant("oa").needs_visibility_gt)
        self.assertTrue(get_model_variant("hyp").needs_visibility_gt)


if __name__ == "__main__":
    unittest.main()
