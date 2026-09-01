"""Independent A/B/C switches for the complete Vis-MVSNet ablation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    """One row of the three-factor ablation table.

    A: source-specific occlusion-aware pair supervision.
    B: uncertainty-adaptive cascade depth ranges.
    C: depth-hypothesis-aware source-view fusion.
    """

    occlusion_supervision: bool
    adaptive_range: bool
    hypothesis_fusion: bool

    @property
    def needs_visibility_gt(self):
        return self.occlusion_supervision or self.hypothesis_fusion

    @property
    def code(self):
        return "{}{}{}".format(
            int(self.occlusion_supervision),
            int(self.adaptive_range),
            int(self.hypothesis_fusion),
        )


MODEL_VARIANTS = {
    "vis": ModelVariant(False, False, False),
    "oa": ModelVariant(True, False, False),
    "range": ModelVariant(False, True, False),
    "hyp": ModelVariant(False, False, True),
    "oa_range": ModelVariant(True, True, False),
    "oa_hyp": ModelVariant(True, False, True),
    "range_hyp": ModelVariant(False, True, True),
    "oa_full": ModelVariant(True, True, True),
}

MODEL_TYPE_CHOICES = tuple(MODEL_VARIANTS)


def get_model_variant(model_type):
    try:
        return MODEL_VARIANTS[model_type]
    except KeyError as exc:
        raise ValueError(
            "unknown model type {!r}; expected one of {}".format(
                model_type, ", ".join(MODEL_TYPE_CHOICES))) from exc
