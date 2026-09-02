"""Independent M1/M2/M3 switches for the final Vis-MVSNet ablation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    """One row of the final three-factor ablation table.

    M1: depth-hypothesis-aware source-view fusion.
    M2: geometrically supervised source visibility and conservative soft gating.
    M3: coverage-preserving local/extended hybrid depth sampling.
    """

    hypothesis_fusion: bool
    visibility_modeling: bool
    hybrid_sampling: bool

    @property
    def needs_visibility_gt(self):
        return self.hypothesis_fusion or self.visibility_modeling

    @property
    def code(self):
        return "{}{}{}".format(
            int(self.hypothesis_fusion),
            int(self.visibility_modeling),
            int(self.hybrid_sampling),
        )


MODEL_VARIANTS = {
    "vis": ModelVariant(False, False, False),
    "m1_hyp": ModelVariant(True, False, False),
    "m2_visibility": ModelVariant(False, True, False),
    "m3_hybrid": ModelVariant(False, False, True),
    "m1_m2": ModelVariant(True, True, False),
    "m1_m3": ModelVariant(True, False, True),
    "m2_m3": ModelVariant(False, True, True),
    "full": ModelVariant(True, True, True),
}

MODEL_TYPE_CHOICES = tuple(MODEL_VARIANTS)


def get_model_variant(model_type):
    try:
        return MODEL_VARIANTS[model_type]
    except KeyError as exc:
        raise ValueError(
            "unknown model type {!r}; expected one of {}".format(
                model_type, ", ".join(MODEL_TYPE_CHOICES))) from exc
