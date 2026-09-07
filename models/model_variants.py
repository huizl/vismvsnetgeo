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
    revision: str = 'legacy'
    guided_centers: bool = False

    @property
    def projection_validity(self):
        return self.revision == 'v2' and self.hypothesis_fusion

    @property
    def uses_visibility_gate(self):
        return self.revision == 'legacy' and self.visibility_modeling

    @property
    def hypothesis_visibility_supervision(self):
        return self.revision == 'legacy' and self.hypothesis_fusion

    @property
    def needs_visibility_gt(self):
        return self.hypothesis_visibility_supervision or self.visibility_modeling

    @property
    def code(self):
        return "{}{}{}".format(
            int(self.hypothesis_fusion),
            int(self.visibility_modeling),
            int(self.hybrid_sampling or self.guided_centers),
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

# Separate names prevent the revised M3 and auxiliary-only M2 from silently
# changing the semantics of existing checkpoints and CSVs.
for name, flags in {
    'v2_vis': (False, False, False),
    'v2_m1': (True, False, False),
    'v2_m2': (False, True, False),
    'v2_m3': (False, False, True),
    'v2_m1_m2': (True, True, False),
    'v2_m1_m3': (True, False, True),
    'v2_m2_m3': (False, True, True),
    'v2_full': (True, True, True),
}.items():
    MODEL_VARIANTS[name] = ModelVariant(flags[0], flags[1], False,
                                        revision='v2', guided_centers=flags[2])

MODEL_TYPE_CHOICES = tuple(MODEL_VARIANTS)


def get_model_variant(model_type):
    try:
        return MODEL_VARIANTS[model_type]
    except KeyError as exc:
        raise ValueError(
            "unknown model type {!r}; expected one of {}".format(
                model_type, ", ".join(MODEL_TYPE_CHOICES))) from exc
