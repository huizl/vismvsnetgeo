"""Paper display names; preserve checkpoint/CSV identity separately.

Display names follow tables 1–4 of the supplied Word manuscript. The paper
codes are row labels and do not encode the experimental factor settings.
Never use analysis_code to construct a network or change checkpoint tensors.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperConfig:
    model_type: str
    name: str
    analysis_code: str
    csv_code: str


CONFIGS = (
    PaperConfig('range', 'Base', '000', '010'),
    PaperConfig('oa_range', 'Base+A', '100', '110'),
    PaperConfig('vis', 'Base+B', '010', '000'),
    PaperConfig('range_hyp', 'Base+C', '001', '011'),
    PaperConfig('oa', 'Base+A+B', '110', '100'),
    PaperConfig('oa_full', 'Base+A+C', '101', '111'),
    PaperConfig('oa_hyp', 'Base+B+C', '011', '101'),
    PaperConfig('hyp', 'Base+A+B+C', '111', '001'),
)
BY_MODEL = {c.model_type: c for c in CONFIGS}
BY_NAME = {c.name: c for c in CONFIGS}


def validate_csv_identity(row, config):
    if row['model_type'] != config.model_type:
        raise ValueError('model_type does not match paper configuration')
    code = str(row['ablation_code']).zfill(3)
    if code != config.csv_code:
        raise ValueError(f'{config.name}: expected raw CSV code {config.csv_code}, got {code}')
