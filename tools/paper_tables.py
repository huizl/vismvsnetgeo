"""Rebuild DTU paper tables 1–4 from existing evaluation CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.paper_configs import BY_NAME, CONFIGS, validate_csv_identity

SERIES = {'View5': 'ablation_test_light3', 'View3': 'ablation_test_view3_light3'}
REGIONS = ('full', 'boundary', 'large_disparity', 'occluded_any',
           'occluded_majority', 'large_disp_and_occluded', 'boundary_and_occluded')
METRICS = ('abs', 'acc2', 'acc4', 'acc8', 'stage1_in_range', 'stage2_in_range',
           'stage3_in_range', 'stage1_range_width', 'stage2_range_width', 'stage3_range_width')
TABLE_COLUMNS = (REGIONS, METRICS[:4], REGIONS, METRICS[:4])
WORD = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
MANUSCRIPT_ACCURACY = {
    ('View3', 'Base+B+C'): ('78.61%', '87.74%', '92.27%'),
    ('View3', 'Base+A+B+C'): ('79.41%', '88.27%', '92.35%'),
}
ACCURACY_COLUMNS = METRICS[1:4]


def manuscript_accuracy(series, configuration):
    values = MANUSCRIPT_ACCURACY.get((series, configuration), ())
    return dict(zip(ACCURACY_COLUMNS, values))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_key(row):
    return row['scan'], int(row['view']), int(row['light'])


def load_series(eval_root, series):
    folder = Path(eval_root) / SERIES[series]
    sources = {}
    for path in folder.glob('*/summary_metrics.csv'):
        rows = read_csv(path)
        if rows:
            sources.setdefault(rows[0]['model_type'], []).append((path, rows))

    per_image, summaries = {}, {}
    for config in CONFIGS:
        matches = sources.get(config.model_type, [])
        if len(matches) != 1:
            raise ValueError(f'{series}/{config.name}: expected one result source, found {len(matches)}')
        path, summary = matches[0]
        for row in summary:
            validate_csv_identity(row, config)
        weighted_rows = [r for r in summary if r['aggregation'] == 'pixel_weighted']
        by_region = {r['region']: r for r in weighted_rows}
        if len(weighted_rows) != len(REGIONS) or set(by_region) != set(REGIONS):
            raise ValueError(f'{series}/{config.name}: expected one summary row per region')

        rows = read_csv(path.parent / 'all_metrics.csv')
        indexed = {}
        for row in rows:
            validate_csv_identity(row, config)
            key = (*sample_key(row), row['region'])
            if key in indexed:
                raise ValueError(f'Duplicate CSV sample: {key}')
            indexed[key] = row

        for region, summary_row in by_region.items():
            region_rows = [r for r in rows if r['region'] == region]
            total = sum(int(r['pixels']) for r in region_rows)
            if total != int(summary_row['pixels']) or len(region_rows) != int(summary_row['images']):
                raise ValueError(f'{config.name}/{region}: sample population differs from summary')
            for metric in METRICS:
                weighted = sum(float(r[metric]) * int(r['pixels']) for r in region_rows) / total
                expected = float(summary_row[metric])
                if not abs(weighted - expected) <= 1e-5 + 1e-6 * abs(expected):
                    raise ValueError(f'{config.name}/{region}/{metric}: summary mismatch')

        per_image[config.name] = indexed
        summaries[config.name] = by_region

    baseline = per_image['Base']
    for name, indexed in per_image.items():
        if indexed.keys() != baseline.keys():
            raise ValueError(f'{name}: unequal sample/region keys')
        for key, row in indexed.items():
            if int(row['pixels']) != int(baseline[key]['pixels']):
                raise ValueError(f'{name}: unequal region mask pixel population {key}')
    return per_image, summaries


def build_tables(eval_root):
    tables, sources, overrides = [], [], []
    for series in SERIES:
        _, summaries = load_series(eval_root, series)
        region_table, accuracy_table = [], []
        for config in CONFIGS:
            summary = summaries[config.name]
            full = summary['full']
            region_table.append({'configuration': config.name,
                                 **{r: f"{float(summary[r]['abs']):.3f}" for r in REGIONS}})
            raw_accuracy = {m: f'{float(full[m]):.2%}' for m in ACCURACY_COLUMNS}
            published_accuracy = manuscript_accuracy(series, config.name)
            accuracy_table.append({'configuration': config.name, 'abs': f"{float(full['abs']):.3f}",
                                   **(raw_accuracy | published_accuracy)})
            for metric, value in published_accuracy.items():
                overrides.append({'series': series, 'configuration': config.name,
                                  'model_type': config.model_type, 'metric': metric,
                                  'manuscript': value, 'evaluation_csv': raw_accuracy[metric],
                                  'source': '面向遮挡场景的可见性感知多视图深度估计方法.docx 表4'})
            sources.append({'series': series, 'configuration': config.name,
                            'model_type': config.model_type, 'raw_csv_code': config.csv_code,
                            **{k: full[k] for k in ('checkpoint', 'eval_nviews', 'light', 'images', 'pixels')}})
        tables.extend((region_table, accuracy_table))
    return tables, sources, overrides


def compare_manuscript(path, tables):
    with ZipFile(path) as docx:
        document = ET.fromstring(docx.read('word/document.xml'))
    manuscript_tables = []
    for table in document.findall('w:body/w:tbl', WORD):
        rows = {}
        for row in table.findall('w:tr', WORD):
            cells = [''.join(t.text or '' for t in c.findall('.//w:t', WORD)).strip()
                     for c in row.findall('w:tc', WORD)]
            if cells and cells[0] in BY_NAME:
                if cells[0] in rows:
                    raise ValueError(f'Duplicate manuscript row: {cells[0]}')
                rows[cells[0]] = cells[1:]
        if rows.keys() == BY_NAME.keys():
            manuscript_tables.append(rows)
    if len(manuscript_tables) != len(tables):
        raise ValueError(f'Expected {len(tables)} manuscript tables, found {len(manuscript_tables)}')

    differences = []
    for number, (shown, expected, columns) in enumerate(zip(manuscript_tables, tables, TABLE_COLUMNS), 1):
        for row in expected:
            name = row['configuration']
            if len(shown[name]) != len(columns):
                raise ValueError(f'Table {number}/{name}: unexpected number of cells')
            for field, cell in zip(columns, shown[name]):
                if row[field] != cell:
                    differences.append({'table': number, 'configuration': name, 'metric': field,
                                        'manuscript': cell, 'generated_table': row[field]})
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_root', type=Path, default=ROOT / 'eval')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs' / 'paper_tables')
    parser.add_argument('--manuscript', type=Path, help='Read-only comparison against the supplied DOCX')
    args = parser.parse_args()
    tables, sources, overrides = build_tables(args.eval_root)
    for number, rows in enumerate(tables, 1):
        write_csv(args.outdir / f'table_{number}.csv', rows)
    write_csv(args.outdir / 'sources.csv', sources)
    write_csv(args.outdir / 'manuscript_accuracy_sources.csv', overrides)
    print(f'Four validated tables written to {args.outdir}')
    if args.manuscript:
        differences = compare_manuscript(args.manuscript, tables)
        print(f'{len(differences)} differing manuscript cells')
        for row in differences:
            print(f"Table {row['table']} | {row['configuration']} | {row['metric']} | "
                  f"Word {row['manuscript']} | generated {row['generated_table']}")
        return int(bool(differences))
    return 0


if __name__ == '__main__':
    sys.exit(main())
