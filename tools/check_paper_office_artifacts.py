"""Check lossless manuscript conversion and native, editable Visio shapes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import unicodedata

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

from export_paper_office import SOURCE, OUT, DOCX, ROOT, parse_blocks, MathParser, split_math


def normalize(value):
    return re.sub(r'\s+','',value.replace('**','').replace('`',''))


def formula_text(node):
    # Word saves script and double-struck Unicode as base glyphs plus OMML styles.
    return unicodedata.normalize('NFKC',''.join(t.text or '' for t in node.iter(qn('m:t'))))


def source_math(text):
    return [MathParser(m).parse() for m in re.findall(r'\$([^$]+)\$',text)]


def main():
    blocks=parse_blocks(SOURCE.read_text(encoding='utf8'))
    manuscript=Document(DOCX)
    rows=[]
    for table in manuscript.tables:
        if len(table.rows)>1 and not table._tbl.findall('.//'+qn('w:drawing')):
            rows.append([[normalize(c.text) for c in row.cells] for row in table.rows])
    expected_tables=[b[1] for b in blocks if b[0]=='table']
    assert len(rows)==len(expected_tables)==28
    compared=0
    for index,(expected,actual) in enumerate(zip(expected_tables,rows)):
        assert len(expected)==len(actual),(index,'row count')
        for i,(e,a) in enumerate(zip(expected,actual)):
            for j,(ee,aa) in enumerate(zip(e,a)):
                if '$' in ee: continue
                assert normalize(ee)==aa,(index,i,j,ee,aa)
                compared+=1
    expected_math=[]
    for b in blocks:
        if b[0]=='equation':
            expected_math.extend(MathParser(p).parse() for p in split_math(b[1]))
        elif b[0]=='table':
            for row in b[1]:
                for cell in row: expected_math.extend(source_math(cell))
        elif b[0] in ('paragraph','caption'): expected_math.extend(source_math(b[1]))
    actual_math=manuscript._element.findall('.//'+qn('m:oMath'))
    assert len(expected_math)==len(actual_math)==149
    mismatches=[]
    for i,(e,a) in enumerate(zip(expected_math,actual_math)):
        if normalize(formula_text(e))!=normalize(formula_text(a)):
            mismatches.append({'index':i,'source':formula_text(e),'word':formula_text(a)})
    assert not mismatches,mismatches
    body_text=normalize(''.join(t.text or '' for t in manuscript._element.iter(qn('w:t'))))
    source_text_blocks=0
    for b in blocks:
        if b[0] not in ('paragraph','caption','heading'): continue
        value=b[2] if b[0]=='heading' else b[1]
        value=re.sub(r'\$[^$]+\$','',value)
        value=re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',value)
        assert normalize(value) in body_text, value
        source_text_blocks+=1
    with zipfile.ZipFile(DOCX) as archive:
        media={hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith('word/media/')}
    scene_panel_count=0
    for b in blocks:
        if b[0]=='image' and b[2].endswith('overview_3x5.png'):
            m=re.search(r'(View[35])/(?:scenes/)?([^/]+)/(scan[^/]+)/',b[2])
            series,category,key=m.groups()
            folder=ROOT/'outputs/paper_visualizations_ranked'/series/'scenes'/category/key/'panels'
            panels=list(folder.glob('*.png'))
            assert len(panels)==15,(folder,len(panels))
            for image in panels:
                assert hashlib.sha256(image.read_bytes()).hexdigest() in media,image
                scene_panel_count+=1
    vns={'v':'http://schemas.microsoft.com/office/visio/2012/main'}
    shape_count=0
    with zipfile.ZipFile(OUT/'论文图表_可编辑.vsdx') as archive:
        pages=[n for n in archive.namelist() if re.fullmatch(r'visio/pages/page\d+\.xml',n)]
        assert len(pages)==10
        for name in pages:
            page=etree.fromstring(archive.read(name))
            shapes=page.findall('.//v:Shape',vns)
            assert len(shapes)>10
            assert not page.findall('.//v:ForeignData',vns),name
            shape_count+=len(shapes)
    report={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'data_tables':28,'non_math_table_cells_compared':compared,
            'native_formulas_matched':149,'scene_panels_verified_original':scene_panel_count,
            'source_text_blocks_preserved':source_text_blocks,
            'visio_pages':10,'visio_native_shapes':shape_count,'visio_foreign_bitmap_shapes':0,
            'all_checks_passed':True}
    (OUT/'qa/final_integrity_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__': main()
