"""Validate equation/citation/figure numbering and local image paths."""
import argparse
from pathlib import Path
import re


def check(path):
    path=Path(path)
    text=path.read_text(encoding='utf-8')
    body,bibliography=text.split('## 参考文献',1)
    first=list(dict.fromkeys(int(x) for x in re.findall(r'\[(\d+)\]',body)))
    reference_ids=[int(x) for x in re.findall(r'(?m)^\[(\d+)\]',bibliography)]
    if first != list(range(1,len(first)+1)) or first != reference_ids:
        raise ValueError('References must be consecutive in first-appearance order and match bibliography')
    equations=[int(x) for x in re.findall(r'\\tag\{(\d+)\}',body)]
    if equations != list(range(1,len(equations)+1)):
        raise ValueError('Equation tags are not consecutive')
    captions=[int(x) for x in re.findall(r'\*\*图(\d+)\.',body)]
    if captions != list(range(1,len(captions)+1)):
        raise ValueError('Figure captions are not consecutive')
    images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',body)
    if len(images) != len(captions):
        raise ValueError('Image and caption counts differ')
    for image in images:
        if not (path.parent/image).is_file():
            raise FileNotFoundError(image)
    if re.search(r'待填|图\s*X|总体框架，连线草图',body):
        raise ValueError('Manuscript contains placeholders')
    print(f'OK: {len(equations)} equations, {len(first)} ordered references, {len(images)} figures')
    return len(equations),len(first),len(images)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',nargs='?',default=str(Path(__file__).resolve().parents[1]/'docs'/'PAPER_COMPLETE_WITH_RESULTS.md'))
    check(parser.parse_args().path)
