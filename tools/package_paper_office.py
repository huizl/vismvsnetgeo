"""Package the manuscript, native Visio drawings and original scene panels."""
import re
import zipfile
from export_paper_office import ROOT, OUT, SOURCE, DOCX, parse_blocks


def main():
    target=OUT/'论文_Word与Visio图表.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [DOCX,OUT/'论文图表_可编辑.vsdx',OUT/'使用说明.md']:
            archive.write(path,path.name)
        for path in sorted((OUT/'visio_previews').glob('*')):
            if path.suffix in ('.png','.svg'):
                archive.write(path,str(path.relative_to(OUT)))
        for block in parse_blocks(SOURCE.read_text(encoding='utf8')):
            if block[0]!='image' or not block[2].endswith('overview_3x5.png'):
                continue
            match=re.search(r'(View[35])/(?:scenes/)?([^/]+)/(scan[^/]+)/',block[2])
            series,category,key=match.groups()
            folder=ROOT/'outputs/paper_visualizations_ranked'/series/'scenes'/category/key/'panels'
            base=f'scene_panels/{series}/{category}/{key}'
            for panel in sorted(folder.glob('*.png')):
                archive.write(panel,f'{base}/panels/{panel.name}')
            overview=ROOT/'docs'/block[2]
            archive.write(overview,f'{base}/overview_3x5.png')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if '/panels/' in n])==180
        print(f'{target}: {len(archive.namelist())} files; {target.stat().st_size} bytes')


if __name__=='__main__':
    main()
