"""Use the document skill renderer with the exact native Word PDF on Windows.

The current Windows runtime does not bundle LibreOffice. The companion Office
automation exports the newly saved DOCX through Word. The packaged render_docx
pipeline performs its normal OOXML inspection and PDF-to-PNG rasterization.
"""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/paper_word_visio'
RENDER_TEMP=OUT/'qa/render_temp'
RENDER_TEMP.mkdir(parents=True,exist_ok=True)
tempfile.tempdir=str(RENDER_TEMP)
SKILL=Path.home()/'.codex/plugins/cache/openai-primary-runtime/documents/26.909.11809/skills/documents/render_docx.py'
spec=importlib.util.spec_from_file_location('paper_skill_renderer',SKILL)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def native_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    source=OUT/'qa/paper_native_word.pdf'
    if not source.is_file(): raise FileNotFoundError(source)
    target=Path(convert_tmp_dir)/(stem+'.pdf')
    shutil.copy2(source,target)
    return str(target),'Native Microsoft Word ExportAsFixedFormat, exact saved DOCX'


module.convert_to_pdf=native_pdf
sys.argv=[str(SKILL),str(OUT/'大视差与复杂遮挡深度估计论文.docx'),
          '--output_dir',str(OUT/'qa/pages'),'--dpi','120']
module.main()
