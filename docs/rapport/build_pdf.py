"""Construit rapport.pdf : Markdown -> HTML (pandoc) -> PDF (Chromium via Playwright)."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["pyproject.toml", ".env.example", ".gitignore", "examples/constitution.md", "examples/brief.md",
         "src/sdd/models.py", "src/sdd/ingest.py", "src/sdd/prompts.py", "src/sdd/llm.py",
         "src/sdd/validators.py", "src/sdd/guardian.py", "src/sdd/render.py", "src/sdd/graph.py", "src/sdd/cli.py", "src/sdd/runner.py", "src/sdd/ui.py",
         "tests/test_pipeline.py", "tests/test_guardian.py", "tests/test_ui.py", "tests/test_llm.py"]
LANG = {".py": "python", ".toml": "toml", ".md": "markdown"}

md = (HERE / "rapport.md").read_text(encoding="utf-8")
annex = []
for f in FILES:
    code = (ROOT / f).read_text(encoding="utf-8")
    fence = "````"
    annex.append(f"## `{f}` {{.annex-file}}\n\n{fence}{LANG.get(Path(f).suffix, 'text')}\n{code.rstrip()}\n{fence}\n")
md = md.replace("<!-- GRAPH_SVG -->", (HERE / "graph.svg").read_text(encoding="utf-8"))
md = md.replace("<!-- GUARDIAN_SVG -->", (HERE / "guardian.svg").read_text(encoding="utf-8"))
md = md.replace("<!-- ANNEXE_CODE -->", "\n".join(annex))
res = HERE / "resultats.md"
md = md.replace("<!-- RESULTATS_REELS -->", res.read_text(encoding="utf-8") if res.exists() else "")
build = HERE / "build"; build.mkdir(exist_ok=True)
(build / "full.md").write_text(md, encoding="utf-8")
subprocess.run(["pandoc", str(build / "full.md"), "-s", "--toc", "--toc-depth=2", "--highlight-style=tango",
                "--embed-resources", "--resource-path", str(HERE), "--css", str(HERE / "style.css"), "-o", str(build / "rapport.html")], check=True)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rapport-sdd-pipeline.pdf"
js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://{build / "rapport.html"}', {{ waitUntil: 'networkidle' }});
  await p.pdf({{ path: '{out}', format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size:7pt;color:#888;width:100%;padding:0 18mm;font-family:DejaVu Sans">SDD Pipeline déterministe et son Gardien — LangChain / LangGraph</div>',
    footerTemplate: '<div style="font-size:7pt;color:#888;width:100%;text-align:center;font-family:DejaVu Sans"><span class="pageNumber"></span> / <span class="totalPages"></span></div>' }});
  await b.close();
}})();
"""
subprocess.run(["node", "-e", js], check=True)
print(out)
