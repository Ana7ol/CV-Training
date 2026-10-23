from pathlib import Path
import ast, base64, html, json, zipfile
from content import LESSONS
ROOT = Path(__file__).resolve().parent
for i, lesson in enumerate(LESSONS, 1):
    ast.parse(lesson['code'])
    (ROOT/'examples'/f'{i:02d}_example.py').write_text('# '+lesson['title']+'\n'+lesson['code']+'\n')
css = (ROOT/'style.css').read_text()
metrics = json.loads((ROOT/'outputs_3d/metrics.json').read_text())
capstone = {'metrics': metrics, 'image': 'data:image/png;base64,'+base64.b64encode((ROOT/'outputs_3d/case_00.png').read_bytes()).decode()}
js = 'const capstoneResult = '+json.dumps(capstone)+';\n'+(ROOT/'app.js').read_text()
data = json.dumps(LESSONS, ensure_ascii=False).replace('</', '<\\/')
page = '''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="An offline computer vision course: 28 lessons from kernels to 3D segmentation, with equations, code, diagrams and interactive labs."><title>Computer Vision · From pixels to volumes</title><style>CSS_CONTENT</style></head>
<body><a class="skip" href="#main">Skip to course content</a><aside aria-label="Course navigation"><a class="brand" href="#home"><span class="brandmark">◈</span><span>Vision / Fieldnotes<small>A course from first principles</small></span></a><input class="search" id="search" type="search" placeholder="Search topics, equations, code…" aria-label="Search lessons"><nav id="nav" aria-label="Lessons"></nav><div class="asidefoot"><span id="progresstext">Your learning progress</span><div class="progress"><span id="progressbar"></span></div>Saved in this browser · works offline</div></aside>
<div class="workspace"><header class="topbar"><button class="menu" id="menu" aria-label="Toggle course navigation" aria-expanded="false">☰ Contents</button><span class="crumb">COMPUTER VISION / THE COMPLETE LEARNING PATH</span><div class="right"><button id="print">Print course / PDF</button><button id="reset">Reset progress</button></div></header><main class="content" id="main" tabindex="-1"><section id="home"></section><section id="labs" hidden></section><section id="glossary" hidden></section><section id="resources" hidden></section><div id="lessons"></div><footer>Vision / Fieldnotes · Original teaching material · Course assembled October 2026.<br>All browser examples are local and synthetic. Python training runs separately; references require internet access.</footer></main></div><script id="course-data" type="application/json">DATA_CONTENT</script><script>JS_CONTENT</script></body></html>'''
page=page.replace('CSS_CONTENT', css).replace('DATA_CONTENT',data).replace('JS_CONTENT', js)
(ROOT/'index.html').write_text(page)
with zipfile.ZipFile(ROOT.parent/'computer-vision-course.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            z.write(p, 'computer-vision-course/'+str(p.relative_to(ROOT)))
print(f'Built {len(LESSONS)} lessons · {len(page):,} HTML characters · self-contained index.html')
