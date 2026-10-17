"""Build offline study HTML and a source atlas using only the Python stdlib."""
from pathlib import Path
import ast
import hashlib
import html
import json
import re
from html.parser import HTMLParser

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
DATE = '14 October 2026'
FILES = [ROOT / 'backend' / name for name in (
    'app.py', 'preprocessing.py', 'audio_utils.py', 'model_manager.py',
    'youtube_utils.py', 'asr_manager.py', 'test_asr.py', 'ASR_NOTES.md', 'requirements.txt')]
FILES += [ROOT / 'frontend' / name for name in ('app.py', 'requirements.txt', '.streamlit/config.toml')]
FILES += [ROOT / 'models' / name for name in (
    'model_audio_hatespeech/config.json', 'model_audio_hatespeech/processor_config.json',
    'model_audio_deepfake/config.json', 'model_audio_deepfake/preprocessor_config.json',
    'model_text_hatespeech/text_model_zip/config.json')]

CSS = '''
:root{--ink:#182b3e;--muted:#52677c;--accent:#14626b;--line:#cfdae3;--paper:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f5;color:var(--ink);font:16px/1.7 'Segoe UI','Nirmala UI',Arial,sans-serif}
a{color:#086678;text-underline-offset:3px}aside{position:fixed;inset:0 auto 0 0;width:286px;background:#102c40;color:#eaf4fb;padding:26px 22px;overflow:auto}
aside a{color:#d1e6ef;text-decoration:none;display:block;font-size:13px;line-height:1.45;margin:10px 0}aside a:hover{color:white;text-decoration:underline}aside strong{font-size:19px}aside .tag{font-size:11px;letter-spacing:1.5px;color:#96cdd4;text-transform:uppercase}
main{margin:28px 32px 50px 318px;max-width:1100px;background:var(--paper);padding:45px 55px;border:1px solid #dce5eb;border-radius:8px}
h1{font-size:38px;line-height:1.2;letter-spacing:-1px;margin:0 0 18px}h2{font-size:26px;line-height:1.35;border-top:2px solid var(--line);padding-top:28px;margin-top:48px;scroll-margin-top:20px}h3{font-size:20px;line-height:1.4;margin-top:30px;color:var(--accent);scroll-margin-top:20px}p{margin:12px 0}li{margin:7px 0}strong{font-weight:650}
.subtitle{font-size:14px;color:var(--muted);border-left:4px solid var(--accent);padding-left:14px;margin:22px 0 32px}.toolbar{display:flex;gap:15px;align-items:center;margin-bottom:30px;flex-wrap:wrap}.toolbar a,button{font:600 13px 'Segoe UI',sans-serif;border:1px solid var(--line);border-radius:5px;padding:8px 12px;background:#f5fafb;color:var(--accent);text-decoration:none;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.55;margin:20px 0;table-layout:auto}th{background:#e5f0f3;color:#183b49;text-align:left}th,td{padding:10px 12px;border:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}tr:nth-child(even) td{background:#f8fafc}thead{display:table-header-group}
code{font:0.87em/1.5 Consolas,'Nirmala UI',monospace;background:#edf3f6;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{border:1px solid #d2e0e8;border-left:4px solid #4a8994;background:#f4f8fa;padding:16px;overflow:auto;font:13px/1.6 Consolas,'Nirmala UI',monospace;border-radius:4px}pre code{background:transparent;padding:0;font-size:inherit;overflow-wrap:normal}blockquote{border-left:4px solid #a7bdca;padding:2px 18px;color:var(--muted)}
.source{padding:14px 0;font-size:12px;tab-size:4;white-space:normal}.line{display:block;min-height:1.6em;padding-right:15px;white-space:pre}.line:target{background:#fff2b2}.ln{display:inline-block;width:57px;text-align:right;padding-right:15px;color:#6e8292;text-decoration:none;user-select:none}.source code{white-space:normal}summary{cursor:pointer;color:var(--accent);font-weight:650;padding:8px}.note{font-size:13px;color:var(--muted)}
@media(max-width:1000px){aside{position:static;width:auto;max-height:300px}main{margin:18px;padding:26px}h1{font-size:31px}}
@media print{body{background:#fff;font-size:10pt;line-height:1.5}aside,.toolbar{display:none}main{margin:0;max-width:none;padding:0;border:0}h1{font-size:26pt}h2{font-size:17pt;break-after:avoid;break-before:page;margin-top:0}h3{font-size:13pt;break-after:avoid}table{font-size:8.5pt}tr{break-inside:avoid}pre{white-space:pre-wrap;font-size:8pt;overflow:visible}pre code,.source code{white-space:pre-wrap;overflow-wrap:anywhere}a{color:inherit}p{orphans:3;widows:3}@page{size:A4;margin:18mm}}
'''

def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

def inline(text):
    parts = []
    def keep(value):
        parts.append(value)
        return f'ZZTOKEN{len(parts)-1}ZZ'
    text = re.sub(r'`([^`]+)`', lambda m: keep('<code>'+html.escape(m[1])+'</code>'), text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: keep('<a href="'+html.escape(m[2], quote=True)+'">'+html.escape(m[1])+'</a>'), text)
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    for i, part in enumerate(parts):
        text = text.replace(f'ZZTOKEN{i}ZZ', part)
    return text

def markdown(source):
    lines = source.splitlines()
    out, headings = [], []
    i = 0
    while i < len(lines):
        s = lines[i]
        if not s.strip():
            i += 1
            continue
        if s.startswith('```'):
            language, block = s[3:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i += 1
            out.append('<pre><code class="language-'+html.escape(language)+'">'+html.escape('\n'.join(block))+'</code></pre>')
            i += 1
            continue
        m = re.match(r'^(#{1,3}) (.*)', s)
        if m:
            level, label = len(m[1]), m[2]
            anchor = slug(label)
            out.append(f'<h{level} id="{anchor}">{inline(label)}</h{level}>')
            if level == 2:
                headings.append((anchor, label))
            i += 1
            continue
        if s.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                row = [c.strip().replace('\\|','|') for c in re.split(r'(?<!\\)\|', lines[i].strip().strip('|'))]
                if not all(re.fullmatch(r':?-+:?', c) for c in row):
                    rows.append(row)
                i += 1
            out.append('<table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in rows[0])+'</tr></thead><tbody>')
            for row in rows[1:]:
                out.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
            out.append('</tbody></table>')
            continue
        if re.match(r'^(- |\d+\. )', s):
            tag = 'ul' if s.startswith('- ') else 'ol'
            out.append(f'<{tag}>')
            while i < len(lines) and re.match(r'^(- |\d+\. )', lines[i]):
                text = re.sub(r'^(- |\d+\. )', '', lines[i])
                out.append('<li>'+inline(text)+'</li>'); i += 1
            out.append(f'</{tag}>')
            continue
        paragraph = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|- |\d+\. )', lines[i]):
            paragraph.append(lines[i]); i += 1
        out.append('<p>'+inline(' '.join(paragraph))+'</p>')
    return '\n'.join(out), headings

def shell(title, body, nav, kind):
    other = ('SOURCE_ATLAS.html', 'Open source atlas') if kind == 'guide' else ('VIVA_GUIDE.html', 'Open study guide')
    links = ''.join(f'<a href="#{anchor}">{html.escape(label)}</a>' for anchor,label in nav)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<aside><p class="tag">R26-DS-001 / Viva preparation</p><strong>{html.escape(title)}</strong><p class="note" style="color:#b9cdd9">Use Ctrl+F to find a function or question.</p>{links}</aside><main>
<div class="toolbar"><a href="{other[0]}">{other[1]}</a><a href="VIVA_GUIDE.md">Editable Markdown</a><button onclick="window.print()">Print / Save as PDF</button></div>
{body}<p class="subtitle">Prepared from the local source snapshot on {DATE}. Read explicit labels and limitations alongside model scores.</p></main></body></html>'''

class Inventory(ast.NodeVisitor):
    def __init__(self, path):
        self.path, self.scope, self.functions, self.calls = path, [], [], []
    def visit_ClassDef(self,node):
        self.scope.append(node.name); self.generic_visit(node); self.scope.pop()
    def visit_FunctionDef(self,node):
        self.scope.append(node.name)
        self.functions.append({'file':self.path, 'name':node.name, 'qualified':'.'.join(self.scope),
                               'line':node.lineno, 'end':node.end_lineno,
                               'decorators':[ast.unparse(d) for d in node.decorator_list]})
        self.generic_visit(node); self.scope.pop()
    def visit_Call(self,node):
        self.calls.append({'file':self.path,'caller':'.'.join(self.scope) or '<module/UI>',
                           'target':ast.unparse(node.func),'line':node.lineno})
        self.generic_visit(node)

def main():
    manifests, sources, funcs, calls = [], {}, [], []
    for path in FILES:
        rel = path.relative_to(ROOT).as_posix()
        content = path.read_text(encoding='utf-8')
        sources[rel] = content
        manifests.append({'path':rel,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'lines':len(content.splitlines())})
        if path.suffix == '.py':
            inv = Inventory(rel); inv.visit(ast.parse(content))
            funcs.extend(inv.functions); calls.extend(inv.calls)
    (OUT/'source_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    (OUT/'function_inventory.json').write_text(json.dumps(funcs,ensure_ascii=False,indent=2),encoding='utf-8')
    text = (OUT/'VIVA_GUIDE.md').read_text(encoding='utf-8')
    body, nav = markdown(text)
    words = len(re.findall(r'\S+', text))
    intro = f'<p class="subtitle">A guided walkthrough of every application file, {len(funcs)} function/method definitions, 44 mock questions and the current implementation’s limitations. About {words:,} words, plus the complete source atlas.</p>'
    body = body.replace('</h1>', '</h1>'+intro, 1)
    (OUT/'VIVA_GUIDE.html').write_text(shell('Final viva study guide',body,nav,'guide'),encoding='utf-8')
    atlas = ['<h1>Code source atlas</h1><p class="subtitle">Complete numbered source and saved configuration evidence. Click any line number to bookmark it. The study guide explains the behavior; this atlas lets you locate and rehearse the exact code.</p>']
    atlas += ['<h2 id="function-index">Function index and call references</h2>',
              '<p>Call references below are static syntactic matches, not a runtime trace. Repeated method names such as decode can be ambiguous. Flask calls route handlers through registration; ReportLab calls page_decor as a callback; unittest invokes test_ methods through discovery. These framework calls need not appear as direct calls in Python source.</p>',
              '<table><thead><tr><th>Definition</th><th>Local call/reference candidates</th></tr></thead><tbody>']
    def loc(file,line,label=None):
        return '<a href="#'+slug(file)+'-L'+str(line)+'">'+html.escape(label or file+':'+str(line))+'</a>'
    for f in funcs:
        refs = [c for c in calls if c['target'] == f['name'] or c['target'].endswith('.'+f['name'])]
        refhtml = '<br>'.join(loc(c['file'],c['line'],c['file']+':'+str(c['line'])+' — '+c['caller']) for c in refs)
        if f['decorators']:
            refhtml += '<br><code>'+html.escape('; '.join(f['decorators']))+'</code>'
        if f['name'] == 'page_decor': refhtml += '<br>Callback: doc.build(onFirstPage=page_decor, onLaterPages=page_decor)'
        if f['name'].startswith('test_'): refhtml += '<br>Invoked by unittest discovery'
        if not refhtml: refhtml = 'No direct local call found; see guide for framework entry or unused helper.'
        atlas.append('<tr><td>'+loc(f['file'],f['line'],f['file']+' — '+f['qualified'])+f'<br>Lines {f["line"]}–{f["end"]}</td><td>'+refhtml+'</td></tr>')
    atlas.append('</tbody></table>')
    atlasnav = [('function-index','Function index and callers')]
    for file,content in sources.items():
        anchor = slug(file); atlasnav.append((anchor,file))
        atlas.append(f'<h2 id="{anchor}">{html.escape(file)}</h2>')
        atlas.append('<pre class="source"><code>')
        for n,line in enumerate(content.splitlines(),1):
            atlas.append(f'<span class="line" id="{anchor}-L{n}"><a class="ln" href="#{anchor}-L{n}">{n}</a>{html.escape(line)}</span>')
        atlas.append('</code></pre>')
    atlas.append('<h2 id="snapshot">Snapshot verification</h2><p>SHA-256 identifies the exact file bytes reviewed. If source files change, regenerate this guide’s references before relying on old line numbers.</p><pre>'+html.escape(json.dumps(manifests,indent=2))+'</pre>')
    atlasnav.append(('snapshot','Snapshot hashes'))
    (OUT/'SOURCE_ATLAS.html').write_text(shell('Code source atlas','\n'.join(atlas),atlasnav,'atlas'),encoding='utf-8')
    class Check(HTMLParser):
        def __init__(self): super().__init__(); self.ids=[]; self.links=[]; self.depth=0
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if 'id' in attrs: self.ids.append(attrs['id'])
            if tag=='a': self.links.append(attrs.get('href',''))
    for file in ('VIVA_GUIDE.html','SOURCE_ATLAS.html'):
        parsed=Check(); parsed.feed((OUT/file).read_text(encoding='utf-8'))
        assert len(parsed.ids)==len(set(parsed.ids)), f'Duplicate IDs in {file}'
        assert all(link[1:] in parsed.ids for link in parsed.links if link.startswith('#')), f'Broken anchors in {file}'
    missing = [f['qualified'] for f in funcs if f['name'] not in text]
    assert not missing, missing
    print(json.dumps({'words':words,'functions':len(funcs),'source_files':len(FILES),
                      'chapters':len(nav),'coverage':'all function names covered',
                      'html':'unique IDs and local anchors verified'},indent=2))

if __name__=='__main__':
    main()
