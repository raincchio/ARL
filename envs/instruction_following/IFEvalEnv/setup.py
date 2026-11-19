"""Install pinned official checker and data in the ignored benchmark cache."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT/'.cache/benchmarks'
manifest = json.loads(Path(__file__).with_name('upstream.json').read_text())
for name, sha in manifest['files'].items():
    dst = CACHE/'ifeval-upstream'/name
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
        continue
    url = f'https://raw.githubusercontent.com/google-research/google-research/{manifest["commit"]}/{name}'
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f'hash mismatch: {name}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
python = CACHE/'ifeval-venv/bin/python'
if not python.exists():
    venv.create(python.parents[1], with_pip=True)
subprocess.run([str(python), '-m', 'pip', 'install', 'absl-py==2.3.1', 'langdetect==1.0.9',
                'nltk==3.9.2', 'immutabledict==4.2.2'], check=True)
subprocess.run([str(python), '-c', 'import nltk,sys; assert nltk.download("punkt_tab", download_dir=sys.argv[1], quiet=True)',
                str(CACHE/'ifeval-nltk')], check=True)
print('IFEval checker and data ready:', manifest['commit'])
