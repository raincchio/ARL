"""Install official ProgramBench release in the ignored, isolated cache."""
import json
from pathlib import Path
import subprocess
import venv

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT/'.cache/benchmarks'
manifest = json.loads(Path(__file__).with_name('upstream.json').read_text())
upstream = CACHE/'ProgramBench'
if not (upstream/'.git').exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v'+manifest['version'], manifest['repository'], str(upstream)], check=True)
if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != manifest['commit']:
    raise ValueError('ProgramBench cache revision differs from pinned release; use a clean pinned checkout')
subprocess.run(['git', '-C', str(upstream), 'diff', '--exit-code', 'HEAD', '--', 'src'], check=True)
python = CACHE/'programbench-venv/bin/python'
if not python.exists(): venv.create(python.parents[1], with_pip=True)
subprocess.run([str(python), '-m', 'pip', 'install', str(upstream)], check=True)
print('ProgramBench Python:', python)
print('Prepare cleanroom images and official test blobs on the Docker host before model runs.')
