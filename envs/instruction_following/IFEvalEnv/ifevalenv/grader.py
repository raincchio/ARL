"""Run pinned official checks in an isolated Python subprocess."""
import asyncio
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / '.cache/benchmarks'
ENTRY = Path(__file__).resolve().parents[1]
UPSTREAM = CACHE / 'ifeval-upstream'
PYTHON = CACHE / 'ifeval-venv/bin/python'
NLTK_DATA = CACHE / 'ifeval-nltk'


def verify_upstream(root=UPSTREAM):
    root = Path(root)
    manifest = json.loads((ENTRY/'upstream.json').read_text())
    for name, expected in manifest['files'].items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'IFEval upstream missing/modified: {path}; run script/setup_ifeval.sh')
    return manifest


class OfficialGrader:
    def __init__(self, *, python=PYTHON, upstream=UPSTREAM, timeout=60, concurrency=4):
        if timeout <= 0 or concurrency < 1:
            raise ValueError('IFEval grading timeout/concurrency must be positive')
        self.python, self.upstream = str(python), Path(upstream)
        self.timeout = timeout
        self.sem = asyncio.Semaphore(concurrency)
        self.processes = set()
        self.closed = False

    async def request(self, payload):
        async with self.sem:
            if self.closed:
                raise RuntimeError('IFEval grader is closed')
            env = {**os.environ, 'PYTHONPATH': str(self.upstream), 'NLTK_DATA': str(NLTK_DATA)}
            proc = await asyncio.create_subprocess_exec(self.python, str(ENTRY/'worker.py'),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            self.processes.add(proc)
            try:
                out, err = await asyncio.wait_for(proc.communicate(json.dumps(payload).encode()), self.timeout)
                if proc.returncode:
                    raise RuntimeError(f'IFEval worker failed: {err.decode(errors="replace")[-1500:]}')
                result = json.loads(out)
                if result.get('error'):
                    raise RuntimeError(f'IFEval checker error: {result["error"]}')
                return result
            finally:
                if proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                self.processes.discard(proc)

    async def check(self, tasks):
        manifest = verify_upstream(self.upstream)
        result = await self.request({'action': 'check', 'tasks': [t.record() for t in tasks]})
        return {**result, 'commit': manifest['commit']}

    async def score(self, task, response):
        return await self.request({'action': 'score', 'task': task.record(), 'response': response})

    async def close(self):
        self.closed = True
        procs = list(self.processes)
        for proc in procs:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        await asyncio.gather(*(proc.wait() for proc in procs))
