"""On-policy MultiCodeIF episodes using the pinned checker and repair prompt."""
import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import tempfile

from arl.benchmarks import multicodeif as mc


@dataclass
class MultiCodeIFSourceConfig:
    train_set: str = 'multicodeif'
    test_set: str = ''
    upstream: str = ''
    dependencies: str = ''
    suite: str = 'single_level'
    constraints: list[str] | None = None
    languages: list[str] | None = None
    test_fraction: float = .1
    split_seed: int = 0
    limit: int = 0
    test_limit: int = 0
    max_rounds: int = 3
    reward_mode: str = 'strict'
    timeout: float = 600
    max_concurrency: int = 4
    judge_base_url: str = ''
    judge_model: str = ''
    judge_api_key_env: str = 'MULTICODEIF_JUDGE_API_KEY'


@dataclass
class StepOutput:
    reward: float = 0.
    done: bool = False
    metrics: dict = field(default_factory=dict)
    infra_error: bool = False
    tool_results: list = field(default_factory=list)


class MultiCodeIFEnv:
    tools = []

    def __init__(self, task, source):
        self.task, self.source = task, source
        self.done = False
        self.rounds, self.response, self.result, self.error = 0, None, None, ''
        self.reward, self.correct = 0., False

    async def reset(self):
        if self.rounds or self.done:
            raise RuntimeError('MultiCodeIF episodes are single-use')

    def messages(self, aggressiveness=0):
        from arl.harnesses import get_harness
        previous = self.result
        return get_harness('multicodeif_self_repair').make_messages(prompt=self.task.row['prompt'],
            previous_response=self.response if previous else None,
            failed_checks=[c for c in previous['checks'] if not c['passed']] if previous else None,
            repair_template=previous['repair_template'] if previous else None)

    async def step(self, text, *, truncated=False):
        if self.done:
            raise RuntimeError('step after MultiCodeIF episode ended')
        self.rounds += 1
        self.response = text
        if truncated:
            # An incomplete answer is not an evaluator outage. No repair feedback
            # is fabricated from partial code; rollout owns truncation loss masks.
            self.done, self.reward, self.correct = True, 0., False
            return StepOutput(done=True, metrics={'correct': False, 'truncated': True})
        try:
            result = await self.source.score(self.task, text)
            checks = result['checks']
            if (result.get('task_id') != self.task.task_id or result['status'] != 'scored'
                    or len(checks) != len(self.task.checks)
                    or any(c['status'] != 'scored' or not isinstance(c['passed'], bool) for c in checks)
                    or any((c.get('category'), c.get('constraint'), c.get('answer')) !=
                           (r['category'], r.get('constraint', r['category']), r['answer'])
                           for c, r in zip(checks, self.task.checks))
                    or not isinstance(result['passed'], bool) or result['passed'] != all(c['passed'] for c in checks)):
                raise ValueError('incomplete or misaligned official MultiCodeIF result')
            self.result = result
            self.correct = result['passed']
            fraction = sum(c['passed'] for c in checks)/len(checks)
            self.reward = float(self.correct) if self.source.cfg.reward_mode == 'strict' else fraction
            self.done = self.correct or self.rounds >= self.source.cfg.max_rounds
            if not self.done and not isinstance(result.get('repair_template'), str):
                raise ValueError('official repair template missing')
            return StepOutput(reward=self.reward if self.done else 0., done=self.done,
                metrics={'correct': self.correct, 'multicodeif/check_fraction': fraction,
                         'multicodeif/rounds': self.rounds},
                tool_results=[('constraint_feedback', json.dumps(checks))])
        except Exception as exc:
            self.done, self.error = True, f'{type(exc).__name__}: {exc}'
            return StepOutput(done=True, infra_error=True, metrics={'correct': None, 'grading_error': self.error})

    async def finish(self, reason=''):
        self.done = True
        # End before a response is scored (e.g. context overflow) has no usable
        # episode outcome. Never assign a stale earlier repair score to it.
        return StepOutput(done=True, infra_error=True, metrics={'correct': None, 'finish_reason': reason})

    async def close(self): pass

    def summary(self):
        return {'task': self.task.task_id, 'response': self.response, 'rounds': self.rounds,
                'checks': self.result, 'error': self.error, 'evaluation_scope': 'official_constraint_checks',
                'functional_correctness': 'not_evaluated'}


class MultiCodeIFTaskSource:
    def __init__(self, cfg):
        if cfg.suite not in mc.SUITES or cfg.reward_mode not in ('strict', 'fraction'):
            raise ValueError('invalid MultiCodeIF suite/reward mode')
        if (not 0 <= cfg.test_fraction < 1 or min(cfg.limit, cfg.test_limit) < 0
                or not 1 <= cfg.max_rounds <= 16 or cfg.timeout <= 0 or cfg.max_concurrency < 1):
            raise ValueError('invalid MultiCodeIF split/limits/rounds/timeout/concurrency')
        for values in (cfg.constraints, cfg.languages):
            if values is not None and (not isinstance(values, (list, tuple)) or not all(isinstance(v, str) for v in values)):
                raise ValueError('MultiCodeIF constraints/languages must be lists of strings')
        self.cfg = cfg
        self.repo = Path(cfg.upstream or mc.CACHE/'MultiCodeIF')
        self.deps = Path(cfg.dependencies or mc.CACHE/'multicodeif-deps')
        self._train, self._test = [], []
        self._initialized = False
        self._slots = asyncio.Semaphore(cfg.max_concurrency)
        self._processes = set()

    async def initialize(self):
        if self._initialized: return
        self.commit = mc.revision(self.repo)
        kwargs = dict(suite=self.cfg.suite, constraints=self.cfg.constraints, languages=self.cfg.languages)
        tasks = mc.load_tasks(self.repo, **kwargs)
        by_id = {t.task_id: t for t in tasks}
        def select(name):
            if name == 'multicodeif': return tasks
            path = Path(name)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix == '.jsonl' else json.loads(path.read_text())
            ids = [r if isinstance(r, str) else r['task_id'] for r in rows]
            if len(set(ids)) != len(ids) or set(ids)-by_id.keys():
                raise ValueError('duplicate/unknown MultiCodeIF task IDs')
            return [by_id[key] for key in ids]
        def group(task): return (task.file, task.checks[0]['id']) if self.cfg.suite == 'multi_level' else task.task_id
        train = select(self.cfg.train_set)
        if self.cfg.test_set:
            test = select(self.cfg.test_set)
            if {group(t) for t in train} & {group(t) for t in test}:
                raise ValueError('MultiCodeIF train/test task or constraint-chain overlap')
        else:
            import random
            groups = list(dict.fromkeys(group(task) for task in train))
            random.Random(self.cfg.split_seed).shuffle(groups)
            count = int(len(groups)*self.cfg.test_fraction)
            if self.cfg.test_fraction and len(groups)>1: count=max(1, count)
            heldout = set(groups[:count])
            train, test = [t for t in train if group(t) not in heldout], [t for t in train if group(t) in heldout]
        self._train = train[:self.cfg.limit] if self.cfg.limit else train
        self._test = test[:self.cfg.test_limit] if self.cfg.test_limit else test
        if not self._train: raise ValueError('no MultiCodeIF training tasks')
        self._initialized = True

    async def score(self, task, response):
        payload = {'repo': str(self.repo), 'config': {
            'judge_base_url': self.cfg.judge_base_url, 'judge_model': self.cfg.judge_model,
            'judge_api_key_env': self.cfg.judge_api_key_env}, 'requests': [{
                'task_id': task.task_id, 'suite': self.cfg.suite, 'response': response, 'checks': task.checks}]}
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(self.deps), str(mc.ROOT), env.get('PYTHONPATH', '')))
        async with self._slots:
            with tempfile.TemporaryDirectory(prefix='arl-multicodeif-') as directory:
                process = await asyncio.create_subprocess_exec(sys.executable, str(mc.ENTRY/'worker.py'),
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    cwd=directory, env=env)
                self._processes.add(process)
                try:
                    stdout, stderr = await asyncio.wait_for(process.communicate(json.dumps(payload).encode()), self.cfg.timeout)
                    if process.returncode: raise RuntimeError('official MultiCodeIF checker failed: '+stderr.decode(errors='replace')[-1000:])
                    result = json.loads(stdout)
                    if 'error' in result or len(result.get('results', [])) != 1:
                        raise RuntimeError(result.get('error', 'invalid official MultiCodeIF results'))
                    return result['results'][0]
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.communicate()
                    self._processes.discard(process)

    async def check(self):
        # Probe checker availability before allocating the model. Unavailable
        # task checks are reported explicitly; per-episode outages are dropped.
        if not self._initialized: await self.initialize()
        checked = set()
        for task in self._train+self._test:
            signature = tuple((r['category'], r.get('constraint', r['category']), r['answer'].split(';', 1)[0]) for r in task.checks)
            if signature in checked: continue
            checked.add(signature)
            result = await self.score(task, 'def arl_dependency_probe(): pass')
            if result.get('status') != 'scored':
                raise ValueError('MultiCodeIF checker unavailable for selected tasks: '+json.dumps(result['checks']))
        return {'ready': True, 'commit': self.commit, 'train': len(self._train), 'test': len(self._test)}

    def train_tasks(self): return self._train
    def test_tasks(self): return self._test
    def task_key(self, task): return task.task_id
    def make_env(self, task, uid): return MultiCodeIFEnv(task, self)

    async def close(self):
        for process in tuple(self._processes):
            if process.returncode is None: process.kill()
        await asyncio.gather(*(p.wait() for p in tuple(self._processes)), return_exceptions=True)


def make_source_from_config(cfg):
    return MultiCodeIFTaskSource(MultiCodeIFSourceConfig(
        train_set='multicodeif' if cfg.train_set == 'ops6k' else cfg.train_set,
        test_set=cfg.test_set, test_fraction=cfg.test_fraction, limit=cfg.task_limit, test_limit=cfg.test_limit,
        max_concurrency=cfg.eval_max_concurrency or 4, **{name.removeprefix('multicodeif_'): getattr(cfg, name)
            for name in ('multicodeif_upstream', 'multicodeif_dependencies', 'multicodeif_suite',
                'multicodeif_constraints', 'multicodeif_languages', 'multicodeif_split_seed', 'multicodeif_max_rounds',
                'multicodeif_reward_mode', 'multicodeif_timeout', 'multicodeif_judge_base_url',
                'multicodeif_judge_model', 'multicodeif_judge_api_key_env')}))
