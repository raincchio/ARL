"""ProgramBench task source with pinned official metadata and final test rewards."""
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sys
import weakref

ROOT = Path(__file__).resolve().parents[3]


def _import_programbenchenv():
    # Shared transport, tool parsing and message history live in the existing workspace package.
    from .swe import _import_sweenv
    _import_sweenv()
    root = str(ROOT/'envs/coding/ProgramBenchEnv')
    if root not in sys.path: sys.path.insert(0, root)
    import programbenchenv
    return programbenchenv


@dataclass(frozen=True)
class ProgramTask:
    instance_id: str

    def __post_init__(self):
        if not isinstance(self.instance_id, str) or not re.fullmatch(r'[a-z0-9_.-]+__[a-z0-9_.-]+\.[a-f0-9]{7,40}', self.instance_id):
            raise ValueError('invalid ProgramBench instance_id')


@dataclass
class ProgramBenchSourceConfig:
    train_set: str = 'programbench'
    test_set: str = ''
    test_fraction: float = .1
    split_seed: int = 0
    limit: int = 0
    test_limit: int = 0
    task_ids: str = ''
    ssh_host: str = ''
    ssh_port: int = 22
    python: str = ''
    work_dir: str = '/tmp/arl-programbench-runs'
    blob_dir: str = ''
    memory: str = '60g'
    cpus: float = 20
    max_concurrency: int = 2
    max_turns: int = 1000
    command_timeout: int = 180
    eval_timeout: int = 3600
    eval_cpus: int = 4
    branch_workers: int = 1
    reward_mode: str = 'fraction'


class ProgramBenchTaskSource:
    def __init__(self, cfg):
        if not 0 <= cfg.test_fraction < 1 or min(cfg.limit, cfg.test_limit) < 0:
            raise ValueError('invalid ProgramBench split fraction/task limits')
        if cfg.reward_mode not in ('fraction', 'strict'):
            raise ValueError('programbench_reward_mode must be fraction or strict')
        self.ce = _import_programbenchenv()
        from sweenv.env import EnvConfig
        self.cfg = cfg
        self.backend = self.ce.ProgramBenchBackend(self.ce.BackendConfig(ssh_host=cfg.ssh_host,
            ssh_port=cfg.ssh_port, python=cfg.python or str(ROOT/'.cache/benchmarks/programbench-venv/bin/python'), work_dir=cfg.work_dir, memory=cfg.memory,
            cpus=cfg.cpus, max_concurrency=cfg.max_concurrency), blob_dir=cfg.blob_dir,
            eval_cpus=cfg.eval_cpus, branch_workers=cfg.branch_workers)
        self.env_cfg = EnvConfig(max_turns=cfg.max_turns, command_timeout=cfg.command_timeout, eval_timeout=cfg.eval_timeout)
        self._train, self._test, self._envs = [], [], weakref.WeakSet()
        self._initialized = False

    def _select(self, name, inventory):
        if name == 'programbench':
            ids = sorted(inventory)
            if self.cfg.task_ids:
                ids = [iid.strip() for iid in self.cfg.task_ids.split(',')]
        else:
            p = Path(name)
            records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.suffix=='.jsonl' else json.loads(p.read_text())
            ids = [r if isinstance(r, str) else r['instance_id'] for r in records]
        if len(set(ids)) != len(ids): raise ValueError('duplicate ProgramBench task IDs')
        if set(ids)-inventory: raise ValueError(f'unknown ProgramBench IDs: {sorted(set(ids)-inventory)}')
        return [ProgramTask(iid) for iid in sorted(ids)]

    async def initialize(self):
        if self._initialized: return
        inventory = {row['instance_id'] for row in await self.backend.inventory()}
        tasks = self._select(self.cfg.train_set, inventory)
        if self.cfg.test_set:
            train, test = tasks, self._select(self.cfg.test_set, inventory)
            if {t.instance_id for t in train} & {t.instance_id for t in test}: raise ValueError('ProgramBench train/test overlap')
        else:
            shuffled = list(tasks); random.Random(self.cfg.split_seed).shuffle(shuffled)
            count = int(len(shuffled)*self.cfg.test_fraction)
            if self.cfg.test_fraction and len(shuffled)>1: count = max(1, count)
            heldout = {t.instance_id for t in shuffled[:count]}
            train = [t for t in tasks if t.instance_id not in heldout]
            test = [t for t in tasks if t.instance_id in heldout]
        self._train = train[:self.cfg.limit] if self.cfg.limit else train
        self._test = test[:self.cfg.test_limit] if self.cfg.test_limit else test
        if not self._train: raise ValueError('no ProgramBench training tasks')
        self._initialized = True

    async def check(self): return await self.backend.health(t.instance_id for t in self._train+self._test)
    def train_tasks(self): return self._train
    def test_tasks(self): return self._test
    def task_key(self, task): return task.instance_id

    def make_env(self, task, uid):
        env = self.ce.ProgramBenchEnv(task, self.backend.workspace(task), self.env_cfg, reward_mode=self.cfg.reward_mode)
        self._envs.add(env)
        return env

    async def close(self):
        import asyncio
        try:
            await asyncio.gather(*(env.close() for env in self._envs))
        finally:
            await self.backend.close()


def make_source_from_config(cfg):
    options = {name.removeprefix('programbench_'): getattr(cfg, name) for name in (
        'programbench_ssh_host', 'programbench_ssh_port', 'programbench_python', 'programbench_work_dir',
        'programbench_blob_dir', 'programbench_memory', 'programbench_cpus', 'programbench_task_ids',
        'programbench_split_seed', 'programbench_command_timeout', 'programbench_eval_timeout',
        'programbench_eval_cpus', 'programbench_branch_workers', 'programbench_reward_mode')}
    return ProgramBenchTaskSource(ProgramBenchSourceConfig(train_set='programbench' if cfg.train_set=='ops6k' else cfg.train_set,
        test_set=cfg.test_set, test_fraction=cfg.test_fraction, limit=cfg.task_limit, test_limit=cfg.test_limit,
        max_concurrency=cfg.eval_max_concurrency or 2,
        max_turns=cfg.env_max_turns if cfg.env_max_turns is not None else 1000, **options))
