"""IFEval TaskSource: official checks, local data and held-out experiment splits."""
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]/'envs/instruction_following/IFEvalEnv'


def _import_ifevalenv():
    root = str(_DEFAULT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import ifevalenv
    return ifevalenv


@dataclass
class IFEvalSourceConfig:
    train_set: str = 'ifeval'
    test_set: str = ''
    test_fraction: float = .1
    limit: int = 0
    test_limit: int = 0
    split_seed: int = 0
    reward_mode: str = 'strict'
    python: str = ''
    upstream: str = ''
    timeout: float = 60
    max_concurrency: int = 4


class IFEvalTaskSource:
    def __init__(self, cfg: IFEvalSourceConfig):
        self.cfg = cfg
        self.ce = _import_ifevalenv()
        if cfg.reward_mode not in ('strict', 'fraction'):
            raise ValueError('ifeval_reward_mode must be strict or fraction')
        if not 0 <= cfg.test_fraction < 1 or cfg.limit < 0 or cfg.test_limit < 0:
            raise ValueError('invalid IFEval split fraction/task limits')
        self.upstream = Path(cfg.upstream) if cfg.upstream else self.ce.UPSTREAM
        self.grader = self.ce.OfficialGrader(python=cfg.python or self.ce.PYTHON,
            upstream=self.upstream, timeout=cfg.timeout, concurrency=cfg.max_concurrency)
        self._train, self._test = [], []
        self._initialized = False

    def _load(self, name):
        path = self.upstream/'instruction_following_eval/data/input_data.jsonl' if name == 'ifeval' else Path(name)
        if not path.exists():
            raise ValueError(f'IFEval data missing: {path}; run script/setup_ifeval.sh')
        return self.ce.load_tasks(path)

    async def initialize(self):
        if self._initialized:
            return
        tasks = self._load(self.cfg.train_set)
        if self.cfg.test_set:
            train, test = tasks, self._load(self.cfg.test_set)
            if {t.key for t in train} & {t.key for t in test} or {t.prompt_id for t in train} & {t.prompt_id for t in test}:
                raise ValueError('IFEval train/test overlap in task keys or prompts')
        else:
            train, test = self.ce.split_tasks(tasks, self.cfg.test_fraction, self.cfg.split_seed)
        # Limits never change split membership.
        self._train = train[:self.cfg.limit] if self.cfg.limit else train
        self._test = test[:self.cfg.test_limit] if self.cfg.test_limit else test
        if not self._train:
            raise ValueError('no IFEval training tasks')
        self._initialized = True
        logger.info('[ifeval] %d train / %d held-out; reward=%s', len(self._train), len(self._test), self.cfg.reward_mode)

    async def check(self):
        return await self.grader.check(self._train+self._test)

    def train_tasks(self): return self._train
    def test_tasks(self): return self._test
    def task_key(self, task): return task.key

    def make_env(self, task, uid):
        return self.ce.IFEvalEnv(task, self.grader, reward_mode=self.cfg.reward_mode)

    async def close(self):
        await self.grader.close()


def make_source_from_config(cfg):
    return IFEvalTaskSource(IFEvalSourceConfig(
        train_set='ifeval' if cfg.train_set == 'ops6k' else cfg.train_set,
        test_set=cfg.test_set, test_fraction=cfg.test_fraction,
        split_seed=cfg.ifeval_split_seed, limit=cfg.task_limit, test_limit=cfg.test_limit,
        reward_mode=cfg.ifeval_reward_mode, python=cfg.ifeval_python, upstream=cfg.ifeval_upstream,
        timeout=cfg.ifeval_timeout, max_concurrency=cfg.eval_max_concurrency or 4))
