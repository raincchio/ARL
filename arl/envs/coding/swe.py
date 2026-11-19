"""TaskSource adapter for envs/coding/SweEnv and standard SWE-bench Python instances."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import random
import sys
import weakref

logger = logging.getLogger(__name__)


def _import_sweenv():
    root = str(Path(__file__).resolve().parents[3] / "envs" / "coding" / "SweEnv")
    if root not in sys.path:
        sys.path.insert(0, root)
    import sweenv
    return sweenv


@dataclass
class SweSourceConfig:
    train_set: str = "verified"        # local JSON[L], HF dataset ID/alias, or remote:/host/file.jsonl
    test_set: str = ""
    test_fraction: float = .02
    limit: int = 0
    test_limit: int = 0
    split_seed: int = 0
    max_turns: int = 250
    max_concurrency: int = 8
    dataset_split: str = "test"
    endpoint: str = ""
    token_file: str = ""
    ssh_host: str = ""
    ssh_port: int = 22
    python: str = "python3"
    worker_script: str = "/root/workspace/SweContainerService/swe_service/worker.py"
    work_dir: str = "/tmp/arl-swe-runs"
    proxy: str = ""
    memory: str = "4g"
    cpus: float = 2.
    command_timeout: int = 60
    eval_timeout: int = 1800


class SweTaskSource:
    def __init__(self, cfg):
        if not 0 <= cfg.test_fraction < 1:
            raise ValueError("SWE test_fraction must be in [0, 1)")
        if min(cfg.limit, cfg.test_limit) < 0:
            raise ValueError("SWE task limits must be >= 0")
        self.cfg = cfg
        self._ce = ce = _import_sweenv()
        self._backend = ce.DockerBackend(ce.BackendConfig(
            endpoint=cfg.endpoint, token_file=cfg.token_file,
            ssh_host=cfg.ssh_host, ssh_port=cfg.ssh_port, python=cfg.python, worker_script=cfg.worker_script, work_dir=cfg.work_dir,
            proxy=cfg.proxy, memory=cfg.memory, cpus=cfg.cpus, max_concurrency=cfg.max_concurrency or 8))
        self._env_cfg = ce.EnvConfig(max_turns=cfg.max_turns, command_timeout=cfg.command_timeout,
                                     eval_timeout=cfg.eval_timeout)
        self._train, self._test = [], []
        self._envs = weakref.WeakSet()
        self._initialized = False

    async def _load(self, name, limit):
        if name.startswith("remote:"):
            rows = await self._backend.load_tasks(name.removeprefix("remote:"), limit)
            tasks = [self._ce.SweTask.from_record(row) for row in rows]
            if len({t.instance_id for t in tasks}) != len(tasks):
                raise ValueError("duplicate SWE-bench instance_id")
            return tasks
        # Initialization runs before sampling/training begins, so no background
        # executor is needed for dataset I/O.
        return self._ce.load_task_set(name, split=self.cfg.dataset_split, limit=limit)

    async def initialize(self):
        if self._initialized:
            return
        tasks = await self._load(self.cfg.train_set, self.cfg.limit)
        if self.cfg.test_set:
            self._train = tasks
            self._test = await self._load(self.cfg.test_set, self.cfg.test_limit)
            if {t.instance_id for t in self._train} & {t.instance_id for t in self._test}:
                raise ValueError("SWE train_set and test_set overlap; use disjoint instance files")
        else:
            ids = list(range(len(tasks)))
            random.Random(self.cfg.split_seed).shuffle(ids)
            n_test = int(len(tasks) * self.cfg.test_fraction)
            if self.cfg.test_limit:
                n_test = min(n_test, self.cfg.test_limit)
            heldout = set(ids[:n_test])
            self._train = [task for i, task in enumerate(tasks) if i not in heldout]
            self._test = [tasks[i] for i in sorted(heldout)]
        if not self._train:
            raise ValueError("no SWE training instances")
        self._initialized = True
        logger.info("[sweenv] %d train / %d held-out; worker=%s; reward=resolved (0/1)",
                    len(self._train), len(self._test), self.cfg.endpoint or self.cfg.ssh_host or "local")

    async def check(self):
        if self.cfg.endpoint and self._initialized:
            result = await self._backend.rpc("preflight", instance_ids=[t.instance_id for t in self._train + self._test])
            if not result["ready"]:
                raise self._ce.BackendError(f"SWE host is missing {len(result['missing'])} instance images; "
                    f"preload before training: {result['missing'][:10]}")
        return await self._backend.health()

    def train_tasks(self):
        return self._train

    def test_tasks(self):
        return self._test

    def task_key(self, task):
        return task.instance_id

    def make_env(self, task, uid):
        env = self._ce.SweWorkspaceEnv(task, self._backend.workspace(task), self._env_cfg)
        self._envs.add(env)
        return env

    async def close(self):
        await asyncio.gather(*(env.close() for env in self._envs))
        self._envs.clear()
        await self._backend.close()


def make_source_from_config(cfg):
    """Translate training settings into SWE-specific source configuration."""
    return SweTaskSource(SweSourceConfig(
        train_set="verified" if cfg.train_set == "ops6k" else cfg.train_set,
        test_set=cfg.test_set, test_fraction=cfg.test_fraction,
        limit=cfg.task_limit, test_limit=cfg.test_limit,
        max_concurrency=cfg.eval_max_concurrency, max_turns=cfg.env_max_turns if cfg.env_max_turns is not None else 250,
        **{name.removeprefix("swe_"): getattr(cfg, name) for name in (
            "swe_dataset_split", "swe_endpoint", "swe_token_file", "swe_ssh_host", "swe_ssh_port", "swe_python", "swe_worker_script", "swe_work_dir",
            "swe_proxy", "swe_memory", "swe_cpus", "swe_command_timeout", "swe_eval_timeout")}))
