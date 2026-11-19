"""CudaEnv 适配: CudaTasks 题库 + CudaWorkspaceEnv -> ARL 的 TaskSource.

CudaEnv 不是 ARL 的依赖 (ARL 只认协议); 这里按 ``CUDAENV_ROOT`` (缺省仓内
``envs/coding/CudaEnv``) 把它放上 sys.path 再 import。
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = str(Path(__file__).resolve().parents[3] / "envs" / "coding" / "CudaEnv")


def _import_cudaenv():
    root = os.environ.get("CUDAENV_ROOT", _DEFAULT_ROOT)
    if root not in sys.path and os.path.isdir(root):
        sys.path.insert(0, root)
    import cudaenv  # noqa: F401
    return cudaenv


@dataclass
class CudaSourceConfig:
    train_set: str = "ops6k"           # cudaenv.load_task_set 的写法: ops6k / kernelmix:l1 / *.jsonl
    test_set: str = ""                 # 空 = 从 train_set 里切 test_fraction 出来
    test_fraction: float = 0.02
    limit: int = 0                     # 只取前 N 道 (冒烟)
    test_limit: int = 0
    split_seed: int = 0
    endpoint: str = ""                 # 空 = CUDAENV_EVAL_ENDPOINT / EVAL_ENDPOINT / 缺省主节点
    max_concurrency: int = 0           # 本进程同时在飞的评测数上限 (0 = 不限)
    max_turns: int = 16
    enable_analyze: bool = False
    reward_mode: str = "best"


class CudaTaskSource:
    def __init__(self, cfg: CudaSourceConfig):
        ce = _import_cudaenv()
        self._ce = ce
        self.cfg = cfg
        tasks = ce.load_task_set(cfg.train_set, limit=cfg.limit)
        if cfg.test_set:
            self._train = tasks
            self._test = ce.load_task_set(cfg.test_set, limit=cfg.test_limit)
        else:
            idx = list(range(len(tasks)))
            random.Random(cfg.split_seed).shuffle(idx)
            n_test = int(len(tasks) * cfg.test_fraction)
            if cfg.test_limit:
                n_test = min(n_test, cfg.test_limit)
            test_idx = set(idx[:n_test])
            self._test = [tasks[i] for i in sorted(test_idx)]
            self._train = [tasks[i] for i in range(len(tasks)) if i not in test_idx]
        self._client = ce.CudaEvalClient(cfg.endpoint or None,
                                         max_concurrency=cfg.max_concurrency)
        self._env_cfg = ce.EnvConfig(max_turns=cfg.max_turns,
                                     enable_analyze=cfg.enable_analyze,
                                     reward_mode=cfg.reward_mode)
        logger.info(f"[cudaenv] {len(self._train)} train / {len(self._test)} test tasks "
                    f"({cfg.train_set}{' | ' + cfg.test_set if cfg.test_set else ''}), "
                    f"endpoint {self._client.endpoint}, max_turns={cfg.max_turns}, "
                    f"analyze={cfg.enable_analyze}, reward={cfg.reward_mode}")

    def train_tasks(self) -> list:
        return self._train

    def test_tasks(self) -> list:
        return self._test

    def task_key(self, task) -> str:
        return task.instance_id

    def make_env(self, task, uid: str):
        return self._ce.CudaWorkspaceEnv(task, self._client, self._env_cfg)

    def check(self) -> dict:
        from cudaenv.client import check_health
        return check_health(self._client.endpoint)

    async def close(self) -> None:
        await self._client.close()


def make_source_from_config(cfg):
    """Translate training settings into CUDA-specific source configuration."""
    return CudaTaskSource(CudaSourceConfig(
        train_set=cfg.train_set, test_set=cfg.test_set, test_fraction=cfg.test_fraction,
        limit=cfg.task_limit, test_limit=cfg.test_limit,
        max_concurrency=cfg.eval_max_concurrency, max_turns=cfg.env_max_turns if cfg.env_max_turns is not None else 16,
        endpoint=cfg.eval_endpoint, enable_analyze=cfg.env_enable_analyze,
        reward_mode=cfg.env_reward_mode))
