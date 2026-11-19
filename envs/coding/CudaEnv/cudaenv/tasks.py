"""题目: CudaTasks 注册表里的一个仓库 (``tasks/<set>/registry/<instance_id>/``).

只读本地共享盘上的 ``task.json`` + ``model.py`` (参考实现进 prompt / read_file);
判分时服务端自己再按 instance_id 从注册表取原版, 客户端传什么都改不了受保护文件。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

TASKS_ROOT_DEFAULT = os.environ.get("CUDAENV_TASKS_ROOT", "/root/workspace/CudaTasks/tasks")


@dataclass(frozen=True)
class CudaTask:
    instance_id: str
    ref_code: str
    registry: str                 # 含 <instance_id>/ 的注册表目录 (随请求发给服务端)
    level: int = 0
    problem_id: int = 0
    timeout: int = 600            # task.json 的评测超时
    baselines: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:     # 和 CudaAgent 老字段同名, 方便日志/归档复用
        return self.instance_id


def load_repo(repo: str) -> CudaTask:
    """从一道题的仓库目录构造 CudaTask。"""
    repo = os.path.abspath(repo)
    with open(os.path.join(repo, "task.json")) as f:
        tj = json.load(f)
    with open(os.path.join(repo, tj.get("ref_file", "model.py"))) as f:
        ref = f.read()
    meta = tj.get("meta") or {}
    return CudaTask(
        instance_id=tj.get("instance_id") or os.path.basename(repo),
        ref_code=ref,
        registry=os.path.dirname(repo),
        level=int(meta.get("level") or 0),
        problem_id=int(meta.get("problem_id") or 0),
        timeout=int(tj.get("timeout") or 600),
        baselines=dict(tj.get("baselines") or {}),
        meta=meta,
    )


def load_instances(path: str, limit: int = 0) -> list[CudaTask]:
    """``instances.jsonl`` (CudaTasks 生成器产出, 每行带 ``repo``) -> 题目列表。"""
    tasks: list[CudaTask] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            repo = rec.get("repo")
            if not repo:
                reg = os.path.join(os.path.dirname(os.path.abspath(path)), "registry")
                repo = os.path.join(reg, rec["instance_id"])
            tasks.append(load_repo(repo))
            if limit and len(tasks) >= limit:
                break
    return tasks


def load_task_set(name_or_path: str, *, tasks_root: str = TASKS_ROOT_DEFAULT,
                  limit: int = 0) -> list[CudaTask]:
    """``"ops6k"`` / ``"kernelmix"`` / ``"toy"`` (按 tasks_root 解析), 或直接给 instances*.jsonl 路径。

    子集写法: ``kernelmix:l1`` -> ``tasks/kernelmix/instances_l1.jsonl``。
    """
    if name_or_path.endswith(".jsonl") or os.path.sep in name_or_path:
        return load_instances(name_or_path, limit)
    name, _, sub = name_or_path.partition(":")
    fn = f"instances_{sub}.jsonl" if sub else "instances.jsonl"
    return load_instances(os.path.join(tasks_root, name, fn), limit)
