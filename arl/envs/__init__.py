"""环境注册表: 名字 -> TaskSource 工厂。新环境实现 ``arl.agent.protocol`` 的协议后在这里登记。"""

from __future__ import annotations


def make_task_source(name: str, **kwargs):
    if name == "cuda":
        from .cuda import CudaSourceConfig, CudaTaskSource
        return CudaTaskSource(CudaSourceConfig(**kwargs))
    raise ValueError(f"unknown env {name!r} (available: cuda)")
