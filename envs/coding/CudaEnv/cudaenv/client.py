"""CudaEval 客户端: ``POST /repo/evaluate`` (迭代模式, files 方式交工作区).

为什么走 ``/repo/evaluate`` 而不是老的 ``/workspace/evaluate``: 后者 2026-08-30 起降为集群
内部转发端点 (CudaEval API.md §2)。repo 模式下 ref_code / cfg / 基线 / 受保护文件**全由
服务端按 instance_id 从注册表取**, 客户端只交 agent 能改的文件 —— 判定口径只剩一份。

连接纪律沿用 CudaAgent ``r3lm/envs/eval/remote.py`` 的实测结论:
  * ``force_close=True`` 不复用连接 (uvicorn keep-alive 5s < aiohttp 池 15s, 复用必撞死连接);
  * 客户端总超时 = 服务端 timeout + margin, margin 按"训练步阻塞 event loop 的最长窗口"给;
  * 网络/HTTP 失败一律退化成 ``infra_error=True`` 的结果, 绝不抛给 rollout。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENDPOINT_ENV = "CUDAENV_EVAL_ENDPOINT"
ENDPOINT_DEFAULT = "http://10.0.134.90:8000"
MARGIN_ENV = "CUDAENV_CLIENT_MARGIN"

# 工具名 -> 服务端动作 (与 CudaTasks cudatool 的 ACTION_MAP 一致)。
# profile / analyze 固定带 verify: 进奖励的 speedup 永远是"刚验证过的那个 .so"测出来的。
ACTIONS = {
    "compile": ["compile"],
    "verify": ["verify"],
    "profile": ["verify", "profile"],
    "analyze": ["verify", "acu"],
    "grade": ["compile", "verify", "profile"],   # 判分模式 (tests/resolved), 交卷时可选
}

_CONNECT_RETRIES = 1


def _infra(msg: str, instance_id: str | None = None) -> dict[str, Any]:
    return {"instance_id": instance_id, "ok": False, "workspace": {}, "milestone": None,
            "infra_error": True, "error_message": msg}


class CudaEvalClient:
    def __init__(self, endpoint: str | None = None, *, client_margin: float | None = None,
                 max_concurrency: int = 0):
        self.endpoint = (endpoint or os.environ.get(ENDPOINT_ENV)
                         or os.environ.get("EVAL_ENDPOINT") or ENDPOINT_DEFAULT).rstrip("/")
        self.client_margin = float(client_margin if client_margin is not None
                                   else os.environ.get(MARGIN_ENV, 1800))
        self._session = None
        self._loop = None
        # 0 = 不限; 限流放在客户端, 服务端排队是按来源 IP 轮询的, 一台机器打太多只会自己排队
        self._sem = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None

    async def _get_session(self):
        loop = asyncio.get_running_loop()
        if self._session is None or self._loop is not loop or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, force_close=True))
            self._loop = loop
        return self._session

    async def evaluate(self, instance_id: str, files: dict[str, str], tool: str, *,
                       registry: str | None = None, timeout: float = 600,
                       cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """一次评测。返回 ``/repo/evaluate`` 的响应 (``workspace`` 内是 §10 形状的原始反馈)。"""
        body: dict[str, Any] = {"instance_id": instance_id, "files": files,
                                "actions": ACTIONS[tool], "timeout": int(timeout)}
        if registry:
            body["registry"] = registry
        if cfg:
            body["cfg"] = cfg
        if self._sem is None:
            return await self._post("/repo/evaluate", body, timeout, instance_id)
        async with self._sem:
            return await self._post("/repo/evaluate", body, timeout, instance_id)

    async def _post(self, path: str, body: dict, timeout: float, iid: str | None) -> dict:
        import aiohttp

        client_to = aiohttp.ClientTimeout(total=timeout + self.client_margin)
        for attempt in range(1 + _CONNECT_RETRIES):
            try:
                session = await self._get_session()
                async with session.post(f"{self.endpoint}{path}", json=body,
                                        timeout=client_to) as resp:
                    if resp.status != 200:
                        text = (await resp.text())[:200]
                        return _infra(f"eval server HTTP {resp.status}: {text}", iid)
                    return await resp.json()
            except Exception as e:  # noqa: BLE001 -- 评测绝不能炸 rollout
                conn_err = isinstance(e, aiohttp.ClientConnectionError) and not isinstance(
                    e, asyncio.TimeoutError)
                if conn_err and attempt < _CONNECT_RETRIES:
                    continue
                return _infra(f"remote eval unreachable: {type(e).__name__}: {e}", iid)
        raise AssertionError("unreachable")

    async def health(self) -> dict[str, Any] | None:
        import aiohttp
        try:
            session = await self._get_session()
            async with session.get(f"{self.endpoint}/health",
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                return await resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[cudaenv] health check failed: {e}")
            return None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


def check_health(endpoint: str | None = None) -> dict[str, Any]:
    """同步探活 (启动脚本用): 要求 repo 模式可用且至少一个活节点带 workspace 支持。"""
    import json
    import urllib.request

    ep = (endpoint or os.environ.get(ENDPOINT_ENV) or os.environ.get("EVAL_ENDPOINT")
          or ENDPOINT_DEFAULT).rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{ep}/health", timeout=15) as r:
        d = json.load(r)
    nodes = [n for n in (d.get("nodes") or []) if n.get("alive")]
    ws = max([int(n.get("ws_version") or 0) for n in nodes] or [int(d.get("ws_version") or 0)])
    if not d.get("repo_version"):
        raise RuntimeError(f"{ep} 不支持 /repo/evaluate (health 无 repo_version)")
    if ws < 1:
        raise RuntimeError(f"{ep} 没有带 workspace 支持的活节点")
    return {"endpoint": ep, "repo_version": d.get("repo_version"), "ws_version": ws,
            "nodes_alive": len(nodes) or None, "workers_alive": d.get("workers_alive"),
            "queued": (d.get("queue_status") or {}).get("pending")}
