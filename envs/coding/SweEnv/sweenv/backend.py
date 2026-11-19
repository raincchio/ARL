"""Async transport to an owned Docker workspace and the official SWE-bench grader."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shlex
import tempfile
import uuid
import urllib.error
import urllib.request
import time
import threading

logger = logging.getLogger(__name__)


class BackendError(RuntimeError):
    pass


@dataclass
class BackendConfig:
    endpoint: str = ""                 # persistent host service; takes precedence over SSH
    token_file: str = ""               # secret stays out of training configuration/logs
    ssh_host: str = ""                 # empty = local Docker host
    ssh_port: int = 22
    python: str = "python3"             # Python with swebench + docker installed on that host
    worker_script: str = "/root/workspace/SweContainerService/swe_service/worker.py"  # host-owned legacy stdio entry
    work_dir: str = "/tmp/arl-swe-runs"
    proxy: str = ""
    memory: str = "4g"
    cpus: float = 2.0
    max_concurrency: int = 8           # active trajectories, not individual commands
    admission_timeout: int = 14400     # host-wide backpressure can exceed a single trajectory


class DockerBackend:
    def __init__(self, cfg: BackendConfig, *, worker_path=None, kind="SWE"):
        if cfg.max_concurrency < 1 or cfg.cpus <= 0:
            raise ValueError("SWE max_concurrency and cpus must be positive")
        self.cfg = cfg
        self.kind = kind
        if cfg.endpoint and (kind != "SWE" or cfg.ssh_host):
            raise ValueError("SWE service endpoint cannot be combined with SSH or another worker kind")
        self._token = (Path(cfg.token_file).read_text().strip() if cfg.token_file else
                       os.environ.get("SWE_WORKER_TOKEN", "")) if cfg.endpoint else ""
        if cfg.endpoint and not self._token:
            raise ValueError("SWE service requires token_file or SWE_WORKER_TOKEN")
        # SWE server code belongs to the host service project. The optional
        # injected source remains only for other environment-specific workers.
        self._worker = Path(worker_path).read_text() if worker_path is not None else None
        self._slots = asyncio.Semaphore(cfg.max_concurrency)
        # Agent threads synchronously wait for workspace RPCs. Sharing their
        # default executor deadlocks when every worker is occupied by an agent.
        self._service_executor = (ThreadPoolExecutor(max_workers=64, thread_name_prefix="swe-http")
                                  if cfg.endpoint else None)
        self._ssh_dir = tempfile.TemporaryDirectory(prefix="arl-swe-ssh-") if cfg.ssh_host else None

    def _argv(self):
        command = ([self.cfg.python, "-c", self._worker] if self._worker is not None else
                   [self.cfg.python, self.cfg.worker_script])
        if not self.cfg.ssh_host:
            return command
        # Quote the complete remote argv; issue text/commands travel only on stdin.
        return ["ssh", "-p", str(self.cfg.ssh_port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
                "-o", "ControlMaster=auto", "-o", "ControlPersist=60",
                "-o", f"ControlPath={self._ssh_dir.name}/%C",
                "--", self.cfg.ssh_host, shlex.join(command)]

    async def rpc(self, action, *, timeout=180, **payload):
        if self.cfg.endpoint:
            cancelled = threading.Event()
            try:
                context = contextvars.copy_context()
                return await asyncio.get_running_loop().run_in_executor(
                    self._service_executor, context.run,
                    self._service_rpc, action, timeout, payload, cancelled)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        proc = await asyncio.create_subprocess_exec(
            *self._argv(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(json.dumps(
                {"action": action, **payload}).encode()), timeout=timeout)
            if proc.returncode:
                raise BackendError(f"{self.kind} worker exited {proc.returncode}: {err.decode(errors='replace')[-2000:]}")
            try:
                result = json.loads(out)
            except ValueError as e:
                raise BackendError(f"invalid {self.kind} worker response: {out[-500:]!r}") from e
            if result.get("infra_error"):
                raise BackendError(result.get("error", f"{self.kind} infrastructure failed"))
            return result
        except asyncio.TimeoutError as e:
            raise BackendError(f"{self.kind} {action} timed out after {timeout}s") from e
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.communicate()

    def _service_rpc(self, action, timeout, payload, cancelled):
        deadline = time.monotonic() + timeout
        body = json.dumps({"action": action, **payload}).encode()
        while True:
            if cancelled.is_set():
                raise BackendError("SWE service request cancelled")
            request = urllib.request.Request(self.cfg.endpoint.rstrip("/") + "/rpc", data=body,
                headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"})
            try:
                # Cluster traffic must not be routed through a download proxy.
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                        request, timeout=max(.1, deadline - time.monotonic())) as response:
                    result = json.load(response)
                if result.get("infra_error"):
                    raise BackendError(result.get("error", "SWE service failed"))
                return result
            except urllib.error.HTTPError as exc:
                # Only explicit admission failures are safe to retry. A dropped
                # response may mean an operation ran; never blindly repeat it.
                if exc.code == 429 and time.monotonic() + 1 < deadline:
                    # Source-IP limits can also reject exec/patch before they
                    # run. Only an explicit 429 is replayed; transport failures
                    # still have an unknown outcome and must not be retried.
                    exc.close()
                    cancelled.wait(1)
                    continue
                raise BackendError(f"SWE service HTTP {exc.code}: {exc.read(2000).decode(errors='replace')}") from exc
            except (OSError, ValueError) as exc:
                raise BackendError(f"SWE service {action}: {exc}") from exc

    async def health(self):
        return await self.rpc("health", timeout=30)

    async def load_tasks(self, path, limit=0):
        return (await self.rpc("tasks", path=path, limit=limit))["tasks"]

    def workspace(self, task):
        return DockerWorkspace(self, task)

    async def close(self):
        if self._service_executor is not None:
            self._service_executor.shutdown(wait=False, cancel_futures=True)
        if self._ssh_dir is not None:
            for socket in Path(self._ssh_dir.name).iterdir():
                if not socket.is_socket():
                    continue
                proc = await asyncio.create_subprocess_exec(
                    "ssh", "-S", str(socket), "-O", "exit", "--", self.cfg.ssh_host,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            self._ssh_dir.cleanup()
            self._ssh_dir = None


class DockerWorkspace:
    def __init__(self, backend, task):
        self.backend, self.task = backend, task
        self.name = "arl-swe-" + uuid.uuid4().hex
        self._started = self._slot = False
        self._closed = False
        self.placement = {}

    async def start(self):
        if self._closed:
            raise BackendError("workspace is closed")
        if self._started:
            return
        await self.backend._slots.acquire()
        self._slot = True
        # Set before issuing start so cancellation/failed transport also attempts cleanup.
        self._started = True
        try:
            self.placement = await self.backend.rpc("start", timeout=(self.backend.cfg.admission_timeout
                if self.backend.cfg.endpoint else 180), name=self.name, task={
                "instance_id": self.task.instance_id, "base_commit": self.task.base_commit},
                memory=self.backend.cfg.memory, cpus=self.backend.cfg.cpus)
        except BaseException:
            await self.close()
            raise

    async def execute(self, command, timeout=120, output_limit=16000, *, environment=None, interpreter=None):
        await self.start()
        return await self.backend.rpc("exec", name=self.name, command=command,
                                      timeout=timeout + 60, command_timeout=timeout, output_limit=output_limit,
                                      environment=environment, interpreter=interpreter)

    async def patch(self):
        if not self._started:
            return {"patch": ""}
        return await self.backend.rpc("patch", name=self.name, base_commit=self.task.base_commit)

    async def grade(self, patch, timeout):
        return await self.backend.rpc("grade", task=self.task.record, patch=patch,
            work_dir=self.backend.cfg.work_dir, proxy=self.backend.cfg.proxy,
            timeout=timeout + 660, grade_timeout=timeout, run_id=self.name)

    async def close(self):
        if self._closed:
            return
        try:
            if self._started:
                await self.backend.rpc("close", name=self.name, instance_id=self.task.instance_id)
            self._closed = True
        finally:
            if self._slot:
                self._slot = False
                self.backend._slots.release()
