"""本机按卡的租约: 同一台机器上的几个服务 (④ 的异步评估 eval_service、CudaMaster) 共用的抢卡协调。

为什么要有它: 两边各自看 nvidia-smi 判断卡空不空, 但 vLLM 从起进程到把显存吃满要几十秒到几分钟,
这段时间卡上只有一个 CUDA context —— 另一边看到的是一张空卡, 起上去, 后起的 OOM; 更糟的是
CudaMaster 作业结束时按卡清场 (gpu.reclaim), 会把撞上来的评估整组杀掉。先看后起天然有空档,
只能靠一把两边都认的锁。

做法: 一张卡一个 UUID.lock 文件, 对它 flock(LOCK_EX)。默认使用
/root/share/arm_async_eval/v1/gpu_leases; 无共享目录时必须显式配置 GPU_LEASE_DIR。
GPU_LEASE_DIR 可覆盖; 共享文件系统必须先验证 flock。锁文件不得删除或替换。

  * 要么全拿, 要么全不拿: 逐张非阻塞地试, 有一张拿不到就把已拿到的全放掉 —— 不会死锁。
  * 进程死了 (含 kill -9) 内核自动放锁, 没有陈旧锁要清。把 :attr:`Lease.fds` 经 ``pass_fds``
    传给干活的子进程, 锁就跟着子进程活: 调度进程挂了, 子进程还在卡上, 锁也还在。
  * 文件内容 = 持有者自报的 ``{"owner", "job", "pid", "since"}``, 只有文件**被锁着**时才作数;
    :func:`holders` 读它, 拿来在 status / 看板上显示 "这张卡被谁占着、占了多久"。
  * 它只管合作的服务之间; 不走这把锁的人 (手动起的训练) 仍要靠各自的 nvidia-smi 检查挡。
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import socket
import subprocess
from functools import lru_cache
from pathlib import Path


def process_scope() -> str:
    """Only processes in the same host boot and PID namespace may address each other by PID."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return f"{boot}:{os.readlink('/proc/self/ns/pid')}:{os.getuid()}"


def lease_dir() -> Path:
    configured = os.environ.get("GPU_LEASE_DIR")
    if configured:
        return Path(configured)
    if not Path("/root/share").is_dir():
        raise RuntimeError("/root/share unavailable; mount the shared directory or explicitly set GPU_LEASE_DIR")
    return Path("/root/share/arm_async_eval/v1/gpu_leases")


@lru_cache(maxsize=1)
def device_uuids() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15)
    devices = dict(tuple(x.strip() for x in row.split(",", 1))
                   for row in result.stdout.splitlines() if row.strip())
    if not devices or any(not u.startswith("GPU-") or "/" in u for u in devices.values()):
        raise RuntimeError("cannot resolve physical GPU UUIDs; refusing unlocked scheduling")
    return devices


def _path(card: str) -> Path:
    return lease_dir() / (device_uuids()[str(card)] + ".lock")


def _open(card: str) -> int:
    d = lease_dir()
    # Set permissions explicitly: the caller's umask must not exclude other users.
    d.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(PermissionError):
        os.chmod(d, 0o1777)
    fd = os.open(_path(card), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if os.fstat(fd).st_uid == os.geteuid():
            os.fchmod(fd, 0o666)
        os.set_inheritable(fd, False)
    except BaseException:
        os.close(fd)
        raise
    return fd


class Lease:
    """一组卡的租约。:meth:`release` 之前 (以及 fds 传下去的子进程都退出之前) 卡归持有者。"""

    def __init__(self, cards: list[str], fds: list[int], owner: str, job: str):
        self.cards, self.fds, self.owner, self.job = cards, fds, owner, job

    def release(self) -> None:
        fds, self.fds = self.fds, []
        for fd in fds:
            with contextlib.suppress(OSError):
                os.close(fd)              # 关 fd 即放锁 (子进程手里的副本关掉之前锁仍在)

    def __repr__(self) -> str:
        return f"Lease({','.join(self.cards)} {self.owner}:{self.job})"


def try_acquire(cards: list[str], owner: str, job: str) -> Lease | None:
    """非阻塞地拿下 ``cards`` 的全部租约; 有一张被别人拿着就一张不留, 返回 None。"""
    fds: list[int] = []
    try:
        for c in dict.fromkeys(cards):
            fd = _open(c)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                for f in fds:
                    os.close(f)
                return None
            fds.append(fd)
    except BaseException:
        for f in fds:
            os.close(f)
        raise
    try:
        info = json.dumps({"owner": owner, "job": job, "pid": os.getpid(), "cards": list(cards),
                           "scope": process_scope(), "container": socket.gethostname(),
                           "since": time.time()}).encode()
        for fd in fds:
            os.ftruncate(fd, 0)
            os.pwrite(fd, info, 0)
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
    return Lease(list(cards), fds, owner, job)


def holders(cards: list[str] | None = None) -> dict[str, dict]:
    """此刻被租着的卡: ``{卡号: {"owner", "job", "pid", "since"}}``。``cards`` 缺省 = 目录里所有的。

    探测方式是试着拿一下锁 (拿到立刻放), 所以和别人的 :func:`try_acquire` 撞在同一瞬间时对方
    可能失败一次 —— 双方都是轮询, 下一轮就过去了。
    """
    d = lease_dir()
    if cards is None:
        cards = list(device_uuids()) if d.is_dir() else []
    out: dict[str, dict] = {}
    for c in cards:
        if not _path(c).exists():
            continue
        fd = _open(c)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                try:
                    info = json.loads(os.pread(fd, 4096, 0) or b"{}")
                except ValueError:
                    info = {}             # 持有者正写到一半
                out[str(c)] = {"owner": info.get("owner", "?"), "job": info.get("job", "?"),
                               "pid": info.get("pid"), "since": info.get("since"),
                               "scope": info.get("scope"), "container": info.get("container")}
                continue
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    return out


def describe(info: dict, now: float | None = None) -> str:
    """一条持有者信息的人话: ``cudamaster teamA-0922 (12m)``。"""
    since = info.get("since")
    age = ""
    if since:
        s = max((now or time.time()) - since, 0)
        age = f" ({s / 3600:.1f}h)" if s >= 3600 else f" ({s / 60:.0f}m)"
    return f"{info.get('owner', '?')}@{info.get('container', '?')} {info.get('job', '?')}{age}"


def summarize(held: dict[str, dict], now: float | None = None) -> str:
    """``{卡: info}`` 按持有者归并成一行: ``GPU0-3: cudamaster teamA (12m); GPU8,9: ...``。"""
    groups: dict[tuple, list[str]] = {}
    for c, info in held.items():
        groups.setdefault((info.get("owner"), info.get("job"), info.get("pid"), info.get("scope")), []).append(c)
    parts = []
    for key, cs in groups.items():
        cs = sorted(cs, key=int)
        ids = [int(c) for c in cs]
        span = f"{cs[0]}-{cs[-1]}" if len(cs) > 1 and ids == list(range(ids[0], ids[-1] + 1)) else ",".join(cs)
        parts.append(f"GPU{span}: {describe(held[cs[0]], now)}")
    return "; ".join(parts)
