# Adapted from ARM rlhf/ppo/eval_service.py; GPU queue/leases/locks/lifecycle retained.
#!/usr/bin/env python3
"""ARM-derived asynchronous SWE evaluation daemon.

Run on the evaluation machine through script/eval_swe10_async.sh.
Retains ARM queue/globs, GPU UUID leases, process-group cleanup, retries,
heartbeat locks, progress, redo/archive and status. Reads frozen SWE10 specs
and train/checkpoints.jsonl instead of KernelBench PPO configuration.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import gzip
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener
from dataclasses import dataclass, field
from pathlib import Path

from arl.utils import gpu_lease
from arl.experiments import eval_defaults

LEASE_OWNER = "arl-eval"     # 本机卡租约里报的名字 (CudaMaster 那边是 "cudamaster")
LOCK_STALE_S = 15 * 60       # 锁超过这么久没刷新 = 持有者已死
PROBE_TIMEOUT_S = 120
REPORT_EVERY_S = 600         # 排队状态多久往日志里报一次


def machine_ip() -> str:
    """本机第一个非回环 IPv4 (和 script/_lib.sh 的 machine_ip 同口径: hostname -I 的第一个,
    拿不到用默认路由的源地址)。机器身份用 IP 不用 hostname: 集群里的容器 hostname 全是同一个
    (rgt_xing), 按 hostname 分的状态目录在共享盘上会被所有机器共用 —— 别的机器 add 时看到
    daemon.lock 被占, 以为本机服务在跑, 结果只入队不起服务。"""
    with contextlib.suppress(Exception):
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=10).stdout.split()
        if out:
            return out[0]
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))   # UDP connect 不发包, 只是让内核选源地址
        return s.getsockname()[0]
    raise SystemExit("cannot determine this machine's IP (hostname -I / default route both failed); "
                     "set ARL_EVAL_SERVICE_DIR to pick a state dir by hand")


HOST = machine_ip()          # 本机身份: 状态目录名、running 锁里的 host 字段、日志里的 host=
REPO = Path(__file__).resolve().parents[2]
# ARL_EVAL_SERVICE_DIR: 换一个状态目录 (测试用, 免得碰本机真服务的队列)
SCOPE = gpu_lease.process_scope()
INSTANCE = hashlib.sha256(SCOPE.encode()).hexdigest()[:16]
LEASE_OWNER = f"arl-eval:{INSTANCE}"
STATE = Path(os.environ.get("ARL_EVAL_SERVICE_DIR") or REPO / "logs" / "eval_service" / HOST / INSTANCE)
# 老版本按 hostname 分的状态目录 (所有机器共用一份); 里面还有活的守护进程时 status 提醒一句
LEGACY_STATE = REPO / "logs" / "eval_service" / socket.gethostname()


def log(msg: str) -> None:
    try:
        print(f">>> [{time.strftime('%F %T')}] {msg}", flush=True)
    except (BrokenPipeError, OSError):
        pass


def _atomic_write(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(p)


def _fmt_dur(s: float | None) -> str:
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# ── 队列 ─────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _queue_locked():
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "queue.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        yield


def read_queue() -> list[str]:
    try:
        return [x.strip() for x in (STATE / "queue").read_text().splitlines() if x.strip()]
    except FileNotFoundError:
        return []


def _write_queue(lines: list[str]) -> None:
    _atomic_write(STATE / "queue", "".join(x + "\n" for x in lines))


def expand_queue(lines: list[str]) -> list[Path]:
    """队列 -> run 目录 (去重, 保持顺序); 通配符当场展开, ``!`` 行排除。"""
    excl = {str(Path(x[1:]).resolve()) for x in lines if x.startswith("!")}
    out: dict[Path, None] = {}
    for x in lines:
        if x.startswith("!"):
            continue
        hits = sorted(glob.glob(x)) if glob.has_magic(x) else [x]
        for h in hits:
            p = Path(h).resolve()
            if p.is_dir() and str(p) not in excl:
                out[p] = None
    return list(out)


def _abs(x: str) -> str:
    return str(Path(x).resolve()) if not glob.has_magic(x) else os.path.abspath(x)


def queue_add(items: list[str]) -> list[str]:
    msgs = []
    with _queue_locked():
        q = read_queue()
        for it in map(_abs, items):
            if f"!{it}" in q:
                q.remove(f"!{it}"); msgs.append(f"un-excluded {it}")
            if it in q or (not glob.has_magic(it) and Path(it) in expand_queue(q)):
                msgs.append(f"already queued: {it}")
                continue
            if not glob.has_magic(it) and not Path(it).is_dir():
                msgs.append(f"!!! not a directory, not added: {it}")
                continue
            q.append(it); msgs.append(f"queued {it}")
        _write_queue(q)
    return msgs


def queue_rm(items: list[str]) -> list[str]:
    """按 队列里那一行 / run 目录 / run 名字 移除; 被通配符罩住的 run 记一条 ``!目录`` 排除。"""
    msgs = []
    with _queue_locked():
        q = read_queue()
        runs = expand_queue(q)
        for it in items:
            a = _abs(it)
            if a in q:
                q.remove(a); msgs.append(f"removed {a}")
                continue
            hit = [p for p in runs if str(p) == a or p.name == it]
            if not hit:
                msgs.append(f"!!! not in the queue: {it}")
                continue
            for p in hit:
                if str(p) in q:
                    q.remove(str(p))
                if p in expand_queue(q):         # 还被某个通配符罩着
                    q.append(f"!{p}")
                msgs.append(f"removed {p}")
        _write_queue(q)
    return msgs


# ── 卡 ────────────────────────────────────────────────────────────────────────

def _smi(query: str) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def build_pools(gpus: str) -> list[list[str]]:
    """auto = 0-7 / 8-15 两组 (本机有完整 8 张的组才算; 一组都凑不出来时退回全部卡一个池);
    all = 全部卡一个池; 否则 = 给定的卡号列表一个池。"""
    every = [r[0] for r in _smi("index")]
    if gpus == "auto":
        pools = [[str(i) for i in range(b, b + 8)] for b in (0, 8)]
        pools = [p for p in pools if all(g in every for g in p)]
        return pools or [every]
    if gpus == "all":
        return [every]
    return [[g.strip() for g in gpus.split(",") if g.strip()]]


def free_devices(util: float) -> set[str]:
    """空余显存 >= util x 总显存 + 2GiB (vLLM 起得来的条件) 且上面没有计算进程的卡。

    进程那条挡的是 "刚起、显存还没吃上来" 的别人 —— 只看显存, 一个正在加载权重的 vLLM 看着
    就是张空卡。"""
    rows = _smi("index,uuid,memory.free,memory.total")
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    taken = {line.strip() for line in out.splitlines() if line.strip()}
    return {i for i, u, f, t in rows if u not in taken and float(f) >= util * float(t) + 2048}


def probe(devs: list[str]) -> list[str]:
    """每张卡单独起一个进程试一次 (CUDA context + 小运算), 返回试不过的卡。umd 的报错有时
    只打在 stderr 而进程照样退 0, 所以也看输出里的关键字。"""
    code = ("import torch; x = torch.ones(1 << 20, device='cuda'); torch.cuda.synchronize(); "
            "assert (x * 2).sum().item() == 2 * (1 << 20)")
    procs = {g: subprocess.Popen([sys.executable, "-c", code], env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for g in devs}
    bad = []
    for g, p in procs.items():
        try:
            out, _ = p.communicate(timeout=PROBE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill(); out = "probe timed out"
        low = out.lower()
        if p.returncode != 0 or any(k in low for k in ("umd error", "unreliable state", "reset operation")):
            bad.append(g)
            tail = "\n".join("      " + x for x in out.strip().splitlines()[-3:])
            log(f"!!! device {g} failed the probe:\n{tail}")
    return bad


# ── run ──────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    path: Path
    spec: dict
    every: int
    at_start: bool
    policy: str
    dp: int
    dp_src: str
    giveup_streak: int = 0
    stopped: str = ""                       # 非空 = 停下的原因
    finished: bool = False
    unreachable: bool = False               # 评测服务连不上 (只在状态变化时打日志)
    state: str = "idle"                     # status 里显示的一句话
    pending_since: dict = field(default_factory=dict)   # name -> 第一次发现它待评的时间

    @property
    def out(self) -> Path:
        return self.path / "async_eval"


def load_run(path: Path, sampler_dp: int | None, ngpus: int, sampler_tp: int | None = None) -> Run | str:
    spec_p = path / "async_eval/spec.json"
    try:
        spec = json.loads(spec_p.read_text())
        if spec.get('schema') != 'arl.swe10.async_eval.v1':
            return 'not an ARL SWE10 asynchronous evaluation spec'
        policy = spec['baseline']['models']['policy']['name']
        every = int(spec['eval_every'])
        if every <= 0:
            return 'eval_every must be positive'
        if spec['baseline_requested'] and not (Path(policy)/'config.json').is_file():
            return f'base model is not visible on evaluator: {policy}'
        if not all((path/name).is_file() for name in
                   ('tasks.private.jsonl',spec.get('evaluation_tasks_file','tasks.private.jsonl'))):
            return 'shared frozen task file is not visible on evaluator'
    except (OSError,ValueError,KeyError) as exc:
        return f'cannot load {spec_p}: {exc}'
    if sampler_dp is not None:
        dp,src=sampler_dp,'command line'
    else:
        tp=sampler_tp if sampler_tp is not None else spec.get('sampler_tp',eval_defaults.SAMPLER_TP)
        src='command line TP' if sampler_tp is not None else ('spec sampler_tp' if 'sampler_tp' in spec else 'default sampler_tp')
        if isinstance(tp,bool) or not isinstance(tp,int) or tp < 1 or ngpus % tp:
            return f'sampler_tp={tp} must be positive and divide NGPUS={ngpus}'
        dp=ngpus//tp
    if dp < 1 or ngpus % dp:
        return f'SAMPLER_DP={dp} does not divide NGPUS={ngpus}'
    return Run(path=path,spec=spec,every=every,at_start=bool(spec['baseline_requested']),
               policy=policy,dp=dp,dp_src=src)


def gave_up(run: Run, name: str, retries: int) -> bool:
    try:
        return int((run.out / "failed" / name).read_text().strip() or 0) >= retries
    except (FileNotFoundError, ValueError):
        return False


def _done_set(run: Run) -> set[str]:
    d = run.out / "done"
    return set(os.listdir(d)) if d.is_dir() else set()


def _checkpoints(run: Run) -> dict[str, dict]:
    # ARM publishes each index line only after the checkpoint write completes.
    path=run.path/'train/checkpoints.jsonl'
    try: text=path.read_text()
    except FileNotFoundError: return {}
    lines=text.splitlines(keepends=True)
    return {r['name']:r for line in lines if line.endswith('\n') and line.strip()
            for r in [json.loads(line)]}


def eval_points(run: Run) -> list[str]:
    recs=_checkpoints(run)
    points=['base'] if run.at_start else []
    # Like ARM, evaluate the final batch once. Prefer the final policy name so
    # the SWE before/after comparison always points at the final checkpoint.
    final_batch=int(run.spec['expected_batches'])
    for name,record in sorted(recs.items(),key=lambda kv:int(kv[1]['batch'])):
        if name!='final' and int(record['batch'])<final_batch and int(record['batch'])%run.every==0:
            points.append(name)
    if 'final' in recs: points.append('final')
    return points


def next_ckpt(run: Run, retries: int) -> str | None:
    """Return the next runnable checkpoint, or FINISHED when all points are done."""
    done = _done_set(run)
    pts = eval_points(run)
    for n in pts:
        if n not in done and not gave_up(run, n, retries):
            return n
    return "FINISHED" if "final" in pts and all(n in done for n in pts) else None


# ── 锁 ───────────────────────────────────────────────────────────────────────

def lock_path(run: Run, name: str) -> Path:
    return run.out / "running" / name


def _lock_dead(p: Path) -> str:
    """锁是否已失效: 超过 LOCK_STALE_S 没心跳, 或持有者就在本机且进程已经不在了。
    失效返回原因, 否则 ""。"""
    age = time.time() - p.stat().st_mtime
    if age >= LOCK_STALE_S:
        return f"{age / 60:.0f} min without heartbeat"
    try:
        owner = json.loads(p.read_text())
        if owner.get("scope") == SCOPE and owner.get("pid") != os.getpid():
            os.kill(int(owner["pid"]), 0)
    except ProcessLookupError:
        return "its service process on this host is gone"
    except Exception:  # noqa: BLE001  (锁内容坏了 / 没权限探: 按 mtime 判)
        pass
    return ""


def try_lock(run: Run, name: str) -> bool:
    p = lock_path(run, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        why = _lock_dead(p)
        if not why:
            return False
        log(f"{run.path.name}/{name}: removing stale lock ({why}): {p.read_text().strip()}")
        p.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"host": HOST, "scope": SCOPE, "pid": os.getpid(), "start": time.strftime("%F %T")}) + "\n")
    return True


def locked_by_other(run: Run, name: str) -> str:
    """别人持有的有效锁的内容; 没有 (或已失效) 返回 ""。"""
    p = lock_path(run, name)
    try:
        if not _lock_dead(p):
            return p.read_text().strip()
    except FileNotFoundError:
        pass
    return ""


# ── 进度 / 耗时 ──────────────────────────────────────────────────────────────

def read_progress(run: Run, name: str) -> dict:
    try:
        return json.loads((run.out / "progress" / f"{name}.json").read_text())
    except (FileNotFoundError, ValueError):
        return {}


def past_durations(run: Run) -> list[float]:
    out = []
    d = run.out / "done"
    for f in (d.iterdir() if d.is_dir() else []):
        try:
            t = json.loads(f.read_text()).get("time/eval")
            if t:
                out.append(float(t))
        except (ValueError, OSError):
            pass
    return out


def last_metrics(run: Run) -> dict:
    try:
        with open(run.out / "metrics.jsonl") as f:
            lines = [x for x in f if x.strip()]
        return json.loads(lines[-1]) if lines else {}
    except (FileNotFoundError, ValueError):
        return {}


# ── 重评 ─────────────────────────────────────────────────────────────────────

def resolve_run(it: str) -> Path:
    """redo 的 run 参数: 目录, 或 run 名字 (先在本机队列里找, 再找 <仓库>/runs/<名字>)。"""
    a = Path(_abs(it))
    if a.is_dir():
        return a
    hit = [p for p in expand_queue(read_queue()) if p.name == it]
    if len(hit) == 1:
        return hit[0]
    p = REPO / "runs" / it
    if p.is_dir():
        return p.resolve()
    raise SystemExit(f"redo: run not found: {it}" + (f" (ambiguous in the queue: {hit})" if hit else ""))


def redo(items: list[str]) -> list[str]:
    if len(items)<2: raise SystemExit('redo: RUN base|CHECKPOINT|final|all')
    # Archiving needs no GPU allocation; leave the actual TP/DP layout to the daemon.
    path=resolve_run(items[0]);run=load_run(path,1,1)
    if isinstance(run,str): raise SystemExit(run)
    points=eval_points(run)
    names=points if items[1:]==['all'] else [n.zfill(6) if n.isdigit() else n for n in items[1:]]
    if any(n not in points for n in names): raise SystemExit('unknown evaluation checkpoint')
    running=run.out/'running'
    if running.exists() and any(locked_by_other(run,p.name) for p in running.iterdir()):
        raise SystemExit('run is being evaluated; rm it from the queue first')
    archive=run.out/'redo'/str(time.time_ns());archive.mkdir(parents=True)
    metrics=run.out/'metrics.jsonl'
    if metrics.exists():
        rows=[json.loads(line) for line in metrics.read_text().splitlines() if line]
        (archive/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['checkpoint'] in names))
        _atomic_write(metrics,''.join(json.dumps(r)+'\n' for r in rows if r['checkpoint'] not in names))
    for name in names:
        for sub,file in (('done',name),('failed',name),('progress',name+'.json'),('results',name)):
            src=run.out/sub/file
            if src.exists():
                (archive/sub).mkdir(exist_ok=True);src.rename(archive/sub/file)
    if (path/'comparison.json').exists(): (path/'comparison.json').rename(archive/'comparison.json')
    return [f'archived {names} -> {archive}']


# ── 守护进程 ─────────────────────────────────────────────────────────────────

@dataclass
class Job:
    run: Run
    name: str
    devs: list[str]
    proc: subprocess.Popen
    log_file: Path
    t0: float
    lease: gpu_lease.Lease


def eval_reachable(endpoint: str) -> bool:
    if not endpoint:
        return True  # SSH and local SWE backends are checked by the evaluator.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f"{endpoint.rstrip('/')}/healthz", timeout=5):
            return True
    except HTTPError as exc:
        exc.close()
        return exc.code in (401, 403, 404, 405)  # Leave older masters to the child preflight.
    except OSError:
        return False


def child_env(run: Run, devs: list[str], util: float) -> dict:
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')
         and k not in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','MASTER_ADDR','MASTER_PORT')}
    env.update(PYTHONPATH=str(REPO), HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUDA_VISIBLE_DEVICES=','.join(devs),KERNEL_RL_SAMPLER_GPUS=','.join(devs),
        KERNEL_RL_SAMPLER_DP=str(run.dp),KERNEL_RL_SAMPLING_GPU_UTIL=str(util),
        KERNEL_RL_COLOCATE='0',NO_PROXY='127.0.0.1',no_proxy='127.0.0.1')
    # Match the training sampler's transport defaults on the PPU fabric.
    env.setdefault('NCCL_SHM_DISABLE', '1')
    env.setdefault('NCCL_P2P_LEVEL', 'SYS')
    return env


def _group_alive(pgid: int) -> bool:
    """Linux: zombies cannot hold GPU contexts and must not delay cleanup."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm may contain spaces or parentheses; fields after it start at state.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
            return True
    return False


def _kill_group(pgid: int, grace: float = 30) -> None:
    """Wait for the whole evaluation group, even after its leader has exited."""
    if pgid <= 1 or pgid == os.getpgrp():
        raise ValueError(f"refusing to kill unsafe process group {pgid}")
    if not _group_alive(pgid):
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while _group_alive(pgid):
        if time.monotonic() >= deadline:
            log(f"evaluation process group {pgid} survived SIGTERM; sending SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 10
            while _group_alive(pgid):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"evaluation process group {pgid} still alive after SIGKILL")
                time.sleep(0.1)
            return
        time.sleep(0.1)


def _kill(proc: subprocess.Popen, grace: float = 30) -> None:
    """Keep leases/job records until all live members of the session are gone."""
    _kill_group(proc.pid, grace)
    proc.wait()


class Daemon:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.pools = build_pools(a.gpus)
        for p in self.pools:
            if a.ngpus > len(p):
                raise SystemExit(f"NGPUS={a.ngpus} > the {len(p)} devices in pool {','.join(p)}")
        self.bad: set[str] = set()
        self.runs: dict[Path, Run] = {}
        self.skipped: dict[Path, str] = {}
        self.jobs: list[Job] = []
        self.wait_why = ""
        self.last_report = 0.0
        self.t_start = time.time()

    # 队列 -> self.runs
    def sync_queue(self) -> None:
        with _queue_locked():
            want = expand_queue(read_queue())
        for p in list(self.runs):
            if p not in want:
                self.drop(self.runs.pop(p), "removed from the queue")
            elif not p.is_dir():
                self.drop(self.runs.pop(p), "run directory is gone")
        for p in want:
            if p in self.runs:
                continue
            r = load_run(p, self.a.sampler_dp, self.a.ngpus, getattr(self.a,'sampler_tp',None))
            if isinstance(r, str):
                if self.skipped.get(p) != r:
                    self.skipped[p] = r
                    log(f"!!! skip {p.name}: {r}")
                continue
            self.skipped.pop(p, None)
            (r.out / "done").mkdir(parents=True, exist_ok=True)
            failed = r.out / "failed"
            failed.mkdir(exist_ok=True)
            reset = sorted(x.name for x in failed.iterdir())
            for x in failed.iterdir():
                x.unlink()
            self.runs[p] = r
            log(f"+ run {p}  eval_every={r.every} eval_at_start={r.at_start}  SAMPLER_DP={r.dp} ({r.dp_src})"
                + (f"  retrying previously failed: {','.join(reset)}" if reset else ""))
        for p in list(self.skipped):
            if p not in want:
                self.skipped.pop(p)

    def drop(self, run: Run, why: str) -> None:
        for j in [j for j in self.jobs if j.run is run]:
            _kill(j.proc)
            j.lease.release()
            self.jobs.remove(j)
            self.save_jobs()
            lock_path(run, j.name).unlink(missing_ok=True)
            log(f"- {run.path.name}: {why}; aborted the running evaluation of {j.name} "
                f"(not counted as a failure; it is redone if the run is added back)")
            return
        log(f"- {run.path.name}: {why}")

    def save_jobs(self) -> None:
        _atomic_write(STATE / "jobs.json", json.dumps(
            [{"scope": SCOPE, "pgid": j.proc.pid, "run": str(j.run.path), "name": j.name} for j in self.jobs]))

    def busy_devs(self) -> set[str]:
        return {d for j in self.jobs for d in j.devs}

    def pick(self, job: str) -> gpu_lease.Lease | None:
        """在某个池里挑 NGPUS 张空卡并把租约拿下; 凑不出来返回 None, 原因 (含别的服务占着哪些卡)
        记在 self.wait_why。"""
        held = gpu_lease.holders()
        free = free_devices(self.a.util) - self.bad - self.busy_devs() - set(held)
        for p in self.pools:
            ok = [g for g in p if g in free]
            if len(ok) >= self.a.ngpus:
                lease = gpu_lease.try_acquire(ok[:self.a.ngpus], LEASE_OWNER, job)
                if lease is not None:
                    try:
                        if set(lease.cards) <= free_devices(self.a.util):
                            return lease
                    except BaseException:
                        lease.release()
                        raise
                    lease.release()
                # 看完 holders() 到这一下之间被别人租走了: 下一轮再挑
        others = {c: i for c, i in held.items() if i.get("owner") != LEASE_OWNER}
        self.wait_why = f"no free pool of {self.a.ngpus} devices" + (
            f"; held by: {gpu_lease.summarize(others)}" if others else "")
        return None

    def launch(self, run: Run, name: str, lease: gpu_lease.Lease) -> None:
        devs = lease.cards
        logs = run.out / "logs"
        logs.mkdir(exist_ok=True)
        lf = logs / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(lf, "w") as fh:
            fh.write(f">>> host={HOST} devices={','.join(devs)} SAMPLER_DP={run.dp} "
                     f"start={time.strftime('%F %T')}\n"); fh.flush()
            proc = subprocess.Popen([sys.executable, "-m", self.a.child_module, str(run.path), name],
                                    cwd=REPO, env=child_env(run, devs, self.a.util),
                                    stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                                    pass_fds=lease.fds)    # 租约跟着子进程: 守护进程挂了它也还在
        (run.out / "progress" / f"{name}.json").unlink(missing_ok=True)   # 上一次失败留下的
        self.jobs.append(Job(run, name, devs, proc, lf, time.time(), lease))
        self.save_jobs()
        run.pending_since.pop(name, None)
        log(f"evaluating {run.path.name}/{name} on devices {','.join(devs)} "
            f"({run.dp} x TP={self.a.ngpus // run.dp})  log -> {lf}")

    def reap(self) -> None:
        for j in list(self.jobs):
            rc = j.proc.poll()
            if rc is None:
                with contextlib.suppress(FileNotFoundError):
                    lock_path(j.run, j.name).touch(exist_ok=True)   # 心跳
                continue
            # os._exit in the evaluator can leave vLLM engines/workers alive.
            # Clean the group before releasing leases or deleting recovery records.
            _kill(j.proc)
            j.lease.release()
            self.jobs.remove(j)
            self.save_jobs()
            lock_path(j.run, j.name).unlink(missing_ok=True)
            dur = _fmt_dur(time.time() - j.t0)
            if rc == 0:
                j.run.giveup_streak = 0
                log(f"{j.run.path.name}/{j.name} done ({dur})")
                continue
            fp = j.run.out / "failed" / j.name
            try:
                n = int(fp.read_text().strip() or 0) + 1
            except (FileNotFoundError, ValueError):
                n = 1
            fp.write_text(str(n))
            log(f"!!! {j.run.path.name}/{j.name} failed after {dur} (rc={rc}, attempt {n}/{self.a.retries}; "
                f"see {j.log_file})")
            if n >= self.a.retries:
                j.run.giveup_streak += 1
                if j.run.giveup_streak >= 2:
                    j.run.stopped = "two checkpoints in a row gave up -- environment or weights likely broken"
                    log(f"!!! stopping {j.run.path.name}: {j.run.stopped}. Fix it, then `rm` + `add` the run "
                        f"(or restart the service) to retry.")
                else:
                    log(f"!!! giving up on {j.run.path.name}/{j.name}, moving on")

    def tick(self) -> None:
        self.reap()
        self.sync_queue()
        running = {j.run.path: j for j in self.jobs}
        queue = []
        for r in self.runs.values():
            if r.path in running:
                r.state = "running"
                continue
            if r.stopped:
                r.state = f"stopped: {r.stopped}"
                continue
            name = next_ckpt(r, self.a.retries)
            if name == "FINISHED":
                if not r.finished:
                    log(f"{r.path.name}: final checkpoint evaluated -- done. results: {r.out / 'metrics.jsonl'}")
                r.finished, r.state = True, "finished"
                continue
            r.finished = False
            if name is None:
                done = _done_set(r)
                failed = [p for p in eval_points(r) if p not in done and gave_up(r, p, self.a.retries)]
                if failed:
                    r.state = f"retry limit reached for {', '.join(failed)} (use redo to retry)"
                else:
                    r.state = "idle (no new checkpoint to evaluate yet)"
                continue
            other = locked_by_other(r, name)
            if other:
                r.state = f"{name} being evaluated by another service: {other}"
                continue
            r.state = f"waiting to evaluate {name}"
            queue.append((r.pending_since.setdefault(name, time.time()), r, name))
        queue.sort(key=lambda x: x[0])
        for _, r, name in queue:
            config = r.spec['baseline' if name == 'base' else 'evaluation']
            endpoint = config['evaluate'].get('options', {}).get('swe_endpoint') or ''
            if not eval_reachable(endpoint):
                if not r.unreachable:
                    log(f"!!! {r.path.name}: evaluation service unreachable ({endpoint}); waiting for it")
                r.unreachable, r.state = True, f"waiting to evaluate {name} (evaluation service unreachable)"
                continue
            if r.unreachable:
                log(f"{r.path.name}: evaluation service reachable again")
                r.unreachable = False
            lease = self.pick(f"{r.path.name}/{name}")
            if lease is None:
                r.state = f"waiting to evaluate {name} ({self.wait_why})"
                continue
            newbad = probe(lease.cards)
            if newbad:
                lease.release()
                self.bad |= set(newbad)
                log(f"!!! excluding device(s) {','.join(newbad)} for the lifetime of this service "
                    f"(excluded so far: {','.join(sorted(self.bad, key=int))}); picking again next round")
                break
            if not try_lock(r, name):
                lease.release()
                continue
            self.launch(r, name, lease)
            r.state = "running"
        now = time.time()
        waiting = [(t, r, n) for t, r, n in queue if r.state != "running"]
        if waiting and now - self.last_report >= REPORT_EVERY_S:
            self.last_report = now
            log("waiting: " + ", ".join(f"{r.path.name}/{n} ({_fmt_dur(now - t)})" for t, r, n in waiting)
                + "  | running: " + (", ".join(f"{j.run.path.name}/{j.name}" for j in self.jobs) or "-")
                + f"  | {self.wait_why}")
        self.write_status()

    def write_status(self) -> None:
        now = time.time()
        try:
            held = gpu_lease.holders()
        except OSError:
            held = {}
        try:
            free = free_devices(self.a.util) - self.bad - self.busy_devs() - set(held)
        except Exception:  # noqa: BLE001
            free = set()
        others = {c: i for c, i in held.items() if i.get("owner") != LEASE_OWNER}
        jobs = {j.run.path: j for j in self.jobs}
        runs = []
        for r in self.runs.values():
            pts, done = eval_points(r), _done_set(r)
            durs = past_durations(r)
            failed = {}
            for p in pts:
                with contextlib.suppress(FileNotFoundError):
                    failed[p] = (r.out / "failed" / p).read_text().strip()
            ent = {"run": str(r.path), "name": r.path.name, "state": r.state,
                   "done": sum(p in done for p in pts), "points": len(pts),
                   "failed": failed,
                   "avg_eval_s": sum(durs) / len(durs) if durs else None,
                   "last": {k: v for k, v in last_metrics(r).items()
                            if k in ("checkpoint", "step", "eval/resolved_rate",
                                     "eval/pass_any_rate", "time/eval")}}
            j = jobs.get(r.path)
            if j is not None:
                pr = read_progress(r, j.name)
                ent["job"] = {"checkpoint": j.name, "devices": f"{j.devs[0]}-{j.devs[-1]}",
                              "elapsed_s": now - j.t0, "log": str(j.log_file),
                              "trajectories_done": pr.get("trajectories_done"),
                              "trajectories_total": pr.get("trajectories_total"),
                              "turns_done": pr.get("turns_done"), "turns_total": pr.get("turns_total")}
            runs.append(ent)
        st = {"host": HOST, "pid": os.getpid(), "updated": now, "uptime_s": now - self.t_start,
              "pools": [{"devices": f"{p[0]}-{p[-1]}", "free": sum(g in free for g in p), "size": len(p),
                         "busy": next((f"{j.run.path.name}/{j.name}" for j in self.jobs if j.devs[0] in p), "")}
                        for p in self.pools],
              "held_by_others": {c: gpu_lease.describe(i, now) for c, i in others.items()},
              "held_by_others_summary": gpu_lease.summarize(others, now),
              "bad_devices": sorted(self.bad, key=int), "runs": runs,
              "skipped": {str(p): why for p, why in self.skipped.items()}}
        _atomic_write(STATE / "status.json", json.dumps(st, indent=1, ensure_ascii=False))

    def shutdown(self, *_):
        for j in self.jobs:          # 先杀子进程再打日志 (日志写不出去也要把卡腾出来)
            _kill(j.proc)
            j.lease.release()
            lock_path(j.run, j.name).unlink(missing_ok=True)
        self.jobs = []
        self.save_jobs()
        log("stopped: running evaluations were terminated (they are redone when the service starts again)")
        with contextlib.suppress(FileNotFoundError):
            st = json.loads((STATE / "status.json").read_text())
            st["stopped"] = time.time()
            _atomic_write(STATE / "status.json", json.dumps(st, indent=1, ensure_ascii=False))
        os._exit(0)

    def reap_orphans(self) -> None:
        """上一代守护进程被 kill -9 时留下的评估子进程: 整组杀掉, 锁一并清掉 (那个点会重评)。"""
        try:
            prev = json.loads((STATE / "jobs.json").read_text())
        except (FileNotFoundError, ValueError):
            return
        for e in prev:
            if e.get("scope") != SCOPE:
                log("ignoring orphan record from another PID namespace")
                continue
            _kill_group(int(e["pgid"]))
            (Path(e["run"]) / "async_eval" / "running" / e["name"]).unlink(missing_ok=True)
            log(f"killed an orphaned evaluation left by the previous service: {Path(e['run']).name}/{e['name']}")
        self.save_jobs()

    def serve(self) -> None:
        self.reap_orphans()
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        log(f"service up: host={HOST} pid={os.getpid()}  pools: {' | '.join(','.join(p) for p in self.pools)}  "
            f"NGPUS={self.a.ngpus}  poll={self.a.poll}s  queue: {STATE / 'queue'}")
        while True:
            self.tick()
            time.sleep(self.a.poll)


# ── 守护进程的启停 ───────────────────────────────────────────────────────────

def daemon_pid(state: Path = STATE) -> int | None:
    """守护进程在跑就返回 pid (以 daemon.lock 上的 flock 为准, pid 文件只是记录)。"""
    lk = state / "daemon.lock"
    if not lk.exists():
        return None
    with open(lk) as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                return int(json.loads((state / "daemon.json").read_text())["pid"])
            except Exception:  # noqa: BLE001
                return -1
        fcntl.flock(f, fcntl.LOCK_UN)
    return None


def daemon_opts(a: argparse.Namespace) -> list[str]:
    o = ["--gpus", a.gpus, "--ngpus", str(a.ngpus), "--util", str(a.util), "--poll", str(a.poll),
         "--retries", str(a.retries), "--child-module", a.child_module]
    if a.sampler_dp is not None:
        o += ["--sampler-dp", str(a.sampler_dp)]
    if getattr(a,'sampler_tp',None) is not None:
        o += ["--sampler-tp", str(a.sampler_tp)]
    return o


def start_daemon(a: argparse.Namespace) -> None:
    if note := legacy_note():
        raise SystemExit(note)
    pid = daemon_pid()
    if pid is not None:
        print(f"service already running on {HOST} (pid {pid}); its options stay as they were started "
              f"(`restart` to change them)")
        return
    STATE.mkdir(parents=True, exist_ok=True)
    logf = open(STATE / "daemon.log", "a")
    p = subprocess.Popen([sys.executable, "-u", "-m", "arl.experiments.eval_service", "_serve", *daemon_opts(a)],
                         cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True,
                         env={**os.environ, "PYTHONPATH": ":".join(filter(None, (str(REPO), os.environ.get("PYTHONPATH", ""))))})
    for _ in range(50):
        time.sleep(0.2)
        if daemon_pid() is not None:
            print(f"service started on {HOST} (pid {p.pid}); log: {STATE / 'daemon.log'}")
            return
        if p.poll() is not None:
            break
    print(f"!!! service failed to start; see {STATE / 'daemon.log'}:")
    print("".join(open(STATE / "daemon.log").readlines()[-15:]))
    sys.exit(1)


def stop_daemon() -> bool:
    pid = daemon_pid()
    if pid is None:
        print(f"service not running on {HOST}")
        return False
    meta = json.loads((STATE / "daemon.json").read_text())
    if pid <= 0 or meta.get("scope") != SCOPE:
        raise SystemExit("refusing to stop a service from another/unknown PID namespace; use its own container")
    os.kill(pid, signal.SIGTERM)
    for _ in range(600):
        if daemon_pid() is None:
            print(f"service stopped (pid {pid}); queue kept: {STATE / 'queue'}")
            return True
        time.sleep(0.2)
    print(f"!!! pid {pid} still alive after 120s")
    return False


def serve(a: argparse.Namespace) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    lk = open(STATE / "daemon.lock", "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another service is already running on {HOST}")
    _atomic_write(STATE / "daemon.json", json.dumps({"scope": SCOPE, "pid": os.getpid(), "argv": sys.argv[2:],
                                                    "start": time.strftime("%F %T")}))
    Daemon(a).serve()


# ── status ───────────────────────────────────────────────────────────────────

def legacy_note() -> str:
    """Do not adopt queues or send signals across unknown PID namespaces."""
    if os.environ.get("ARL_EVAL_SERVICE_DIR"):
        return ""
    notes = []
    for old in dict.fromkeys([REPO / "logs" / "eval_service" / HOST, LEGACY_STATE]):
        if old == STATE:
            continue
        pid = daemon_pid(old)
        if pid is not None:
            notes.append(f"note: legacy service pid={pid} at {old}; stop it from its original "
                         "container using the old version before starting the UUID lease protocol. "
                         "Its queue is not automatically migrated.")
    return "\n".join(notes)


def print_status() -> None:
    if note := legacy_note():
        print(note)
    pid = daemon_pid()
    try:
        st = json.loads((STATE / "status.json").read_text())
    except (FileNotFoundError, ValueError):
        st = None
    q = read_queue()
    if pid is None:
        print(f"service: NOT running on {HOST}" + ("  (start: ./script/eval_swe10_async.sh start)" if q else ""))
    else:
        age = time.time() - st["updated"] if st else None
        print(f"service: running on {HOST}, pid {pid}, up {_fmt_dur(st and st['uptime_s'])}, "
              f"snapshot {_fmt_dur(age)} old")
    if not q:
        print("queue: empty  (add: ./script/eval_swe10_async.sh add <run>)")
        return
    print("queue: " + "  ".join(q))
    if st is None or pid is None:
        if st is None:
            return
        print("(last snapshot before the service stopped:)")
    print("pools: " + "   ".join(f"{p['devices']} {p['free']}/{p['size']} free"
                                 + (f" [{p['busy']}]" if p["busy"] else "") for p in st["pools"])
          + (f"   bad: {','.join(st['bad_devices'])}" if st["bad_devices"] else ""))
    if st.get("held_by_others_summary"):
        print(f"held by other services: {st['held_by_others_summary']}")
    for r in st["runs"]:
        last = r.get("last") or {}
        res = (f"  last {last.get('checkpoint')}: resolved {last.get('eval/resolved_rate', 0):.3f}"
               f" pass_any {last.get('eval/pass_any_rate', 0):.3f}") if last else ""
        print(f"\n  {r['name']}   {r['done']}/{r['points']} evaluated, avg {_fmt_dur(r['avg_eval_s'])}/ckpt{res}")
        j = r.get("job")
        if j:
            progress = dict(j)
            # Read live counts so status also works with an older running daemon.
            if pid:
                try:
                    progress.update(json.loads((Path(r['run']) / 'async_eval' / 'progress' /
                                                f"{j['checkpoint']}.json").read_text()))
                except (OSError, ValueError):
                    pass
            td, tt = progress.get("trajectories_done"), progress.get("trajectories_total")
            el = j["elapsed_s"] + (time.time() - st["updated"] if pid else 0)
            if td is not None and tt:
                frac = min(td / tt, 1.0)
                eta = el * (1 - frac) / frac if frac > 0.02 else (
                    max(r["avg_eval_s"] - el, 0) if r["avg_eval_s"] else None)
                prog = f"{td}/{tt} trajectories ({frac:.0%}), ~{_fmt_dur(eta)} left"
            else:
                prog = "starting engine" + (f", ~{_fmt_dur(max(r['avg_eval_s'] - el, 0))} left" if r["avg_eval_s"] else "")
            print(f"    ▶ {j['checkpoint']} on {j['devices']}, {_fmt_dur(el)} elapsed, {prog}")
            print(f"      log: {j['log']}")
        else:
            print(f"    {r['state']}")
        if r["failed"]:
            print("    failed attempts: " + ", ".join(f"{k}×{v}" for k, v in r["failed"].items()))
    for p, why in st.get("skipped", {}).items():
        print(f"\n  {Path(p).name}   skipped: {why}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["add", "rm", "redo", "status", "log", "start", "stop", "restart", "_serve"])
    ap.add_argument("items", nargs="*", help="add / rm: run 目录, run 名字 (rm), 或带引号的通配符; "
                    "redo: run 目录或名字, 后面跟 checkpoint 名 / batch 号 / base / final / all")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--ngpus", type=int, default=eval_defaults.GPUS_PER_JOB)
    ap.add_argument("--sampler-dp", type=int, default=None)
    ap.add_argument("--sampler-tp", type=int, default=None, help="GPUs per engine; defaults to run spec, then TP=4")
    ap.add_argument("--util", type=float, default=eval_defaults.GPU_MEMORY_UTILIZATION)
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--child-module", default="arl.experiments.async_swe_eval", help=argparse.SUPPRESS)  # 测试用
    ap.add_argument("--no-start", action="store_true", help=argparse.SUPPRESS)              # 测试用
    a = ap.parse_args()
    if a.sampler_dp is not None and a.sampler_tp is not None:
        ap.error('specify either SAMPLER_DP or SAMPLER_TP, not both')
    if a.ngpus < 1:
        raise SystemExit("NGPUS must be a positive integer")
    if a.cmd == "_serve":
        serve(a)
        return
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `status | head` 不打 BrokenPipe 栈
    if a.cmd == "add":
        if not a.items:
            raise SystemExit("add: give at least one run directory or quoted glob")
        print("\n".join(queue_add(a.items)))
        if not a.no_start:
            start_daemon(a)
    elif a.cmd == "rm":
        if not a.items:
            raise SystemExit("rm: give at least one run (directory, name or queued glob)")
        print("\n".join(queue_rm(a.items)))
        if daemon_pid() is not None:
            print(f"(the service applies it within {a.poll}s; a running evaluation of a removed run is aborted)")
    elif a.cmd == "redo":
        print("\n".join(redo(a.items)))
        if daemon_pid() is not None:
            print(f"(the service picks it up within {a.poll}s if the run is in its queue)")
    elif a.cmd == "status":
        print_status()
    elif a.cmd == "log":
        os.execvp("tail", ["tail", "-n", "40", "-F", str(STATE / "daemon.log")])
    elif a.cmd == "start":
        start_daemon(a)
    elif a.cmd == "stop":
        stop_daemon()
    elif a.cmd == "restart":
        stop_daemon()
        start_daemon(a)


if __name__ == "__main__":
    main()
