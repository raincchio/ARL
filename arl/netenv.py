"""进程启动时的网络环境加固 —— 同一份代码在 DNS/代理各不相同的机器上都跑得动。

这里修的都是"机器给的环境不对, 但程序本可以不受影响"的事。每一条都先检测再动手,
环境正常的机器上整个模块是 no-op:

  1. **RES_OPTIONS**: glibc 解析器的默认是 ``timeout:5 attempts:2``。碰上一台"对没有
     AAAA 记录的域名直接丢包 (而不是回 NODATA)"的内网 DNS 时, 每次冷解析都要赔上十几
     秒 —— 实测 ``api.wandb.ai`` / ``hf-mirror.com`` 的 AAAA 查询固定 12.01s 才超时,
     ``getent hosts api.wandb.ai`` 首查 14.2s。压到 ``timeout:1 attempts:2`` 之后最坏
     ~6s, 而 DNS 正常的机器本来就 <50ms, 感觉不到差别。

  2. **代理 env 的大小写分裂**: ``http_proxy`` 和 ``HTTP_PROXY`` 指到不同地址时, 会按
     语言分叉 —— Python (urllib/requests) 和 curl 取小写, Go 的 net/http/httpproxy
     (wandb-core 这类) 取大写。于是同一台机器上一半请求走活代理、另一半对着死代理干等
     到超时, 而且症状按库随机出现。统一成同一个地址 (优先保留连得上的那个)。

  3. **no_proxy 补上自家服务**: RM / π_ref / 评测服务都是内网 IP, 不该走代理。代码里
     能显式禁代理的地方都禁了 (rm_client 的 ``proxies=None``、httpx 的
     ``trust_env=False``、脚本里的 ``curl --noproxy '*'``), 这里是给第三方库和后续新
     代码的兜底: 少一处漏网, 就少一次"代理把内网请求吞了"的超时。

  4. **本机 hostname 解析不了时的短路**: 主机名带下划线 (DNS 非法, 容器里很常见) 或
     /etc/hosts 里没有对应记录时, ``gethostbyname(gethostname())`` 要卡满整个解析超时
     再抛 gaierror。自家代码只在 ``ml_log._host_ip`` 的兜底分支上碰它, 但 torch elastic
     的 rendezvous、ray、部分 wandb 路径都会走这一步。检测到解析不了, 就把本机名钉到
     出口 IP 上 —— 效果等同于往 /etc/hosts 写一行, 但不用 root, 也跟着代码走。

作用范围: 1-3 改的是环境变量, 本进程和它 fork/spawn 出来的所有子进程 (torchrun 的
rank、vLLM 的 worker、wandb-core) 都跟着生效; 4 是 socket 层的补丁, 只在调用过
``harden_network_env()`` 的那个进程里有效 —— 所以每个入口自己调一次, 别指望继承。
``script/_lib.sh`` 另外 export 了一份 RES_OPTIONS, 让 curl 和 torchrun 这类"在我们的
代码之前就开始解析"的进程也吃得到。

整体开关: ``KERNEL_RL_NET_HARDEN=0`` 关掉全部四条。
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from collections.abc import Iterable
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# timeout:1 是"一个正常 DNS 一定答得完"的量级 (内网 DNS 实测 <50ms), attempts:2 留一次
# 重试容忍丢包, single-request-reopen 让 A/AAAA 不共用一个 socket —— 有些中间设备只回
# 其中一个答案就关连接, 共用时另一个查询会一直等到超时。
_RES_OPTIONS = "timeout:1 attempts:2 single-request-reopen"

_PROBE_TIMEOUT = 1.0      # 探一个代理地址连不连得上
_RESOLVE_TIMEOUT = 2.0    # 等本机名解析出结果的上限 (超了就当解析不了)

_PROXY_PAIRS = (("http_proxy", "HTTP_PROXY"),
                ("https_proxy", "HTTPS_PROXY"),
                ("all_proxy", "ALL_PROXY"))

# 自家服务的地址都从这些 env 来 (训练脚本设的), 一律不该经过代理。
_ENDPOINT_ENVS = ("KERNEL_RL_EVAL_ENDPOINT", "KERNEL_RL_RM_URL", "KERNEL_RL_REF_URL",
                  "MASTER_ADDR")

_done = False


def harden_network_env(extra_no_proxy: Iterable[str] = ()) -> None:
    """按需修正本进程的 DNS / 代理环境。幂等, 可以在每个入口放心调。

    ``extra_no_proxy`` 给调用方补自己知道的主机名 (rm_service 知道 master 的地址,
    但那时它还没进 env)。
    """
    global _done
    if os.environ.get("KERNEL_RL_NET_HARDEN", "1").strip() == "0":
        return
    if _done:
        _extend_no_proxy(extra_no_proxy)   # 这条是纯累加的, 后来的地址还能补进去
        return
    _done = True

    _set_res_options()
    _unify_proxy_env()
    _extend_no_proxy(extra_no_proxy)
    _pin_local_hostname()

    # litellm 被上游 (vllm 等) import 时会去 raw.githubusercontent.com 拉价格表, 拉不到
    # 就是一条启动期的 read timeout —— 这个仓库从不用它计价, 直接用包内自带的那份。
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


# ── 1. 解析器超时 ──────────────────────────────────────────────────────────
def _set_res_options() -> None:
    if os.environ.get("RES_OPTIONS", "").strip():
        return                              # 机器/运维已经有主张了, 不插手
    os.environ["RES_OPTIONS"] = _RES_OPTIONS
    logger.debug(f"RES_OPTIONS={_RES_OPTIONS} (缩短 DNS 冷解析最坏耗时)")


# ── 2. 代理 env 统一 ───────────────────────────────────────────────────────
def _reachable(url: str, timeout: float = _PROBE_TIMEOUT) -> bool:
    """代理地址连得上吗 —— 只做 TCP 握手, 不发请求。"""
    u = urlsplit(url if "://" in url else f"http://{url}")
    if not u.hostname:
        return False
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        with socket.create_connection((u.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def _unify_proxy_env() -> None:
    """同一个代理的大小写两份值不一致时, 统一成连得上的那个。

    都连不上 (或都连得上) 时留小写的: 那本来就是 Python 和 curl 在用的值, 保持现状只
    消除分叉, 不改变本机已经被验证过的行为。
    """
    for lower, upper in _PROXY_PAIRS:
        lo, up = os.environ.get(lower, "").strip(), os.environ.get(upper, "").strip()
        if lo == up or not (lo or up):
            continue
        if lo and up:
            keep = lo if _reachable(lo) else (up if _reachable(up) else lo)
        else:
            keep = lo or up
        dropped = up if keep == lo else lo
        os.environ[lower] = os.environ[upper] = keep
        logger.warning(
            f"{lower}/{upper} 指向不同地址, 已统一为 {keep} (弃用 {dropped or '空值'}) —— "
            f"分叉时 Python/curl 走小写、Go 走大写, 一半请求会对着不通的代理干等超时")


# ── 3. no_proxy 兜底 ───────────────────────────────────────────────────────
def _host_of(value: str) -> str:
    """从 ``http://10.0.134.90:30600`` / ``10.0.134.90:8000`` / ``10.0.134.90`` 里取主机名。"""
    value = value.strip()
    if not value:
        return ""
    u = urlsplit(value if "://" in value else f"http://{value}")
    return u.hostname or ""


def _extend_no_proxy(extra: Iterable[str] = ()) -> None:
    """把回环和自家服务的地址加进 no_proxy (大小写两份都写)。"""
    wanted = ["localhost", "127.0.0.1", "::1"]
    # 端点可以是逗号分隔的多台 (评测主节点的主/备, 见 r3lm/envs/eval/remote.py), 每台都要进。
    raw = [os.environ.get(k, "") for k in _ENDPOINT_ENVS] + [x or "" for x in extra]
    wanted += [_host_of(part) for value in raw for part in value.split(",")]

    current = os.environ.get("no_proxy", "") or os.environ.get("NO_PROXY", "")
    merged: list[str] = []
    for item in [*current.split(","), *wanted]:
        item = item.strip()
        if item and item not in merged:
            merged.append(item)
    value = ",".join(merged)
    if value != current:
        logger.debug(f"no_proxy={value}")
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = value


# ── 4. 本机 hostname ───────────────────────────────────────────────────────
def _outbound_ip() -> str | None:
    """默认路由的源 IP。UDP "connect" 不发包, 只是让内核选路 —— 比 ``hostname -I``
    (多网卡时常给 docker 网桥) 和 gethostbyname (常给 127.0.1.1) 都靠谱。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _resolve_within(host: str, timeout: float) -> str | None:
    """限时解析。解析卡住时 getaddrinfo 本身没法打断 (C 库调用), 所以丢给守护线程去
    等, 主线程到点就走人 —— 我们只需要知道"它在合理时间内答不上来"。"""
    box: dict[str, str] = {}

    def run() -> None:
        try:
            box["ip"] = socket.getaddrinfo(host, None, socket.AF_INET,
                                           socket.SOCK_STREAM)[0][4][0]
        except OSError:
            pass

    t = threading.Thread(target=run, daemon=True, name="resolve-selfname")
    t.start()
    t.join(timeout)
    return box.get("ip")


def _port_num(port) -> int:
    """getaddrinfo 的 port 可以是 int / 数字串 / 服务名 / None。服务名查 /etc/services,
    不碰 DNS。"""
    if port is None:
        return 0
    if isinstance(port, int):
        return port
    text = str(port)
    if text.isdigit():
        return int(text)
    try:
        return socket.getservbyname(text)
    except OSError:
        return 0


def _pin_local_hostname() -> None:
    """本机名解析不了时, 在 socket 层把它钉到出口 IP 上。

    只拦本机名, 别的域名原样走原函数 —— 这是个补丁, 不是自己实现一个解析器。
    """
    name = socket.gethostname()
    if not name:
        return
    short = name.split(".")[0]
    if _resolve_within(name, _RESOLVE_TIMEOUT):
        return                                   # 解析得了, 什么都不用做

    ip = _outbound_ip() or "127.0.0.1"
    aliases = {name, short}
    real_gai, real_gbn = socket.getaddrinfo, socket.gethostbyname
    real_gbn_ex, real_getfqdn = socket.gethostbyname_ex, socket.getfqdn

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        if host in aliases:
            if family == socket.AF_INET6:
                # 和真实的"只有 A 没有 AAAA"表现一致, 免得调用方以为有 v6 地址。
                raise socket.gaierror(socket.EAI_NODATA, "No address associated with hostname")
            types = [type] if type else [socket.SOCK_STREAM, socket.SOCK_DGRAM]
            canon = host if flags & socket.AI_CANONNAME else ""
            return [(socket.AF_INET, t, proto, canon, (ip, _port_num(port))) for t in types]
        return real_gai(host, port, family, type, proto, flags)

    def gethostbyname(host):
        return ip if host in aliases else real_gbn(host)

    def gethostbyname_ex(host):
        return (host, [], [ip]) if host in aliases else real_gbn_ex(host)

    def getfqdn(host=""):
        return name if (host or name) in aliases else real_getfqdn(host)

    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex
    socket.getfqdn = getfqdn

    logger.warning(
        f"本机名 {name!r} 在 {_RESOLVE_TIMEOUT:.0f}s 内解析不出来 (DNS 没有它, /etc/hosts 也没写), "
        f"已在本进程内把它钉到 {ip} —— 否则每次 gethostbyname(gethostname()) 都要卡满解析超时再报错 "
        f"(torch elastic / ray / wandb 都会走这一步)。根治是往 /etc/hosts 加一行 '{ip} {short}'")
