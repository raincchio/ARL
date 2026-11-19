"""共卡的阶段切换与权重同步。训练客户端 (LocalTrainingClient) 持有一个 :class:`Colocator`。

阶段 (全 rank 一致, 切换都是 COLLECTIVE):

    "train"   vLLM 暂停调度、KV unmap; 训练侧可以用满显存
    "sample"  训练侧 empty_cache; vLLM KV 映射回来、恢复调度

启动 (``bootstrap``): rank 0 起完引擎后马上进 "train" (KV 放掉), 其余 rank 等它放完才碰
显存 —— vLLM 起引擎时按空闲显存的变化做 profiling, 别的进程这时在同一张卡上分配/释放会让
它直接报错 (79 上撞过: "Initial free memory ... current free memory ...")。这时训练的进程组
还没建 (vLLM 必须先于 DeepSpeed 的 PG 起, 见 training_client), 所以用文件做这道屏障。

权重同步 (``sync_weights``, 必须在 "train" 阶段): 按组 ZeRO-3 gather 参数, 采样卡上的训练 rank
把每个参数的 CUDA IPC handle 经 gloo 交给 rank 0, rank 0 让每台引擎的 worker 按卡 UUID
认领自己那份 load_weights。数据从头到尾不出这张卡。
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time

import torch

from arl.colocate import worker_ops
from arl.utils import distributed as dist_utils

logger = logging.getLogger(__name__)

_READY_POLL_S = 2.0


def _gib(n: float) -> str:
    return f"{n / 2**30:.1f}G"


class Colocator:
    def __init__(self, client):
        self.client = client
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g != ""]
        self.my_gpu = visible[self.local_rank] if visible else str(self.local_rank)
        self.on_sampler_card = self.my_gpu in client._sampler_gpus()
        self.phase = "sample"  # 引擎刚起来时 KV 是映射着的
        # 每张采样卡 kv_sleep 放掉的字节数 (rank 0 记, 唤醒前广播给各 rank 自查)
        self._kv_bytes: dict[str, int] = {}
        self.cleanup_ready_file()  # 上一个同名 job 留下的 (rank 0 才删)

    # ---- 引擎侧 (只有 rank 0 有引擎) -------------------------------------- #
    def _engines(self):
        eng = self.client._sampling_engine
        return [] if eng is None else eng.engines

    async def _rpc_all(self, fn, *args):
        """对每台引擎 collective_rpc 同一个函数, 结果按 worker 摊平。"""
        outs = await asyncio.gather(*[e.engine.collective_rpc(fn, args=args) for e in self._engines()])
        return [r for o in outs for r in o]

    async def _sampler_sleep(self) -> None:
        engines = self._engines()
        busy = sum(e.n_inflight for e in engines)
        if busy:
            logger.warning(f"[colocate] {busy} sampling request(s) still in flight when the sampler is put to sleep -- aborting them")
        # 先停调度、中止在飞请求、清前缀缓存 (KV 内容马上就没了), 再拿走 KV 的显存。
        await asyncio.gather(*[e.engine.pause_generation(mode="abort", clear_cache=True) for e in engines])
        for uuid, nbytes in await self._rpc_all(worker_ops.kv_sleep):
            self._kv_bytes[uuid] = nbytes

    async def _sampler_wake(self) -> None:
        engines = self._engines()
        await self._rpc_all(worker_ops.kv_wake)
        await asyncio.gather(*[e.engine.resume_generation() for e in engines])
        # 权重在这之间换过, 而前缀缓存的 key 只有 token hash —— 必须丢 (见 SamplingEngine.reset_prefix_cache)
        await asyncio.gather(*[e.reset_prefix_cache() for e in engines])

    # ---- 启动 -------------------------------------------------------------- #
    def _ready_file(self) -> str:
        # torchrun 的同一个 agent 起所有 local rank, getppid 相同; 带上端口防同机两个 job 撞名
        return f"/tmp/kernel_rl_colocate_{os.getppid()}_{os.environ.get('MASTER_PORT', '0')}.ready"

    async def bootstrap(self) -> None:
        """rank 0: KV 放掉后发信号; 其它 rank: 等信号。之后才能初始化 DeepSpeed。"""
        path = self._ready_file()
        if self.rank == 0:
            t = time.time()
            await self._sampler_sleep()
            self.phase = "train"
            with open(path, "w") as f:
                f.write(str(os.getpid()))
            logger.info(
                f"[colocate] sampler KV released ({', '.join(_gib(v) for v in self._kv_bytes.values())} per card, "
                f"{time.time() - t:.1f}s); trainer init may proceed")
            return
        t = time.time()
        while not os.path.exists(path):
            time.sleep(_READY_POLL_S)
        self.phase = "train"
        logger.info(f"[colocate] rank {self.rank}: sampler is asleep after {time.time() - t:.0f}s, initializing the trainer")

    def cap_trainer_memory(self) -> None:
        """给训练进程的 PyTorch 分配器设上限 = 现在的空闲显存 - 余量 (KERNEL_RL_COLOCATE_HEADROOM_GB,
        默认 8G; <=0 = 不设)。bootstrap 之后、DeepSpeed 初始化之前调: 这时 vLLM 已放掉 KV, 空闲
        显存里扣掉的正好是同卡 worker 常驻的权重 / CUDA graph 和 CUDA context。

        为什么要: 训练侧缓存远大于实际分配 (32k pack 实测分配峰值 43.7G, 缓存到 ~70G; DeepSpeed
        多条 stream 各留各的段), 会把整卡吃到只剩几 G。PyTorch 自己被驱动拒了还会清缓存重试,
        NCCL / 驱动 / 同卡的 vLLM 不会 —— 87 / 76 上 qwen38_grpo_1023_v3 / v4 训练阶段 dmesg
        反复 "alloc failed ... ret -12", 都在第二步训练里卡死在 ZeRO-3 reduce-scatter。设了上限,
        分配器到线就先清自己的缓存, 卡上始终留着这段真空闲的显存。"""
        headroom = float(os.environ.get("KERNEL_RL_COLOCATE_HEADROOM_GB", "8")) * 2**30
        if headroom <= 0:
            return
        free, total = torch.cuda.mem_get_info(self.local_rank)
        cap = free - headroom
        if cap <= 0:
            raise RuntimeError(
                f"[colocate] rank {self.rank}: only {_gib(free)} free after the sampler released its KV cache, "
                f"less than KERNEL_RL_COLOCATE_HEADROOM_GB={_gib(headroom)} -- nothing left for the trainer")
        torch.cuda.set_per_process_memory_fraction(cap / total, self.local_rank)
        logger.info(f"[colocate] rank {self.rank} ({'sampler' if self.on_sampler_card else 'train-only'} card): "
                    f"trainer allocator capped at {_gib(cap)} (free {_gib(free)} - headroom {_gib(headroom)})")

    def log_train_memory(self) -> None:
        """训练阶段 (含权重同步) 各 rank 的显存峰值 (实际分配 / 分配器缓存), rank 0 打每项的最大值,
        然后清峰值统计。缓存峰值贴着 cap_trainer_memory 的上限 = 分配器在靠清缓存周转。
        COLLECTIVE (gloo)。"""
        mine = (torch.cuda.max_memory_allocated(self.local_rank), torch.cuda.max_memory_reserved(self.local_rank))
        torch.cuda.reset_peak_memory_stats(self.local_rank)
        rows = dist_utils.all_gather_object(mine, group=dist_utils.gloo_group())
        if self.rank == 0:
            parts = []
            for i, what in enumerate(("allocated peak", "reserved peak")):
                r = max(range(len(rows)), key=lambda k: rows[k][i])
                parts.append(f"{what} {_gib(rows[r][i])} (rank {r})")
            logger.info(f"[colocate] train phase memory, max over ranks: {', '.join(parts)}")

    def cleanup_ready_file(self) -> None:
        if self.rank == 0:
            try:
                os.remove(self._ready_file())
            except OSError:
                pass

    # ---- 阶段切换 ---------------------------------------------------------- #
    async def enter_train(self) -> None:
        if self.phase == "train":
            return
        t = time.time()
        dist_utils.barrier()  # 所有 rank 都离开了采样 (DistributedSamplingClient 的 gloo 往返已结束)
        if self.rank == 0:
            await self._sampler_sleep()
        dist_utils.barrier()
        self.phase = "train"
        if self.rank == 0:
            logger.info(f"[colocate] -> train phase ({time.time() - t:.1f}s)")

    async def enter_sample(self) -> None:
        if self.phase == "sample":
            return
        t = time.time()
        self.log_train_memory()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # 唤醒前每张采样卡自查: 空闲显存放得下 KV 吗。放不下时 cuMemCreate 在 worker 里炸,
        # 报错远没有这里清楚。COLLECTIVE (broadcast + all_reduce)。
        kv_bytes = dist_utils.scatter_object_from_rank0(
            [self._kv_bytes] * int(os.environ.get("WORLD_SIZE", "1")) if self.rank == 0 else None,
            group=dist_utils.gloo_group())
        need = 0
        if self.on_sampler_card:
            need = kv_bytes.get(str(torch.cuda.get_device_properties(self.local_rank).uuid), 0)
        free = torch.cuda.mem_get_info(self.local_rank)[0]
        short = max(0, need - free)
        worst = dist_utils.all_reduce_max(float(short))
        if worst > 0:
            raise RuntimeError(
                f"[colocate] not enough free memory to wake the sampler's KV cache: short by up to {_gib(worst)} on a card "
                f"(rank {self.rank}: needs {_gib(need)}, free {_gib(free)}, trainer still holds "
                f"{_gib(torch.cuda.memory_reserved(self.local_rank))}). Lower SAMPLING_UTIL so the KV cache leaves room "
                f"for what the trainer keeps resident between phases.")
        if self.rank == 0:
            await self._sampler_wake()
        dist_utils.barrier()
        self.phase = "sample"
        if self.rank == 0:
            logger.info(f"[colocate] -> sample phase ({time.time() - t:.1f}s)")

    # ---- 权重同步 ---------------------------------------------------------- #
    async def sync_weights(self, named_params: list, budget_gb: float = 6.0) -> None:
        """把训练侧当前参数灌进同卡的 vLLM worker (CUDA IPC)。完成后切到 "sample"。"""
        import deepspeed
        from torch.multiprocessing.reductions import reduce_tensor

        await self.enter_train()  # gather 要显存, 而且 worker 的 load_weights 不能和采样并发
        my_uuid = str(torch.cuda.get_device_properties(self.local_rank).uuid)
        gloo = dist_utils.gloo_group()

        groups, cur, cur_bytes = [], [], 0
        for name, p in named_params:
            nb = getattr(p, "ds_numel", p.numel()) * p.element_size()
            if cur and cur_bytes + nb > budget_gb * 1e9:
                groups.append(cur)
                cur, cur_bytes = [], 0
            cur.append((name, p))
            cur_bytes += nb
        if cur:
            groups.append(cur)

        # 训练侧开着 expandable_segments (conf 的 PYTORCH_CUDA_ALLOC_CONF 默认开, 治训练碎片) 时,
        # 它分出来的显存走 cuMem 的另一套共享接口, PPU 驱动不认: worker 重建 IPC 张量时
        # "CUDA driver error: unknown error" (74 qwen36_grpo_1021_v3)。同步期间临时关掉它, 把要发的
        # 参数 clone 到普通段 (cudaMalloc, 已验证能 IPC) 上再发, 发完放掉、再打开。没开时零拷贝直接发。
        expandable = "expandable_segments:true" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").replace(" ", "").lower()
        if expandable:
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        t0 = time.time()
        t_rpc = 0.0
        try:
            for group in groups:
                # COLLECTIVE: 每个 rank 都要进 gather, 不管它的卡上有没有 worker。
                with deepspeed.zero.GatheredParameters([p for _, p in group], modifier_rank=None):
                    mine, staged = None, []
                    if self.on_sampler_card:
                        staged = [(name, p.data.detach().clone() if expandable else p.data.detach()) for name, p in group]
                        mine = (my_uuid, [(name, reduce_tensor(t)) for name, t in staged])
                    gathered = dist_utils.gather_object_to_rank0(mine, group=gloo)
                    if self.rank == 0:
                        handles = {u: h for u, h in (g for g in gathered if g is not None)}
                        _t = time.time()
                        await self._rpc_all(worker_ops.ipc_load_weights, handles)
                        t_rpc += time.time() - _t
                    # worker 读完之前发送方不能释放 gathered 参数 / 拷贝
                    dist_utils.barrier()
                    del staged, mine
        finally:
            torch.cuda.ipc_collect()
            if expandable:
                torch.cuda.empty_cache()  # 拷贝用的普通段还给驱动, 免得之后一直占着
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        if self.rank == 0:
            logger.info(
                f"[sync-timing] colocate IPC sync={time.time() - t0:.1f}s "
                f"(load_weights {t_rpc:.1f}s, {len(groups)} groups, {len(named_params)} params)")
        await self.enter_sample()
