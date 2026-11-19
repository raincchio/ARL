"""训练 / 采样共卡 (KERNEL_RL_COLOCATE=1, conf 的 COLOCATE=1)。

默认关: 训练卡和采样卡互不重叠 (分离式, 权重走跨卡 NCCL 广播)。打开后 SAMPLER_GPUS 必须
是 TRAIN_GPUS 的子集, 同一张卡上同时住着一个训练 rank 和一个 vLLM TP worker, 两个阶段
轮流用显存:

    rollout   vLLM KV 映射着; 训练侧只留常驻部分 (ZeRO-3 的参数/梯度分片等)
    train     vLLM 暂停调度 + 只 unmap KV cache, 权重和 CUDA graph 留着; 训练侧用满显存
    sync      训练卡上的完整参数 -> CUDA IPC handle -> 同卡的 vLLM worker load_weights (不跨卡)

为什么只放 KV、不用 vLLM 自带的 sleep (2026-10-21, 10.0.134.79 实测, 探针在
/root/workspace/colo_probe/): vLLM 的 sleep 会把 cumem 池里**全部**分配 (权重 + KV) 都
unmap 再 remap, 之后重放 CUDA graph 时 PPU 运行时在 memcpy 节点上段错误
(hgiMemcpyOperandGetRootDevice), FULL / PIECEWISE 都会, 2~4 轮就炸; eager 不炸但采样
慢 2.1x。只 unmap tag=kv_cache 的分配, Qwen3.8-27B TP=8 PIECEWISE 连续 30 轮无事, 每卡
放出 70.8G (含 GDN 状态缓存), 训练侧实测能拿到 83G/卡; greedy 输出的分叉位置和不 sleep
的对照完全一样。代价是权重常驻 (27B TP=8 每卡 6.75G, 加 graph 池等共 ~13G)。

IPC 灌权重实测: 27B 851 个张量 0.8s; 灌零化 down_proj 输出立刻变、灌回原权重分叉位置
回到对照水平。

只支持 ZeRO-3 (训练侧本来就只有它): ZeRO-2 每卡一份完整 bf16 参数, 27B 就是 50G, 加上
vLLM 常驻后留给激活的不到 30G, 生产的 32k~40k 序列放不下, 更大的模型更不用说。

模块:
    worker_ops   跑在 vLLM worker 里的函数 (collective_rpc)
    controller   阶段切换 + 权重同步, 训练客户端持有一个
"""

from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("KERNEL_RL_COLOCATE", "0").strip().lower() in ("1", "true", "yes")
