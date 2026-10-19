"""Local training client: HF model + DeepSpeed ZeRO-3 **full-parameter** fine-tuning.

Drop-in replacement for the subset of ``tinker.TrainingClient`` that
``arl.algorithms.loop`` uses:

    get_tokenizer()
    forward_backward_async(data, loss_fn, loss_fn_config) -> future.result_async()
    optim_step_async(adam_params) -> future.result_async()
    save_weights_and_get_sampling_client_async() -> SamplingClient
    create_sampling_client_async(path) -> SamplingClient

The whole model is trainable (no LoRA). On-policy weight sync is done by writing
the consolidated (ZeRO-3-gathered) policy weights to disk as a standard HF
checkpoint and reloading them **in place** into the shared vLLM engine via
``reload_weights`` -- vLLM's official RLHF weight-update path (see ``sampling.py``).
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

import torch
import torch.utils.checkpoint

from arl.models import types as tinker
from arl.algorithms.loss import SamplingReplay, gather_target_logprobs, ppo_surrogate
from arl.models.types import Datum, ForwardBackwardOutput, TensorData

logger = logging.getLogger(__name__)


def ensure_distributed_env() -> None:
    """DeepSpeed needs a torch.distributed rendezvous even for single process.

    Without launcher-provided env vars it falls back to MPI (mpi4py), which is
    not installed on this accelerator. Populate sane single-process defaults.
    """
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # Single-process run: pick a guaranteed-free port to avoid EADDRINUSE from a
    # stale/hardcoded MASTER_PORT. A real multi-proc launcher (WORLD_SIZE>1) sets
    # its own rendezvous, so leave that alone.
    if os.environ.get("WORLD_SIZE", "1") == "1":
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()
    else:
        os.environ.setdefault("MASTER_PORT", "29500")


def _env_int(name: str, default: int) -> int:
    """Read an int env override, tolerating "2e8"-style scientific notation."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        logger.warning(f"[trainer] {name}={raw!r} not a number; using {default}")
        return default


def _head_chunk_logprobs(lm_head, h: torch.Tensor, t: torch.Tensor,
                         replay=None, stats=None) -> torch.Tensor:
    """logits = lm_head(h) for ONE row-chunk, reduced straight to target logprobs.
    Runs under torch.utils.checkpoint so the [chunk, V] logits are dropped after the
    forward and recomputed in backward (see forward_backward_async, HEAD_CHUNKS).
    ``replay``/``stats``: see ``gather_target_logprobs`` (sampling distribution replay)."""
    return gather_target_logprobs(lm_head(h), t, replay, stats)


def _env_float(name: str, default: float) -> float:
    """Read a float env override, falling back (with a warning) on garbage."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[trainer] {name}={raw!r} not a number; using {default}")
        return default


def _row_logits(logits: torch.Tensor, row: int, off: int, T: int) -> torch.Tensor:
    """``logits[row, off:off+T, :]`` without SliceBackward's full-size grad buffer.

    Ported from CudaAgent (2026-08-22 RFT OOM fix). Indexing into the [mb, maxlen, V]
    logits builds a SliceBackward node whose backward allocates a ZEROS copy of the
    whole base tensor and scatters the slice grad into it -- at micro_bsz=1 without
    packing the slice covers everything, so that is a pure waste of one full [T, V]
    (bf16 T=33k, V=248320 -> 15.5 GiB). When the slice is the whole tensor, reshape
    instead: a zero-alloc view whose backward is also a view.
    """
    if logits.shape[0] == 1 and off == 0 and logits.shape[1] == T:
        return logits.reshape(T, logits.shape[-1])
    return logits[row, off : off + T, :]


def _sampler_enforce_eager(inference_only: bool) -> bool:
    """vLLM 的 ``enforce_eager``: True = 不用 CUDA graph, 每步逐 kernel 下发。

    默认**一律 False (开 CUDA graph)** —— 2026-08-08 起训练侧也开, 依据是两次实测:

    * inference_only (① rollout / ⑤ 评估), 2026-08-07 (QwQ-32B, TP=4,
      32 条真实 prompt x 4096 token):
          eager        生成 123.6s, 1061 tok/s, KV 1,073,744 tok
          CUDA graph   生成  80.8s, 1621 tok/s, KV 1,071,520 tok
      **快 1.53x**, graph pool 只吃 0.16 GiB (KV 只少 0.2%)。这条路只在启动时加载
      一次权重, 本来就没有风险。

    * 训练 (④/④′), 2026-08-08 (QwQ-32B, TP=8, 32 条 x 2048 token 满长):
      eager 1077 tok/s vs graph 2343 tok/s = **2.18x**, 折算每批 (采样占 43%)
      省约 23% 墙钟; 代价是起引擎多 ~42s (graph 捕获, 一次性)。

      训练侧曾长期强制 eager, 顾虑是: 每 batch 权重**热更新** (``reload_weights``
      / ``update_weights_nccl``), 而 CUDA graph 捕获的是显存**地址** —— 若
      ``load_weights`` 重新分配了某个参数, graph 会继续读旧地址, **不报错**, 只是
      采样静默用上一版权重。该顾虑已被实测排除 (本 vLLM 0.19 fork, graph 模式下):
        - 经 ``model.load_weights`` 热改一个 layernorm, 固定序列的 prompt logprob
          立刻变化 (最大差 1.92, 重复打分噪声地板 0.0) —— graph 读到的是新权重;
        - ``reload_weights`` 原 checkpoint 后 logprob **精确**还原 (残差 0.0);
        - 全部被测参数的 data_ptr 在热更新/reload 前后不变 = load_weights 是原地
          copy_, graph 有效是结构保证; reload 后采样速度不掉 (graph 不需要重录)。
      验证覆盖 ``load_weights`` 这个入口 (checkpoint 路径直接测过; NCCL 路径的
      worker 侧也是同一个 load_weights, 但 NCCL 传输本身没在 graph 下跑过整批) ——
      升级 vLLM 后重跑一遍 bench (它测的是 load_weights 的原地性, 版本相关),
      再决定要不要继续开。

    ``KERNEL_RL_ENFORCE_EAGER=1/0`` 可以强制覆盖 (排查 graph 相关的诡异现象时,
    ``KERNEL_RL_ENFORCE_EAGER=1`` 退回 eager 对照)。
    """
    raw = os.environ.get("KERNEL_RL_ENFORCE_EAGER", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    if raw:
        logger.warning(
            f"[sampler] KERNEL_RL_ENFORCE_EAGER={raw!r} could not be parsed (expected 1/0), using the default"
        )
    return False


def _zero3_comm_tuning() -> dict:
    """ZeRO-3 communication knobs (stage 3 only). Element counts, bf16 = 2 bytes each.

    DeepSpeed 0.14.4 already defaults ``overlap_comm``/``contiguous_gradients`` to
    True under stage 3, so those two are pinned here only so a version bump can't
    silently flip them -- they are not a change in behaviour. The two that ARE
    raised above the defaults buy overlap depth with GPU memory we have spare
    (32B full-FT on 8 ranks sits around 45GB of a 96GiB card):

    prefetch_bucket_size  5e7 -> 2e8 elements (~100MB -> ~400MB in flight)
        How far ahead the next layers' params are all-gathered. Too small and the
        backward stalls on a gather it could have started three layers ago; this
        is the knob that matters most when ranks span more than one NVLink group.
    max_live_parameters   1e9 -> 2e9 elements (~2GB -> ~4GB gathered at once)
        Params allowed to stay gathered before ZeRO releases them. Raising it
        avoids re-gathering the same layer inside one forward/backward. Must stay
        above the largest single param -- QwQ-32B's lm_head is 152064x5120 =
        779M elements, so both the old and new value clear it.

    param_persistence_threshold (1e5) is DeepSpeed's default and is left alone: it
    already covers this model's per-layer norms (5120 elements), which is the case
    it exists for -- below the threshold a param is never partitioned, so thousands
    of tiny all-gathers disappear.

    ``zero_hpz_partition_size`` (ZeRO++ hierarchical partitioning) is OFF by
    default. Setting it to N keeps a secondary param replica within each group of
    N ranks, so the backward all-gather runs inside the group instead of across
    the whole world -- exactly the win this machine's topology offers, where
    TRAIN_GPUS spans two fully-NVLinked quads joined by a single NV1 link. It is
    NOT free: the secondary partition costs model_bytes/N per card (~16GB per card
    at N=4 for 32B), so it needs the headroom checked before it is turned on.
    """
    cfg = {
        # Defaults under stage 3; pinned so they can't regress on a version bump.
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
        # 5e8 elements = 1GB per buffer, x2 while overlap_comm double-buffers.
        "reduce_bucket_size": _env_int("KERNEL_RL_ZERO3_REDUCE_BUCKET", 500_000_000),
        "stage3_param_persistence_threshold": _env_int(
            "KERNEL_RL_ZERO3_PERSIST_THRESHOLD", 100_000
        ),
        # Raised above DeepSpeed's defaults -- see docstring.
        "stage3_prefetch_bucket_size": _env_int("KERNEL_RL_ZERO3_PREFETCH", 200_000_000),
        "stage3_max_live_parameters": _env_int("KERNEL_RL_ZERO3_MAX_LIVE", 2_000_000_000),
        "stage3_max_reuse_distance": _env_int("KERNEL_RL_ZERO3_MAX_REUSE", 1_000_000_000),
    }
    hpz = _env_int("KERNEL_RL_ZERO_HPZ", 1)
    if hpz > 1:
        cfg["zero_hpz_partition_size"] = hpz
    return cfg


def _set_z3_leaf_moe(model) -> int:
    """Mark MoE blocks as ZeRO-3 leaf modules. KERNEL_RL_Z3_LEAF=1/0 forces;
    unset = auto (on iff the model has MoE blocks).

    Why: ZeRO-3 gathers/releases params via per-module forward hooks, prefetched
    by replaying the trace of the previous step. An MoE block's expert execution
    is DATA-DEPENDENT (top-k routing decides which experts see tokens), so the
    module sequence differs between the original forward and the gradient-
    checkpointing recompute -- params that were gathered in forward can be left
    partitioned (shape [0] placeholder) during recompute. Observed on
    Qwen3.6-35B-A3B smoke 0827: torch.utils.checkpoint metadata check failed on
    the first backward with recomputed shape [0] (see CheckpointError in
    logs/smoke_qwen36moe_0827_101008.log). Marking the MoE block as a z3 leaf
    makes ZeRO-3 gather the whole block's params at its boundary and not hook
    inside -- DeepSpeed's documented requirement for Mixtral-family models.

    Cost: one block's params live gathered during its forward/backward
    (Qwen3.6-35B-A3B: 256 experts x 3 x 2048x512 x bf16 ~= 1.6 GB per layer,
    transient) -- the price of correctness here.

    Detection is duck-typed (module has both ``experts`` and ``gate``) so it
    survives class renames across the Qwen MoE family; only the sparse-MoE
    block realistically has both.

    Must run before deepspeed.initialize. Returns the number of leaf classes
    marked (0 = not a MoE / disabled).
    """
    raw = os.environ.get("KERNEL_RL_Z3_LEAF", "").strip().lower()
    if raw in ("0", "false", "no"):
        return 0
    if raw and raw not in ("1", "true", "yes"):
        logger.warning(
            f"[trainer] KERNEL_RL_Z3_LEAF={raw!r} could not be parsed (expected 1/0), falling back to auto-detection"
        )
        raw = ""
    classes = sorted(
        {
            type(m)
            for m in model.modules()
            if hasattr(m, "experts") and hasattr(m, "gate")
        },
        key=lambda c: c.__name__,
    )
    if not classes:
        if raw:
            logger.warning(
                "[trainer] KERNEL_RL_Z3_LEAF=1 but no MoE block was found (no submodule carrying both "
                "experts and gate) -- nothing was marked"
            )
        return 0
    from deepspeed.utils import set_z3_leaf_modules

    set_z3_leaf_modules(model, classes)
    logger.info(
        f"[trainer] ZeRO-3 leaf modules: {[c.__name__ for c in classes]} "
        f"(KERNEL_RL_Z3_LEAF=0 to disable)"
    )
    return len(classes)


def _freeze_moe_router(model) -> int:
    """Freeze MoE routing params. KERNEL_RL_FREEZE_ROUTER=1/0 forces; unset = auto
    (freeze iff the model has router params, i.e. it is a MoE).

    Why default-on: this training loop has NO load-balancing aux loss (the thing
    that keeps routing stable during MoE pretraining), and RL batches are tiny
    (batch_size x group_size rollouts). A trainable router without a balancing
    term can drift / collapse experts silently -- no error, the curves just
    degrade. The router is also negligible capacity (Qwen3.6-35B-A3B: 40 layers
    x 2048x256 ~= 21M params), so freezing costs nothing and saves its optimizer
    state.

    What matches (Qwen3.5-MoE naming; the dots around ``.mlp.gate.`` matter -- a
    bare "gate" substring would also hit the experts' own ``gate_up_proj`` /
    ``shared_expert.gate_proj``, which are the main capacity and must stay
    trainable):
      *.mlp.gate.*                 top-k router
      *.mlp.shared_expert_gate.*   shared expert's sigmoid gate (routing weight too)

    Must run BEFORE deepspeed.initialize: the optimizer param list is filtered on
    requires_grad, so a later flip never reaches the optimizer. Fine under ZeRO-3
    partitioned init (params are shards; the flag applies per-shard). Weight sync
    to vLLM is untouched -- frozen weights broadcast as usual, values unchanged.

    Returns the number of frozen params (0 = nothing matched / disabled).
    """
    raw = os.environ.get("KERNEL_RL_FREEZE_ROUTER", "").strip().lower()
    if raw in ("0", "false", "no"):
        return 0
    if raw and raw not in ("1", "true", "yes"):
        logger.warning(
            f"[trainer] KERNEL_RL_FREEZE_ROUTER={raw!r} could not be parsed (expected 1/0), falling back to auto-detection"
        )
        raw = ""
    frozen: list[tuple[str, int]] = []
    for name, p in model.named_parameters():
        if ".mlp.gate." in name or ".mlp.shared_expert_gate." in name:
            p.requires_grad = False
            frozen.append((name, getattr(p, "ds_numel", p.numel())))
    if not frozen:
        if raw:  # explicitly requested but nothing matched: not a MoE, or naming drifted
            logger.warning(
                "[trainer] KERNEL_RL_FREEZE_ROUTER=1 but no parameter matches *.mlp.gate.* / "
                "*.mlp.shared_expert_gate.* (not an MoE? a model family with different naming?) -- nothing was frozen"
            )
        return 0
    n_elem = sum(k for _, k in frozen)
    logger.info(
        f"[trainer] MoE router frozen: {len(frozen)} params, {n_elem:,} elements "
        f"(e.g. {frozen[0][0]}; KERNEL_RL_FREEZE_ROUTER=0 to train them)"
    )
    return len(frozen)


def _freeze_shared_expert(model) -> int:
    """Freeze the MoE shared expert's MLP (``*.mlp.shared_expert.{gate,up,down}_proj``).
    KERNEL_RL_FREEZE_SHARED_EXPERT=1/0 forces; unset = auto (freeze iff matched).

    Why (2026-09-07): every non-finite / absurd gradient seen in the Qwen3.6-35B-A3B
    runs (0903_v2 / 0904_v2 / 0905_v1 / 0907_dp2, 2400+ flagged params) landed on
    exactly these three matrices and nothing else -- 30% of substeps skipped with
    sequence packing, 6-9% without, plus finite-but-huge spikes the inf/nan guard
    cannot see. Root cause is still open (single-card and 8-card ZeRO-3 repros on
    .70 stay clean). Taking the shared expert out of the optimizer removes the
    only parameters that ever went bad: no gradient is computed for them, so no
    skipped substeps and no garbage updates. Capacity cost is small: 40 layers x
    3 x 512x2048 = 126M of 34.6B trainable params; the shared expert still runs in
    forward with its pretrained weights, and the routed experts (the bulk of the
    MLP capacity) keep training. Its sigmoid gate was already frozen with the
    router.

    Same constraints as _freeze_moe_router: run BEFORE deepspeed.initialize (the
    optimizer filters on requires_grad); weight sync unaffected. The trailing dot
    in ``.mlp.shared_expert.`` keeps ``shared_expert_gate`` out of this match (it
    belongs to the router freeze).
    """
    raw = os.environ.get("KERNEL_RL_FREEZE_SHARED_EXPERT", "").strip().lower()
    if raw in ("0", "false", "no"):
        return 0
    if raw and raw not in ("1", "true", "yes"):
        logger.warning(
            f"[trainer] KERNEL_RL_FREEZE_SHARED_EXPERT={raw!r} could not be parsed (expected 1/0), falling back to auto-detection"
        )
        raw = ""
    frozen: list[tuple[str, int]] = []
    for name, p in model.named_parameters():
        if ".mlp.shared_expert." in name:
            p.requires_grad = False
            frozen.append((name, getattr(p, "ds_numel", p.numel())))
    if not frozen:
        if raw:
            logger.warning(
                "[trainer] KERNEL_RL_FREEZE_SHARED_EXPERT=1 but no parameter matches *.mlp.shared_expert.* "
                "(not an MoE with a shared expert?) -- nothing was frozen"
            )
        return 0
    n_elem = sum(k for _, k in frozen)
    logger.info(
        f"[trainer] MoE shared expert frozen: {len(frozen)} params, {n_elem:,} elements "
        f"(e.g. {frozen[0][0]}; KERNEL_RL_FREEZE_SHARED_EXPERT=0 to train them)"
    )
    return len(frozen)


class _ImmediateFuture:
    """Trivial future: the work already ran synchronously; expose result_async()."""

    def __init__(self, value: Any):
        self._value = value

    async def result_async(self) -> Any:
        return self._value

    def result(self) -> Any:
        return self._value


class CausalLMWithValueHead(torch.nn.Module):
    """Policy backbone that also exposes one hidden state (PPO critic input).

    forward returns ``(logits, hidden)``: ``logits`` from the LM head, ``hidden`` =
    ``hidden_states[value_layer]`` [B, T, H] (-1 = after the final norm; L = output of
    decoder layer L). The value net itself is NOT in here: it lives
    on the client (``TrainingClient._v_head``), outside the DeepSpeed engine, with
    its own optimizer, and reads ``hidden.detach()`` -- so the value loss trains the
    head only and never touches the trunk (or its grad_clip budget). Weight sync to
    vLLM saves ONLY ``self.base`` (the plain CausalLM).
    """

    def __init__(self, base, value_layer: int = -1):
        super().__init__()
        self.base = base
        self.config = base.config
        self.value_layer = value_layer

    def forward(self, input_ids=None, attention_mask=None):
        out = self.base(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        return out.logits, out.hidden_states[self.value_layer]

    def gradient_checkpointing_enable(self, **kw):
        self.base.gradient_checkpointing_enable(**kw)


def _nvlink_matrix() -> dict[str, dict[str, int]]:
    """``nvidia-smi topo -m`` -> {gpu: {peer: NVLink 条数}} (只记 NV* 链路; PIX/PXB/NODE/SYS
    都不算)。解析不了就返回 {}, 调用方按"没有 NVLink"处理。"""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:  # noqa: BLE001 -- 拓扑只是优化提示, 拿不到就退回老行为
        return {}
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return {}
    cols = [h[3:] for h in lines[0].split() if h.startswith("GPU") and h[3:].isdigit()]
    mat: dict[str, dict[str, int]] = {}
    for line in lines[1:]:
        parts = line.split()
        if not parts or not parts[0].startswith("GPU") or not parts[0][3:].isdigit():
            continue
        row: dict[str, int] = {}
        for peer, cell in zip(cols, parts[1:1 + len(cols)]):
            if cell.startswith("NV"):
                try:
                    row[peer] = int(cell[2:] or "1")
                except ValueError:
                    row[peer] = 1
        mat[parts[0][3:]] = row
    return mat


def _sync_owner_ranks(replicas: list[list[str]], world: int, visible: list[str],
                      mat: dict[str, dict[str, int]] | None = None,
                      override: str = "") -> list[int]:
    """每台采样副本的权重广播源 = 哪个训练 rank (2026-09-07)。

    ZeRO-3 gather 之后每个训练 rank 手里都是完整参数, 广播源不必是 rank 0。NCCL 广播是
    流水线环, 吞吐由环上最慢的一跳决定: 训练卡到副本四元组的第一跳没有 NVLink 时, 跨组
    只能走 PCIe P2P (.87 上 43GB/s) 或 socket (.70 上 P2P 建不起来, 3.5GB/s, 35B 一台副本
    20s); 进了四元组之后全是 NVLink。所以按 ``nvidia-smi topo -m`` 给每台副本挑"到它的
    卡有最多 NVLink 条数"的训练 rank (并列取 rank 小的); 一条都没有就退回 rank 0 (老行为)。
    各副本源不同时广播天然并行, 同步时间从"各副本之和"变成"最慢一台"。

    ``override`` (env KERNEL_RL_SAMPLER_SYNC_RANKS="1,0"): 显式指定, 绕过拓扑探测。
    训练 rank r 的物理卡 = CUDA_VISIBLE_DEVICES[r] (launcher 把训练卡放在采样卡前面)。
    所有 rank 都要算出同一个结果 (env 一样, topo 一样), 它决定谁进哪个 NCCL 组。
    """
    if override.strip():
        owners = [int(x) for x in override.split(",")]
        if len(owners) != len(replicas) or any(not 0 <= o < world for o in owners):
            raise ValueError(
                f"KERNEL_RL_SAMPLER_SYNC_RANKS={override!r} expects {len(replicas)} ranks in [0, {world})"
            )
        return owners
    mat = _nvlink_matrix() if mat is None else mat
    train_gpu = {r: (visible[r] if r < len(visible) else str(r)) for r in range(world)}
    owners = []
    for rep in replicas:
        best, best_score = 0, 0
        for r in range(world):
            score = sum(mat.get(train_gpu[r], {}).get(g, 0) for g in rep)
            if score > best_score:
                best, best_score = r, score
        owners.append(best)
    return owners


class LocalTrainingClient:
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        zero_stage: int = 3,
        grad_clip: float = 1.0,
        learning_rate: float = 1e-6,
        gradient_checkpointing: bool = True,
        sampling_gpu_memory_utilization: float = 0.3,
        sampling_max_model_len: int | None = None,
        sampling_tensor_parallel_size: int = 1,
        max_tokens: int = 4096,
        enable_value_head: bool = False,
        value_head_coef: float = 0.5,
        value_lr: float = 1e-4,
        value_arch: str = "linear",
        value_layer: int = -1,
        value_tower_layers: int = 2,
        value_tower_dim: int = 1024,
        value_tower_heads: int = 8,
        value_head_arch: str = "linear",
        value_head_hidden: int = 1024,
        value_init_path: str = "",
        inference_only: bool = False,
    ):
        import deepspeed
        from transformers import AutoModelForCausalLM

        from arl.models.tokenizer import get_tokenizer

        ensure_distributed_env()
        self.model_name = model_name
        self.dtype = dtype
        # PPO value head: a scalar critic (Linear) on the policy's last hidden state.
        # Off by default (pure policy-gradient / GRPO). It is trained by its OWN
        # torch AdamW (lr = value_lr, typically >> the policy lr, so it can fit the
        # reward scale quickly) on the DETACHED hidden state: the value loss updates
        # the head only, never the trunk. Weight sync to vLLM never sees it.
        self._has_value_head = enable_value_head
        self._vf_coef = float(value_head_coef)
        self._value_lr = float(value_lr)
        # Value net = arl.models.value_net.ValueNet on hidden_states[value_layer]:
        # "linear" = the original single Linear on the last hidden state; "tower" = a
        # few causal transformer blocks over the whole sequence + an MLP head (offline:
        # ev 0.22 -> 0.41, see rlhf/ppo/value_lora.py). Both read the DETACHED hidden.
        self._value_cfg = dict(arch=value_arch, tower_layers=value_tower_layers,
                               tower_dim=value_tower_dim, tower_heads=value_tower_heads,
                               head_arch=value_head_arch, head_hidden=value_head_hidden)
        self._value_layer = int(value_layer)
        self._value_init_path = value_init_path or ""
        # Dr-GRPO (improvement #2) constant-length denominator: the loss divides each
        # response's summed surrogate by a FIXED constant (the generation budget),
        # not by the response's own token count, so long and short responses carry
        # equal per-token weight -- removes the length bias original GRPO's 1/|o_i|
        # introduces. Env override wins for experiments; else the generation budget.
        #
        # This is the DEFAULT/fallback. A caller that changes the generation budget
        # mid-run (see ppo_loop's max_tokens_late) passes the budget in effect for
        # that batch via loss_fn_config["dr_grpo_const_len"] -- Dr-GRPO's constant is
        # *the generation budget*, so it follows the budget rather than staying at
        # whatever the budget happened to be at startup. The env override, being an
        # explicit experimental pin, still wins over both.
        self._const_len_pinned = "KERNEL_RL_DR_GRPO_CONST_LEN" in os.environ
        self._dr_grpo_const_len = max(
            1, int(os.environ.get("KERNEL_RL_DR_GRPO_CONST_LEN", str(max_tokens)))
        )
        # Placement: training runs on cards 0..W-1 and each rank's sampler runs on a
        # DISJOINT shared sampler (rank 0 owns one TP engine over
        # KERNEL_RL_SAMPLER_GPUS), so the two never contend for memory. Per-step
        # weight sync goes over a cross-GPU NCCL broadcast (rank 0 -> TP workers);
        # set KERNEL_RL_NCCL_SYNC=0 to fall back to the disk path.
        self._nccl_sync = os.environ.get("KERNEL_RL_NCCL_SYNC", "1") == "1"
        # Must be set before any _ensure_sampling_engine() call below: engine placement
        # (own card vs. disjoint inference card) branches on it.
        self._inference_only = inference_only
        self._sampling_cfg = dict(
            gpu_memory_utilization=sampling_gpu_memory_utilization,
            max_model_len=sampling_max_model_len,
            tensor_parallel_size=sampling_tensor_parallel_size,
            # CUDA graph 默认全开 (见 _sampler_enforce_eager: 热更新的原地性已实测
            # 验证, 训练侧快 2.18x)。KERNEL_RL_ENFORCE_EAGER=1 可退回 eager。
            enforce_eager=_sampler_enforce_eager(inference_only),
        )
        self._tokenizer = get_tokenizer(model_name)

        # Shared vLLM sampling engine (weights hot-reloaded in place each sync).
        self._sampling_engine = None
        self._sync_counter = 0
        # Trainer-side NCCL communicators for disaggregated weight sync (lazy init;
        # one per sampler replica, see _ensure_weight_comm).
        self._weight_comms: dict | None = None
        self._sync_owners: list[int] = []
        # Single reusable dir for on-policy weight sync (bounds disk to one copy).
        self._sync_dir = os.environ.get("KERNEL_RL_SYNC_DIR", "/tmp/kernel_rl_sync")
        # In-flight background checkpoint write (rank 0 only; see _save_full_model).
        self._save_thread: threading.Thread | None = None
        self._save_error: BaseException | None = None

        # SPMD: create this rank's vLLM engine BEFORE DeepSpeed initializes the
        # multi-rank training NCCL process group. vLLM's spawned EngineCore inits
        # its own (world_size=1) NCCL cleanly only when no multi-rank parent PG
        # exists yet; once DeepSpeed's PG is up, the child engine fails to start.
        # (Single-process runs keep the lazy path -- there is no such conflict.)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            self._ensure_sampling_engine()

        # Sampling-only mode (rollout / eval): build NO optimizer, NO DeepSpeed engine,
        # and no trainer-side HF model at all -- vLLM loads ``model_name`` itself, so a
        # second copy on the training side would be pure waste. A full-FT engine also
        # allocates fp32 master weights + Adam m/v (~6x model bytes: ~190GB for 32B)
        # that a rollout never touches -- that allocation is what OOM'd 32B rollouts.
        # Weight sync is a no-op here: nothing mutates the weights, so what vLLM loaded
        # from disk IS the current policy.
        if inference_only:
            self.engine = None
            self._hf_ds_cfg = None
            self._ensure_sampling_engine()  # idempotent; already built when WORLD_SIZE>1
            # DeepSpeed normally inits the process group. It isn't running, so do it
            # here -- the rollout's cross-rank all_reduce (sample counting) needs it.
            # AFTER the vLLM spawn above, per the EngineCore ordering note.
            if int(os.environ.get("WORLD_SIZE", "1")) > 1:
                import torch.distributed as dist

                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
            self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            logger.info(f"[sampler] inference-only client for {model_name} (no optimizer)")
            return

        zero_cfg = {
            "stage": zero_stage,
            # Gather full bf16 weights on save so we can write an HF checkpoint.
            "stage3_gather_16bit_weights_on_model_save": True,
        }
        if zero_stage == 3:
            zero_cfg.update(_zero3_comm_tuning())
        # Offload optimizer (and optionally params) to CPU so a large model can be
        # full-FT'd in a single process. Required to fit 7B+ on one GPU; harmless
        # (just slower) for small models. Controlled by KERNEL_RL_ZERO_OFFLOAD:
        #   "none" (default) | "optimizer" | "all" (optimizer + params)
        offload = os.environ.get("KERNEL_RL_ZERO_OFFLOAD", "none").lower()
        if offload in ("optimizer", "all"):
            zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if offload == "all":
            zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": dtype == torch.bfloat16},
            "fp16": {"enabled": dtype == torch.float16},
            "zero_optimization": zero_cfg,
            "gradient_clipping": grad_clip,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": learning_rate, "betas": [0.9, 0.95], "eps": 1e-8},
            },
        }

        logger.info(f"[trainer] loading base model {model_name} (full-parameter FT)")
        # ZeRO-3 partitioned init: with a live HfDeepSpeedConfig, from_pretrained shards
        # params ACROSS ranks as it builds the model, so no rank ever materializes the
        # full model on GPU. Essential for 32B -- otherwise the transient full-model
        # load peak (~26GB) coexists with the awake vLLM engine and OOMs. (Only bites
        # under ZeRO-3 multi-rank; harmless otherwise.)
        self._hf_ds_cfg = None
        if zero_stage == 3:
            from transformers.integrations import HfDeepSpeedConfig

            self._hf_ds_cfg = HfDeepSpeedConfig(ds_config)  # must stay alive through from_pretrained
        # FlashAttention-2: the varlen kernel takes cu_seqlens, so padding tokens in a
        # mixed-length batch never enter attention -- unlike sdpa, whose fast path dies
        # the moment attention_mask has any padding (measured 2.9x slower on the real
        # 6k+16k pairing). FA2 only supports fp16/bf16, hence the dtype guard.
        attn_impl = (
            "flash_attention_2" if dtype in (torch.bfloat16, torch.float16) else None
        )
        # Multimodal checkpoints (Qwen3.6 etc.) carry a COMPOSITE config whose text
        # params live under ``text_config``. AutoModelForCausalLM maps them to the
        # text-only class, but ``from_pretrained`` passes the composite config
        # straight into it and crashes on ``config.vocab_size`` (``from_config``
        # extracts the text config; ``from_pretrained`` does not -- transformers
        # 5.14.1). Extract it ourselves; the ``model.language_model.*`` ->
        # ``model.*`` weight renaming and the mtp/visual ignore-list are already
        # registered per model_type, so loading needs nothing else.
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(model_name)
        text_cfg = hf_cfg.get_text_config()
        self._mm_wrapped = text_cfg is not hf_cfg
        load_kwargs = {"config": text_cfg} if self._mm_wrapped else {}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation=attn_impl, **load_kwargs
        )
        model.config.use_cache = False
        # Gradient checkpointing trades compute for memory -- essential for full-FT
        # of 7B+ models. Small overhead for tiny models.
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        # MoE: freeze routing params (auto-on for MoE, KERNEL_RL_FREEZE_ROUTER=0 to
        # train them -- see _freeze_moe_router for why). Before the value-head wrap
        # so names match the raw model; before deepspeed.initialize so the optimizer
        # filter below sees the flags.
        _freeze_moe_router(model)
        # 共享专家的三个矩阵也冻结 (2026-09-07): 迄今所有坏梯度都只落在它们上, 见 _freeze_shared_expert。
        _freeze_shared_expert(model)
        # MoE x ZeRO-3 x 梯度检查点: expert 执行数据依赖, 重算时参数可能没被重新
        # gather (0827 smoke 首次 backward 的 CheckpointError) —— 整块标成 z3 leaf。
        _set_z3_leaf_moe(model)
        # PPO: wrap the policy so its forward also returns the last hidden state (the
        # value head's input). The head itself stays outside the engine -- see below.
        if self._has_value_head:
            n_hidden = model.config.num_hidden_layers
            if not (self._value_layer == -1 or 1 <= self._value_layer <= n_hidden):
                raise ValueError(f"value_layer={self._value_layer} -- -1 (final) or 1..{n_hidden}")
            model = CausalLMWithValueHead(model, self._value_layer)
        # Under ZeRO-3 partitioned init each local param tensor is a shard; ds_numel
        # (if present) reports the true global element count.
        n_train = sum(
            getattr(p, "ds_numel", p.numel()) for p in model.parameters() if p.requires_grad
        )
        logger.info(f"[trainer] trainable params (full model): {n_train:,}")

        # Only trainable params go to the optimizer: ZeRO would otherwise build fp32
        # master copies + Adam state for frozen ones (router, see above).
        model_parameters = [p for p in model.parameters() if p.requires_grad]

        self.engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model_parameters,
            config=ds_config,
        )
        self.engine.train()
        self.device = self.engine.device

        # Value head: OUTSIDE the DeepSpeed engine, with its own optimizer that
        # optimizes nothing but the head. fp32, replicated on every rank (H+1
        # params): rank 0's init is broadcast, grads are averaged across ranks in
        # _value_head_step (same mean DeepSpeed takes for the trunk), so all
        # replicas stay identical. Not subject to the trunk's grad_clip.
        self._v_head = None
        self._v_opt = None
        if self._has_value_head:
            import torch.distributed as dist

            from arl.models.value_net import ValueNet

            self._v_head = ValueNet(model.config.hidden_size, **self._value_cfg).to(
                self.device, torch.float32)
            if dist.is_available() and dist.is_initialized():
                for p in self._v_head.parameters():
                    dist.broadcast(p.data, src=0)
            self._v_opt = torch.optim.AdamW(
                self._v_head.parameters(), lr=self._value_lr,
                betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
            if self._value_init_path:
                self._load_value_init(self._value_init_path)
            n_v = sum(p.numel() for p in self._v_head.parameters())
            logger.info(f"[trainer] value net: {self._value_cfg} on hidden_states[{self._value_layer}], "
                        f"{n_v:,} params, separate AdamW (lr={self._value_lr:g}), trunk hidden detached")

    # ---- tokenizer -------------------------------------------------------- #
    def get_tokenizer(self):
        return self._tokenizer

    def _require_engine(self, what: str) -> None:
        """Fail loudly on a training call against an inference-only client."""
        if self._inference_only:
            raise RuntimeError(
                f"{what} requires a training engine, but this client was built with "
                "inference_only=True (sampling only, no optimizer)."
            )

    # ---- forward / backward ---------------------------------------------- #
    @staticmethod
    def _pack_by_tokens(lengths: list[int], budget: int) -> list[list[int]]:
        """First-fit-decreasing: group datum indices so each group's total length
        stays within ``budget`` tokens. A datum longer than the budget gets a group
        of its own (it fits in memory alone today; the budget is for the packing).
        Deterministic given the input order."""
        order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
        groups: list[list[int]] = []
        fill: list[int] = []
        for i in order:
            for g, f in enumerate(fill):
                if f + lengths[i] <= budget:
                    groups[g].append(i)
                    fill[g] = f + lengths[i]
                    break
            else:
                groups.append([i])
                fill.append(lengths[i])
        return groups

    async def forward_backward_async(
        self,
        data: list[Datum],
        loss_fn: str = "ppo",
        loss_fn_config: dict[str, float] | None = None,
    ) -> _ImmediateFuture:
        self._require_engine("forward_backward_async")
        if loss_fn != "ppo":
            raise NotImplementedError(f"loss_fn={loss_fn!r} not supported (only 'ppo')")
        cfg = loss_fn_config or {}
        clip_low = float(cfg.get("clip_low_threshold", 0.2))
        clip_high = float(cfg.get("clip_high_threshold", 0.2))
        # In-loss KL penalty vs the reference policy (GRPO-style): adds
        # kl_beta * k3 with k3_t = exp(ref_t - lp_t) - (ref_t - lp_t) - 1 per active
        # token, normalized exactly like the surrogate (sum/const_len per response,
        # then 1/G over responses) so kl_beta's scale is comparable to the policy
        # term. k3 uses the CURRENT policy's logprob, so unlike an in-reward penalty
        # it is differentiable and stays fresh across substeps. Needs
        # loss_fn_inputs["ref_logprobs"] on each datum (datums without it -- e.g.
        # SPMD padding dummies -- simply contribute nothing).
        kl_beta = float(cfg.get("kl_beta", 0.0))
        # k3 的 delta = ref - lp 在进 exp 之前 clamp 到 ±这个值 (0 = 关)。兜的是训练端
        # forward 偶发的假 lp: 2026-08-20 诊断 (0820_continue, policy 逐字节 == π_ref,
        # 真实 KL≈0) 确认各 run 间歇性 kl_ref 飙升 0.1-81 是假测量 —— 个别 token 瞬时
        # lp ≈ ref-10 以下, k3=exp(δ)-δ-1 把几十个 token 指数放大成整批梯度爆炸
        # (grad_norm 曾到 1e3-1e4)。正常 token 的 |δ| 在 0.5 以内, 离 10 差一个多量级,
        # clamp 不触碰真实测量; 被 clamp 的 token 值封顶 exp(10)-11 且梯度为 0
        # (clamp 越界处零梯度), 爆不动了。异常本身不靠这条曲线看 —— 见 kl_delta_max
        # 指标与超阈 dump。
        kl_delta_clamp = _env_float("KERNEL_RL_KL_DELTA_CLAMP", 10.0)
        # Dr-GRPO's constant denominator = the generation budget IN EFFECT for this
        # batch (see __init__). The env pin overrides it; otherwise the caller's value
        # wins, falling back to the startup budget when it doesn't pass one.
        const_len = self._dr_grpo_const_len
        if not self._const_len_pinned:
            const_len = max(1.0, float(cfg.get("dr_grpo_const_len", const_len)))
        # 采样分布回放 (Cognition SWE-1.7 的熵保持, 见 loss.SamplingReplay): 给了 replay_*
        # 三个键就在 rollout 的 temperature + top_k/top_p kept set 内重新归一 log p, 集合外
        # 零梯度。old logprobs 必须是同一口径 (采样端 logprobs_mode=processed_logprobs),
        # 由 ppo_loop 保证两边一起开。
        replay = None
        if "replay_temperature" in cfg:
            replay = SamplingReplay(
                temperature=float(cfg["replay_temperature"]),
                top_k=int(cfg.get("replay_top_k", -1)),
                top_p=float(cfg.get("replay_top_p", 1.0)),
            )
        rp_tok = rp_outside = rp_single = 0
        rp_kept_sum = 0.0

        if not data:
            return _ImmediateFuture(ForwardBackwardOutput(loss_fn_outputs=[]))

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id or 0

        # Micro-batch the forward/backward so peak memory is per-chunk, not
        # whole-batch. Full-parameter 7B + large vocab makes a single [B, T, V]
        # logits tensor (e.g. 16*4096*152k*2B ~= 19GB) OOM the GPU; processing a
        # few sequences at a time with DeepSpeed gradient accumulation avoids that.
        micro_bsz = max(1, int(os.environ.get("KERNEL_RL_MICRO_BSZ", "1")))

        # Pre-extract per-datum tensors (kept on CPU) and count active tokens.
        # Dr-GRPO normalizes each response by a FIXED constant (const_len, above)
        # rather than its OWN active-token count (original GRPO's 1/|o_i|), then averages
        # over responses (1/G) -- so the objective is (1/G) Σ_i (1/L_const) Σ_t surrogate.
        # We still record each datum's active-token count (to detect empty responses)
        # and the number of active responses (n_resp) here. n_active_total is kept only
        # for the token-level diagnostics (ratio/clip/kl).
        per = []
        n_active_total = 0
        n_resp_active = 0
        for d in data:
            ids = torch.tensor(d.model_input.to_ints(), dtype=torch.long)
            li = d.loss_fn_inputs
            tgt = li["target_tokens"].to_torch().long()
            old = li["logprobs"].to_torch().float()
            adv = li["advantages"].to_torch().float()
            ret = li["returns"].to_torch().float() if "returns" in li else None
            ref = li["ref_logprobs"].to_torch().float() if "ref_logprobs" in li else None
            n_act = int((adv != 0).sum().item())
            n_active_total += n_act
            if n_act > 0:
                n_resp_active += 1
            per.append((ids, tgt, old, adv, len(ids), n_act, ret, ref))
        denom = max(1, n_active_total)
        n_resp = max(1, n_resp_active)

        per_datum_new_lp: list[Any] = [None] * len(per)
        ratio_sum = 0.0
        clip_count = 0
        kl_sum = 0.0
        loss_sum = 0.0
        vloss_sum = 0.0
        kl3_tok_sum = 0.0  # detached k3 sum over active tokens (diagnostic)
        kl_pen_sum = 0.0   # the loss term actually added (kl_beta included)
        kl_delta_max = 0.0  # max |ref - lp| over active tokens, BEFORE clamp -- 假测量探测器
        kl_clamped_tok = 0  # tokens whose delta hit the clamp this call
        kl_dump_left = 16   # 超阈现场 dump 的行数预算 (per forward_backward, 防刷屏)
        # 非有限探测器 (2026-08-29): 训练端 forward 给出的 lp 里 nan/inf 的 token 数
        # (全部 / 仅 active)、datum 数, 外加最多 8 行现场 dump。Qwen3.6 MoE 的 4 个 run
        # 都是这么炸的: 个别 token 的 lp 非有限 -> loss nan -> DeepSpeed 的 bf16 路径
        # 不查 overflow -> NaN 梯度直接进权重 -> 采样端跟着全废。这里只记不改:
        # backward 照走 (ZeRO-3 的 backward 是集合通信, 单 rank 跳过会死锁), 由
        # train_step 按全 rank 一致的判定去跳过 optim_step (见 optim_step_async(skip=)).
        nonfinite_tok = 0
        nonfinite_active = 0
        nonfinite_datums = 0
        nonfinite_dump_left = 8

        # Sequence packing (KERNEL_RL_PACK_TOKENS = token budget per forward, 0 = off):
        # instead of one padded [mb, maxlen] forward per micro-batch, concatenate
        # several datums into ONE [1, S] row (S <= budget) with no padding at all and
        # let the varlen kernels keep them apart -- position_ids restart at 0 per
        # sequence (the FA2 path detects that and switches to flash_varlen), and
        # cu_seq_lens_{q,k} / seq_idx go to the linear-attention (FLA chunk kernel) /
        # causal_conv1d layers of hybrid models (Qwen3.5/3.6). Same math as micro_bsz
        # (the loss normalization is whole-batch), fewer forwards: each forward under
        # ZeRO-3 costs a full parameter all-gather regardless of token count, and
        # MoE experts see budget-many tokens instead of one sequence's worth.
        # Ranks pack to different chunk counts, so they agree on the max and the
        # short ones run zero-loss dummy chunks (lockstep collectives, as in
        # compute_values). Value-head mode is not wired for it (the wrapper only
        # takes input_ids/attention_mask).
        pack_budget = max(0, int(os.environ.get("KERNEL_RL_PACK_TOKENS", "0")))
        head_chunks = max(0, int(os.environ.get("KERNEL_RL_HEAD_CHUNKS", "0")))
        if pack_budget > 0 and self._has_value_head:
            raise NotImplementedError("KERNEL_RL_PACK_TOKENS is not supported with a value head")
        if pack_budget > 0:
            from arl.utils.distributed import all_reduce_max

            chunks = self._pack_by_tokens([p[4] for p in per], pack_budget)
            n_global = int(all_reduce_max(len(chunks)))
            chunks += [[] for _ in range(n_global - len(chunks))]
        else:
            chunks = [list(range(i, min(i + micro_bsz, len(per)))) for i in range(0, len(per), micro_bsz)]
        for ci, idxs in enumerate(chunks):
            # Manual accumulation boundary: reduce grads + allow step() only on the
            # last micro-batch; earlier chunks just accumulate into the grad buffers.
            self.engine.set_gradient_accumulation_boundary(ci == len(chunks) - 1)

            if not idxs:
                # Packing lockstep dummy: this rank ran out of chunks while others still
                # have some. It MUST run the exact same module sequence as a real packed
                # chunk -- same varlen kwargs, same norm hook with the same N lm_head
                # calls, same checkpoint recompute in backward. Under ZeRO-3 every
                # module call is a parameter all-gather (a collective); the plain
                # full-logits forward this used to run made 1 lm_head call against the
                # N+1 of a fused-head chunk, and the ranks deadlocked on the first
                # training step (qwen36_grpo_0903_v1: NCCL RAS abort after 15 min).
                # Reproduced and fixed by script/test_pack_lockstep.py.
                # A synthetic pad sequence of length max(2, N) (so every row-chunk of
                # tensor_split is non-empty) with zero loss.
                seqs = [torch.full((max(2, head_chunks),), pad_id, dtype=torch.long)]
                tgts = [seqs[0].clone()]
                lengths = [len(seqs[0])]
                idxs_eff: list[int] = []
            else:
                seqs = [per[j][0] for j in idxs]
                tgts = [per[j][1][:per[j][4]] for j in idxs]
                lengths = [per[j][4] for j in idxs]
                idxs_eff = list(idxs)
            maxlen = max(lengths)
            mb = len(lengths)
            if pack_budget > 0:
                # One row, sequences back to back. slots[k] = (row, offset) of datum k.
                input_ids = torch.cat(seqs).unsqueeze(0).to(self.device)
                position_ids = torch.cat(
                    [torch.arange(L, dtype=torch.long) for L in lengths]).unsqueeze(0).to(self.device)
                offsets = [0]
                for L in lengths:
                    offsets.append(offsets[-1] + L)
                cu = torch.tensor(offsets, dtype=torch.int32, device=self.device)
                seq_idx = torch.repeat_interleave(
                    torch.arange(mb, dtype=torch.int32), torch.tensor(lengths)
                ).unsqueeze(0).to(self.device)
                values = None
                slots = [(0, offsets[k]) for k in range(mb)]
                varlen_kwargs = dict(
                    input_ids=input_ids, position_ids=position_ids,
                    cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                    max_length_q=maxlen, max_length_k=maxlen, seq_idx=seq_idx,
                )
                # One gather over the whole row, then slice the fp32 [S] result per
                # datum. Gathering per datum on logits[0, off:off+T] costs a THIRD
                # [S,V] bf16 tensor in backward (SliceBackward materializes a
                # zero-filled full-size grad per slice, 2+ slices => an extra copy
                # alive next to the logits and their grad; measured 3x vs 2x [S,V]
                # on a synthetic [8k,8k], and 71 vs 58GB at S=28k for 4 vs 2 datums).
                tgt_row = torch.cat(tgts).to(self.device)
                # KERNEL_RL_HEAD_CHUNKS = N > 0: never materialize the [S,V] logits at
                # all. The model runs with logits_to_keep=1 (lm_head on one token), and
                # a forward hook on the final norm -- i.e. still INSIDE the top-level
                # forward, so ZeRO-3's module trace / prefetch see a fixed sequence of
                # N+1 lm_head calls every step -- splits the post-norm hidden states
                # into exactly N row chunks and runs lm_head + target-logprob per chunk
                # under torch.utils.checkpoint: only [S/N, V] exists at a time, the
                # rest is recomputed in backward (one extra lm_head matmul, trivial
                # next to 40 layers). Remaining per-token memory is the activations.
                if head_chunks > 0:
                    base = self.engine.module
                    lm_head, final_norm = base.lm_head, base.model.norm
                    stash: dict[str, torch.Tensor] = {}
                    rp_stats: dict | None = {} if replay is not None else None

                    def _norm_hook(_mod, _inp, out):
                        h = out[0] if out.dim() == 3 else out  # [S,H]
                        parts = [
                            torch.utils.checkpoint.checkpoint(
                                _head_chunk_logprobs, lm_head, hc, tc, replay, rp_stats,
                                use_reentrant=False)
                            for hc, tc in zip(torch.tensor_split(h, head_chunks),
                                              torch.tensor_split(tgt_row, head_chunks))
                        ]
                        stash["lp"] = torch.cat(parts)

                    handle = final_norm.register_forward_hook(_norm_hook)
                    try:
                        logits = self.engine(**varlen_kwargs, logits_to_keep=1).logits  # [1,1,V]
                    finally:
                        handle.remove()
                    lp_row = stash.pop("lp")  # [S]
                else:
                    rp_stats = {} if replay is not None else None
                    logits = self.engine(**varlen_kwargs).logits  # [1,S,V]
                    lp_row = gather_target_logprobs(
                        logits, tgt_row.view(1, -1), replay, rp_stats).view(-1)  # [S]
                if rp_stats:
                    rp_in_row = torch.cat(rp_stats["in_set"])
                    rp_kept_row = torch.cat(rp_stats["n_kept"])
            else:
                input_ids = torch.full((mb, maxlen), pad_id, dtype=torch.long, device=self.device)
                attn = torch.zeros((mb, maxlen), dtype=torch.long, device=self.device)
                for k, sq in enumerate(seqs):
                    input_ids[k, : lengths[k]] = sq.to(self.device)
                    attn[k, : lengths[k]] = 1

                if self._has_value_head:
                    logits, hidden = self.engine(input_ids=input_ids, attention_mask=attn)
                    values = self._values_from_hidden(hidden)
                    del hidden
                else:
                    logits = self.engine(input_ids=input_ids, attention_mask=attn).logits  # [mb,T,V]
                    values = None
                slots = [(k, 0) for k in range(mb)]

            if not idxs:
                # dummy: zero loss through the same graph (backward gathers the same
                # params in the same order as a real chunk), nothing to account for.
                self.engine.backward(lp_row.sum() * 0.0)
                del logits, lp_row
                continue

            surrogate_sum = None
            value_sum = None
            kl_term_sum = None
            for k, j in enumerate(idxs_eff):
                T = lengths[k]
                row, off = slots[k]
                tgt = per[j][1][:T].to(self.device)
                old = per[j][2][:T].to(self.device)
                adv = per[j][3][:T].to(self.device)
                if pack_budget > 0:
                    lp = lp_row[off : off + T]
                    if replay is not None:
                        rp_in, rp_kept = rp_in_row[off : off + T], rp_kept_row[off : off + T]
                else:
                    rp_one: dict | None = {} if replay is not None else None
                    lp = gather_target_logprobs(_row_logits(logits, row, off, T), tgt, replay, rp_one)  # [T]
                    if replay is not None:
                        rp_in, rp_kept = torch.cat(rp_one["in_set"]), torch.cat(rp_one["n_kept"])
                if replay is not None:
                    with torch.no_grad():
                        act = adv != 0
                        rp_tok += int(act.sum().item())
                        rp_outside += int((act & ~rp_in).sum().item())
                        rp_single += int((act & rp_in & (rp_kept == 1)).sum().item())
                        rp_kept_sum += float(rp_kept[act].float().sum().item())
                per_datum_new_lp[j] = TensorData.from_torch(lp.detach())
                bad = ~torch.isfinite(lp.detach())
                if bool(bad.any()):
                    n_bad = int(bad.sum().item())
                    nonfinite_tok += n_bad
                    nonfinite_active += int((bad & (adv != 0)).sum().item())
                    nonfinite_datums += 1
                    if nonfinite_dump_left > 0:
                        nonfinite_dump_left -= 1
                        pos = bad.nonzero().flatten()[:8].tolist()
                        logger.warning(
                            "[trainer] non-finite lp: datum %d/%d T=%d n_bad=%d "
                            "(active %d) first_pos=%s lp=%s old=%s tok=%s",
                            j, len(per), T, n_bad,
                            int((bad & (adv != 0)).sum().item()), pos,
                            [float(lp[p]) for p in pos], [float(old[p]) for p in pos],
                            [int(tgt[p]) for p in pos],
                        )
                surrogate_tok = ppo_surrogate(lp, old, adv, clip_low, clip_high)
                # Dr-GRPO: divide THIS response's summed surrogate by a FIXED constant
                # (generation budget), not its own token count -- no per-response length bias.
                n_act_j = per[j][5]
                s = surrogate_tok.sum() / const_len if n_act_j > 0 else surrogate_tok.sum() * 0.0
                surrogate_sum = s if surrogate_sum is None else surrogate_sum + s
                # Value loss: MSE(V_t, return_t) over response tokens, same normalization
                # as the policy term (sum/const_len per response) so vf_coef is comparable.
                if values is not None:
                    ret = (
                        per[j][6][:T].to(self.device)
                        if per[j][6] is not None
                        else torch.zeros(T, device=self.device)
                    )
                    mask = (adv != 0).float()
                    vtok = (values[row, off : off + T] - ret) ** 2 * mask
                    vs = vtok.sum() / const_len if n_act_j > 0 else vtok.sum() * 0.0
                    value_sum = vs if value_sum is None else value_sum + vs
                # k3 KL penalty vs pi_ref, on active tokens. Zeroing delta BEFORE
                # exp makes masked positions contribute exactly 0 (k3(0) = 0), so no
                # separate masking of the sum is needed and no gradient leaks into
                # prompt/pad positions.
                ref = per[j][7]
                if kl_beta != 0.0 and ref is not None and n_act_j > 0:
                    delta = (ref[:T].to(self.device) - lp) * (adv != 0).float()
                    dmax = float(delta.detach().abs().max().item())
                    kl_delta_max = max(kl_delta_max, dmax)
                    if kl_delta_clamp > 0 and dmax > kl_delta_clamp:
                        # 抓现场: clamp 掉之前把最差几个 token 的 (datum, pos, lp, ref)
                        # 打进日志 —— 假测量的唯一未验证环节就是训练端 forward 的 lp,
                        # 下次飙升时这里直接给出肇事 token。
                        with torch.no_grad():
                            over_n = int((delta.abs() > kl_delta_clamp).sum().item())
                            kl_clamped_tok += over_n
                            k = min(kl_dump_left, 4, over_n)
                            if k > 0:
                                kl_dump_left -= k
                                worst = torch.topk(delta.abs(), k)
                                for pos in worst.indices.tolist():
                                    logger.warning(
                                        "[trainer] k3 delta over clamp: "
                                        f"datum={j} pos={pos} "
                                        f"token_id={int(per[j][1][pos])} "
                                        f"lp={float(lp[pos]):.3f} "
                                        f"ref={float(ref[pos]):.3f} "
                                        f"delta={float(delta[pos]):+.3f}")
                        delta = delta.clamp(-kl_delta_clamp, kl_delta_clamp)
                    k3 = torch.exp(delta) - delta - 1.0
                    ks = k3.sum() / const_len
                    kl_term_sum = ks if kl_term_sum is None else kl_term_sum + ks
                    kl3_tok_sum += float(k3.detach().sum().item())
                with torch.no_grad():
                    active = adv != 0
                    r = torch.exp(lp - old)[active]
                    ratio_sum += float(r.sum().item())
                    clip_count += int(((r < 1.0 - clip_low) | (r > 1.0 + clip_high)).sum().item())
                    kl_sum += float((old - lp)[active].sum().item())

            # Dropped before backward rather than after the chunk. This no longer
            # frees anything by itself -- gather_target_logprobs saves the logits for
            # its own backward (it recomputes softmax per chunk instead of keeping a
            # float32 [T,V] around), so autograd holds the last reference either way.
            # Kept because it costs nothing and stays correct if that ever changes:
            # were the loss to stop needing the logits, this is the line that would
            # hand ~6GB per sequence back before the 64-layer backward starts.
            del logits

            # Average the per-response constant-normalized surrogates over responses (1/G).
            loss_chunk = -(surrogate_sum / n_resp)
            if value_sum is not None:
                v_chunk = self._vf_coef * (value_sum / n_resp)
                loss_chunk = loss_chunk + v_chunk
                vloss_sum += float(v_chunk.detach().item())
            if kl_term_sum is not None:
                k_chunk = kl_beta * (kl_term_sum / n_resp)
                loss_chunk = loss_chunk + k_chunk
                kl_pen_sum += float(k_chunk.detach().item())
            self.engine.backward(loss_chunk)
            loss_sum += float(loss_chunk.detach().item())
            del loss_chunk, surrogate_sum

        metrics = {
            "loss": loss_sum,
            "ppo/mean_ratio": ratio_sum / denom,
            "ppo/clip_frac": clip_count / denom,
            "ppo/approx_kl": kl_sum / denom,
            "ppo/n_active_tokens": float(n_active_total),
            "ppo/value_loss": vloss_sum,
            # 非有限探测器 (见上)。恒发 (各 rank key 集一致, 上层按 sum 合并)。
            "ppo/nonfinite_lp_tokens": float(nonfinite_tok),
            "ppo/nonfinite_lp_active": float(nonfinite_active),
            "ppo/nonfinite_datums": float(nonfinite_datums),
            "ppo/nonfinite_loss": 0.0 if math.isfinite(loss_sum) else 1.0,
        }
        if replay is not None:
            # 采样分布回放的现场 (只数 active token, 恒发 -> 各 rank key 集一致):
            #   outside_frac  采样到的 token 不在 trainer 重算的 kept set 里 (被强制并入) 的
            #                 比例 = 近似回放的误差面, 应远小于 1%;
            #   zero_grad_frac 集合只有它自己 (top-1 已超 top_p) -> 概率 1、零梯度的比例;
            #   kept_mean     平均集合大小 (<= top_k)。
            metrics["ppo/replay_tokens"] = float(rp_tok)
            metrics["ppo/replay_outside_frac"] = rp_outside / max(1, rp_tok)
            metrics["ppo/replay_zero_grad_frac"] = rp_single / max(1, rp_tok)
            metrics["ppo/replay_kept_mean"] = rp_kept_sum / max(1, rp_tok)
        if kl_beta != 0.0:
            # Emitted whenever the penalty is configured -- even 0.0 on a rank whose
            # datums are all padding -- so every rank reports the same key set
            # (reduce_weighted_metrics needs that; weight-0 ranks don't skew it).
            metrics["ppo/kl_ref_per_token"] = kl3_tok_sum / denom
            metrics["ppo/kl_penalty"] = kl_pen_sum
            # 假测量探测器: clamp **之前**的最大 |delta| 与被 clamp 的 token 数。
            # kl_ref 被 clamp 兜住之后异常在它自己那条曲线上变小了, 只有这两条还
            # 看得见 (正常 kl_delta_max ~0.5, 假测量时 >10 / clamped_tokens > 0)。
            metrics["ppo/kl_delta_max"] = kl_delta_max
            metrics["ppo/kl_clamped_tokens"] = float(kl_clamped_tok)
        out = ForwardBackwardOutput(
            loss_fn_outputs=[{"logprobs": lp} for lp in per_datum_new_lp],
            metrics=metrics,
        )
        return _ImmediateFuture(out)

    # ---- value head ------------------------------------------------------- #
    def _values_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """[B, T, H] hidden state -> [B, T] values. ``detach`` is the whole point: the
        value loss reaches the value net's params and stops there. With a tower the
        attention is causal, so right padding never leaks into real positions."""
        return self._v_head(hidden.detach())

    def _value_head_step(self, skip: bool) -> None:
        """Step the value head's own optimizer (COLLECTIVE: every rank calls it).

        Grads are averaged across ranks first. A rank whose chunks were all dummies
        has no grad -> contributes zeros, like it does to DeepSpeed's reduce."""
        import torch.distributed as dist

        params = list(self._v_head.parameters())
        if not skip:
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            flat = torch.cat([g.reshape(-1) for g in grads])
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(flat)
                flat /= dist.get_world_size()
            # Same reduced grad on every rank -> same verdict, replicas stay in sync.
            if bool(torch.isfinite(flat).all()):
                off = 0
                for p in params:
                    n = p.numel()
                    p.grad = flat[off:off + n].view_as(p).clone()
                    off += n
                self._v_opt.step()
            else:
                logger.warning("[trainer] value head grad non-finite; value head step skipped")
        self._v_opt.zero_grad(set_to_none=True)

    def _load_value_init(self, path: str) -> None:
        """Warm-start the value net from an offline run (rlhf/ppo/value_lora.py's head.pt:
        {"tower": ..., "head": ...}) or from a ValueNet state_dict. Strict: a shape or name
        mismatch means a different architecture, which must not load silently.
        Every rank loads the same file, so replicas start identical without a broadcast."""
        st = torch.load(path, map_location="cpu", weights_only=False)
        if "head" in st and isinstance(st["head"], dict):  # offline value_lora checkpoint
            sd = {f"head.{k}": v for k, v in st["head"].items()}
            if st.get("tower") is not None:
                sd.update({f"tower.{k}": v for k, v in st["tower"].items()})
            pop = st.get("popart") or {}
            if pop.get("mu") is not None:
                raise ValueError(f"{path} was trained with PopArt (normalized targets); the online "
                                 f"critic regresses raw returns -- train it with --popart 0")
        else:
            sd = st
        self._v_head.load_state_dict(sd, strict=True)
        logger.info(f"[trainer] value net warm-started from {path} ({len(sd)} tensors)")

    def _value_head_path(self, d: str) -> str:
        return os.path.join(d, "value_head.pt")

    def _save_value_head(self, out_dir: str) -> None:
        if self._has_value_head and int(os.environ.get("RANK", "0")) == 0:
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"v_head": self._v_head.state_dict(), "opt": self._v_opt.state_dict()},
                       self._value_head_path(out_dir))

    def _load_value_head(self, in_dir: str) -> None:
        path = self._value_head_path(in_dir)
        if not self._has_value_head:
            return
        if not os.path.exists(path):
            logger.warning(f"[trainer] {path} not found -- value head starts from its fresh init")
            return
        st = torch.load(path, map_location=self.device)
        self._v_head.load_state_dict(st["v_head"])
        self._v_opt.load_state_dict(st["opt"])

    # ---- value forward (for PPO GAE at rollout time) --------------------- #
    @torch.no_grad()
    def compute_values(self, data: list[Datum]) -> list[torch.Tensor]:
        """Per-datum per-token value estimates V_t (aligned to targets, length T).

        Forward-only pass over each datum's input, returning the value head output
        at every position. Used at rollout to get the OLD values for GAE advantages
        (A_t = return_t - V_t). Returns [] and is a no-op if no value head.
        """
        if not self._has_value_head:
            return [None] * len(data)
        self._require_engine("compute_values")
        from arl.utils.distributed import all_reduce_max

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id or 0
        micro_bsz = max(1, int(os.environ.get("KERNEL_RL_MICRO_BSZ", "1")))
        was_training = self.engine.training
        self.engine.eval()
        out: list[torch.Tensor] = [None] * len(data)
        idx_all = list(range(len(data)))
        # ZeRO-3: every engine forward is a COLLECTIVE (param all-gathers), and ranks
        # can hold different datum counts (sampling failures drop responses). All
        # ranks must therefore run the same number of forwards: agree on the max
        # micro-step count, and ranks with fewer datums run dummy 1-token forwards
        # (the all-gather schedule depends only on the module graph, not the input).
        # Otherwise a fast rank moves on to a different collective and the whole
        # job deadlocks until the NCCL watchdog kills it. Must be called by EVERY
        # rank each batch, even with data=[].
        n_steps = (len(idx_all) + micro_bsz - 1) // micro_bsz
        g_steps = int(all_reduce_max(n_steps))
        for step in range(g_steps):
            idxs = idx_all[step * micro_bsz : (step + 1) * micro_bsz]
            if not idxs:  # dummy forward: keep this rank in lockstep, discard output
                dummy = torch.full((1, 1), pad_id, dtype=torch.long, device=self.device)
                self.engine(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                continue
            seqs = [torch.tensor(data[j].model_input.to_ints(), dtype=torch.long) for j in idxs]
            lengths = [len(s) for s in seqs]
            maxlen = max(lengths)
            input_ids = torch.full((len(idxs), maxlen), pad_id, dtype=torch.long, device=self.device)
            attn = torch.zeros((len(idxs), maxlen), dtype=torch.long, device=self.device)
            for k, s in enumerate(seqs):
                input_ids[k, : lengths[k]] = s.to(self.device)
                attn[k, : lengths[k]] = 1
            _, hidden = self.engine(input_ids=input_ids, attention_mask=attn)
            values = self._values_from_hidden(hidden)
            for k, j in enumerate(idxs):
                out[j] = values[k, : lengths[k]].float().detach().cpu()
        if was_training:
            self.engine.train()
        return out

    # ---- optimizer step --------------------------------------------------- #
    async def optim_step_async(self, adam_params: tinker.AdamParams,
                               skip: bool = False) -> _ImmediateFuture:
        """One optimizer step over the accumulated gradients.

        ``skip=True``: 丢掉这一步的梯度, 权重不动 —— 给 train_step 的非有限守卫用
        (loss / lp 里有 nan/inf 时)。走的是 DeepSpeed 自己的 overflow 分支: bf16 下
        stage3.step() 从不查 overflow, 但只要 optimizer.overflow 已是 True 它就会
        zero_grad + 清 averaged_gradients 然后提前返回, engine 侧记 skipped_steps、
        不动 lr scheduler —— 和 fp16 溢出跳步是同一条路, 分片梯度缓冲的状态由它自己
        收拾干净 (下一次 backward 的 micro_step_id 已归 0, 是 copy_ 不是 add_)。
        必须所有 rank 一起 skip (step 内部有集合通信; 判定见 train_step 的 all_reduce)。
        """
        self._require_engine("optim_step_async")
        if not skip:
            # 梯度侧守卫 (2026-08-31, qwen36_grpo_0830_sign batch 12): substep0 的
            # forward 全有限 (loss 0.56, ratio 0.9995), optim_step 之后权重却全 NaN ——
            # NaN 出在 backward/梯度, 不在 forward, 上面按 lp/loss 的判定看不见它。
            # 借 DeepSpeed fp16 路径自己的 has_overflow: 扫本 rank 的分片梯度缓冲
            # (grad_partitions_flat_buffer) 的 inf/nan 并跨 dp group all_reduce(MAX),
            # bf16 路径只是没人调它而已。COLLECTIVE, 各 rank 结论一致。
            if self._grad_has_inf_or_nan():
                skip = True
        if skip:
            opt = self.engine.optimizer
            opt.overflow = True
            try:
                self.engine.step()
            finally:
                opt.overflow = False
            if self._has_value_head:
                self._value_head_step(skip=True)
            logger.warning(
                "[trainer] optim_step skipped (non-finite loss/logprobs); grads dropped, "
                "weights untouched (engine.skipped_steps=%s)",
                getattr(self.engine, "skipped_steps", None),
            )
            return _ImmediateFuture({"grad_norm": None, "skipped": True})
        # Apply the requested learning rate (DeepSpeed AdamW built at init). The value
        # head is not in this optimizer; it has its own (fixed lr = value_lr).
        for group in self.engine.optimizer.param_groups:
            group["lr"] = adam_params.learning_rate
        self.engine.step()
        if self._has_value_head:
            self._value_head_step(skip=False)
        # Grad norm is populated by step() (after gradient clipping).
        grad_norm = self.engine.get_global_grad_norm()
        gn = float(grad_norm) if grad_norm is not None else None
        # DeepSpeed 在范数 inf/nan 时想返回 -1 (stage3 的 err 哨兵), 但它的哨兵算术对
        # inf/nan 会得到 nan (见 _grad_has_inf_or_nan)。走到这里说明两道 step 前守卫都没拦住
        # 而 step 已经做了。分两种: 权重仍有限 -> 只是范数算术出的 nan (有限大值, Adam 的
        # 逐元素归一化把这步更新幅度压在 ~lr 量级), 记警告继续; 权重已非有限 -> 停跑,
        # 不能把坏权重 weight_sync 到采样端再空转几十批。判定 COLLECTIVE (all_reduce MAX)。
        if gn is not None and (not math.isfinite(gn) or gn < 0):
            import torch.distributed as dist

            corrupt = 0.0
            try:
                for t in getattr(self.engine.optimizer, "fp32_partitioned_groups_flat", []):
                    if t is not None and not bool(torch.isfinite(t).all()):
                        corrupt = 1.0
                        break
            except Exception as e:  # noqa: BLE001
                logger.warning("[trainer] post-step weight finiteness check failed; treating the weights as corrupted: %r", e)
                corrupt = 1.0
            flag = torch.tensor([corrupt], device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if float(flag.item()) > 0:
                raise RuntimeError(
                    f"after optim_step grad_norm={gn} and the fp32 master weights contain inf/nan: neither pre-step guard"
                    " caught it and this step has already been written into the weights -- stopping; resume from the previous checkpoint."
                )
            logger.warning(
                "[trainer] after optim_step grad_norm=%s but all master weights are finite: this is a float32 overflow of the gradient norm plus "
                "DeepSpeed's sentinel arithmetic (a finite but large gradient entered Adam unclipped); continuing training and recording this step's "
                "grad_norm as None", gn,
            )
            return _ImmediateFuture({"grad_norm": None, "norm_overflow": True})
        return _ImmediateFuture({"grad_norm": gn})

    def _grad_has_inf_or_nan(self) -> bool:
        """step 之前查分片梯度里有没有 inf/nan (COLLECTIVE)。命中时把本 rank 上坏掉的
        参数名打进日志 (至多 16 个) —— 下次再炸直接知道是哪一层/哪种参数。"""
        opt = self.engine.optimizer
        if not hasattr(opt, "has_overflow"):
            return False
        try:
            bad = bool(opt.has_overflow(partition_gradients=True))
        except Exception as e:  # noqa: BLE001 -- 守卫自己不能把训练弄崩
            logger.warning("[trainer] has_overflow check failed; continuing as if there were no overflow: %r", e)
            return False
        if not bad:
            # 第二道: 逐参数分片范数 (2026-09-07, qwen36_grpo_0907_dp2 batch 3)。has_overflow
            # 只认 inf/nan; 而 shared_expert 的坏梯度有时是**有限但离谱**的数 (0905_v1 见过
            # 一次 grad_norm 1290, 0907_dp2 更大): DeepSpeed 算全局范数时 double -> float32
            # 会溢成 inf, 再经它的哨兵算术 (inf_or_nan * -1 + (not) * inf = -1 + nan) 变成
            # nan, 结果是 clip 失效 (nan > 1 为 False) 且 step 后 grad_norm=nan。这里用
            # reduce 时就算好的 norm_for_param_grads (offload 路径, double) 查每个参数:
            # 非有限, 或 > KERNEL_RL_GRAD_PARAM_NORM_MAX (默认 1e4; 正常整批 grad_norm 是
            # 0.05~0.6, 单个参数分片到 1e4 只能是坏值) 就跳过这一步。COLLECTIVE (all_reduce
            # MAX), 各 rank 结论一致。非 offload 路径没有这份记账, 只走上面的 has_overflow。
            return self._grad_param_norm_bad()
        names = []
        try:
            id2name = {p.ds_id: n for n, p in self.engine.module.named_parameters()
                       if hasattr(p, "ds_id")}
            parts = getattr(opt, "_DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition", {})
            for pid, g in parts.items():
                if g.numel() and not bool(torch.isfinite(g).all()):
                    names.append(id2name.get(pid, f"ds_id={pid}"))
                    if len(names) >= 16:
                        break
        except Exception as e:  # noqa: BLE001
            names = [f"<lookup failed: {e!r}>"]
        logger.warning(
            "[trainer] grad inf/nan before optim_step (rank %s): %d+ bad parameters on this rank, "
            "e.g. %s", os.environ.get("RANK", "0"), len(names), names,
        )
        return True

    def _grad_param_norm_bad(self) -> bool:
        """step 之前按参数查分片梯度范数 (见 _grad_has_inf_or_nan 里的说明)。COLLECTIVE。"""
        import torch.distributed as dist

        opt = self.engine.optimizer
        norms = getattr(opt, "norm_for_param_grads", None)
        limit = float(os.environ.get("KERNEL_RL_GRAD_PARAM_NORM_MAX", "1e4"))
        local_bad = 0.0
        names: list[str] = []
        try:
            if norms:
                ids = list(norms.keys())
                vals = torch.stack([torch.as_tensor(norms[i], dtype=torch.float64).reshape(()) for i in ids])
                vals = vals.to(self.device) if vals.device.type != "cuda" else vals
                flag = (~torch.isfinite(vals)) | (vals > limit)
                if bool(flag.any()):
                    local_bad = 1.0
                    id2name = {p.ds_id: n for n, p in self.engine.module.named_parameters()
                               if hasattr(p, "ds_id")}
                    v = vals.tolist()
                    for k, i in enumerate(ids):
                        if flag[k]:
                            names.append(f"{id2name.get(i, f'ds_id={i}')}={v[k]:.3g}")
                            if len(names) >= 16:
                                break
        except Exception as e:  # noqa: BLE001 -- 守卫自己不能把训练弄崩
            logger.warning("[trainer] per-parameter gradient norm check failed; continuing normally: %r", e)
            local_bad = 0.0
        t = torch.tensor([local_bad], device=self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if float(t.item()) <= 0:
            return False
        if local_bad:
            logger.warning(
                "[trainer] gradient shard norm non-finite or over the limit (rank %s, limit %g): %d+ of them, e.g. %s",
                os.environ.get("RANK", "0"), limit, len(names), names,
            )
        return True

    # ---- weight sync / sampling ------------------------------------------ #
    def _own_gpus_for_rank(self) -> list[str]:
        """Physical GPU ordinals of THIS rank's own sampling engine (inference-only).

        Each rank builds its own engine over a DISJOINT slice of ``tp`` cards
        (``KERNEL_RL_SAMPLER_TP``, default 1 = the legacy one-card-per-rank layout).
        With ``CUDA_VISIBLE_DEVICES=0,1,2,3``:

            tp=1, 4 ranks -> rank i gets [i]              (数据并行 4 份, 老行为)
            tp=4, 1 rank  -> rank 0 gets [0,1,2,3]        (一个 TP=4 引擎)
            tp=2, 2 ranks -> rank 0 [0,1], rank 1 [2,3]   (2 份 x TP=2)

        Why per-rank and not one shared engine like the training path: inference-only
        callers (rollout ①, eval ⑤) sample through ``SamplingClient`` directly, not
        through the collective ``DistributedSamplingClient`` -- every rank needs a real
        engine of its own. So parallelism is chosen by picking ``world_size x tp``.

        ``LOCAL_RANK`` indexes the process's VISIBLE devices, not physical ordinals,
        so translate through the ambient ``CUDA_VISIBLE_DEVICES``: with
        ``CUDA_VISIBLE_DEVICES=4,5,6,7`` local_rank 0 is physical card 4. The
        sampler's vLLM subprocess gets a fresh CUDA_VISIBLE_DEVICES and resolves it
        against ALL physical cards, so it must be handed physical ordinals.

        TP legality is NOT checked here -- vLLM needs ``num_attention_heads`` and
        ``num_key_value_heads`` both divisible by tp (QwQ-32B: 40 and 8 -> only
        1/2/4/8 are legal, 16 is NOT). The launcher scripts check it against the
        model's config.json so the failure lands before a multi-minute model load.
        """
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        tp = max(1, _env_int("KERNEL_RL_SAMPLER_TP", 1))
        visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g != ""]
        if not visible:  # every card visible -> index == physical ordinal
            return [str(local_rank * tp + i) for i in range(tp)]
        lo = local_rank * tp
        if lo + tp > len(visible):
            raise ValueError(
                f"CUDA_VISIBLE_DEVICES={visible} ({len(visible)} devices) are not enough for "
                f"LOCAL_RANK={local_rank} at KERNEL_RL_SAMPLER_TP={tp} each "
                f"-- requires world_size x tp <= the number of visible devices"
            )
        return visible[lo:lo + tp]

    def _sampler_gpus(self) -> list[str]:
        """Physical GPU ordinals of the SHARED sampling engine (training runs only).

        ``KERNEL_RL_SAMPLER_GPUS`` is the sampler card list (comma separated);
        the single shared vLLM engine runs tensor-parallel over ALL of them
        (TP = list length) and is owned by rank 0. Falls back to the legacy
        ``KERNEL_RL_INFER_GPUS`` name with a warning. Defaults to the card right
        after the training range.

        Inference-only runs (rollout) don't come through here -- with no trainer
        on the card, each rank's sampler simply takes the rank's own card.
        """
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        raw = os.environ.get("KERNEL_RL_SAMPLER_GPUS", "")
        if not raw and os.environ.get("KERNEL_RL_INFER_GPUS"):
            raw = os.environ["KERNEL_RL_INFER_GPUS"]
            logger.warning(
                "[sampler] KERNEL_RL_INFER_GPUS is deprecated; the whole list now "
                "backs ONE shared TP engine -- rename it to KERNEL_RL_SAMPLER_GPUS"
            )
        pool = [g for g in raw.split(",") if g != ""]
        return pool if pool else [str(world_size)]

    def _sampler_replicas(self) -> list[list[str]]:
        """训练路径下采样卡按 KERNEL_RL_SAMPLER_DP 切成的副本 (每台 TP = 卡数/DP)。所有
        rank 都能算 (只看 env), 权重同步按它决定谁进哪个 NCCL 组。"""
        gpus = self._sampler_gpus()
        dp = max(1, _env_int("KERNEL_RL_SAMPLER_DP", 1))
        if len(gpus) % dp != 0:
            raise ValueError(
                f"KERNEL_RL_SAMPLER_DP={dp} does not divide the number of sampler devices {len(gpus)} "
                f"(KERNEL_RL_SAMPLER_GPUS={','.join(gpus)})"
            )
        tp = len(gpus) // dp
        return [gpus[i * tp:(i + 1) * tp] for i in range(dp)]

    def _ensure_sampling_engine(self):
        """Build the vLLM engine.

        inference_only (rollout ①, eval ⑤): every rank builds its OWN engine over
        its own slice of ``KERNEL_RL_SAMPLER_TP`` cards (default 1 = the legacy
        one-card-per-rank layout). See ``_own_gpus_for_rank``.
        training: ONLY rank 0 builds -- one shared engine, tensor-parallel over
        ``_sampler_gpus()``; other ranks return None and reach it through the
        collective ``DistributedSamplingClient.sample_batch``.
        """
        if self._sampling_engine is None:
            from arl.models.sampling import SamplingEngine

            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            # 训练路径下 SAMPLER_GPUS 可以切成 KERNEL_RL_SAMPLER_DP 份副本 (各自 TP =
            # 卡数/DP, 卡不重叠), 都由 rank 0 起、rank 0 调度 (MultiSamplingEngine);
            # 默认 1 = 老行为 (一台 TP=卡数 的引擎)。推理路径 (每 rank 自己一台) 不切。
            dp = 1
            if self._inference_only:
                sampler_gpus = self._own_gpus_for_rank()
                self._sampling_cfg["tensor_parallel_size"] = len(sampler_gpus)
            else:
                if int(os.environ.get("RANK", "0")) != 0:
                    return None
                sampler_gpus = self._sampler_gpus()
                replicas_ = self._sampler_replicas()
                dp = len(replicas_)
                self._sampling_cfg["tensor_parallel_size"] = len(replicas_[0])
            tp = self._sampling_cfg["tensor_parallel_size"]
            replicas = [sampler_gpus[i * tp:(i + 1) * tp] for i in range(dp)]
            sampler_gpu = ",".join(sampler_gpus)
            # vLLM's EngineCore is spawned as a subprocess that reads the ambient
            # torch.distributed / torchelastic env at startup and builds its OWN
            # (trivial, world_size=1) process group. Under the deepspeed/torchrun
            # launcher that env describes the TRAINING group; if ANY of it leaks in,
            # the EngineCore tries to join the launcher's rendezvous and hangs/dies.
            # Fix (verified): fully CLEAR the launcher vars for the spawn so vLLM sees
            # a clean single-process env, and pin this rank's engine to its GPU.
            # The parent's CUDA context + training PG are already initialized, so
            # mutating the env now does not disturb this process's own training; we
            # restore everything afterward.
            LAUNCHER_VARS = [
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
                "ROLE_RANK", "ROLE_NAME", "ROLE_WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_ERROR_FILE",
                "TORCHELASTIC_RUN_ID", "TORCHELASTIC_MAX_RESTARTS",
                "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_SIGNALS_TO_HANDLE",
            ]
            touched = list(LAUNCHER_VARS) + [
                "CUDA_VISIBLE_DEVICES", "VLLM_HOST_IP", "NCCL_SHM_DISABLE",
                "NCCL_P2P_LEVEL",
            ]
            saved = {k: os.environ.get(k) for k in touched}
            try:
                if world_size > 1:
                    for k in LAUNCHER_VARS:
                        os.environ.pop(k, None)
                    os.environ["VLLM_HOST_IP"] = "127.0.0.1"  # isolated rendezvous
                if not self._inference_only:
                    # TP>1 workers allocate ~26MB NCCL shm buffers EACH; containers
                    # commonly cap /dev/shm at 64MB, and TP=4 x 26MB SIGBUS'd the
                    # whole job (train_ppo_20260723_080507.log). Force the engine's
                    # NCCL onto P2P/net instead. NOTE: NCCL does NOT negotiate
                    # transports -- every rank of a communicator must see the SAME
                    # value, so _ensure_weight_comm mirrors this setdefault on the
                    # trainer side (a mismatch corrupts the connect handshake and
                    # kills the weight sync, train_ppo_20260723_171644.log).
                    # Override by exporting NCCL_SHM_DISABLE=0 (both sides honor
                    # it) if the container has a big /dev/shm -- shm is faster
                    # than the socket fallback for the no-NVLink sync links.
                    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
                    # This PPU fabric supports P2P between EVERY card pair (the
                    # "SYS" label in topo -m is misleading): measured 2026-07-23,
                    # trainer->sampler NCCL broadcast is 43 GB/s via P2P vs
                    # 3.5 GB/s via the socket fallback. Default P2P policy
                    # refuses SYS paths, so raise the level. Must match the
                    # trainer side (mirrored in _ensure_weight_comm) -- a
                    # mismatch corrupts the transport handshake.
                    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
                # 副本逐台起 (from_engine_args 阻塞到引擎就绪): 每台 spawn 前把
                # CUDA_VISIBLE_DEVICES 钉到它自己那几张卡。DP=1 时就是原来那一台。
                engines = []
                for i, gpus in enumerate(replicas):
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)  # pin this replica's card(s)
                    logger.info(
                        f"[sampler] rank {os.environ.get('RANK','0')} sampler"
                        f"{f' replica {i}/{dp}' if dp > 1 else ''} -> GPU {','.join(gpus)} "
                        f"(TP={tp}, "
                        f"{'eager' if self._sampling_cfg.get('enforce_eager', True) else 'CUDA graph'})"
                    )
                    engines.append(SamplingEngine(
                        self.model_name,
                        dtype=self.dtype,
                        **self._sampling_cfg,
                    ))
                if dp > 1:
                    from arl.models.sampling import MultiSamplingEngine

                    self._sampling_engine = MultiSamplingEngine(engines)
                    logger.info(
                        f"[sampler] {dp} replicas x TP={tp} ready (devices {sampler_gpu}); requests go to whichever has the fewest in flight"
                    )
                else:
                    self._sampling_engine = engines[0]
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        return self._sampling_engine

    def _save_full_model(self, out_dir: str, background: bool = False,
                         on_done=None) -> str:
        """Consolidate ZeRO-3-sharded weights and write a standard HF checkpoint.

        Uses DeepSpeed's consolidation (gathers param-by-param to bound peak
        memory) then ``save_pretrained`` so vLLM can reload it via its normal
        checkpoint loader (which handles the HF->vLLM name/qkv-fusion mapping).

        ``background=True``: only the serialize+write half moves to a background
        thread (rank 0). The consolidation CANNOT overlap training -- it is a
        collective every rank must enter, and it reads the very params training
        mutates -- but its product is a full bf16 copy on rank 0's CPU
        (``state_dict on cpu``, per DeepSpeed), so writing that copy is safe
        while the next batch trains. ``on_done`` (if given) runs in the writer
        thread after a successful write -- callers publish the checkpoint there
        (index line) so readers never see half-written files. At most one write
        is in flight: entry joins the previous one first (each holds a model-
        sized CPU copy; also protects the reused ``_sync_dir`` overwrite path),
        which is also where a failed previous write raises.
        """
        import time as _t

        self.wait_pending_save()
        os.makedirs(out_dir, exist_ok=True)
        is_rank0 = int(os.environ.get("RANK", "0")) == 0
        # Consolidated bf16 state dict, materialized on rank 0.
        _t0 = _t.time()
        state_dict = self.engine._zero3_consolidated_16bit_state_dict()
        _t1 = _t.time()
        # With a value head the module is CausalLMWithValueHead: save ONLY the base
        # CausalLM (strip the "base." prefix) so vLLM's loader sees a
        # standard checkpoint. Pure-policy runs save the module directly.
        module_to_save = self.engine.module
        if self._has_value_head:
            module_to_save = self.engine.module.base
            # Consolidation materializes the dict on rank 0 ONLY (None elsewhere);
            # non-rank0 ranks still had to enter the collective call above.
            if state_dict is not None:
                state_dict = {
                    k[len("base."):]: v for k, v in state_dict.items() if k.startswith("base.")
                }
        if not is_rank0:
            # Nothing to write (state_dict is None here); on_done belongs to the
            # rank that owns the files/index, i.e. rank 0.
            return out_dir

        def _write() -> None:
            # save_pretrained only serializes the passed-in CPU state_dict (plus
            # the static config); it does not read live params, so it cannot race
            # with the training step running concurrently in the main thread.
            module_to_save.save_pretrained(
                out_dir, state_dict=state_dict, safe_serialization=True
            )
            self._tokenizer.save_pretrained(out_dir)
            if getattr(self, "_mm_wrapped", False):
                self._recompose_mm_config(out_dir)
            logger.info(
                f"[sync-timing] consolidate(NCCL gather)={_t1 - _t0:.1f}s  "
                f"save_pretrained(serialize+write)={_t.time() - _t1:.1f}s"
                + (" (background thread)" if background else "")
            )
            if on_done is not None:
                on_done()

        if background:
            def _guarded() -> None:
                try:
                    _write()
                except BaseException as e:  # noqa: BLE001
                    # Surface at the next wait_pending_save (next save / teardown)
                    # instead of dying silently in the thread.
                    self._save_error = e
                    logger.error(f"[checkpoint] background write failed {out_dir}: {e}")
            # Non-daemon on purpose: if the main thread crashes mid-write, the
            # interpreter waits for the writer, so the checkpoint on disk is
            # complete instead of truncated.
            self._save_thread = threading.Thread(target=_guarded, name="ckpt-writer")
            self._save_thread.start()
        else:
            _write()
        return out_dir

    def _recompose_mm_config(self, out_dir: str) -> None:
        """多模态包装模型: 把刚存的纯文本 config.json 重组为底座同款复合 config。

        ④″ 的训练模块按 text_config 加载 (见 __init__ 的 _mm_wrapped), 所以
        save_pretrained 写出的是文本塔 config (qwen3_5_moe_text /
        *ForCausalLM) —— 但权重名仍是 model.language_model.* 的 checkpoint
        格式, 而 vLLM 0.19 对这一族架构 (含 *ForCausalLM) 一律走多模态
        processor, 要求带 vision_config 的复合 config, 纯文本 config 在
        renderer 初始化时直接 TypeError (2026-08-31 ⑤′ 离线评估实测)。这里
        用底座的复合 config 作外壳、刚写出的文本 config 塞回 text_config
        (保留 use_cache 等运行时改动), 并补上 processor 的配置文件, 让
        checkpoint 目录能被 vLLM / ⑤′ 直接加载。resume 时复合 config 走
        __init__ 的 get_text_config 路径, 与直接加载底座完全一致。
        """
        import json
        import shutil

        base_dir = self.model_name.split(":")[0]
        with open(os.path.join(base_dir, "config.json")) as f:
            composite = json.load(f)
        out_path = os.path.join(out_dir, "config.json")
        with open(out_path) as f:
            text = json.load(f)
        composite["text_config"] = {
            k: v for k, v in text.items()
            if k not in ("architectures", "transformers_version")
        }
        with open(out_path, "w") as f:
            json.dump(composite, f, indent=2, ensure_ascii=False)
        for fname in (
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "processor_config.json",
        ):
            src = os.path.join(base_dir, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy(src, dst)

    def wait_pending_save(self) -> None:
        """Join the in-flight background checkpoint write, if any (rank 0).

        Re-raises the writer's failure here: a broken checkpoint write becomes
        loud at the next save (or teardown) rather than staying a silent
        half-written directory.
        """
        t = self._save_thread
        if t is not None:
            t.join()
            self._save_thread = None
        if self._save_error is not None:
            e, self._save_error = self._save_error, None
            raise RuntimeError(f"the previous checkpoint's background write failed: {e}") from e

    def _write_meta(self, out_dir: str) -> None:
        import json

        if int(os.environ.get("RANK", "0")) == 0:
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump({"model_name": self.model_name}, f)

    def save_policy_weights(self, out_dir: str, background: bool = False,
                            on_done=None) -> str:
        """Public: save the full policy weights (sampler weights) + run metadata.

        ``background``/``on_done``: see ``_save_full_model``. meta.json is tiny
        and written inline either way -- "checkpoint ready" is signalled by
        ``on_done`` (the index line), not by any file's presence.
        """
        self._require_engine("save_policy_weights")
        self._save_full_model(out_dir, background=background, on_done=on_done)
        self._write_meta(out_dir)
        # The critic lives outside the engine, so the policy-only ("sampler") checkpoint
        # would otherwise never contain it. It is small (~33M params + Adam), so keep it
        # next to policy/ in every checkpoint (rank 0 writes; replicas are identical).
        if self._has_value_head:
            self._save_value_head(os.path.dirname(os.path.normpath(out_dir)))
        return out_dir

    def save_optimizer_state(self, out_dir: str) -> str:
        """Save DeepSpeed optimizer/engine state for exact resume."""
        self._require_engine("save_optimizer_state")
        self.engine.save_checkpoint(out_dir, tag="ds_state")
        self._save_value_head(out_dir)
        return out_dir

    def load_policy_weights(self, weights_dir: str) -> None:
        """Load a full HF checkpoint's weights into the live (ZeRO-3) model."""
        self._require_engine("load_policy_weights")
        from transformers import AutoModelForCausalLM
        from deepspeed import zero

        if not os.path.isdir(weights_dir):
            logger.warning(f"[trainer] no policy weights at {weights_dir}; skipping load")
            return
        # Load the checkpoint onto CPU, then scatter into the sharded params.
        src = AutoModelForCausalLM.from_pretrained(weights_dir, torch_dtype=self.dtype)
        src_state = src.state_dict()
        del src
        module = self.engine.module
        params = list(module.named_parameters())
        # With a value head, param names are "base.*"; the checkpoint has CausalLM
        # names (no prefix). Strip "base." to match (the value head is not in here).
        strip = "base." if self._has_value_head else ""
        with zero.GatheredParameters([p for _, p in params], modifier_rank=0):
            if int(os.environ.get("RANK", "0")) == 0:
                for name, p in params:
                    key = name[len(strip):] if strip and name.startswith(strip) else name
                    if key in src_state:
                        p.data.copy_(src_state[key].to(p.device, p.dtype))

    async def _sync_to_sampler(self, weights_dir: str) -> None:
        """Reload ``weights_dir`` into the vLLM engine, in place.

        Barrier first: the weights are written by rank 0 only, so the write must
        complete before any reload. Training runs (shared engine) then reload on
        rank 0 only; inference-only runs reload every rank's own replica. A final
        barrier keeps ranks from sampling before the reload lands.
        """
        from arl.utils.distributed import barrier

        barrier()
        engine = self._ensure_sampling_engine()
        import time as _t

        _tr = _t.time()
        if engine is not None:
            if getattr(self, "_mm_wrapped", False):
                # 磁盘路径存的是 HF 文本类的 checkpoint (model.layers...), 而 vLLM 的
                # 多模态包装类只认 checkpoint 格式的 key (model.language_model.*) ——
                # NCCL 路径已在 worker 侧改名 (sampling._nccl_update_weights_worker),
                # 这条回退路没改。真要走它得先把改名搬到这里。
                logger.warning(
                    "[sync] multimodal wrapper model falls back to disk-based sync: the weight key prefixes do not line up with vLLM's "
                    "mapper, so a reload may refuse or silently miss weights -- use the default NCCL sync "
                    "(KERNEL_RL_NCCL_SYNC=1), or add key renaming to this path first."
                )
            await engine.reload_weights(weights_dir)
            # 权重换了, 前缀 KV 没换 —— 必须显式丢掉, 否则下一批采样会命中旧策略算的 KV
            # (prefix caching 在 vLLM V1 默认开着)。见 sampling.reset_prefix_cache。
            await engine.reset_prefix_cache()
        barrier()
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info(
                f"[sync-timing] reload_weights(read+process)={_t.time() - _tr:.1f}s"
            )

    async def _ensure_weight_comm(self):
        """Build the trainer-side NCCL communicators for the sampler replicas.

        COLLECTIVE (every training rank calls it; ends with a barrier). Per replica
        ONE group of {owner training rank = group rank 0, TP worker i = 1+i}, replica
        r rendezvous on port+r. The owner of replica r is chosen by NVLink topology
        (``_sync_owner_ranks``; env KERNEL_RL_SAMPLER_SYNC_RANKS overrides) -- rank 0
        still dispatches the worker-side joins (it owns the engines), but the
        trainer-side ``StatelessProcessGroup.create`` runs on the owner's process.
        DP=1 with no NVLink into the sampler cards degenerates to the old single
        group owned by rank 0.
        """
        if self._weight_comms is not None:
            return
        import asyncio

        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        from arl.utils.distributed import barrier

        rank = int(os.environ.get("RANK", "0"))
        world_train = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        replicas = self._sampler_replicas()
        visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g != ""]
        owners = _sync_owner_ranks(replicas, world_train, visible,
                                   override=os.environ.get("KERNEL_RL_SAMPLER_SYNC_RANKS", ""))
        tp = len(replicas[0])
        world = 1 + tp
        base_port = int(os.environ.get("KERNEL_RL_NCCL_SYNC_PORT", "30500"))
        host = "127.0.0.1"
        # The engine workers inherited NCCL_SHM_DISABLE=1 / NCCL_P2P_LEVEL=SYS at spawn
        # (see _ensure_sampling_engine) but that env was RESTORED in rank 0 and never
        # set on the other ranks. NCCL does NOT negotiate transports: each rank reads
        # its own env, and a mismatch inside one communicator corrupts the connect
        # handshake -- the trainer picks SHM for the (no-NVLink) trainer<->worker
        # links, parses the worker's NET handle as an shm path and dies attaching
        # "/dev/shm/nccl-" (train_ppo_20260723_171644.log). Mirror the exact
        # setdefaults the workers saw on EVERY owner; the training group's NCCL
        # already read its (separate) copy of the env and is unaffected.
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")  # P2P works on every pair on .87

        # rank 0 dispatches every replica's worker-side join up front (each blocks in
        # its rendezvous until that replica's owner creates group rank 0); the owners
        # then create their side off the event loop so rank 0's loop keeps driving
        # the RPCs. No cycle: a worker join depends only on its owner, an owner on
        # nothing.
        worker_tasks = []
        if rank == 0:
            for r, rep in enumerate(self._sampling_engine.engines):
                worker_tasks.append(asyncio.ensure_future(
                    rep.init_weight_update_group(host, base_port + r, world_size=world)))
        comms: dict[int, object] = {}
        loop = asyncio.get_event_loop()
        for r, owner in enumerate(owners):
            if owner != rank:
                continue
            pg = await loop.run_in_executor(
                None, StatelessProcessGroup.create, host, base_port + r, 0, world
            )
            # NCCL comm on this rank's TRAINING card (where the gathered params live).
            comms[r] = PyNcclCommunicator(pg, device=torch.device(f"cuda:{local_rank}"))
        for t in worker_tasks:
            await t
        barrier()
        self._weight_comms = comms
        self._sync_owners = owners
        if rank == 0:
            mat = _nvlink_matrix()
            desc = "; ".join(
                f" replicas{r} (GPU {','.join(rep)}) <- rank {o} (GPU {visible[o] if o < len(visible) else o}, "
                f"NVLink {sum(mat.get(visible[o] if o < len(visible) else str(o), {}).get(g, 0) for g in rep)})"
                for r, (rep, o) in enumerate(zip(replicas, owners))
            )
            logger.info(
                f"[sync] disaggregated NCCL weight-update group{'s' if len(replicas) > 1 else ''} ready "
                f"({len(replicas)} x (1 trainer + {tp} TP workers), ports {base_port}..{base_port + len(replicas) - 1}); "
                f"broadcast source chosen by NVLink topology: {desc}"
            )

    async def _sync_to_sampler_nccl(self, budget_gb: float = 6.0) -> None:
        """Push weights into the sampler replicas over cross-GPU NCCL broadcasts.

        Every training rank participates in the ZeRO-3 ``GatheredParameters`` (a
        training-group collective); replica r's OWNER rank (see _ensure_weight_comm)
        then broadcasts the gathered full tensors on comm r, and each TP worker
        receives them and ``load_weights`` TP-slices them locally. rank 0 (engine
        owner) dispatches the workers' recv RPCs. Owners of different replicas
        broadcast concurrently, so DP replicas cost max(), not sum(). Params are
        streamed in ~``budget_gb`` groups: gather a group, broadcast, sync the
        stream, release, next -- peak GPU stays bounded (no full gathered copy).
        """
        import asyncio
        import time as _t

        import deepspeed

        from arl.utils.distributed import barrier

        rank = int(os.environ.get("RANK", "0"))
        is_rank0 = rank == 0
        barrier()
        await self._ensure_weight_comm()
        my_reps = [r for r, o in enumerate(self._sync_owners) if o == rank]

        named = list(self.engine.module.named_parameters())
        if self._has_value_head:
            # CausalLMWithValueHead names params "base.*"; vLLM's loader
            # expects plain CausalLM names. Strip "base."
            # (mirrors the disk-sync path in _save_full_model).
            named = [
                (name[len("base."):], p) for name, p in named if name.startswith("base.")
            ]
        groups: list[list] = []
        cur: list = []
        cur_bytes = 0
        for name, p in named:
            nb = getattr(p, "ds_numel", p.numel()) * p.element_size()
            if cur and cur_bytes + nb > budget_gb * 1e9:
                groups.append(cur)
                cur, cur_bytes = [], 0
            cur.append((name, p))
            cur_bytes += nb
        if cur:
            groups.append(cur)

        stream = torch.cuda.current_stream()
        _t0 = _t.time()
        for group in groups:
            # COLLECTIVE across the training group: every rank must enter the
            # gather even if it neither dispatches nor broadcasts.
            with deepspeed.zero.GatheredParameters([p for _, p in group], modifier_rank=None):
                if not is_rank0 and not my_reps:
                    continue
                rpc_tasks = []
                if is_rank0:
                    metadata = [
                        (name, str(p.dtype).split(".")[-1], tuple(p.shape)) for name, p in group
                    ]
                    # Dispatch the workers' recv for EVERY replica first (they block
                    # on the NCCL recvs until their owner posts the broadcasts);
                    # sleep(0) flushes the dispatch before anything blocks below.
                    rpc_tasks = [
                        asyncio.ensure_future(rep.update_weights_nccl(metadata))
                        for rep in self._sampling_engine.engines
                    ]
                    await asyncio.sleep(0)
                # Owners post the matching broadcasts (async stream enqueues) for
                # their replicas. Sync the stream before releasing the gathered
                # params so NCCL can't read freed memory.
                for r in my_reps:
                    comm = self._weight_comms[r]
                    for _, p in group:
                        comm.broadcast(p.data, src=0, stream=stream)
                for t in rpc_tasks:
                    await t
                if my_reps:
                    stream.synchronize()
        if is_rank0:
            # 同 _sync_to_sampler: 权重进去了, 前缀 KV 还是旧的, 显式丢掉。放在 barrier
            # 之前 —— 别的 rank 必须等它清完才能开始下一批采样。
            await self._sampling_engine.reset_prefix_cache()
        # No rank may start sampling before the workers have the new weights.
        barrier()
        if is_rank0:
            logger.info(
                f"[sync-timing] NCCL direct sync={_t.time() - _t0:.1f}s "
                f"({len(groups)} groups, {len(named)} params, {len(self._sync_owners)} replica(s), "
                f"broadcast source rank {self._sync_owners})"
            )

    async def save_weights_and_get_sampling_client_async(self, out_dir: str | None = None):
        from arl.models.sampling import SamplingClient

        self._sync_counter += 1
        if self._inference_only:
            # Nothing trains, so the weights vLLM already loaded ARE the policy. The
            # save+reload round-trip below would consolidate the base model to disk
            # and read it straight back -- a no-op costing a full model write.
            return SamplingClient(self._ensure_sampling_engine())
        if self._nccl_sync:
            # Cross-GPU NCCL broadcast: trainer cards -> shared sampler TP workers.
            await self._sync_to_sampler_nccl()
        else:
            # Overwrite a single dir each step: reload below fully consumes it before
            # the next sync, so one model copy on disk is enough (bounds disk usage).
            weights_dir = out_dir or self._sync_dir
            self._save_full_model(weights_dir)
            await self._sync_to_sampler(weights_dir)
        from arl.models.dist_sampling import DistributedSamplingClient

        return DistributedSamplingClient(self._ensure_sampling_engine())

    async def create_sampling_client_async(self, path: str):
        """Reload the policy weights at ``path`` into the sampler and return a client."""
        from arl.models.dist_sampling import DistributedSamplingClient

        await self._sync_to_sampler(path)
        return DistributedSamplingClient(self._ensure_sampling_engine())

    # ---- checkpoint state ------------------------------------------------- #
    def save_state(self, out_dir: str) -> str:
        """Save full policy weights + DeepSpeed optimizer/engine state for resuming."""
        self._require_engine("save_state")
        policy_dir = os.path.join(out_dir, "policy")
        self._save_full_model(policy_dir)
        self.engine.save_checkpoint(out_dir, tag="ds_state")
        self._save_value_head(out_dir)
        return out_dir

    def load_state(self, in_dir: str) -> None:
        self._require_engine("load_state")
        self.engine.load_checkpoint(in_dir, tag="ds_state")
        self._load_value_head(in_dir)

    def shutdown(self) -> None:
        """Release the vLLM sampling engine (stops its EngineCore subprocesses).

        Best-effort and idempotent; called during graceful teardown so the
        process can exit instead of hanging on lingering engine procs.
        """
        # Backstop: don't tear the engine down under a still-writing checkpoint.
        # (Normally the final sync save already joined it.) Never let a failed
        # write block the shutdown itself.
        try:
            self.wait_pending_save()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[trainer] background checkpoint write failed (teardown continues): {e}")
        if self._sampling_engine is not None:
            self._sampling_engine.shutdown()
            self._sampling_engine = None
