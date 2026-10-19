"""PPO / GRPO clipped importance-sampling loss.

Matches the semantics of Tinker's ``loss_fn="ppo"`` as consumed by
``arl.algorithms.loop``: per-token advantages and sampler (old) logprobs are
supplied on each Datum; non-action tokens carry advantage 0 (so masking is
implicit). ``loss_fn_config`` supplies asymmetric clip thresholds.

This module is pure (tensors in, tensors out) so it can be unit-tested without a
model or DeepSpeed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

DEFAULT_CLIP_LOW = 0.2
DEFAULT_CLIP_HIGH = 0.2

# Rows (sequence positions) whose float32 logits are materialized at once inside
# gather_target_logprobs; the full [T, V] never exists in float32 at any point.
# Peak extra memory is ~2 x CHUNK x V x 4B (two float32 temporaries are live at
# once in backward). Same value as reference_policy._LOGIT_CHUNK, which does this
# for the no-grad pi_ref path.
#
# Measured on one PPU card, T=19000, V=152064, bf16, fwd+bwd -- the op is memory
# bandwidth bound, not launch bound, so a small chunk costs essentially nothing:
#     chunk   512: 119.9 ms   extra peak  0.87 GB
#     chunk  1024: 117.5 ms   extra peak  1.74 GB
#     chunk  4096: 115.7 ms   extra peak  6.96 GB
#     chunk  8192: 115.7 ms   extra peak 13.92 GB
# 16x the chunk buys 3.5% of runtime, so take the memory instead.
_LOGP_CHUNK = max(1, int(os.environ.get("KERNEL_RL_LOGP_CHUNK", "512")))


class _ChunkedTargetLogprobs(torch.autograd.Function):
    """log p(target) from ``[T, V]`` logits without a full float32 [T, V].

    The straightforward ``log_softmax(logits.float()).gather(...)`` materializes
    two float32 [T, V] tensors (the cast and the log_softmax output) and autograd
    keeps the second one alive until backward -- at T=19k, V=152k that is 11.6GB
    saved plus another 11.6GB allocated for its gradient, which is what used to
    pin KERNEL_RL_MICRO_BSZ at 1 on a 98GB card.

    Here forward saves only the (bf16) logits and recomputes softmax per chunk in
    backward, so float32 only ever exists CHUNK rows at a time. Measured peak for
    one fwd+bwd at T=19000, V=152064 (bf16, one PPU card): 37.7GB -> 11.6GB; at
    mb=2, 75.3GB -> 22.4GB. The remaining unavoidable [T, V] allocations are the
    incoming bf16 logits and the bf16 gradient handed back to the lm_head matmul;
    removing those too would mean fusing the lm_head itself (cut-cross-entropy
    style), which needs the hidden states and the ZeRO-3-sharded lm_head weight.

    Cost is the extra softmax pass: 72ms -> 118ms for this op at T=19000, i.e.
    ~+46ms per datum, ~+1.5s per rank per batch against a ~3000s batch.

    Numerics (vs a float64 reference on identical inputs): forward differs from
    the log_softmax path by ~1e-6 absolute on logprobs -- float32 ulp noise, not
    bit-identical, and four orders of magnitude below the bf16 quantization of the
    logits feeding it. In bf16 the backward error is identical to the old path to
    three significant figures (7.79e-03, which is bf16 rounding of the returned
    gradient, not the algorithm).
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(logits, target_tokens)
        out = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
        for i in range(0, logits.shape[0], _LOGP_CHUNK):
            sl = slice(i, i + _LOGP_CHUNK)
            z = logits[sl].float()
            # log p_y = z_y - logsumexp(z); logsumexp subtracts the row max, so this
            # is as numerically stable as log_softmax (not bit-identical to it --
            # different summation order, ~1e-6 apart in float32; see the class
            # docstring for the measurement).
            out[sl] = z.gather(-1, target_tokens[sl, None]).squeeze(-1) - z.logsumexp(-1)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        logits, target_tokens = ctx.saved_tensors
        # d(log p_y)/d z_i = onehot_y(i) - softmax(z)_i
        grad = torch.empty_like(logits)
        for i in range(0, logits.shape[0], _LOGP_CHUNK):
            sl = slice(i, i + _LOGP_CHUNK)
            g = grad_out[sl].float().unsqueeze(-1)
            p = torch.softmax(logits[sl].float(), dim=-1)
            p.mul_(-g)
            p.scatter_add_(-1, target_tokens[sl, None], g)
            grad[sl] = p.to(logits.dtype)
        return grad, None


@dataclass(frozen=True)
class SamplingReplay:
    """采样分布回放 (Cognition SWE-1.7 的熵保持做法) 的采样参数: 和 rollout 时
    vLLM 用的 temperature / top_k / top_p 一模一样。

    rollout 时 vLLM 先除 temperature, 再留 top_k, 再在 top_k 里留 top_p 前缀, 只从这个
    "kept set" 里采样; trainer 这边在**同一个集合**上重新归一化 log p, 集合外的 token
    梯度为 0。于是负 advantage 压低一个被采样的 token 时, 挪出来的概率只在 kept set 内部
    重新分配, 不会漏进从来采不到、也就永远不会被纠正的长尾 (空回复/乱码的来源); 集合
    只有一个 token 的位置 (top-1 自己就超过 top_p) 概率恒为 1, 梯度自动为 0。

    集合由 trainer 用**自己的** logits 重算 (近似回放), 不从 vLLM 传回 —— 两边 logits
    只差 bf16 / MoE 路由噪声, 集合只在边界 token 上可能不同; 采样到的 token 不在 trainer
    的集合里时强制并进去 (否则 log p = -inf), 见 ``gather_target_logprobs`` 的 stats。

    ``top_k <= 0`` (vLLM 的"不截断") 时用 ``_REPLAY_TOPK_CAP`` 兜一个上限: 全词表排序
    太贵, 而 top_p=0.95 的 nucleus 超过这个数的位置极少。
    """

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


_REPLAY_TOPK_CAP = max(1, int(os.environ.get("KERNEL_RL_REPLAY_TOPK_CAP", "1024")))


def _replay_kept(z: torch.Tensor, replay: SamplingReplay):
    """z = logits/T (float32, [R, V]) -> (topv [R,K], topi [R,K], kept [R,K] bool).

    和 vLLM ``apply_top_k_top_p_pytorch`` 同一规则: 先 top_k, 再在 top_k 截断后的分布上
    按降序累加, "排在它前面的概率和 < top_p" 的都留 (等价于 vLLM 的升序 cumsum > 1-p),
    第一名恒留。并列值的处理不同 (vLLM 留下所有 >= 第 k 名的值) —— 可忽略。
    """
    V = z.shape[-1]
    k = replay.top_k if replay.top_k and replay.top_k > 0 else _REPLAY_TOPK_CAP
    k = min(k, V)
    topv, topi = z.topk(k, dim=-1)  # sorted descending
    if replay.top_p < 1.0:
        p = torch.softmax(topv, dim=-1)
        excl = torch.cumsum(p, dim=-1) - p
        kept = excl < replay.top_p
    else:
        kept = torch.ones_like(topv, dtype=torch.bool)
    kept[:, 0] = True
    return topv, topi, kept


def _replay_set(z, tgt, replay):
    """-> (idx [R,K+1], set_logits [R,K+1] with -inf outside the set, in_set [R] bool).

    最后一列是 target 自己: target 已在 kept set 里时这一列置 -inf (不重复计), 不在时
    这一列就是把它强制并进集合的那一项。
    """
    topv, topi, kept = _replay_kept(z, replay)
    zy = z.gather(-1, tgt[:, None])
    in_set = ((topi == tgt[:, None]) & kept).any(-1)
    ninf = torch.tensor(float("-inf"), device=z.device, dtype=z.dtype)
    set_logits = torch.cat([torch.where(kept, topv, ninf),
                            torch.where(in_set[:, None], ninf, zy)], dim=-1)
    idx = torch.cat([topi, tgt[:, None]], dim=-1)
    return idx, set_logits, in_set, kept


def _in_backward() -> bool:
    # checkpoint (HEAD_CHUNKS 路径) 会在 backward 里把 forward 重跑一遍; 统计只记第一遍。
    try:
        return torch._C._current_graph_task_id() != -1
    except Exception:  # noqa: BLE001 -- 私有 API 不在时宁可重复计也不炸
        return False


class _ChunkedReplayTargetLogprobs(torch.autograd.Function):
    """log p(target) renormalized over the rollout's kept set (see ``SamplingReplay``).

    Same memory discipline as ``_ChunkedTargetLogprobs``: forward saves only the bf16
    logits, float32 exists CHUNK rows at a time, backward recomputes the set per chunk.
    Gradient w.r.t. the logits is ``(onehot_y - q) / T`` on the set's entries (q = the
    renormalized in-set distribution) and exactly 0 everywhere else.
    """

    @staticmethod
    def forward(ctx, logits, target_tokens, replay: SamplingReplay, stats: dict | None):
        ctx.save_for_backward(logits, target_tokens)
        ctx.replay = replay
        inv_t = 1.0 / max(float(replay.temperature), 1e-6)
        out = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
        record = stats is not None and not _in_backward()
        for i in range(0, logits.shape[0], _LOGP_CHUNK):
            sl = slice(i, i + _LOGP_CHUNK)
            z = logits[sl].float() * inv_t
            tgt = target_tokens[sl]
            _idx, set_logits, in_set, kept = _replay_set(z, tgt, replay)
            out[sl] = z.gather(-1, tgt[:, None]).squeeze(-1) - set_logits.logsumexp(-1)
            if record:
                stats.setdefault("in_set", []).append(in_set.detach())
                stats.setdefault("n_kept", []).append(kept.sum(-1).detach())
        return out

    @staticmethod
    def backward(ctx, grad_out):
        logits, target_tokens = ctx.saved_tensors
        replay = ctx.replay
        inv_t = 1.0 / max(float(replay.temperature), 1e-6)
        grad = torch.zeros_like(logits)
        for i in range(0, logits.shape[0], _LOGP_CHUNK):
            sl = slice(i, i + _LOGP_CHUNK)
            z = logits[sl].float() * inv_t
            tgt = target_tokens[sl]
            idx, set_logits, _in_set, _kept = _replay_set(z, tgt, replay)
            g = grad_out[sl].float().unsqueeze(-1)
            q = torch.softmax(set_logits, dim=-1)  # 0 outside the set
            gz = torch.zeros(z.shape, dtype=torch.float32, device=z.device)
            gz.scatter_add_(-1, idx, -g * q)
            gz.scatter_add_(-1, tgt[:, None], g)
            grad[sl] = (gz * inv_t).to(logits.dtype)
        return grad, None, None, None


def gather_target_logprobs(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    replay: SamplingReplay | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    """Per-token log-probability of ``target_tokens`` under ``logits``.

    Args:
        logits: ``[..., T, V]`` next-token logits.
        target_tokens: ``[..., T]`` int64 target ids.
        replay: 给了就按采样分布回放算 (temperature + top_k/top_p kept set 内重新归一,
            见 ``SamplingReplay``); None = 老口径, 全词表、不除温度。
        stats: 仅 replay 时有用: 往 ``stats["in_set"]`` / ``stats["n_kept"]`` 各追加一段
            逐 token 的 [N] 张量 (采样 token 在不在 trainer 重算的集合里 / 集合大小),
            按调用顺序拼起来和返回值逐位对齐; 调用方自己按 active 掩码统计。
    Returns:
        ``[..., T]`` log-probs (float32).
    """
    tgt = target_tokens.long()
    flat_logits = logits if logits.dim() == 2 else logits.reshape(-1, logits.shape[-1])
    flat_tgt = tgt if logits.dim() == 2 else tgt.reshape(-1)
    if replay is None:
        # Leading batch dims are folded into the row axis so the chunking still bounds
        # memory by rows rather than by whole sequences.
        flat = _ChunkedTargetLogprobs.apply(flat_logits, flat_tgt)
    else:
        flat = _ChunkedReplayTargetLogprobs.apply(flat_logits, flat_tgt, replay, stats)
    return flat if logits.dim() == 2 else flat.view(tgt.shape)


def ppo_surrogate(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
) -> torch.Tensor:
    """Per-token clipped surrogate objective (to be MAXIMIZED).

    ``min(r*A, clip(r, 1-clip_low, 1+clip_high)*A)`` where ``r = exp(new-old)``.
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    return torch.minimum(ratio * advantages, clipped * advantages)


@dataclass
class PPOStats:
    loss: float
    mean_ratio: float
    clip_frac: float
    approx_kl: float
    n_active_tokens: int


def ppo_loss_for_batch(
    new_logprobs_list: list[torch.Tensor],
    old_logprobs_list: list[torch.Tensor],
    advantages_list: list[torch.Tensor],
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
    const_len: float = 1.0,
) -> tuple[torch.Tensor, PPOStats]:
    """Aggregate the clipped surrogate across a batch of variable-length datums.

    Dr-GRPO normalization: each response's summed surrogate is divided by a FIXED
    constant ``const_len`` (the generation budget), NOT its own active-token count,
    then averaged over the number of active responses (1/G) -- objective
    ``(1/G) Σ_i (1/const_len) Σ_t surrogate``. This removes the per-response length
    bias of original GRPO's 1/|o_i|. Mirrors the real training path in
    ``training_client.forward_backward`` (which passes the generation budget as the
    constant). Callers that want the old original-GRPO 1/|o_i| can pass the response
    length as ``const_len`` per response; the default 1.0 leaves the surrogate as a
    plain token sum.

    Returns ``(loss, stats)`` where ``loss`` is the scalar to backprop (negative
    surrogate) and keeps the graph, and ``stats`` are detached diagnostics.
    """
    total_surrogate = None
    n_active = 0
    n_resp = 0
    ratio_sum = 0.0
    clip_count = 0
    kl_sum = 0.0

    for new_lp, old_lp, adv in zip(new_logprobs_list, old_logprobs_list, advantages_list):
        active = adv != 0
        n = int(active.sum().item())
        surrogate_tok = ppo_surrogate(new_lp, old_lp, adv, clip_low, clip_high)
        # Dr-GRPO: divide by a fixed constant (not this response's token count).
        s = surrogate_tok.sum() / const_len if n > 0 else surrogate_tok.sum() * 0.0
        total_surrogate = s if total_surrogate is None else total_surrogate + s
        n_active += n
        if n > 0:
            n_resp += 1
            with torch.no_grad():
                ratio = torch.exp(new_lp - old_lp)[active]
                ratio_sum += float(ratio.sum().item())
                clip_count += int(
                    ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)).sum().item()
                )
                kl_sum += float((old_lp - new_lp)[active].sum().item())

    denom = max(1, n_active)  # token count -- for ratio/clip/kl diagnostics only
    if total_surrogate is None:
        # No datums; return a zero loss that still has a grad path is impossible,
        # so return a detached zero (caller should skip empty batches).
        loss = torch.zeros((), requires_grad=True)
    else:
        # Average the per-response means over the number of responses (GRPO 1/G).
        loss = -(total_surrogate / max(1, n_resp))

    stats = PPOStats(
        loss=float(loss.detach().item()),
        mean_ratio=ratio_sum / denom,
        clip_frac=clip_count / denom,
        approx_kl=kl_sum / denom,
        n_active_tokens=n_active,
    )
    return loss, stats
