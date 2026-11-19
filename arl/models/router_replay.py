"""MoE 路由回放 (router replay, 2026-10-20): 训练前向照抄采样时 vLLM 选的专家。

为什么: MoE 的 top-k 路由是离散的, 训练端 (HF, bf16, ZeRO-3, 打包/梯度检查点) 和采样端
(vLLM, TP, fused kernel) 的隐状态只要差一点, 贴着边界的 token 就会选到不同的专家 ——
同一个 token 在两边走的是**不同的子网络**, old logprob (vLLM 算) 和新 logprob (trainer
算) 之间多出一块和策略更新无关的差, 被 PPO ratio / clip 当成 off-policy 漂移。它随训练
往上走 (qwen36_grpo_1018_v1 的 replay_outside_frac 0.0009 -> 0.0015), 和熵上涨互相放大。
回放之后两边在同一组专家上算, 剩下的差只来自数值误差 (R3, arXiv:2510.11370 的做法)。

怎么做:
  * 采样端 vLLM 开 ``enable_return_routed_experts``, 每条请求结束时回传
    ``[prompt+output-1, num_layers, top_k]`` 的逻辑专家 id (EPLB 映射之前);
    ``SamplingClient`` 存成 uint8 挂在 ``SampledSequence.routed_experts`` 上, 一路带到
    Datum 的 ``loss_fn_inputs["routed_experts"]`` (行数 = datum 的 input 长度)。
  * 训练端 ``install`` 把每层 router (``*.layers.<i>.mlp.gate``, HF 的
    ``Qwen3_5MoeTopKRouter``) 的 forward 换掉: 有回放索引时 top-k 的**下标**用回放的,
    **权重**仍是 trainer 自己的 softmax 在这些下标上取值再归一 —— 梯度照常流回隐状态
    (和 router 自己, 如果它没冻结)。行里第一个 id < 0 = 这个位置没有回放 (pad / 缺数据),
    退回 trainer 自己的 top-k。
  * ``replaying(idx)`` 在一个 chunk 的 forward **和 backward** 期间都要挂着: 梯度检查点
    在 backward 里重算 forward, 重算必须走同一组专家, 否则 backward 和 forward 不是同一张图。

开关: 环境变量 ``KERNEL_RL_ROUTER_REPLAY=1`` (ppo_loop 按 PPOConfig.router_replay 设),
采样引擎和 trainer 读同一个变量, 两边同开同关。

现场指标 (``pop_stats``): 回放覆盖的 token×层 比例, 以及 trainer 自己的 top-k 和 vLLM
选的有多少重合 (agree = 每行重合专家数 / top_k 的均值; set_match = 整组完全一样的比例)。
agree 就是"不回放的话有多少路由会不一样"的直接读数。
"""

from __future__ import annotations

import logging
import os
import re
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ENV = "KERNEL_RL_ROUTER_REPLAY"
# 训练端从 ``model.named_modules()`` 认 router 的名字: HF Qwen3.5/3.6-MoE 的 ``mlp.gate``
# (top-k router; 共享专家的 ``mlp.shared_expert_gate`` 是 sigmoid 门, 不是 top-k, 不碰)。
_ROUTER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


def enabled() -> bool:
    raw = os.environ.get(ENV, "").strip().lower()
    return raw in ("1", "true", "yes")


class _State:
    idx: torch.Tensor | None = None  # [N, num_layers, top_k] int16 on device, -1 = 不回放
    collect: bool = False  # 只在 forward 里记指标, backward 的重算不重复计
    rows = 0.0       # router 调用见过的 token×层 数 (collect 期间)
    replayed = 0.0   # 其中有回放的
    overlap = 0.0    # 回放行上: trainer top-k 与回放 top-k 的重合专家数之和
    set_match = 0.0  # 回放行上: 两组完全一样的行数
    shape_miss = 0   # 回放张量行数和 router 输入行数对不上的调用数 (退回原生路由)
    top_k = 1


_S = _State()


def _router_forward(self, hidden_states):
    """``Qwen3_5MoeTopKRouter.forward`` 的替身, 返回值同口径 (logits, scores, indices)。"""
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    top_value, top_idx = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
    rep = _S.idx
    if rep is not None:
        if rep.shape[0] != top_idx.shape[0]:
            _S.shape_miss += 1
        else:
            r = rep[:, self._rr_layer, :].to(top_idx.device).long()
            valid = r[:, 0] >= 0
            if _S.collect:
                with torch.no_grad():
                    n = float(top_idx.shape[0])
                    _S.rows += n
                    nv = int(valid.sum().item())
                    _S.replayed += nv
                    if nv:
                        native, replay = top_idx[valid], r[valid]
                        hit = torch.zeros(nv, router_probs.shape[-1], dtype=torch.bool,
                                          device=native.device).scatter_(1, native, True)
                        ov = hit.gather(1, replay).sum(-1)
                        _S.overlap += float(ov.sum().item())
                        _S.set_match += float((ov == self.top_k).sum().item())
            top_idx = torch.where(valid[:, None], r.clamp_min(0), top_idx)
            top_value = router_probs.gather(-1, top_idx)
    elif _S.collect:
        _S.rows += float(top_idx.shape[0])
    top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return router_logits, top_value.to(router_logits.dtype), top_idx


def install(model) -> int:
    """把模型里每层 top-k router 的 forward 换成回放版。返回换了几个 (0 = 没有可回放的 router)。

    在 deepspeed.initialize 之前或之后都行 (只换 forward, 不动参数)。名字按 HF 文本类
    (``model.layers.<i>.mlp.gate``); 层号 = vLLM capturer 的 layer_id (同一个 decoder 层序号)。
    """
    n = 0
    for name, mod in model.named_modules():
        m = _ROUTER_NAME.search(name)
        if m is None or not hasattr(mod, "top_k") or not hasattr(mod, "weight"):
            continue
        mod._rr_layer = int(m.group(1))
        mod.forward = types.MethodType(_router_forward, mod)
        _S.top_k = int(mod.top_k)
        n += 1
    if n == 0:
        raise RuntimeError(
            f"{ENV}=1 but no top-k router matching *.layers.<i>.mlp.gate was found "
            "(not an MoE? a model family with different naming?)")
    logger.info(f"[trainer] router replay installed on {n} MoE routers (top_k={_S.top_k})")
    return n


def begin(idx: torch.Tensor | None) -> None:
    """一个 chunk 开始 (forward 之前): 之后的 router 调用用 ``idx`` 回放, 直到 ``end()``
    或下一个 ``begin``。要一直挂到这个 chunk 的 backward 结束 (梯度检查点的重算)。

    ``idx``: [N, num_layers, top_k], N = 这次 forward 进 router 的总行数 (打包时 = 整行
    token 数, 不打包时 = mb x maxlen), 第一个 id < 0 的行不回放。None = 原生路由。
    forward 期间记指标; 调 ``backward_phase()`` 之后不再记。
    """
    _S.idx, _S.collect = idx, True


def end() -> None:
    _S.idx, _S.collect = None, False


@contextmanager
def replaying(idx: torch.Tensor | None):
    """``begin`` / ``end`` 的 with 写法。"""
    begin(idx)
    try:
        yield
    finally:
        end()


def backward_phase() -> None:
    """进 backward 之前调: 回放继续生效 (重算要走同一组专家), 但不再重复计指标。"""
    _S.collect = False


def build_index(routings: list, lengths: list[int], device, pad_to: int | None = None) -> torch.Tensor | None:
    """把一个 chunk 里各 datum 的回放张量拼成 router 要的 [N, L, K] (int16, -1 = 不回放)。

    ``routings[k]``: 第 k 条 datum 的 [>=T_k, L, K] 整数张量, 或 None (没有回放数据)。
    ``pad_to``: 不打包时每条补到 maxlen (按 [mb, maxlen] 展平); None = 打包, 首尾相接。
    全部是 None 时返回 None (这个 chunk 走原生路由, 不用分配)。
    """
    ref = next((r for r in routings if r is not None), None)
    if ref is None:
        return None
    n_layers, top_k = int(ref.shape[1]), int(ref.shape[2])
    parts = []
    for r, T in zip(routings, lengths):
        rows = pad_to if pad_to is not None else T
        out = torch.full((rows, n_layers, top_k), -1, dtype=torch.int16)
        if r is not None and r.shape[0] >= T:
            out[:T] = r[:T].to(torch.int16)
        parts.append(out)
    return torch.cat(parts).to(device, non_blocking=True)


def pop_stats() -> dict[str, float]:
    """取出并清零累计的现场计数 (rank 本地)。键恒在, 便于跨 rank 按 token 加权合并。"""
    k = max(1, _S.top_k)
    out = {
        "ppo/router_replay_rows": _S.rows,
        "ppo/router_replay_frac": _S.replayed / _S.rows if _S.rows else 0.0,
        "ppo/router_agree": _S.overlap / (k * _S.replayed) if _S.replayed else 0.0,
        "ppo/router_set_match": _S.set_match / _S.replayed if _S.replayed else 0.0,
        "ppo/router_replayed_rows": _S.replayed,
        "ppo/router_shape_miss": float(_S.shape_miss),
    }
    _S.rows = _S.replayed = _S.overlap = _S.set_match = 0.0
    _S.shape_miss = 0
    return out
