"""Shared RL training helpers (train step, sampler refresh, padding datum).

Copied from ARM ``r3lm/algorithms/loop.py`` (8821e43); the Kevin-rollout helper
``do_group_rollout_and_filter`` is dropped. Used by ``arl.grpo``:
``train_step`` / ``get_updated_sampling_client`` / ``_make_dummy_datum``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from arl.models import tinker_compat as tinker
from arl.models.types import LossFnType
from arl.models.registry import get_adam_params
from arl.utils.misc_utils import timed
from arl.utils.distributed import all_reduce_max


def _make_dummy_datum(pad_id: int) -> tinker.Datum:
    """A zero-gradient padding datum (all advantages 0 => 0 surrogate => 0 grad).

    Used to pad each rank's training data to a global-max length so every rank
    runs the same number of forward/backward passes (lockstep collectives under
    the deepspeed launcher).
    """
    import torch

    toks = [int(pad_id), int(pad_id)]
    zeros = tinker.TensorData.from_torch(torch.zeros(len(toks)))
    tgt = tinker.TensorData.from_torch(torch.tensor(toks, dtype=torch.long))
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(toks),
        loss_fn_inputs={"target_tokens": tgt, "logprobs": zeros, "advantages": zeros},
    )


def remove_mask(datum: tinker.Datum) -> tinker.Datum:
    """Remove mask from datum loss_fn_inputs before sending to forward_backward.

    The Tinker API doesn't expect the mask key in loss_fn_inputs, so we need to
    remove it before sending the datum to forward_backward.
    """
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


logger = logging.getLogger(__name__)


async def train_step(
    data: list[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
    loss_fn_config: dict[str, float] | None,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """
    Perform a training step with gradient accumulation.

    Args:
        data: List of training datums
        training_client: Tinker training client
        learning_rate: Learning rate
        num_substeps: Number of optimizer steps
        loss_fn: Loss function type
        loss_fn_config: Optional loss-function configuration

    Returns:
        (training_logprobs, train_metrics) -- the per-datum training logprobs and
        the PPO/loss metrics (loss, ppo/mean_ratio, ppo/clip_frac, ppo/approx_kl)
        aggregated across substeps (token-weighted, matching the loss reduction).
    """
    # Split data into substeps
    substep_size = max(1, len(data) // num_substeps)
    training_logprobs = []

    # Accumulate PPO metrics across substeps. loss is summed; the ratio/clip/kl
    # rates are averaged weighted by each substep's active-token count so the
    # aggregate matches the global token-mean the loss itself uses.
    loss_total = 0.0
    tok_total = 0.0
    ratio_wsum = 0.0
    clip_wsum = 0.0
    kl_wsum = 0.0
    vloss_total = 0.0
    # In-loss k3 KL penalty vs pi_ref (present only when loss_fn_config carries
    # kl_beta != 0): the per-token rate is token-weighted like ratio/clip/kl, the
    # penalty (an additive loss term) is summed like loss/value_loss.
    klref_wsum = 0.0
    klpen_total = 0.0
    saw_kl = False
    # 假测量探测器 (见 training_client 的 k3 clamp): 最大值取各 substep 的 max,
    # 被 clamp 的 token 数求和。
    kl_delta_max = 0.0
    kl_clamped_total = 0.0
    # 采样分布回放的现场 (training_client 发 ppo/replay_*): 按 active token 数加权累加。
    rp_tok = rp_out = rp_zero = rp_kept = 0.0
    # 每个 substep 各自的 (kl_ref, grad_norm)。间歇性 k3 飙升的签名是"两个 substep
    # 只爆一个" (grad_norm ≈ grad_norm_max/2), 均值+最大值定位不到是哪个 —— 拆开记。
    per_substep: list[tuple[float | None, float | None]] = []
    # 梯度范数 (DeepSpeed 的 _global_grad_norm, **裁剪之前**的那个数)。直接和 conf 的
    # grad_clip (= ds_config 的 gradient_clipping) 比就知道裁剪有没有在生效 —— 而裁剪是唯一对梯度整体缩放不免疫的
    # 环节 (Adam 的 m/sqrt(v) 把常数因子约掉了), 所以任何改变 loss 归一化的动作
    # (Dr-GRPO 的 const_len、num_substeps) 到底有没有实际影响, 只有这条曲线回答得了。
    gn_sum = 0.0
    gn_max = 0.0
    gn_n = 0
    # 非有限守卫 (2026-08-29): forward_backward 报的 loss 非有限, 或训练端 lp 里有
    # nan/inf token, 这个 substep 就不做 optim_step (梯度丢掉, 权重不动)。判定要全
    # rank 一致 —— NaN 梯度经 reduce-scatter 已经污染了所有 rank 的分片, 且 step 内
    # 有集合通信 —— 所以先 all_reduce_max 再决定。被跳过的 substep 不计入 loss /
    # ratio / kl 的均值 (别把曲线一起污染成 nan), 单独用 nonfinite_* 计数看得见。
    nonfinite_skipped = 0
    grad_overflow_skipped = 0
    nonfinite_tok = 0.0
    nonfinite_active = 0.0
    nonfinite_datums = 0.0
    n_substeps_run = 0

    for i in range(0, len(data), substep_size):
        batch = data[i : i + substep_size]

        # Forward-backward pass (remove mask key from datums)
        fwd_bwd_kwargs: dict[str, Any] = {"loss_fn": loss_fn}
        if loss_fn_config:
            fwd_bwd_kwargs["loss_fn_config"] = loss_fn_config
        fwd_bwd_future = await training_client.forward_backward_async(
            [remove_mask(d) for d in batch],
            **fwd_bwd_kwargs,
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()

        # Accumulate PPO/loss metrics (see training_client forward_backward output).
        m = fwd_bwd_result.metrics or {}
        n_substeps_run += 1
        nonfinite_tok += float(m.get("ppo/nonfinite_lp_tokens", 0.0))
        nonfinite_active += float(m.get("ppo/nonfinite_lp_active", 0.0))
        nonfinite_datums += float(m.get("ppo/nonfinite_datums", 0.0))
        local_bad = (
            not math.isfinite(float(m.get("loss", 0.0)))
            or float(m.get("ppo/nonfinite_lp_tokens", 0.0)) > 0
            or float(m.get("ppo/nonfinite_loss", 0.0)) > 0
        )
        global_bad = all_reduce_max(1.0 if local_bad else 0.0) > 0  # COLLECTIVE
        if global_bad:
            nonfinite_skipped += 1
            logger.warning(
                "[train_step] substep %d: non-finite loss/logprobs (this rank: loss=%s, "
                "bad lp tokens=%d, active=%d, datums=%d) -> optim_step skipped on all ranks",
                len(per_substep), m.get("loss"),
                int(m.get("ppo/nonfinite_lp_tokens", 0)),
                int(m.get("ppo/nonfinite_lp_active", 0)),
                int(m.get("ppo/nonfinite_datums", 0)),
            )
        n_tok = 0.0 if global_bad else float(m.get("ppo/n_active_tokens", 0.0))
        if not global_bad:
            loss_total += float(m.get("loss", 0.0))
        tok_total += n_tok
        if not global_bad:  # nan * 0 仍是 nan, 跳过的 substep 一项都不能进来
            ratio_wsum += float(m.get("ppo/mean_ratio", 0.0)) * n_tok
            clip_wsum += float(m.get("ppo/clip_frac", 0.0)) * n_tok
            kl_wsum += float(m.get("ppo/approx_kl", 0.0)) * n_tok
            vloss_total += float(m.get("ppo/value_loss", 0.0))
        if "ppo/replay_tokens" in m and not global_bad:
            t_ = float(m["ppo/replay_tokens"])
            rp_tok += t_
            rp_out += float(m.get("ppo/replay_outside_frac", 0.0)) * t_
            rp_zero += float(m.get("ppo/replay_zero_grad_frac", 0.0)) * t_
            rp_kept += float(m.get("ppo/replay_kept_mean", 0.0)) * t_
        if "ppo/kl_penalty" in m and not global_bad:
            saw_kl = True
            klref_wsum += float(m.get("ppo/kl_ref_per_token", 0.0)) * n_tok
            klpen_total += float(m["ppo/kl_penalty"])
            kl_delta_max = max(kl_delta_max, float(m.get("ppo/kl_delta_max", 0.0)))
            kl_clamped_total += float(m.get("ppo/kl_clamped_tokens", 0.0))

        # Extract logprobs
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs.append(output["logprobs"].to_torch())

        # Optimizer step
        adam_params = get_adam_params(learning_rate)
        if global_bad:
            optim_future = await training_client.optim_step_async(adam_params, skip=True)
        else:
            optim_future = await training_client.optim_step_async(adam_params)
        optim_result = await optim_future.result_async()
        gn = (optim_result or {}).get("grad_norm") if isinstance(optim_result, dict) else None
        # 梯度侧守卫在 optim_step 里自己判定 (forward 有限但梯度 inf/nan, 见
        # training_client._grad_has_inf_or_nan); 它跳的步同样计进 nonfinite_skipped,
        # 上层的整批账 / 熔断不用分两套。单独再记一条, 曲线上分得出是哪一侧。
        if not global_bad and isinstance(optim_result, dict) and optim_result.get("skipped"):
            nonfinite_skipped += 1
            grad_overflow_skipped += 1
        # None = 这个后端不报 (或 DeepSpeed 还没填); inf/nan = 这一步梯度已经坏了, 别把
        # 均值也一起污染成 nan —— 但要在日志里看得见, 所以单独计数。
        if gn is not None and math.isfinite(gn):
            gn_sum += float(gn)
            gn_max = max(gn_max, float(gn))
            gn_n += 1
        per_substep.append((
            float(m["ppo/kl_ref_per_token"]) if "ppo/kl_ref_per_token" in m else None,
            float(gn) if gn is not None and math.isfinite(gn) else None,
        ))

    train_metrics: dict[str, float] = {
        "loss": loss_total, "ppo/value_loss": vloss_total,
        # 非有限守卫的计数 (见上)。恒发, 各 rank key 集一致: skipped/substeps 是全
        # rank 一致的判定结果, nonfinite_lp_* / datums 是本 rank 的现场计数 (上层
        # 按 sum 合)。
        "ppo/nonfinite_skipped_substeps": float(nonfinite_skipped),
        "ppo/grad_overflow_substeps": float(grad_overflow_skipped),
        "ppo/substeps": float(n_substeps_run),
        "ppo/nonfinite_lp_tokens": nonfinite_tok,
        "ppo/nonfinite_lp_active": nonfinite_active,
        "ppo/nonfinite_datums": nonfinite_datums,
    }
    if gn_n:
        # 均值和最大值分开: num_substeps>1 时一批里只要有一个 substep 被裁, 均值也看不出来。
        train_metrics["ppo/grad_norm"] = gn_sum / gn_n
        train_metrics["ppo/grad_norm_max"] = gn_max
    if tok_total > 0:
        train_metrics["ppo/mean_ratio"] = ratio_wsum / tok_total
        train_metrics["ppo/clip_frac"] = clip_wsum / tok_total
        train_metrics["ppo/approx_kl"] = kl_wsum / tok_total
        train_metrics["ppo/n_active_tokens"] = tok_total
    if rp_tok > 0:
        train_metrics["ppo/replay_tokens"] = rp_tok
        train_metrics["ppo/replay_outside_frac"] = rp_out / rp_tok
        train_metrics["ppo/replay_zero_grad_frac"] = rp_zero / rp_tok
        train_metrics["ppo/replay_kept_mean"] = rp_kept / rp_tok
    if saw_kl:
        train_metrics["ppo/kl_penalty"] = klpen_total
        train_metrics["ppo/kl_ref_per_token"] = (
            klref_wsum / tok_total if tok_total > 0 else 0.0
        )
        train_metrics["ppo/kl_delta_max"] = kl_delta_max
        train_metrics["ppo/kl_clamped_tokens"] = kl_clamped_total
    # substep 拆分曲线 (rank 本地)。只在真的多于一个 substep 时记, 否则和聚合值重复;
    # 末尾不整除时 substep 数可能比 num_substeps 多 1, key 集合随之多一组 —— wandb
    # 上表现为那条曲线偶尔缺点, 无害。
    if len(per_substep) > 1:
        for si, (kr, g) in enumerate(per_substep):
            if kr is not None:
                train_metrics[f"ppo/substep{si}/kl_ref_per_token"] = kr
            if g is not None:
                train_metrics[f"ppo/substep{si}/grad_norm"] = g

    return training_logprobs, train_metrics


async def get_updated_sampling_client(
    training_client: tinker.TrainingClient,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """Push freshly-trained weights into the sampler and return a client.

    Called every batch: writes the consolidated policy to the sync dir and
    hot-reloads it into the vLLM engine in place. Disk checkpointing is handled
    separately by the caller so retention/ranking can key on the post-reduction
    *global* metrics.
    """
    metrics: dict[str, Any] = {}
    with timed("weight_sync", metrics):
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    return sampling_client, metrics
