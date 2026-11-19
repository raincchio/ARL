"""PPO advantage/return computation: GRPO group baseline or shared value head.

NOTE on time scales -- there are two, and only the outer one is discounted:

  * BETWEEN turns (multi_turn): ppo_loop applies ``gamma`` before calling in here,
    so the ``rewards`` argument already holds R_k = sum_j gamma^(j-k) r_j. Each
    turn is one macro-action of a turn-level MDP.
  * WITHIN a turn (here): the reward is terminal on the last response token with
    gamma=1, which is why there is still no per-token TD/lambda recursion below.
    A per-TOKEN discount or lambda<1 would mean writing real GAE.

The critic is a value head ON the policy backbone (arl training_client), not a
separate model. At rollout we forward the value head to get OLD per-token values V_t.

Rewards are the plain bandit case: ``r`` is the scalar for this turn (RM score, or
the discounted return over the turns that follow it), terminal on the last response
token, constant across the response:

    R_t = r

Two baselines for the advantage (``groups`` argument):

    group mode (GRPO):   A_t = (r - baseline(group)) / (std(group) + eps)
    value-head mode:     A_t = R_t - V_t

The group baseline is the length-weighted mean of the group's rewards by default
(``length_weighted_baseline``, Cognition's SWE-1.5/SWE-2 recipe): b = sum_u L_u R_u
/ sum_u L_u with L_u the unit's action-token count, a free stand-in for the
variance-optimal baseline E[R ||grad log pi||^2] / E[||grad log pi||^2]. Off = the
plain arithmetic mean. See ``group_baseline_advantages``.

Group mode is the GRPO baseline: the comparison set is the sibling samples of the
SAME problem (and, in multi_turn, the same turn index -- returns at different
turns sum different numbers of remaining rewards, so mixing turns into one group
would leave a systematic turn-index bias in the "baseline"). A group with fewer
than 2 members, or one where every sibling got the same reward, carries no
relative information: those datums get advantage 0 rather than a made-up signal
(same principle as ``_group_length_rewards`` and Tinker's
remove_constant_reward_groups). The std division is optional (``group_adv_std``);
Dr-GRPO argues it biases toward low-variance groups -- with it off, the global
whitening below supplies the scale instead.

The KL penalty against the reference policy is NOT part of the reward anymore.
It moved into the LOSS (GRPO-style): forward_backward adds

    kl_beta * k3(pi_theta || pi_ref),   k3_t = exp(ref_t - lp_t) - (ref_t - lp_t) - 1

per token, where ``lp_t = log pi_theta(a_t)`` is the CURRENT policy's logprob --
differentiable and recomputed at every substep, unlike the old in-reward k1 which
was frozen at rollout time. This module's only KL duty is to attach the per-token
``log pi_ref(a_t)`` to each Datum (``loss_fn_inputs["ref_logprobs"]``) so the loss
can compute k3 later. k3 is >= 0 by construction (it is a convex function of the
log-ratio with minimum 0 at ref == lp), so no sign/clamping subtleties.

Advantage normalization (value-head mode): standardized over active tokens **within
each (problem, turn) group** (``whiten_groups``), not across the whole batch. The
group mean absorbs the question's difficulty level, so what survives is "better or
worse than the sibling answers to THIS question" -- the same comparison set GRPO
uses, kept here on top of the value-head baseline (V_t still shapes token-level
credit inside each response). A group left with fewer than 2 datums (drop_truncated
/ no-eval drops) has no siblings to compare against; it falls back to the whole-batch
statistics rather than self-normalizing -- standardizing a single datum against
itself would subtract its own reward and leave pure value-residual noise. Group
mode (GRPO) keeps its whole-batch whitening: there the group mean is already gone,
and Dr-GRPO wants the scale supplied globally precisely to avoid per-group std bias.
Datums whose group carried no relative information (advantage 0, above) sit out the
whitening entirely -- they are excluded from the statistics AND left at 0, because
subtracting batch_mean from a deliberate 0 would hand them back the made-up signal
this module just refused to invent. Keeping them at 0 also leaves the TOKEN-weighted
sum of advantages exactly 0 over the batch, which is what "no net push" means once
the loss weighs every token equally (Dr-GRPO's const_len, see training_client).

Returns go into each Datum as ``loss_fn_inputs["returns"]`` so forward_backward's
value loss ``MSE(V_t, R_t)`` can train the head jointly with the policy.

Datum token alignment (input/target/logprobs/mask) is reused from arl's
``trajectory_to_data`` — we only fill in advantages + returns.
"""

from __future__ import annotations

import torch

from arl.models.types import Datum, TensorData
from arl.algorithms.data_processing import trajectory_to_data
from arl.algorithms.rl_types import Trajectory


def base_datums_for_response(traj: Trajectory) -> list[Datum]:
    """Datums with correct mask/logprobs/target for one response (advantages TBD).

    Placeholder advantage 0.0 — real per-token advantages are filled by
    ``finalize_ppo_data`` once old values are known.
    """
    return trajectory_to_data(traj, 0.0)


def tailrl_weights(rs: list[float]) -> list[float]:
    """TailRL (arXiv:2609.02987, Tail-Likelihood RL) 的组内 rollout 权重 ω(r_i)。

    目标 J = ∫ log Pr[r > τ] dτ (对每个阈值 τ 最大化"超过它"的 log 概率), N 条 rollout
    下的无偏梯度估计是 Σ_i ω(r_i)·∇logπ(z_i), 其中

        ω(r_i) = ∫_0^{r_i} dτ / #{j : r_j > τ}

    即每个阈值上一份 credit 平分给超过它的那几条 —— 越少人够到的阈值分得越多, 所以
    正推力集中在组里罕见的高分 rollout 上, 低分条的负推力比 (r - mean) 小得多。排序后闭式 (论文 Eq. 27):

        ω(r_(i)) = ω(r_(i-1)) + (r_(i) - r_(i-1)) / (N - i + 1)

    论文取 r_(0) = r_min (reward 下界); 这里取 r_(0) = 组内最小值 —— 差别只是给**每条**
    都加同一个常数 (r_(1) - r_min)/N, 被随后的组内居中精确减掉, 于是不需要知道 reward
    的下界 (speedup 类 reward 也没有天然的上界, 论文 App. D: 仿射变换只缩放权重)。
    同分的条拿同一个 ω (间隔为 0)。返回值与 ``rs`` 逐位对齐, 还没居中。
    """
    n = len(rs)
    order = sorted(range(n), key=lambda j: rs[j])
    w = [0.0] * n
    acc, prev = 0.0, rs[order[0]] if n else 0.0
    for k, j in enumerate(order):  # k 从 0 起: 论文的 N - i + 1 = n - k
        acc += (rs[j] - prev) / (n - k)
        prev = rs[j]
        w[j] = acc
    return w


def group_baseline_advantages(
    rewards: list[float], groups: list, use_std: bool = True,
    units: list | None = None, lengths: list[float] | None = None,
    estimator: str = "grpo", std_floor: float = 0.0,
) -> tuple[list[float], int, list[bool], float]:
    """GRPO 组内基线: 每条的标量 advantage = (r - 组内基线) [/ 组内 std]。

    ``groups[i]`` 是第 i 条的组键 (可哈希即可; ppo_loop 按 cfg.adv_unit 用 (题号, 轮号)
    或 (题号,))。

    ``units[i]`` 是第 i 条属于哪个**决策单元** —— 均值/std 按单元算, 不按 datum 算,
    同一单元的 datum 拿同一个 advantage。None = 每条自成一个单元 (旧行为, 逐位等价)。
    要它是因为 adv_unit="trajectory" 下一条轨迹的 4 个轮是 4 条 datum 但**同一个**
    arm: 按 datum 算均值等于把活得久的轨迹在基线里数好几次, 而"活得久"本身和 reward
    正相关。同一单元内的 rewards 必须相同 (调用方保证), 这里只取它的第一条。

    ``lengths[i]`` 是第 i 条的 action token 数 —— 给了就把组内基线从**算术均值**换成
    **长度加权**均值 (Cognition SWE-1.5/SWE-2 的 Length-Weighted Reward Baseline):

        b = Σ_u L_u·R_u / Σ_u L_u,   L_u = 这个单元所有 datum 的 token 数之和

    为什么: 方差最小的基线是 b* = E[R·‖∇logπ‖²] / E[‖∇logπ‖²], 而 ‖∇logπ‖² 基本随
    rollout 的长度走 —— 拿 L 当 ‖∇logπ‖² 的代理就得到近似最优基线, 且一次额外反向都不
    用 (真算 ‖∇logπ‖² 要每条 rollout 一次)。它仍然是个**基线**: 加权方式不看这一条自己
    的动作, 所以不比算术均值多引入偏差 (组内基线本来就有的那点自比偏差不变)。
    落到这条管线上还多一个刚好对得上的性质: loss 侧 token 等权 (Dr-GRPO 的 const_len),
    每条拿到的梯度份额按它的 token 数走, 于是组内净推力 Σ_u L_u·A_u = 0 **精确**成立 ——
    算术均值只保证 Σ_u A_u = 0, 两者差 n·cov(L, R): "写崩的通常更长" 时它系统性为负
    (见 4_train_grpo.conf 的 group_adv_std 段: mean_ratio 每批 < 1、熵单调涨)。全 batch
    白化过去是**事后**把这个和减回 0 的, 这里是每组当场不产生它。
    None = 算术均值 (旧行为, 逐位等价)。

    ``estimator``: 组内各单元的 reward 先过哪个映射再居中。
      * "grpo"   不映射, A = (R - b) [/ std] (旧行为, 逐位等价)。
      * "tailrl" 先换成 TailRL 权重 ω (见 ``tailrl_weights``), A = (ω - b(ω)) [/ std(ω)]。
                 论文 Eq. 30 是 ω 的算术均值居中、不除 std; 长度加权基线 / use_std 仍按
                 各自的开关作用在 ω 上 (都只是基线/尺度, 不改 ω 的相对形状)。

    ``std_floor``: use_std 下的除数换成 ``max(std, std_floor)``。0 = 不设下限 (旧行为,
    逐位等价)。组内 std 很小的组 (例如 8 条 speedup 只差 0.0x、差别基本是测速噪声)
    除 std 后会被放大成 ±1 的 advantage; 下限让这类组按 reward 原始量纲给小推力,
    std 高于下限的组不受影响。``sd < 1e-8`` 的无信号判据看的是真实 std, 不受它影响。

    加权只动**基线**, 不动尺度: std 仍按未加权的均值算 (它是 use_std 下的除数, 不是
    居中的位置), ``sd < 1e-8`` 那条无信号判据也因此原样 —— 全组同分时两种基线相等, 都
    给 0。

    组内不足 2 **个单元**、或单元间 reward 全相等 (std≈0) 时整组给 0 —— 没有相对信息
    就不编造信号。返回 ``(advs, n_zero, has_signal, baseline_shift)``:
      * n_zero = 因此拿到 0 的条数 (监控用: 一直很高说明组太小或 reward 太平,
        大部分样本根本没在训练);
      * baseline_shift = 长度加权把基线挪了多远, 按组内 std 计量, 在有信号的组上取
        平均 (lengths=None 时恒 0)。负 = 长 rollout 的 reward 系统性偏低, 也就是算术
        均值那条路上净推力为负的量; 一直贴着 0 说明这道开关在这份数据上无事发生;
      * has_signal[i] = 第 i 条所在的组有没有相对信息。下游的全 batch 白化必须靠它
        把这些条**同时排除在统计量和变换之外** —— 白化会减去 batch_mean, 而
        (0 - batch_mean)/std != 0, 那正好把上面刚决定"不编造"的信号又编回来了
        (group_adv_std=False 时白化才开着, 所以这条路 2026-09-11 之前是死的)。
    """
    by_group: dict = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)

    advs = [0.0] * len(rewards)
    has_signal = [True] * len(rewards)
    n_zero = 0
    shifts: list[float] = []

    def _no_signal(idx: list[int]) -> None:
        nonlocal n_zero
        n_zero += len(idx)
        for i in idx:
            has_signal[i] = False

    for idx in by_group.values():
        # 组 -> 单元 -> 该单元的 datum 下标 (units=None 时每条自成一单元)。dict 保序,
        # 所以单元的遍历顺序 = 首次出现的顺序, 结果与波次到达顺序无关。
        by_unit: dict = {}
        for i in idx:
            by_unit.setdefault(i if units is None else units[i], []).append(i)
        if len(by_unit) < 2:
            _no_signal(idx)
            continue
        rs = [rewards[js[0]] for js in by_unit.values()]
        if estimator == "tailrl":
            # ω 是 r 的严格单调映射 (同分同 ω), 所以下面 sd≈0 的"全组同分"判据不变。
            rs = tailrl_weights(rs)
        mu = sum(rs) / len(rs)
        sd = (sum((x - mu) ** 2 for x in rs) / len(rs)) ** 0.5
        if sd < 1e-8:
            # 全组同分 = 没有相对信息, 两种模式都走这条路 (2026-09-11 修正: 这个判断
            # 以前只在 use_std 分支里, 理由写的是"除以 eps 会把数值噪声放大成假信号";
            # 不除 std 时 r - mu 本来就是 0, 看着无害 —— 但 has_signal 没被标掉, 下游
            # 白化减 batch_mean 就把这个 0 变成了非 0, 正是这里不想编造的信号。顺带
            # ppo/adv_zero_frac 在 group_adv_std=False 下以前恒少算这些条。)
            _no_signal(idx)
            continue
        b = mu
        if lengths is not None:
            # 单元的长度 = 它所有 datum 的 token 数之和 (adv_unit="trajectory" 下一条
            # 4 轮的轨迹算一个单元、长度是四轮相加 —— 和"它在 loss 里占多少 token"同一
            # 个口径, 净推力那笔账才对得上)。
            ls = [sum(lengths[i] for i in js) for js in by_unit.values()]
            tot = sum(ls)
            if tot > 0:  # 全组零长度 (理论上不该有) 就退回算术均值
                b = sum(r * L for r, L in zip(rs, ls)) / tot
                shifts.append((b - mu) / sd)
        div = max(sd, std_floor)
        for r_u, js in zip(rs, by_unit.values()):
            a = (r_u - b) / div if use_std else r_u - b
            for i in js:
                advs[i] = a
    shift = sum(shifts) / len(shifts) if shifts else 0.0
    return advs, n_zero, has_signal, shift


def value_ev_sums(
    datums: list[Datum], rewards: list[float],
    values_list: list[torch.Tensor], groups: list,
) -> list[float]:
    """本 rank 的 explained-variance 充分统计量 (按 active token 计, 和 value loss 同口径)。

    V 是 rollout 时 head **还没在这批上训过**的旧值, 所以算出来的天然是样本外 EV。
    返回 8 个和, 跨 rank 直接相加后交给 ``value_ev_from_sums``:

      [n, ΣR, ΣR², Σe, Σe², n_w, ΣR_w², Σe_w²]      e = R - V

    后三个是**组内**口径: 每个 (题, 轮) 组先减掉组内 (按 token 加权) 的 R 均值和 V 均值。
    value_head 模式的 advantage 是按同一组做白化的, 组均值会被整个减掉 —— V 只靠"认出
    这道题难不难"挣来的那部分 EV 对梯度没有贡献, 组内 EV 才是它真正降掉的 advantage
    方差。只有 1 条 datum 的组不进组内统计 (R_w 恒为 0, 剩下的只是 V 的轮内抖动)。
    """
    n = sR = sR2 = se = se2 = 0.0
    nw = sRw2 = sew2 = 0.0
    by_group: dict = {}
    for i, d in enumerate(datums):
        if values_list[i] is None:
            continue
        m = d.loss_fn_inputs["mask"].to_torch().bool()
        k = int(m.sum())
        if k == 0:
            continue
        v = values_list[i].double()[m]
        r = float(rewards[i])
        e = r - v
        n += k
        sR += r * k
        sR2 += r * r * k
        se += float(e.sum())
        se2 += float((e * e).sum())
        by_group.setdefault(groups[i], []).append((r, v, k))
    for items in by_group.values():
        if len(items) < 2:
            continue
        tok = sum(k for _, _, k in items)
        mR = sum(r * k for r, _, k in items) / tok
        mV = sum(float(v.sum()) for _, v, _ in items) / tok
        for r, v, k in items:
            rw = r - mR
            ew = rw - (v - mV)
            nw += k
            sRw2 += rw * rw * k
            sew2 += float((ew * ew).sum())
    return [n, sR, sR2, se, se2, nw, sRw2, sew2]


def value_ev_from_sums(s: list[float]) -> dict[str, float]:
    """全局和 -> ``ppo/value_ev`` (= 1 - Var(R-V)/Var(R)) 和 ``ppo/value_ev_within``
    (组内口径, 见 ``value_ev_sums``)。分母为 0 (全批同分 / 没有多于 1 条的组) 时不出该键。"""
    n, sR, sR2, se, se2, nw, sRw2, sew2 = s
    out: dict[str, float] = {}
    if n > 0:
        var_r = sR2 / n - (sR / n) ** 2
        var_e = se2 / n - (se / n) ** 2
        out["ppo/value_resid_rmse"] = (se2 / n) ** 0.5
        if var_r > 1e-12:
            out["ppo/value_ev"] = 1.0 - var_e / var_r
    if nw > 0 and sRw2 > 1e-12:
        out["ppo/value_ev_within"] = 1.0 - sew2 / sRw2
    return out


def finalize_ppo_data(
    datums: list[Datum],
    rewards: list[float],
    values_list: list[torch.Tensor] | None = None,
    whiten: bool = True,
    ref_logprobs_list: list[torch.Tensor] | None = None,
    groups: list | None = None,
    units: list | None = None,
    group_adv_std: bool = True,
    whiten_std: bool = True,
    whiten_groups: list | None = None,
    length_weighted_baseline: bool = True,
    adv_estimator: str = "grpo",
    group_adv_std_floor: float = 0.0,
    mask_positive_incorrect: bool = False,
    response_correctness: list[bool | None] | None = None,
    mask_negative_correct: bool = False,
    distributed_groups: bool = False,
) -> tuple[list[Datum], dict[str, float]]:
    """Fill per-token advantages and returns into each datum.

    Args:
        datums: base datums (one per response; mask marks the response tokens).
        rewards: RM reward r per response.
        values_list: old per-token value V_t per datum (aligned to targets, length
            T). Required in value-head mode (groups=None); ignored in group mode.
        whiten: normalize advantages over active tokens (scope: ``whiten_groups``).
        whiten_std: divide by the whitening std as well, or only subtract the mean.
            False = center-only, which is what Dr-GRPO asks for -- it objects to the
            std DIVISION (low-variance groups get inflated), not to centering. The
            token-weighted centering is what keeps the batch's net push at zero, so
            dropping just the divisor removes every std division from the pipeline
            without handing that property back. Scale then stays in raw reward units:
            per rank that is absorbed by grad_clip (a fixed-norm step), across ranks it
            means a problem whose siblings actually spread out weighs more than one
            where all G answers scored the same -- again what Dr-GRPO wants.
        ref_logprobs_list: per-token log pi_ref per datum, attached to each Datum as
            ``loss_fn_inputs["ref_logprobs"]`` for the in-loss k3 KL penalty
            (see module docstring). None = no reference policy (kl_beta=0).
        groups: per-datum group key -> GRPO group-baseline mode (see module
            docstring). None -> value-head mode ``A_t = R_t - V_t``.
        units: per-datum decision-unit key -- the group mean/std are taken over
            UNITS, not over datums, and every datum of a unit gets the same
            advantage (ppo_loop's ``adv_unit``: one unit per (trajectory, turn), or
            one per trajectory). None = every datum is its own unit.
        group_adv_std: group mode only -- divide by the group std (original GRPO)
            or mean-center only (Dr-GRPO).
        group_adv_std_floor: with group_adv_std, divide by ``max(std, floor)``
            instead of the raw group std (0 = no floor, the old behavior).
        whiten_groups: per-datum group key for the whitening statistics (value-head
            mode passes the same (problem, turn) keys GRPO groups by). None = one
            set of statistics over the whole batch (the pre-2026-08 behavior, and
            what group mode still uses -- see module docstring).
        length_weighted_baseline: group mode only -- weight the group baseline by
            each unit's action-token count instead of taking the plain mean
            (default on; see ``group_baseline_advantages`` for why).
        adv_estimator: group mode only -- "grpo" centers the raw rewards, "tailrl"
            first maps each group's rewards to TailRL weights (``tailrl_weights``).
        mask_positive_incorrect: group mode only; after whitening, zero positive
            advantages and loss-mask positions on incorrect responses. The trainer
            also excludes these tokens from KL because its active mask is adv != 0.
        response_correctness: one label per datum; False means incorrect, True
            means correct, None means unknown (left unchanged by the gate).
        mask_negative_correct: group mode only; after whitening, zero negative
            advantages and loss-mask positions on correct responses, including KL.
        distributed_groups: COLLECTIVE on every rank, including empty ranks.
            Gather only reward/group/unit/action-count metadata on CPU. Compute
            sibling baselines and batch whitening globally; keep token data local.
            Requires group mode and batch whitening (whiten_groups=None).

    Returns:
        (datums, metrics) — group mode reports ``ppo/adv_zero_frac`` (share of
        datums whose group carried no signal) and, with the length-weighted
        baseline on, ``ppo/adv_baseline_shift`` (how far it moved the baseline,
        in group-std units); per-group whitening reports
        ``ppo/adv_whiten_singleton_frac`` (share of datums that fell back to batch
        statistics for lack of siblings). KL diagnostics come out of
        forward_backward, where the k3 penalty actually lives.
    """
    if distributed_groups and (groups is None or whiten_groups is not None):
        raise ValueError("distributed_groups requires group mode and batch whitening")
    for name, enabled in (("mask_positive_incorrect", mask_positive_incorrect),
                          ("mask_negative_correct", mask_negative_correct)):
        if enabled:
            if groups is None:
                raise ValueError(f"{name} requires group advantages (GRPO)")
            if response_correctness is None or len(response_correctness) != len(datums):
                raise ValueError(f"{name} requires one correctness label per datum")
    metrics: dict[str, float] = {}
    scalar_advs: list[float] | None = None
    has_signal = [True] * len(datums)
    # [T] float mask per datum, 1 on response tokens. 算在前面是因为组内基线要按
    # **长度**加权 (见 group_baseline_advantages 的 lengths), 而长度就是 mask 的和。
    masks_f = [d.loss_fn_inputs["mask"].to_torch().float() for d in datums]
    global_whitening = None
    if groups is not None:
        action_counts = [float(m.sum()) for m in masks_f]
        baseline_rewards, baseline_groups, baseline_units = rewards, groups, units
        baseline_counts = action_counts
        local_start = 0
        if distributed_groups:
            from arl.utils import distributed as du
            rank, _, _ = du.dist_info()
            local_units = [(rank, i if units is None else units[i]) for i in range(len(rewards))]
            gathered = du.all_gather_object(
                (rewards, groups, local_units, action_counts), group=du.gloo_group())
            local_start = sum(len(row[0]) for row in gathered[:rank])
            baseline_rewards, baseline_groups, baseline_units, baseline_counts = (
                [item for row in gathered for item in row[column]] for column in range(4))
        lengths = baseline_counts if length_weighted_baseline else None
        scalar_advs, n_zero, has_signal, shift = group_baseline_advantages(
            baseline_rewards, baseline_groups, use_std=group_adv_std, units=baseline_units, lengths=lengths,
            estimator=adv_estimator, std_floor=group_adv_std_floor)
        if baseline_rewards:
            metrics["ppo/adv_zero_frac"] = n_zero / len(baseline_rewards)
        if lengths is not None:
            metrics["ppo/adv_baseline_shift"] = shift
        if distributed_groups:
            # One scalar per trajectory segment, repeated for its action tokens.
            # Two passes avoid cancellation in the sample variance. No token
            # arrays or model inputs cross ranks, even for very long contexts.
            active_stats = [(a, count) for a, count, signal in
                            zip(scalar_advs, baseline_counts, has_signal) if signal]
            count = sum(n for _, n in active_stats)
            mean = sum(a*n for a, n in active_stats) / max(1, count)
            std = (sum(n*(a-mean)**2 for a, n in active_stats) / max(1, count-1))**0.5
            global_whitening = (count, mean, std)
            end = local_start + len(rewards)
            scalar_advs = scalar_advs[local_start:end]
            has_signal = has_signal[local_start:end]
    elif values_list is None:
        raise ValueError("values_list is required in value-head mode (groups=None)")

    advs: list[torch.Tensor] = []
    rets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for i, (d, r) in enumerate(zip(datums, rewards)):
        mask = masks_f[i]
        ret = mask * float(r)  # R_t = r (bandit; KL is in the loss, not the reward)
        if scalar_advs is not None:
            advs.append(mask * scalar_advs[i])  # 组内基线: 每 token 同一个标量
        else:
            v = values_list[i]
            v = v.to(mask.dtype) if v is not None else torch.zeros_like(mask)
            advs.append(mask * (ret - v))  # A_t = R_t - V_t on response tokens, else 0
        rets.append(ret)
        masks.append(mask.bool())

    if whiten:
        def _denom(std):
            # whiten_std=False: 只减均值, 不缩放 (Dr-GRPO 反对的是**除 std**, 不是居中)。
            return (std + 1e-8) if whiten_std else 1.0

        def _standardize(idx: list[int], mean, std) -> None:
            d = _denom(std)
            for i in idx:
                advs[i] = torch.where(masks[i], (advs[i] - mean) / d, advs[i])

        # 参与白化的条: group 模式下"组内没有相对信息"而被置 0 的那些**不参与** ——
        # 既不进统计量, 也不被减去 batch_mean (见 group_baseline_advantages 的
        # has_signal)。它们留在 0 上, 于是 (a) 不编造信号, (b) 按 token 加权的
        # Σ_t adv_t 仍然精确为 0 —— 后者是 loss 侧 token 等权 (const_len) 下"整批净
        # 推力"的定义, 不为 0 就是在无差别地推高/压低所有 token。value-head 模式
        # has_signal 全 True, 行为与 2026-09-11 之前逐位相同。
        w_idx = [i for i in range(len(advs)) if has_signal[i]]
        if global_whitening is not None:
            count, batch_mean, batch_std = global_whitening
            if count > 1:
                _standardize(w_idx, batch_mean, batch_std)
        else:
            active = (torch.cat([advs[i][masks[i]] for i in w_idx])
                      if w_idx else torch.empty(0))
        if global_whitening is None and active.numel() > 1:
            batch_mean, batch_std = active.mean(), active.std()
            if whiten_groups is None:
                _standardize(w_idx, batch_mean, batch_std)
            else:
                by_group: dict = {}
                for i in w_idx:
                    by_group.setdefault(whiten_groups[i], []).append(i)
                n_singleton = 0
                for idx in by_group.values():
                    ga = torch.cat([advs[i][masks[i]] for i in idx])
                    # 组里只剩一条 (drop_truncated/评测失败把兄弟丢光了) 时不能拿自己
                    # 归一自己 -- 那会把自己的 reward 减掉, 只剩 value 残差的噪声。
                    # 退回全 batch 的统计量, 保持和其它组同一个尺度。
                    if len(idx) < 2 or ga.numel() < 2:
                        n_singleton += len(idx)
                        _standardize(idx, batch_mean, batch_std)
                    else:
                        _standardize(idx, ga.mean(), ga.std())
                metrics["ppo/adv_whiten_singleton_frac"] = n_singleton / len(advs)

    # Gate AFTER whitening: centering can change the sign. Keep every response
    # in the original baseline/whitening statistics, without recomputing them.
    # Unknown correctness is left unchanged by either independent gate.
    blocked = [torch.zeros_like(m) for m in masks]
    for name, enabled, correctness, sign in (
        ("positive_incorrect", mask_positive_incorrect, False, 1),
        ("negative_correct", mask_negative_correct, True, -1),
    ):
        if enabled:
            gate = [masks[i] & (a * sign > 0)
                    if response_correctness[i] is correctness else torch.zeros_like(masks[i])
                    for i, a in enumerate(advs)]
            for i, b in enumerate(gate):
                blocked[i] |= b
            metrics[f"ppo/{name}_masked_frac"] = (
                sum(bool(b.any()) for b in gate) / max(1, len(datums)))
            metrics[f"ppo/{name}_masked_token_frac"] = (
                sum(int(b.sum()) for b in gate) / max(1, sum(int(m.sum()) for m in masks)))

    out: list[Datum] = []
    for i, (d, a, r) in enumerate(zip(datums, advs, rets)):
        li = dict(d.loss_fn_inputs)
        if mask_positive_incorrect or mask_negative_correct:
            li["mask"] = TensorData.from_torch(masks_f[i].masked_fill(blocked[i], 0.0))
            a = a.masked_fill(blocked[i], 0.0)
        li["advantages"] = TensorData.from_torch(a)
        li["returns"] = TensorData.from_torch(r)
        if ref_logprobs_list is not None:
            li["ref_logprobs"] = TensorData.from_torch(
                ref_logprobs_list[i].detach().float()
            )
        out.append(Datum(model_input=d.model_input, loss_fn_inputs=li))
    return out, metrics
