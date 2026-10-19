"""Agentic GRPO —— ARM 4c (Qwen3.8-27B) 的 GRPO 训练路径, 对话流程换成多轮工具调用.

和 ARM ``rlhf/ppo/ppo_loop.py`` (8821e43) 的对应:

  保留 (逐项同口径)   Dr-GRPO: 组内只减均值不除 std, 白化只中心化 (whiten_std=False);
                     adv_unit=trajectory (组键 = 题, 一条轨迹一个 arm); 常数分母
                     const_len = 本批生成上限; PPO-clip [1-0.2, 1+0.28]; num_substeps=2;
                     max_tokens -> max_tokens_late 的单向截断率闩; 非有限守卫 + 熔断;
                     空批熔断; wave 异步 rollout + extra_samples; NCCL 权重同步;
                     ZeRO-3 + optimizer offload; 跨 rank 加权合并指标; 后台存 checkpoint。
  换掉               Kevin refine 的 multi_turn 环境 (每轮整段重写 prompt, 不回放思考) ->
                     ``arl.agent.protocol`` 的消息层环境 (CudaEnv 等), harness 式历史:
                     思考原样回放、token 只追加, 一条轨迹一个 datum;
                     每轮奖励 + gamma 折扣 -> 轨迹终局奖励 (里程碑);
                     require_full_turns (agent 会提前交卷) -> 去掉; drop_truncated ->
                     truncated_turn (缺省 mask: 截断轮不进 loss, 轨迹继续)。
  删掉 (4c 下是死码)  RM / π_ref KL / value head / tower critic / 长度奖励 / think_budget /
                     sampling_replay / tailrl / 轮间同步 rollout / curriculum (按 level)。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Any

import chz
import numpy as np
import torch

from arl.agent.render import AgentRenderer
from arl.agent.rollout import BestSolutions, RolloutParams, TurnLogger, run_agent_rollout
from arl.algorithms.loop import _make_dummy_datum, get_updated_sampling_client, train_step
from arl.algorithms.ppo_data import finalize_ppo_data
from arl.env import setup_environment
from arl.models import tinker_compat as tinker
from arl.models.service import sampling_window
from arl.models.types import LossFnType
from arl.utils import checkpoint_utils, ml_log
from arl.utils import distributed as dist_utils

logger = logging.getLogger(__name__)


@chz.chz
class GRPOConfig:
    # ── 模型 / 输出 ──
    model_name: str
    log_path: str
    load_checkpoint_path: str | None = None   # 权重热启动 (不含优化器状态, 同 ARM)
    start_batch: int = 0                       # 续跑: 跳过前 N 批 (数据顺序由 seed 决定, 可复现)
    wandb_project: str | None = None
    wandb_name: str | None = None

    # ── 环境 ──
    env: str = "cuda"
    train_set: str = "ops6k"
    test_set: str = ""
    test_fraction: float = 0.02
    task_limit: int = 0
    test_limit: int = 0
    env_max_turns: int = 16
    env_enable_analyze: bool = False
    env_reward_mode: str = "best"              # best (论文, best-of-trajectory) | final (交卷判分)
    eval_endpoint: str = ""
    eval_max_concurrency: int = 0

    # ── 渲染 ──
    enable_thinking: bool = True
    reasoning_effort: str = ""                 # 空 = 模板缺省 (CudaAgent RFT 口径); ARM 4c 是 medium

    # ── 批形状 ──
    batch_size: int = 8                        # 每批题数 (须 >= rank 数, 按 rank 切)
    group_size: int = 8
    num_epochs: int = 1
    shuffle_seed: int = 0
    extra_samples: int = 0                     # 组内超额采样 (ARM 4c = 2)

    # ── 采样 ──
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 16384                    # 每轮生成上限 (含 <think>)
    max_tokens_late: int = 0                   # 截断率超阈后单向放宽到这里 (0 = 不放宽)
    max_tokens_switch_trunc: float = 0.08
    max_tokens_switch_patience: int = 1
    prompt_budget_tokens: int = 32768          # 每轮 prompt 上限 (= 累积上下文; 超了环境逐级压缩历史)
    sampling_max_model_len: int = 0            # 0 = prompt_budget + 峰值生成上限
    truncated_turn: str = "mask"               # mask | train | drop_traj
    poll_seconds: float = 10.0

    # ── 优化 (ARM 4_train_grpo.conf 的值) ──
    learning_rate: float = 1e-6
    grad_clip: float = 0.05
    num_substeps: int = 2
    clip_low: float = 0.2
    clip_high: float = 0.28
    loss_fn: LossFnType = "ppo"
    loss_fn_config: dict[str, Any] | None = None
    group_adv_std: bool = False                # Dr-GRPO: 不除 std
    group_adv_std_floor: float = 0.0
    whiten_advantages: bool = True
    whiten_std: bool = False
    length_weighted_baseline: bool = False     # ARM 4c 实际生效值 (base conf 显式 False)
    nonfinite_max_consecutive: int = 3

    # ── 评估 / 存档 ──
    eval_every: int = 5
    eval_at_start: bool = True
    eval_num_samples: int = 1
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95
    eval_top_k: int = -1
    save_every: int = 5
    save_trajectories: bool = True
    save_best_solutions: bool = True


def _gen_cap_latch(cfg: GRPOConfig, gen_cap: int, n_over: int, trunc: float) -> tuple[int, int]:
    """ARM 同款: 全局截断率连续 patience 批 >= 阈值 -> 单向切到 max_tokens_late。"""
    if not cfg.max_tokens_late or gen_cap >= cfg.max_tokens_late:
        return gen_cap, n_over
    n_over = n_over + 1 if trunc >= cfg.max_tokens_switch_trunc else 0
    if n_over < cfg.max_tokens_switch_patience:
        return gen_cap, n_over
    logger.info(f"[gen_cap] truncated_rate {trunc:.1%} >= {cfg.max_tokens_switch_trunc:.0%} for "
                f"{n_over} batch(es): generation cap {gen_cap} -> {cfg.max_tokens_late} (one-way)")
    return cfg.max_tokens_late, n_over


def _batches(n_tasks: int, cfg: GRPOConfig) -> list[list[int]]:
    import random
    order: list[int] = []
    for ep in range(cfg.num_epochs):
        idx = list(range(n_tasks))
        random.Random(cfg.shuffle_seed + ep).shuffle(idx)
        order.extend(idx)
    return [order[i:i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]


def _validate(cfg: GRPOConfig, world_size: int) -> None:
    if cfg.group_size < 2:
        raise SystemExit("group_size < 2: 组内基线需要至少 2 条兄弟轨迹, 否则 advantage 恒 0")
    if cfg.batch_size < world_size:
        raise SystemExit(f"batch_size={cfg.batch_size} < {world_size} ranks: 有 rank 一道题都分不到")
    if cfg.batch_size % world_size:
        logger.warning(f"batch_size={cfg.batch_size} 不整除 {world_size} ranks: 各 rank 题数不等, "
                       f"快的 rank 会空等")
    if cfg.truncated_turn not in ("mask", "train", "drop_traj"):
        raise SystemExit(f"truncated_turn={cfg.truncated_turn!r}: mask | train | drop_traj")
    if cfg.loss_fn != "ppo":
        raise SystemExit(f"loss_fn={cfg.loss_fn!r}: 训练客户端只实现了 ppo")


async def _run_eval(source, sampling_client, renderer, cfg: GRPOConfig, gen_cap: int,
                    rank: int, world_size: int, batch_idx: int,
                    best: BestSolutions, tlog: TurnLogger) -> dict[str, float]:
    """held-out 评估 (COLLECTIVE): 测试题按 rank 摊开, 每题 eval_num_samples 条, 不训练。"""
    tasks = dist_utils.shard_list(list(source.test_tasks()), rank, world_size)
    p = RolloutParams(gen_cap=gen_cap, prompt_budget=cfg.prompt_budget_tokens,
                      temperature=cfg.eval_temperature, top_p=cfg.eval_top_p, top_k=cfg.eval_top_k,
                      group_size=cfg.eval_num_samples, poll_seconds=cfg.poll_seconds)
    t0 = time.time()
    res = await run_agent_rollout(tasks, source, sampling_client, renderer, p, train=False,
                                  uid_prefix=f"eval{batch_idx}-", log_turns=tlog.enabled)
    best.update(batch_idx, res.trajs, tag="eval")
    tlog.write(batch_idx, res.trajs, tag="eval")
    n = c1 = c2 = c3 = rsum = 0.0
    log_se = log_sc = n_se = n_sc = 0.0
    for t in res.trajs:
        if t.dropped:
            continue
        n += 1
        rsum += t.reward
        c1 += t.reward >= 1
        c2 += t.reward >= 2
        c3 += t.reward >= 3
        s = t.env.summary() if hasattr(t.env, "summary") else {}
        if t.reward >= 1:
            for key, acc in (("speedup_vs_eager", 0), ("speedup_vs_compile", 1)):
                v = s.get(key)
                if v and v > 0:
                    if acc == 0:
                        log_se += math.log(v); n_se += 1
                    else:
                        log_sc += math.log(v); n_sc += 1
    vals = [n, rsum, c1, c2, c3, log_se, n_se, log_sc, n_sc]
    if world_size > 1:
        vals = dist_utils.all_reduce_sum_list_gloo([float(v) for v in vals])
    n, rsum, c1, c2, c3, log_se, n_se, log_sc, n_sc = vals
    out = {"eval/n": n, "time/eval": time.time() - t0,
           "eval/truncated_rate": res.metrics.get("rollout/truncated_rate", 0.0)}
    if n:
        out.update({"eval/reward_mean": rsum / n, "eval/pass_rate": c1 / n,
                    "eval/faster_eager_rate": c2 / n, "eval/faster_compile_rate": c3 / n})
    if n_se:
        out["eval/geomean_speedup_eager"] = math.exp(log_se / n_se)
    if n_sc:
        out["eval/geomean_speedup_compile"] = math.exp(log_sc / n_sc)
    return out


async def run_grpo(cfg: GRPOConfig) -> None:
    setup_environment()
    rank, world_size, local_rank = dist_utils.dist_info()
    is_main = rank == 0
    if world_size > 1:
        import random
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
    _validate(cfg, world_size)
    os.makedirs(cfg.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(log_dir=cfg.log_path, wandb_project=cfg.wandb_project,
                                     config=cfg, wandb_name=cfg.wandb_name) if is_main else None

    from arl.envs import make_task_source
    source = make_task_source(
        cfg.env, train_set=cfg.train_set, test_set=cfg.test_set, test_fraction=cfg.test_fraction,
        limit=cfg.task_limit, test_limit=cfg.test_limit, endpoint=cfg.eval_endpoint,
        max_concurrency=cfg.eval_max_concurrency, max_turns=cfg.env_max_turns,
        enable_analyze=cfg.env_enable_analyze, reward_mode=cfg.env_reward_mode)
    if is_main and hasattr(source, "check"):
        logger.info(f"[env] eval service: {source.check()}")

    peak = max(cfg.max_tokens, cfg.max_tokens_late)
    win = sampling_window(cfg.model_name, peak,
                          cfg.sampling_max_model_len or cfg.prompt_budget_tokens + peak)
    if win - peak < cfg.prompt_budget_tokens:
        logger.warning(f"sampling window {win} - peak gen {peak} < prompt_budget "
                       f"{cfg.prompt_budget_tokens}: 长 prompt 会被引擎拒掉")
    service_client = tinker.ServiceClient(
        base_url=None, model_name=cfg.model_name, learning_rate=cfg.learning_rate,
        grad_clip=cfg.grad_clip, max_tokens=cfg.max_tokens, sampling_max_model_len=win,
        sampling_gpu_memory_utilization=float(os.environ.get("KERNEL_RL_SAMPLING_GPU_UTIL", "0.3")),
        enable_value_head=False)
    if cfg.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.load_checkpoint_path)
    else:
        training_client = await service_client.create_training_client_async(cfg.model_name)
    tokenizer = training_client.get_tokenizer()
    renderer = AgentRenderer(tokenizer, enable_thinking=cfg.enable_thinking,
                             reasoning_effort=cfg.reasoning_effort or None)
    sc = await get_updated_sampling_client(training_client)
    sampling_client = sc[0] if isinstance(sc, tuple) else sc

    train_tasks = source.train_tasks()
    batches = _batches(len(train_tasks), cfg)
    num_batches = len(batches)
    logger.info(
        f"GRPO(agentic, Dr-GRPO, G={cfg.group_size}+{cfg.extra_samples}, adv per trajectory"
        f"{', std floor %g' % cfg.group_adv_std_floor if cfg.group_adv_std else ', no std division'}"
        f"): {num_batches} batches x {cfg.batch_size} tasks, env={cfg.env}/{cfg.train_set}, "
        f"max_turns={cfg.env_max_turns}, gen_cap={cfg.max_tokens}"
        f"{'->' + str(cfg.max_tokens_late) if cfg.max_tokens_late else ''}, "
        f"prompt_budget={cfg.prompt_budget_tokens}, window={win}")

    best = BestSolutions(cfg.log_path, rank, cfg.save_best_solutions)
    tlog = TurnLogger(cfg.log_path, rank, cfg.save_trajectories)
    gen_cap, n_over = cfg.max_tokens, 0
    eval_cap = peak
    nonfinite_total = nonfinite_consec = empty_consec = 0
    has_test = bool(source.test_tasks())

    if cfg.eval_at_start and cfg.eval_every > 0 and has_test and cfg.start_batch == 0:
        ev = await _run_eval(source, sampling_client, renderer, cfg, eval_cap, rank, world_size,
                             -1, best, tlog)
        if is_main:
            logger.info(f"before training: {_fmt_eval(ev)}")
            if ml_logger is not None:
                ml_logger.log_metrics({**ev, "eval/before_train": 1.0}, step=0)

    for batch_idx in range(cfg.start_batch, num_batches):
        t0 = time.time()
        idxs = dist_utils.shard_list(batches[batch_idx], rank, world_size)
        tasks = [train_tasks[i] for i in idxs]

        # 1) rollout
        p = RolloutParams(gen_cap=gen_cap, prompt_budget=cfg.prompt_budget_tokens,
                          temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                          group_size=cfg.group_size, extra_samples=cfg.extra_samples,
                          truncated_turn=cfg.truncated_turn, poll_seconds=cfg.poll_seconds)
        res = await run_agent_rollout(tasks, source, sampling_client, renderer, p, train=True,
                                      uid_prefix=f"b{batch_idx}-", log_turns=tlog.enabled)
        t_rollout = time.time() - t0
        batch_gen_cap = gen_cap
        gen_cap, n_over = _gen_cap_latch(cfg, gen_cap, n_over,
                                         res.metrics.get("rollout/truncated_rate", 0.0))
        best.update(batch_idx, res.trajs)
        tlog.write(batch_idx, res.trajs)

        # 2) advantages: 组 = 题, 单元 = 轨迹; Dr-GRPO 只减均值
        adv_metrics: dict[str, float] = {}
        if res.datums:
            data, adv_metrics = finalize_ppo_data(
                res.datums, res.returns, whiten=cfg.whiten_advantages and not cfg.group_adv_std,
                groups=res.groups, units=res.units, group_adv_std=cfg.group_adv_std,
                whiten_std=cfg.whiten_std, length_weighted_baseline=cfg.length_weighted_baseline,
                group_adv_std_floor=cfg.group_adv_std_floor)
        else:
            data = []
        _t = time.time()
        if world_size > 1:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            gmax = int(dist_utils.all_reduce_max(len(data)))
            data = data + [_make_dummy_datum(pad_id) for _ in range(gmax - len(data))]
        global_n = dist_utils.all_reduce_sum(len(res.returns)) if world_size > 1 else len(res.returns)
        t_rank_sync = time.time() - _t
        if global_n == 0:
            empty_consec += 1
            logger.warning(f"batch {batch_idx}: no trainable turns on any rank "
                           f"({empty_consec}/{cfg.nonfinite_max_consecutive or '∞'})")
            if cfg.nonfinite_max_consecutive and empty_consec >= cfg.nonfinite_max_consecutive:
                raise RuntimeError(f"{empty_consec} consecutive empty batches -- sampler weights "
                                   f"or the eval service are likely broken; stopping")
            continue
        empty_consec = 0

        # 3) 一次更新
        loss_cfg = dict(cfg.loss_fn_config or {})
        loss_cfg["clip_low_threshold"] = float(cfg.clip_low)
        loss_cfg["clip_high_threshold"] = float(cfg.clip_high)
        loss_cfg["dr_grpo_const_len"] = float(batch_gen_cap)
        _t = time.time()
        _, train_metrics = await train_step(data, training_client, cfg.learning_rate,
                                            cfg.num_substeps, cfg.loss_fn, loss_cfg)
        t_train = time.time() - _t
        n_skip = int(train_metrics.get("ppo/nonfinite_skipped_substeps", 0))
        n_run = int(train_metrics.get("ppo/substeps", 0))
        batch_skipped = n_skip > 0 and n_skip >= n_run
        nonfinite_total += n_skip
        nonfinite_consec = nonfinite_consec + 1 if batch_skipped else 0
        train_metrics["ppo/nonfinite_skips_total"] = float(nonfinite_total)

        # 4) 权重同步
        _t = time.time()
        if batch_skipped:
            logger.warning(f"batch {batch_idx + 1}: all {n_run} substeps non-finite -- weights "
                           f"unchanged, skipping sync")
        else:
            sc = await get_updated_sampling_client(training_client)
            sampling_client = sc[0] if isinstance(sc, tuple) else sc
        t_sync = time.time() - _t
        if cfg.nonfinite_max_consecutive and nonfinite_consec >= cfg.nonfinite_max_consecutive:
            raise RuntimeError(f"{nonfinite_consec} batches with non-finite loss/logprob; stopping")

        # 5) 周期评估
        eval_metrics: dict[str, float] = {}
        if cfg.eval_every > 0 and has_test and (
                (batch_idx + 1) % cfg.eval_every == 0 or batch_idx + 1 == num_batches):
            eval_metrics = await _run_eval(source, sampling_client, renderer, cfg, eval_cap,
                                           rank, world_size, batch_idx, best, tlog)
            if is_main:
                logger.info(f"batch {batch_idx + 1}: {_fmt_eval(eval_metrics)}")

        metrics: dict[str, float] = {
            **res.metrics, **adv_metrics, **train_metrics, **eval_metrics,
            "progress/batch": float(batch_idx),
            "rollout/gen_cap": float(batch_gen_cap),
            "ppo/reward_mean": float(np.mean(res.returns)) if res.returns else 0.0,
            "time/rollout": t_rollout, "time/rank_sync": t_rank_sync,
            "time/train_step": t_train, "time/weight_sync": t_sync,
            "time/total": time.time() - t0,
        }
        if world_size > 1:
            metrics = _reduce(metrics, res, train_metrics)
        if is_main:
            logger.info(
                f"batch {batch_idx + 1}/{num_batches}: reward={metrics.get('rollout/reward_mean', 0):.3f} "
                f"correct={metrics.get('rollout/correct_rate', 0):.1%} "
                f">compile={metrics.get('rollout/faster_compile_rate', 0):.1%} "
                f"turns={metrics.get('rollout/turns_per_traj', 0):.1f} "
                f"trunc={metrics.get('rollout/truncated_rate', 0):.1%} "
                f"clip={metrics.get('ppo/clip_frac', 0):.1%} gn={metrics.get('ppo/grad_norm', 0):.3f} "
                f"adv_zero={metrics.get('ppo/adv_zero_frac', 0):.1%} "
                f"({metrics['time/total']:.0f}s = rollout {t_rollout:.0f} + train {t_train:.0f} "
                f"+ sync {t_sync:.0f})")
            if ml_logger is not None:
                ml_logger.log_metrics(metrics, step=batch_idx)

        if cfg.save_every > 0 and (batch_idx + 1) % cfg.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client, name=f"{batch_idx + 1:06d}",
                log_path=cfg.log_path, loop_state={"batch": batch_idx + 1},
                kind="sampler", background=True)

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client, name="final", log_path=cfg.log_path,
        loop_state={"batch": num_batches}, kind="sampler")
    if is_main and ml_logger is not None:
        ml_logger.close()
    await _teardown(training_client, source)


_MEAN_KEYS = ("rollout/reward_mean", "rollout/correct_rate", "rollout/faster_eager_rate",
              "rollout/faster_compile_rate", "rollout/turns_per_traj", "rollout/response_len_mean",
              "rollout/n_tool_calls_per_turn", "rollout/n_invalid_calls_per_turn",
              "rollout/format_ok_per_turn", "rollout/eval_calls_per_traj", "rollout/append_rate",
              "rollout/datums_per_traj", "ppo/reward_mean",
              "ppo/adv_zero_frac")
_SUM_KEYS = ("rollout/n_trajs", "rollout/ctx_overflow_trajs", "rollout/sample_fail_trajs",
             "rollout/env_error_trajs", "rollout/extra_dropped_trajs", "rollout/infra_dropped_trajs",
             "ppo/nonfinite_lp_tokens", "ppo/nonfinite_lp_active", "ppo/nonfinite_datums")


def _reduce(metrics: dict, res, train_metrics: dict) -> dict:
    """COLLECTIVE: 各 rank 本地的均值类按轨迹数加权合并, 计数类求和; 训练侧按 active token 加权。"""
    for k in _MEAN_KEYS + _SUM_KEYS:
        metrics.setdefault(k, 0.0)
    metrics = dist_utils.reduce_weighted_metrics(
        metrics, mean_keys=list(_MEAN_KEYS),
        local_weight=float(res.metrics.get("rollout/n_trajs", 0.0)), sum_keys=list(_SUM_KEYS))
    tok_keys = ["ppo/mean_ratio", "ppo/clip_frac", "ppo/approx_kl"]
    metrics = dist_utils.reduce_weighted_metrics(
        metrics, mean_keys=tok_keys, local_weight=float(train_metrics.get("ppo/n_active_tokens", 0.0)),
        sum_keys=["loss", "ppo/n_active_tokens"])
    if metrics.get("ppo/n_active_tokens", 0.0) <= 0:
        for k in tok_keys + ["ppo/n_active_tokens"]:
            metrics.pop(k, None)
    for k in ("rollout/response_len_p95", "rollout/prompt_len_max", "rollout/datum_len_max",
              "time/rollout_wall"):
        metrics[k] = dist_utils.all_reduce_max(float(metrics.get(k, 0.0)))
    return metrics


def _fmt_eval(ev: dict) -> str:
    if not ev.get("eval/n"):
        return "eval: no judged trajectories"
    return (f"eval n={int(ev['eval/n'])} pass={ev.get('eval/pass_rate', 0):.1%} "
            f">eager={ev.get('eval/faster_eager_rate', 0):.1%} "
            f">compile={ev.get('eval/faster_compile_rate', 0):.1%} "
            f"reward={ev.get('eval/reward_mean', 0):.3f} "
            f"geo_x_eager={ev.get('eval/geomean_speedup_eager', float('nan')):.2f} "
            f"({ev.get('time/eval', 0):.0f}s)")


async def _teardown(training_client, source) -> None:
    from arl.utils.distributed import barrier, destroy
    try:
        if hasattr(source, "close"):
            await source.close()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[teardown] env close failed (ignored): {e}")
    for step, fn in (("barrier", barrier), ("destroy_pg", destroy),
                     ("engine_shutdown", training_client.shutdown)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[teardown] {step} failed (ignored): {e}")
