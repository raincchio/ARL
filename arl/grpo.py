"""Agentic GRPO —— ARM 4c (Qwen3.8-27B) 的 GRPO 训练路径, 对话流程换成多轮工具调用.

和 ARM ``rlhf/ppo/ppo_loop.py`` 的对应 (2026-11-16 工作区，HEAD 021af58，含未提交训练改动):

  保留 (逐项同口径)   Dr-GRPO: 组内只减均值不除 std, 白化只中心化 (whiten_std=False);
                     adv_unit=trajectory (组键 = 题, 一条轨迹一个 arm); 常数分母
                     const_len = 本批生成上限; PPO-clip [1-0.2, 1+0.28]; num_substeps=2;
                     固定每轮生成上限; 非有限守卫 + 熔断;
                     空批熔断; wave 异步 rollout + extra_samples; NCCL 权重同步;
                     ZeRO-3 + optimizer offload; 跨 rank 加权合并指标; 后台存 checkpoint。
  换掉               Kevin refine 的 multi_turn 环境 (每轮整段重写 prompt, 不回放思考) ->
                     ``arl.agent.protocol`` 的消息层环境 (CudaEnv 等), harness 式历史:
                     思考原样回放、token 只追加, 一条轨迹一个 datum;
                     每轮奖励 + gamma 折扣 -> 轨迹终局奖励 (里程碑);
                     require_full_turns (agent 会提前交卷) -> 去掉; drop_truncated ->
                     truncated_turn (缺省 mask: 截断轮不进 loss, 轨迹继续)。
  可选算法           TailRL、正确性优势屏蔽、thinking cost；前 5 次 policy 更新半学习率；
                     优化器 checkpoint / 恢复、打包后的跨 rank token 负载重排；sampling replay、
                     MoE router replay、固定 RMSNorm 和训练/采样共卡。
  未接入             RM / π_ref KL / value head / tower critic / 长度奖励 /
                     轮间同步 rollout / curriculum (按 level)。
"""

from __future__ import annotations

import asyncio
import logging
import inspect
import math
import os
import time
from typing import Any

import chz
import numpy as np
import torch

from arl.agent.rollout import BestSolutions, RolloutParams, TurnLogger
from arl.algorithms.loop import _make_dummy_datum, get_updated_sampling_client, train_step
from arl.algorithms.ppo_data import finalize_ppo_data
from arl.algorithms.thinking_cost import validate_thinking_cost
from arl.env import setup_environment
from arl.envs import get_env_spec, make_task_source_from_config
from arl.harnesses import get_harness, resolve_pair
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
    load_checkpoint_path: str | None = None   # 权重热启动；resume_optimizer=True 时恢复优化器
    resume_optimizer: bool = False
    start_batch: int = 0                       # 续跑: 跳过前 N 批 (数据顺序由 seed 决定, 可复现)
    wandb_project: str | None = "ARL"
    wandb_name: str | None = None

    # ── 环境 ──
    env: str = "coding.cuda"                 # category-qualified environment name; cuda/swe aliases accepted
    harness: str = ""                     # empty = the selected environment default
    environments: list[dict] | None = None   # weighted, per-batch environment mixture
    mixture_steps_per_epoch: int = 0         # 0 = ceil(total train tasks / batch_size)
    eval_harness: str = ""                   # empty = same harness as training
    train_set: str = "ops6k"
    test_set: str = ""
    test_fraction: float = 0.02
    task_limit: int = 0
    test_limit: int = 0
    env_max_turns: int | None = None       # None = environment/harness default
    env_enable_analyze: bool = False
    env_reward_mode: str = "best"              # best (论文, best-of-trajectory) | final (交卷判分)
    eval_endpoint: str = ""
    eval_max_concurrency: int = 0
    # SWE-bench worker (local Docker or SSH host with the official harness).
    swe_dataset_split: str = "test"
    swe_endpoint: str = ""
    swe_token_file: str = ""
    swe_ssh_host: str = ""
    swe_ssh_port: int = 22
    swe_python: str = "python3"
    swe_worker_script: str = "/root/workspace/SweContainerService/swe_service/worker.py"
    swe_work_dir: str = "/tmp/arl-swe-runs"
    swe_proxy: str = ""
    swe_memory: str = "4g"
    swe_cpus: float = 2.0
    swe_command_timeout: int = 60
    swe_eval_timeout: int = 1800

    # IFEval official checker, isolated from the model runtime.
    ifeval_reward_mode: str = "strict"       # strict all-pass 0/1 | fraction of strict checks
    ifeval_split_seed: int = 0
    ifeval_python: str = ""
    ifeval_upstream: str = ""
    ifeval_timeout: float = 60

    # MultiCodeIF: official constraint checks + official repair prompt, local tokens.
    multicodeif_upstream: str = ""
    multicodeif_dependencies: str = ""
    multicodeif_suite: str = "single_level"
    multicodeif_constraints: list[str] | None = None
    multicodeif_languages: list[str] | None = None
    multicodeif_split_seed: int = 0
    multicodeif_max_rounds: int = 3
    multicodeif_reward_mode: str = "strict"
    multicodeif_timeout: float = 600
    multicodeif_judge_base_url: str = ""
    multicodeif_judge_model: str = ""
    multicodeif_judge_api_key_env: str = "MULTICODEIF_JUDGE_API_KEY"

    # ProgramBench cleanroom workspaces and official evaluator.
    programbench_ssh_host: str = ""
    programbench_ssh_port: int = 22
    programbench_python: str = ""
    programbench_work_dir: str = "/tmp/arl-programbench-runs"
    programbench_blob_dir: str = ""
    programbench_memory: str = "60g"
    programbench_cpus: float = 20
    programbench_task_ids: str = ""
    programbench_split_seed: int = 0
    programbench_command_timeout: int = 180
    programbench_eval_timeout: int = 3600
    programbench_eval_cpus: int = 4
    programbench_branch_workers: int = 1
    programbench_reward_mode: str = "fraction"

    # ── 渲染 ──
    enable_thinking: bool = True
    reasoning_effort: str = ""                 # 空 = 模板缺省 (CudaAgent RFT 口径); ARM 4c 是 medium

    # ── 批形状 ──
    batch_size: int = 8                        # 每批题数；题内轨迹展开后按 rank 分配
    group_size: int = 8
    num_epochs: int = 1
    shuffle_seed: int = 0
    extra_samples: int = 0                     # 组内超额采样 (ARM 4c = 2)

    # ── 采样 ──
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 16384                    # 每轮生成上限 (含 <think>)
    prompt_budget_tokens: int = 32768          # 每轮 prompt 上限 (= 累积上下文; 超了环境逐级压缩历史)
    enable_prefix_caching: bool | None = None # unset follows vLLM/model default
    mamba_cache_mode: str = 'none'             # align for Qwen hybrid prefix caching
    sampling_prefill_tokens: int = 0           # per-engine runtime scheduling cap; 0 = vLLM default
    sampling_prefill_batch_tokens: int = 0     # compile/profile cap, independent of runtime scheduling
    sampling_max_model_len: int = 0            # 0 = prompt_budget + 峰值生成上限
    truncated_turn: str = "mask"               # mask | train | drop_traj
    poll_seconds: float = 10.0
    think_budget: int = 0
    think_force_text: str = (
        "\n\nConsidering the limited time by the user, I have to give the solution "
        "based on the thinking directly now.\n</think>\n\n"
    )
    think_cost_free_tokens: int = 0           # 每次模型回复免计费的 sampled thinking tokens
    sampling_replay: bool = False
    router_replay: bool = False
    think_cost_lambda: float = 0.0            # 每 8192 个 sampled thinking token 的代价；轨迹累加

    # ── 优化 (ARM 4_train_grpo.conf 的值) ──
    learning_rate: float = 1e-6
    warmup_steps: int = 5                     # 按实际 policy 更新次数计，跳过的步不消耗
    warmup_lr_scale: float = 0.5
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
    adv_estimator: str = "grpo"                # grpo | tailrl
    mask_positive_incorrect: bool = False
    mask_negative_correct: bool = False
    nonfinite_max_consecutive: int = 3

    # ── 评估 / 存档 ──
    eval_every: int = 5
    eval_at_start: bool = True
    eval_num_samples: int = 1
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95
    eval_top_k: int = -1
    save_every: int = 5
    save_optimizer: bool = False              # 开启后只保留最新一份优化器状态
    save_trajectories: bool = True
    save_best_solutions: bool = True


def _batches(n_tasks: int, cfg: GRPOConfig) -> list[list[int]]:
    import random
    order: list[int] = []
    for ep in range(cfg.num_epochs):
        idx = list(range(n_tasks))
        random.Random(cfg.shuffle_seed + ep).shuffle(idx)
        order.extend(idx)
    return [order[i:i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]


def _validate(cfg: GRPOConfig, world_size: int) -> None:
    if getattr(cfg, 'environments', None) is not None:
        from arl.envs.mixture import expand_config
        if cfg.mixture_steps_per_epoch < 0 or cfg.num_epochs < 1:
            raise ValueError('invalid mixture_steps_per_epoch/num_epochs')
        for member in expand_config(cfg):
            _validate(member.cfg, world_size)
        return
    get_env_spec(cfg.env)
    if cfg.env_max_turns is not None and cfg.env_max_turns < 1:
        raise ValueError('env_max_turns must be positive or None (environment default)')
    resolve_pair(cfg.env, cfg.harness, operation='train', model_backend='local')
    resolve_pair(cfg.env, cfg.eval_harness or cfg.harness, operation='evaluate', model_backend='local')
    validate_thinking_cost(cfg)
    from arl.algorithms.training_options import validate_training_options
    validate_training_options(cfg)
    if get_env_spec(cfg.env).name == 'instruction_following.multicodeif':
        from arl.envs.instruction_following.multicodeif import make_source_from_config
        # Configuration validation only; no dataset/evaluator/model is loaded.
        make_source_from_config(cfg)
    if get_env_spec(cfg.env).name == 'instruction_following.ifeval':
        if cfg.ifeval_reward_mode not in ('strict', 'fraction'):
            raise ValueError('ifeval_reward_mode must be strict or fraction')
        if not 0 <= cfg.test_fraction < 1 or cfg.ifeval_timeout <= 0:
            raise ValueError('invalid IFEval test_fraction/timeout')
    if get_env_spec(cfg.env).name == 'coding.programbench':
        if cfg.programbench_reward_mode not in ('strict', 'fraction'):
            raise ValueError('programbench_reward_mode must be strict or fraction')
        if not 0 <= cfg.test_fraction < 1 or min(cfg.programbench_eval_timeout, cfg.programbench_command_timeout, cfg.programbench_eval_cpus, cfg.programbench_branch_workers) <= 0:
            raise ValueError('invalid ProgramBench split/timeouts/resources')
    if cfg.num_substeps < 1:
        raise ValueError("num_substeps must be >= 1")
    if cfg.warmup_steps < 0 or not 0 < cfg.warmup_lr_scale <= 1:
        raise ValueError("warmup_steps must be >= 0; warmup_lr_scale must be in (0, 1]")
    if cfg.adv_estimator not in ("grpo", "tailrl"):
        raise ValueError("adv_estimator must be grpo or tailrl")
    if cfg.resume_optimizer and not cfg.load_checkpoint_path:
        raise ValueError("resume_optimizer requires load_checkpoint_path")
    if cfg.resume_optimizer and not os.path.isdir(os.path.join(cfg.load_checkpoint_path, "ds_state")):
        raise ValueError("resume_optimizer requires a checkpoint containing ds_state")
    if cfg.group_size < 2:
        raise SystemExit("group_size < 2: 组内基线需要至少 2 条兄弟轨迹, 否则 advantage 恒 0")
    if cfg.extra_samples < 0:
        raise ValueError("extra_samples must be nonnegative")
    n_trajectories = cfg.batch_size * (cfg.group_size + cfg.extra_samples)
    if n_trajectories < world_size:
        if int(os.environ.get('KERNEL_RL_PACK_TOKENS','0')) <= 0 or os.environ.get('KERNEL_RL_REBALANCE','1') == '0':
            raise SystemExit(f'{n_trajectories} trajectories < {world_size} ranks requires PACK_TOKENS>0 and REBALANCE=1')
    elif n_trajectories % world_size:
        logger.warning(f"{n_trajectories} 条轨迹不整除 {world_size} ranks: 各 rank 轨迹数相差 1")
    if cfg.truncated_turn not in ("mask", "train", "drop_traj"):
        raise SystemExit(f"truncated_turn={cfg.truncated_turn!r}: mask | train | drop_traj")
    if cfg.loss_fn != "ppo":
        raise SystemExit(f"loss_fn={cfg.loss_fn!r}: 训练客户端只实现了 ppo")


def _make_source(cfg):
    return make_task_source_from_config(cfg)


def _run_identity(cfg, *, evaluate=False):
    if cfg.environments is not None:
        from arl.envs.mixture import normalize_members
        return {'env': 'mixture', 'model': cfg.model_name,
                'environments': list(normalize_members(cfg.environments, harness=cfg.harness)),
                'shuffle_seed': cfg.shuffle_seed, 'num_epochs': cfg.num_epochs,
                'batch_size': cfg.batch_size, 'mixture_steps_per_epoch': cfg.mixture_steps_per_epoch}
    pair = resolve_pair(cfg.env, (cfg.eval_harness or cfg.harness) if evaluate else cfg.harness,
                        operation='evaluate' if evaluate else 'train', model_backend='local')
    return {'env': pair.env, 'harness': pair.harness, 'model': cfg.model_name}


async def _run_eval(source, sampling_client, renderer, cfg: GRPOConfig, gen_cap: int,
                    rank: int, world_size: int, batch_idx: int,
                    best: BestSolutions, tlog: TurnLogger, environment_id=None) -> dict[str, float]:
    """held-out 评估 (COLLECTIVE): 测试题按 rank 摊开, 每题 eval_num_samples 条, 不训练。"""
    tasks = dist_utils.shard_list(list(source.test_tasks()), rank, world_size)
    from arl.algorithms.training_options import thinking_sampling_options
    p = RolloutParams(**thinking_sampling_options(cfg, renderer.tokenizer), gen_cap=gen_cap, prompt_budget=cfg.prompt_budget_tokens,
                      temperature=cfg.eval_temperature, top_p=cfg.eval_top_p, top_k=cfg.eval_top_k,
                      group_size=cfg.eval_num_samples, poll_seconds=cfg.poll_seconds,
                      progress_path=os.path.join(cfg.log_path, f'rollout_progress.rank{rank}.json'))
    t0 = time.time()
    selected = get_harness(cfg.eval_harness or cfg.harness)
    eval_renderer = selected.make_renderer(renderer.tokenizer, enable_thinking=cfg.enable_thinking,
                                          reasoning_effort=cfg.reasoning_effort or None)
    res = await selected.run(tasks=tasks, source=source, sampling_client=sampling_client,
                            renderer=eval_renderer, params=p, operation='evaluate', env_name=get_env_spec(cfg.env).name,
                            uid_prefix=f"eval{environment_id or ''}-{batch_idx}-", log_turns=tlog.enabled)
    identity = _run_identity(cfg, evaluate=True)
    if environment_id is not None:
        identity['environment_id'] = environment_id
    best.update(batch_idx, res.trajs, tag="eval", identity=identity)
    tlog.write(batch_idx, res.trajs, tag="eval", identity=identity)
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
        if get_env_spec(cfg.env).name != 'coding.cuda':
            out.pop("eval/faster_eager_rate", None)
            out.pop("eval/faster_compile_rate", None)
            if get_env_spec(cfg.env).name in ("coding.swe", "coding.programbench"):
                out["eval/resolved_rate"] = c1 / n
    if n_se:
        out["eval/geomean_speedup_eager"] = math.exp(log_se / n_se)
    if n_sc:
        out["eval/geomean_speedup_compile"] = math.exp(log_sc / n_sc)
    return out


async def run_grpo(cfg: GRPOConfig) -> None:
    from arl.envs.mixture import expand_config, SourcePool
    _validate(cfg, int(os.environ.get('WORLD_SIZE', '1')))
    pool = SourcePool(expand_config(cfg))
    try:
        await _run_grpo(cfg, pool)
    finally:
        await pool.close()
        if getattr(pool, 'training_client', None) is not None:
            try:
                pool.training_client.shutdown()
            except Exception as e:
                logger.warning('training client cleanup failed: %s', e)


async def _run_grpo(cfg: GRPOConfig, pool) -> None:
    # Reject unsupported pairs before touching model/GPU/network state.
    _validate(cfg, int(os.environ.get('WORLD_SIZE', '1')))
    setup_environment()
    from arl.algorithms.training_options import configure_training_runtime, training_loss_config, training_phase, thinking_sampling_options
    configure_training_runtime(cfg)
    rank, world_size, local_rank = dist_utils.dist_info()
    is_main = rank == 0
    if world_size > 1:
        import random
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
    _validate(cfg, world_size)
    os.makedirs(cfg.log_path, exist_ok=True)
    if is_main:
        from arl.experiments.artifacts import write_identity
        write_identity(os.path.join(cfg.log_path, 'run_identity.json'), _run_identity(cfg))
    ml_logger = ml_log.setup_logging(log_dir=cfg.log_path, wandb_project=cfg.wandb_project,
                                     config=cfg, wandb_name=cfg.wandb_name) if is_main else None

    from arl.envs.mixture import batch_schedule
    base_cfg = cfg
    members = pool.members
    await pool.initialize()
    if is_main:
        for member, member_source in zip(members, pool.sources):
            if hasattr(member_source, 'check'):
                health = member_source.check()
                if inspect.isawaitable(health):
                    health = await health
                logger.info('[env:%s] eval service: %s', member.id, health)

    peak = max(m.cfg.max_tokens for m in members)
    requested_window = max(m.cfg.prompt_budget_tokens + m.cfg.max_tokens for m in members)
    win = sampling_window(cfg.model_name, peak, cfg.sampling_max_model_len or requested_window)
    for member in members:
        if win < member.cfg.prompt_budget_tokens + member.cfg.max_tokens:
            logger.warning('sampling window %s is below %s prompt+generation budget', win, member.id)
    task_sets = [source.train_tasks() for source in pool.sources]
    if cfg.environments is None:
        batches = [(0, indices) for indices in _batches(len(task_sets[0]), cfg)]
    else:
        batches = batch_schedule(members, [len(tasks) for tasks in task_sets], batch_size=cfg.batch_size,
            epochs=cfg.num_epochs, seed=cfg.shuffle_seed, steps_per_epoch=cfg.mixture_steps_per_epoch)
    num_batches = len(batches)
    service_client = tinker.ServiceClient(
        base_url=None, model_name=cfg.model_name, learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps, warmup_lr_scale=cfg.warmup_lr_scale,
        grad_clip=cfg.grad_clip, max_tokens=peak, sampling_max_model_len=win,
        sampling_prefill_tokens=cfg.sampling_prefill_tokens,
        sampling_prefill_batch_tokens=cfg.sampling_prefill_batch_tokens,
        enable_prefix_caching=cfg.enable_prefix_caching, mamba_cache_mode=cfg.mamba_cache_mode,
        sampling_gpu_memory_utilization=float(os.environ.get("KERNEL_RL_SAMPLING_GPU_UTIL", "0.3")),
        enable_value_head=False)
    if cfg.load_checkpoint_path:
        restore = (service_client.create_training_client_from_state_with_optimizer_async
                   if cfg.resume_optimizer else service_client.create_training_client_from_state_async)
        training_client = await restore(cfg.load_checkpoint_path)
    else:
        training_client = await service_client.create_training_client_async(cfg.model_name)
    pool.training_client = training_client
    tokenizer = training_client.get_tokenizer()
    harnesses = [get_harness(member.cfg.harness) for member in members]
    renderers = [harness.make_renderer(tokenizer, enable_thinking=member.cfg.enable_thinking,
                 reasoning_effort=member.cfg.reasoning_effort or None) for member, harness in zip(members, harnesses)]
    sc = await get_updated_sampling_client(training_client)
    sampling_client = sc[0] if isinstance(sc, tuple) else sc
    logger.info('GRPO: %d batches, environments=%s; shared policy=%s, window=%d',
                num_batches, [(m.id, m.weight) for m in members], cfg.model_name, win)
    best = BestSolutions(cfg.log_path, rank, cfg.save_best_solutions)
    tlog = TurnLogger(cfg.log_path, rank, cfg.save_trajectories)

    async def evaluate_all(batch_idx):
        await training_phase(training_client, "sample")
        metrics = {}
        for member, source, renderer in zip(members, pool.sources, renderers):
            if not source.test_tasks():
                continue
            cap = member.cfg.max_tokens
            ev = await _run_eval(source, sampling_client, renderer, member.cfg, cap,
                                 rank, world_size, batch_idx, best, tlog,
                                 environment_id=member.id if base_cfg.environments is not None else None)
            if is_main:
                logger.info('[eval:%s] %s', member.id, _fmt_eval(ev))
            if base_cfg.environments is None:
                metrics.update(ev)
            else:
                metrics.update({key.replace('eval/', f'eval/{member.id}/', 1)
                                if key.startswith('eval/') else f'{key}/{member.id}': value for key, value in ev.items()})
        return metrics

    nonfinite_total = nonfinite_consec = empty_consec = 0
    has_test = any(source.test_tasks() for source in pool.sources)

    if cfg.eval_at_start and cfg.eval_every > 0 and has_test and cfg.start_batch == 0:
        ev = await evaluate_all(-1)
        if is_main:
            logger.info(f"before training: {_fmt_eval(ev)}")
            if ml_logger is not None:
                ml_logger.log_metrics({**ev, "eval/before_train": 1.0}, step=0)

    for batch_idx in range(cfg.start_batch, num_batches):
        t0 = time.time()
        await training_phase(training_client, "sample")
        member_index, global_indices = batches[batch_idx]
        member = members[member_index]
        cfg, source = member.cfg, pool.sources[member_index]
        selected_harness, renderer = harnesses[member_index], renderers[member_index]
        gen_cap = cfg.max_tokens
        identity = {**_run_identity(cfg), 'environment_id': member.id} if base_cfg.environments is not None else _run_identity(cfg)
        tasks = [task_sets[member_index][i] for i in global_indices]

        # 1) rollout
        p = RolloutParams(**thinking_sampling_options(cfg, renderer.tokenizer), gen_cap=gen_cap, prompt_budget=cfg.prompt_budget_tokens,
                          temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                          group_size=cfg.group_size, extra_samples=cfg.extra_samples,
                          shard_trajectories=True,
                          progress_path=os.path.join(cfg.log_path, f'rollout_progress.rank{rank}.json'),
                          truncated_turn=cfg.truncated_turn, poll_seconds=cfg.poll_seconds)
        p.think_cost_lambda = cfg.think_cost_lambda
        p.think_cost_free_tokens = cfg.think_cost_free_tokens
        res = await selected_harness.run(tasks=tasks, source=source, sampling_client=sampling_client,
            renderer=renderer, params=p, operation='train', env_name=get_env_spec(cfg.env).name,
            uid_prefix=f"b{batch_idx}-", log_turns=tlog.enabled)
        from arl.algorithms.rollout_data import validate_rollout
        validate_rollout(res)
        t_rollout = time.time() - t0
        await training_phase(training_client, "train")
        batch_gen_cap = gen_cap
        best.update(batch_idx, res.trajs, identity=identity)
        tlog.write(batch_idx, res.trajs, identity=identity)

        # 2) advantages: 组 = 题, 单元 = 轨迹; Dr-GRPO 只减均值
        adv_metrics: dict[str, float] = {}
        if res.datums or world_size > 1:
            data, adv_metrics = finalize_ppo_data(
                res.datums, res.returns, whiten=cfg.whiten_advantages and not cfg.group_adv_std,
                groups=res.groups, units=res.units, group_adv_std=cfg.group_adv_std,
                whiten_std=cfg.whiten_std, length_weighted_baseline=cfg.length_weighted_baseline,
                group_adv_std_floor=cfg.group_adv_std_floor, adv_estimator=cfg.adv_estimator,
                mask_positive_incorrect=cfg.mask_positive_incorrect,
                mask_negative_correct=cfg.mask_negative_correct,
                distributed_groups=world_size > 1,
                response_correctness=res.correctness)
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
        loss_cfg = training_loss_config(cfg, batch_gen_cap)
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
            eval_metrics = await evaluate_all(batch_idx)
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
        if get_env_spec(cfg.env).name != 'coding.cuda':
            for key in ("rollout/faster_eager_rate", "rollout/faster_compile_rate"):
                metrics.pop(key, None)
        if base_cfg.environments is not None:
            metrics.update({f'env/{member.id}/{key}': value for key, value in list(metrics.items())
                            if key.startswith(('rollout/', 'ppo/'))})
        if is_main:
            task_score = (f"resolved={metrics.get('rollout/resolved_rate', 0):.1%} " if get_env_spec(cfg.env).name in ("coding.swe", "coding.programbench")
                          else f">compile={metrics.get('rollout/faster_compile_rate', 0):.1%} " if get_env_spec(cfg.env).name == 'coding.cuda' else '')
            logger.info(
                f"batch {batch_idx + 1}/{num_batches} env={member.id}: reward={metrics.get('rollout/reward_mean', 0):.3f} "
                f"correct={metrics.get('rollout/correct_rate', 0):.1%} "
                f"{task_score}"
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
                kind="both" if cfg.save_optimizer else "sampler", background=True)

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client, name="final", log_path=cfg.log_path,
        loop_state={"batch": num_batches}, kind="both" if cfg.save_optimizer else "sampler")
    if is_main and ml_logger is not None:
        ml_logger.close()
    await _teardown(training_client, pool)
    pool.training_client = None


_MEAN_KEYS = ("rollout/reward_mean", "rollout/correct_rate", "rollout/faster_eager_rate",
              "rollout/faster_compile_rate", "rollout/turns_per_traj", "rollout/response_len_mean",
              "rollout/n_tool_calls_per_turn", "rollout/n_invalid_calls_per_turn",
              "rollout/format_ok_per_turn", "rollout/eval_calls_per_traj", "rollout/append_rate",
              "rollout/datums_per_traj", "ppo/reward_mean",
              "rollout/resolved_rate",
              "rollout/think_tokens_mean", "rollout/think_cost_mean",
              "ppo/adv_zero_frac")
_SUM_KEYS = ("rollout/n_trajs", "rollout/ctx_overflow_trajs", "rollout/sample_fail_trajs",
             "rollout/invalid_trace_trajs",
             "rollout/env_error_trajs", "rollout/extra_dropped_trajs", "rollout/infra_dropped_trajs",
             "ppo/nonfinite_lp_tokens", "ppo/nonfinite_lp_active", "ppo/nonfinite_datums")


def _reduce(metrics: dict, res, train_metrics: dict) -> dict:
    """COLLECTIVE: 各 rank 本地的均值类按轨迹数加权合并, 计数类求和; 训练侧按 active token 加权。"""
    for k in _MEAN_KEYS + _SUM_KEYS:
        metrics.setdefault(k, 0.0)
    metrics = dist_utils.reduce_weighted_metrics(
        metrics, mean_keys=list(_MEAN_KEYS),
        local_weight=float(res.metrics.get("rollout/n_trajs", 0.0)), sum_keys=list(_SUM_KEYS))
    if "rollout/think_forced_rate" in metrics:
        metrics = dist_utils.reduce_weighted_metrics(metrics, mean_keys=["rollout/think_forced_rate"],
            local_weight=float(sum(len(t.resp_lens) for t in res.trajs if t.dropped != "extra_sample")))
    # Gating fractions use the original datum/token population, before the gate
    # removes positions from the training mask. Empty ranks participate as well.
    gate_keys = [f"ppo/{gate}_masked_frac" for gate in ("positive_incorrect", "negative_correct")]
    gate_tok_keys = [f"ppo/{gate}_masked_token_frac" for gate in ("positive_incorrect", "negative_correct")]
    for k in gate_keys + gate_tok_keys:
        metrics.setdefault(k, 0.0)
    metrics = dist_utils.reduce_weighted_metrics(
        metrics, mean_keys=gate_keys, local_weight=float(len(res.datums)))
    metrics = dist_utils.reduce_weighted_metrics(
        metrics, mean_keys=gate_tok_keys,
        local_weight=float(sum(d.loss_fn_inputs["mask"].to_torch().sum() for d in res.datums)))
    tok_keys = ["ppo/mean_ratio", "ppo/clip_frac", "ppo/approx_kl"]
    metrics = dist_utils.reduce_weighted_metrics(
        metrics, mean_keys=tok_keys, local_weight=float(train_metrics.get("ppo/n_active_tokens", 0.0)),
        sum_keys=["loss", "ppo/n_active_tokens"])
    if metrics.get("ppo/n_active_tokens", 0.0) <= 0:
        for k in tok_keys + ["ppo/n_active_tokens"]:
            metrics.pop(k, None)
    for weight_key, mean_keys, sum_keys in (
        ('ppo/replay_tokens', ['ppo/replay_outside_frac', 'ppo/replay_zero_grad_frac', 'ppo/replay_kept_mean'], []),
        ('ppo/router_replay_rows', ['ppo/router_replay_frac', 'ppo/router_agree', 'ppo/router_set_match'], ['ppo/router_shape_miss']),
    ):
        # All ranks use the same runtime switches, even if a rank has no data.
        enabled = os.environ.get('KERNEL_RL_ROUTER_REPLAY') == '1' if 'router' in weight_key else os.environ.get('KERNEL_RL_SAMPLER_LOGPROBS_MODE') == 'processed_logprobs'
        if enabled:
            for key in mean_keys + sum_keys:
                metrics.setdefault(key, 0.0)
            metrics = dist_utils.reduce_weighted_metrics(metrics, mean_keys=mean_keys,
                local_weight=float(train_metrics.get(weight_key, 0.0)), sum_keys=sum_keys)
    for k in ("rollout/response_len_p95", "rollout/prompt_len_max", "rollout/datum_len_max",
              "time/rollout_wall"):
        metrics[k] = dist_utils.all_reduce_max(float(metrics.get(k, 0.0)))
    return metrics


def _fmt_eval(ev: dict) -> str:
    if 'eval/n' not in ev:
        counts = [key for key in ev if key.startswith('eval/') and key.endswith('/n')]
        if counts:
            return 'eval: ' + ', '.join(f"{key[5:-2]} n={int(ev[key])}" for key in counts)
    if not ev.get("eval/n"):
        return "eval: no judged trajectories"
    if "eval/resolved_rate" in ev:
        return f"eval n={int(ev['eval/n'])} resolved={ev['eval/resolved_rate']:.1%} ({ev.get('time/eval', 0):.0f}s)"
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
