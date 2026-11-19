"""多轮工具调用轨迹的异步采样。

共享 vLLM 的 rollout 逐条轨迹推进，只在批次边界同步。
超额采样通过异步完成通知选出每组最先完成的 G 条有效轨迹。

token 序列是 harness 式的**追加**: 环境历史不改写时, 下一轮 prompt = 上一轮 prompt +
本轮采样 token + 新增部分 (见 ``render.AgentRenderer.continue_prompt``), 整条轨迹前缀递增,
收尾时 ``trajectory_to_data`` 把它合成**一个** datum (prompt / 工具结果 mask 掉, 每轮回复
训练)。历史被压缩 (超预算) 的那一轮起前缀断开, 自动另起一个 datum。奖励是轨迹终局标量,
给这条轨迹的每个 datum (adv_unit=trajectory)。
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
from pathlib import Path
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from arl.algorithms.completers import TokensWithLogprobs
from arl.algorithms.ppo_data import base_datums_for_response
from arl.algorithms.thinking_cost import thinking_cost
from arl.algorithms.rl_types import Trajectory, Transition
from arl.models import tinker_compat as tinker
from arl.models.dist_sampling import SampleRequest, DistributedSamplingClient
from arl.models.sampling_rpc import rollout_sampler
from arl.utils import distributed as dist_utils

from .render import AgentRenderer

logger = logging.getLogger(__name__)


@dataclass
class RolloutParams:
    gen_cap: int                  # 每轮生成上限 (含 <think>)
    prompt_budget: int            # 每轮 prompt 上限 (超了就压历史, 压到底还超 -> 提前收尾)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    group_size: int = 8
    extra_samples: int = 0        # 组内超额采样: 每题多采几条, 先完成的 G 条留下
    truncated_turn: str = "mask"  # "mask" | "train" | "drop_traj" (见 GRPOConfig)
    poll_seconds: float = 10.0
    max_aggressiveness: int = 4
    think_cost_lambda: float = 0.0
    think_cost_free_tokens: int = 0
    think_budget: int = 0
    think_force_ids: list[int] | None = None
    think_end_ids: list[int] | None = None
    progress_path: str = ""      # optional per-rank live status, independent of trajectory logging
    shard_trajectories: bool = False  # tasks is global; distribute individual siblings across ranks
    format_reward_weight: float = 0.0


def trajectory_slots(n_tasks: int, n_per: int, rank: int, world_size: int,
                     shard: bool = False) -> list[tuple[int, int]]:
    """Return (global task position, sibling index), exactly once across ranks."""
    slots = range(rank, n_tasks * n_per, world_size) if shard else range(n_tasks * n_per)
    return [divmod(slot, n_per) for slot in slots]


@dataclass
class _Traj:
    env: Any
    group: int
    tid: int
    task_key: str
    datums: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    prompt_ids: list[int] | None = None
    prompt_text: str | None = None
    last_resp_ids: list[int] | None = None
    n_appended: int = 0           # 走追加路径的轮数 (第 0 轮之后)
    n_rerendered: int = 0         # 走重渲染路径的轮数 (第 0 轮之后)
    done: bool = False
    inflight: bool = False
    proc: asyncio.Task | None = None
    turns: int = 0                # 已走完的轮数 (生成 + env.step)
    dropped: str = ""             # 非空 = 不进训练的原因
    reward: float = 0.0
    think_tokens: int = 0
    think_cost: float = 0.0
    n_think_forced: int = 0
    infra: bool = False
    n_truncated: int = 0
    n_masked: int = 0
    resp_lens: list[int] = field(default_factory=list)
    prompt_lens: list[int] = field(default_factory=list)
    episode_metrics: dict = field(default_factory=dict)
    turn_metrics: list[dict] = field(default_factory=list)
    turn_log: list[dict] = field(default_factory=list)


@dataclass
class RolloutResult:
    datums: list                  # 本 rank 进训练的 datum (advantage 待填)
    returns: list[float]          # 与 datums 平行: 所在轨迹的终局奖励
    groups: list[tuple]           # 与 datums 平行: (题,)
    units: list[tuple]            # 与 datums 平行: (轨迹, 0)
    trajs: list[_Traj]
    metrics: dict[str, float]
    correctness: list[bool | None] = field(default_factory=list)  # 与 datums 平行，终局标签


async def run_agent_rollout(tasks: list, source, sampling_client, renderer: AgentRenderer,
                            p: RolloutParams, *, train: bool = True, uid_prefix: str = "",
                            log_turns: bool = False) -> RolloutResult:
    owned, workers = [], set()
    affinity_scope = (sampling_client.begin_affinity_scope()
                      if isinstance(sampling_client, DistributedSamplingClient) else None)
    try:
        return await _run_agent_rollout(tasks, source, sampling_client, renderer, p,
            train=train, uid_prefix=uid_prefix, log_turns=log_turns, owned=owned, workers=workers,
            affinity_scope=affinity_scope)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        # Sandboxed environments own containers. Also release them on sampler
        # errors and cancellation, before the next rollout can need their slots.
        await asyncio.gather(*(_close_env(env) for env in owned))
        if affinity_scope is not None:
            await sampling_client.end_affinity_scope(affinity_scope)


async def _close_env(env):
    if hasattr(env, "close"):
        try:
            await env.close()
        except Exception as e:
            logger.warning("[rollout] environment cleanup failed: %s", e)


async def _run_agent_rollout(tasks, source, sampling_client, renderer, p, *,
                             train, uid_prefix, log_turns, owned, workers, affinity_scope=None):
    """COLLECTIVE: 所有 rank 都要进 (本 rank 没有题也要进, 空手陪跑)。"""
    rank, world_size, _ = dist_utils.dist_info()
    stop = renderer.stop_token_ids()

    trajs: list[_Traj] = []
    keep_per_group: dict[int, int] = {}
    n_per = p.group_size + (p.extra_samples if train else 0)
    slots = trajectory_slots(len(tasks), n_per, rank, world_size, p.shard_trajectories)
    next_progress = 0.0

    def report_progress(status="running", *, force=False):
        nonlocal next_progress
        if not p.progress_path or (not force and time.monotonic() < next_progress):
            return
        next_progress = time.monotonic() + 2
        def completed(t):
            return bool(t.done or t.dropped) and not t.inflight and (t.proc is None or t.proc.done())
        value = dict(pid=os.getpid(), hostname=socket.gethostname(), rank=rank,
            updated_at=time.time(), batch=uid_prefix, status=status,
            batch_size=len({gi for gi, _ in slots}), samples_per_task=n_per, total=len(slots),
            shard_trajectories=p.shard_trajectories,
            initialized=len(trajs), completed=sum(completed(t) for t in trajs),
            generating=sum(t.inflight for t in trajs),
            environment=sum(not t.inflight and not completed(t) and t.proc is not None for t in trajs),
            dropped=sum(bool(t.dropped) for t in trajs),
            turns_completed=sum(len(t.turn_metrics) for t in trajs),
            generated_tokens=sum(sum(t.resp_lens) for t in trajs))
        try:
            from arl.experiments.artifacts import write_json
            Path(p.progress_path).parent.mkdir(parents=True, exist_ok=True)
            write_json(p.progress_path, value)
        except OSError as exc:
            logger.warning("could not write rollout progress: %s", exc)

    report_progress("initializing", force=True)
    if train and p.extra_samples:
        keep_per_group = {gi: p.group_size for gi in range(len(tasks))}
    for gi, j in slots:
        task = tasks[gi]
        key = source.task_key(task)
        env = source.make_env(task, uid=f"{uid_prefix}r{rank}-g{gi}-{j}")
        owned.append(env)
        await env.reset()
        trajs.append(_Traj(env=env, group=gi, tid=gi*n_per+j, task_key=key))
        report_progress("initializing")

    pending: list[SampleRequest] = []
    req_meta: dict[str, _Traj] = {}
    procs = workers
    outstanding = 0
    n_ctx_overflow = n_sample_fail = n_env_error = 0

    async def _end(t: _Traj, reason: str) -> None:
        """训练器要求提前收尾: 奖励照常结算 (env.finish), 轨迹到此为止。"""
        t.done = True
        try:
            out = await t.env.finish(reason)
            t.reward, t.infra = float(out.reward), bool(out.infra_error)
            t.episode_metrics = dict(out.metrics or {})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"traj {t.tid} ({t.task_key}): env.finish failed: {e}")
            t.dropped = t.dropped or "env_error"
        finally:
            await _close_env(t.env)

    async def _request_for(t: _Traj) -> SampleRequest | None:
        nonlocal n_ctx_overflow
        cont = None
        if t.turns > 0 and t.prompt_ids is not None and t.last_resp_ids is not None:
            cont = renderer.continue_prompt(t.env, t.prompt_ids, t.prompt_text or "",
                                            t.last_resp_ids, p.prompt_budget)
        if cont is not None:
            ids, text = cont
            t.n_appended += 1
        else:
            ids, text, _a = renderer.fit(t.env, p.prompt_budget, p.max_aggressiveness)
            if ids is None:
                n_ctx_overflow += 1
                await _end(t, f"prompt over budget {p.prompt_budget} at every compression level")
                return None
            t.n_rerendered += int(t.turns > 0)
        t.prompt_ids, t.prompt_text = ids, text
        rid = (f"{affinity_scope}.{t.tid}:{t.turns}" if affinity_scope is not None
               else f"{t.tid}:{t.turns}")
        return SampleRequest(
            prompt_ids=ids, num_samples=1, max_tokens=p.gen_cap,
            temperature=p.temperature, top_p=p.top_p, top_k=p.top_k,
            stop=stop, extras={"req_id": rid, **(
                {"affinity_scope": affinity_scope, "affinity_key": f"{affinity_scope}:{t.tid}"}
                if affinity_scope is not None else {})}, think_budget=p.think_budget,
            think_force_ids=p.think_force_ids, think_end_ids=p.think_end_ids)

    def _enqueue(t: _Traj, request: SampleRequest | None) -> None:
        if request is not None:
            req_meta[request.extras['req_id']] = t
            t.inflight = True
            pending.append(request)

    async def _finish_turn(t: _Traj, seq, prompt_ids: list[int]) -> SampleRequest | None:
        nonlocal n_env_error
        try:
            if train:
                from arl.algorithms.rollout_data import validate_sample
                try:
                    validate_sample(seq)
                except ValueError as exc:
                    logger.warning('traj %s: invalid on-policy trace: %s', t.tid, exc)
                    t.done, t.dropped = True, 'invalid_trace'
                    return
            resp = TokensWithLogprobs(tokens=seq.tokens, maybe_logprobs=seq.logprobs,
                                      stop_reason=seq.stop_reason,
                                      maybe_mask=getattr(seq, "loss_mask", None),
                                      maybe_routing=getattr(seq, "routed_experts", None))
            truncated = resp.stop_reason == "length"
            n_think, cost = thinking_cost(resp, renderer.tokenizer, p.think_cost_lambda, p.think_cost_free_tokens)
            t.n_think_forced += int(getattr(seq, "think_forced", False))
            t.think_tokens += n_think
            t.think_cost += cost
            text = renderer.decode(resp.tokens)
            t.resp_lens.append(len(resp.tokens))
            t.prompt_lens.append(len(prompt_ids))
            t.n_truncated += int(truncated)
            try:
                out = await t.env.step(text, truncated=truncated)
            except Exception as e:  # noqa: BLE001 -- 单条轨迹的环境异常不拖垮整批
                n_env_error += 1
                logger.warning(f"traj {t.tid} ({t.task_key}) turn {t.turns}: env.step raised "
                               f"{type(e).__name__}: {e}; trajectory dropped")
                t.done, t.dropped = True, "env_error"
                return
            t.turn_metrics.append(dict(out.metrics or {}))
            if log_turns:
                t.turn_log.append({"turn": t.turns, "response": text, "stop_reason": resp.stop_reason,
                                   "tool_results": list(getattr(out, "tool_results", []) or []),
                                   "prompt_tokens": len(prompt_ids), "response_tokens": len(resp.tokens)})
                if p.think_cost_lambda:
                    t.turn_log[-1].update(think_tokens=n_think, think_cost=cost)

            t.last_resp_ids = list(resp.tokens)
            if train:
                if truncated and p.truncated_turn == "drop_traj":
                    t.dropped = "truncated"
                if truncated and p.truncated_turn == "mask":
                    # 留在序列里 (后面的轮次以它为前缀), 只是不进 loss
                    resp = TokensWithLogprobs(tokens=resp.tokens, maybe_logprobs=resp.maybe_logprobs,
                                              stop_reason=resp.stop_reason,
                                              maybe_mask=[0.0] * len(resp.tokens), maybe_routing=resp.maybe_routing)
                    t.n_masked += 1
                t.transitions.append(Transition(ob=tinker.ModelInput.from_ints(prompt_ids), ac=resp,
                                                reward=0.0, episode_done=False, metrics={}))

            if out.done:
                t.done = True
                t.reward, t.infra = float(out.reward), bool(out.infra_error)
                t.episode_metrics = dict(out.metrics or {})
            elif t.dropped:
                await _end(t, "dropped")
            else:
                t.turns += 1
                return await _request_for(t)
        finally:
            t.proc = None
            if t.done or t.dropped:
                await _close_env(t.env)
        return None

    async def _finish_and_enqueue(t: _Traj, seq, prompt_ids: list[int]) -> None:
        _enqueue(t, await _finish_turn(t, seq, prompt_ids))

    async def _run_trajectory(t: _Traj, sample) -> None:
        nonlocal n_sample_fail
        request = await _request_for(t)
        while request is not None:
            t.inflight = True
            try:
                seq = await sample(request)
            finally:
                t.inflight = False
            t.proc = asyncio.current_task()
            if seq is None:
                n_sample_fail += 1
                t.dropped = 'sampling_failed'
                await _end(t, 'sampling failed')
                return
            request = await _finish_turn(t, seq, request.prompt_ids)

    by_group: dict[int, list[_Traj]] = {}
    for t in trajs:
        by_group.setdefault(t.group, []).append(t)

    async def _run_independently() -> None:
        nonlocal n_extra_dropped
        selection_options = ((p.group_size, n_per, p.shard_trajectories)
                             if train and p.extra_samples else None)
        async with rollout_sampler(sampling_client, selection_options=selection_options) as (sample, select):
            async def run_and_report(t):
                await _run_trajectory(t, sample)
                if selection_options is not None:
                    await select('complete', t.group, t.tid, t.done and not t.dropped and not t.infra)

            trajectories = {t.tid: asyncio.create_task(run_and_report(t)) for t in trajs}

            async def stop_extras(group, siblings):
                nonlocal n_extra_dropped
                decision = await select('wait', group)
                if decision.get('failed'):
                    raise RuntimeError('async rollout failed before oversample selection')
                if decision['keep'] is None:
                    return
                keep = set(decision['keep'])
                extras = [t for t in siblings if t.tid not in keep]
                n_extra_dropped += len(extras)
                for t in extras:
                    t.dropped, t.done = 'extra_sample', True
                    trajectories[t.tid].cancel()
                await asyncio.gather(*(trajectories[t.tid] for t in extras), return_exceptions=True)
                await asyncio.gather(*(_close_env(t.env) for t in extras))

            local = set(trajectories.values())
            if selection_options is not None:
                local.update(asyncio.create_task(stop_extras(group, siblings))
                             for group, siblings in by_group.items())
            workers.update(local)
            report_progress(force=True)
            try:
                while local:
                    done, local = await asyncio.wait(local, timeout=2,
                                                     return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if not task.cancelled():
                            task.result()
                        workers.discard(task)
                    report_progress()
                report_progress(force=True)
            except Exception:
                if selection_options is not None:
                    try:
                        await select('abort')
                    except Exception:
                        logger.exception('could not abort async rollout selection')
                raise
            finally:
                for task in local:
                    task.cancel()
                await asyncio.gather(*local, return_exceptions=True)

    group_decided: set[int] = set()
    complete_order: dict[int, tuple[int, int]] = {}  # tid -> (wave, group)
    dropped_rids: set[str] = set()
    abort_rids: list[str] = []
    n_extra_dropped = 0
    wave = 0

    def _complete(t: _Traj) -> bool:
        return not t.inflight and (t.proc is None or t.proc.done()) and t.done

    async def _run_waves() -> None:
        nonlocal outstanding, n_sample_fail, n_extra_dropped, abort_rids, wave
        for t in trajs:
            _enqueue(t, await _request_for(t))
        report_progress(force=True)
        while True:
            to_submit = list(pending)
            pending.clear()
            outstanding += len(to_submit)
            got = await sampling_client.submit_and_poll(to_submit, timeout=p.poll_seconds,
                                                        abort=abort_rids)
            abort_rids = []
            wave += 1
            for rid, seq in got:
                t = req_meta.pop(rid, None)
                if t is None:
                    if rid not in dropped_rids:
                        logger.warning(f"received a req_id not belonging to this rollout {rid}, discarding")
                    continue
                outstanding -= 1
                t.inflight = False
                if seq is None:
                    n_sample_fail += 1
                    t.dropped = 'sampling_failed'
                    t.proc = asyncio.create_task(_end(t, "sampling failed"))
                else:
                    t.proc = asyncio.create_task(_finish_and_enqueue(t, seq, t.prompt_ids or []))
                procs.add(t.proc)
            for pr in [pr for pr in procs if pr.done()]:
                procs.discard(pr)
                if not pr.cancelled():
                    pr.result()
            # Split groups must agree globally on the first G completed siblings.
            if keep_per_group:
                newly_complete = [(t.tid, t.group) for t in trajs
                                  if t.tid not in complete_order and not t.dropped and _complete(t)]
                completed_by_rank = (dist_utils.all_gather_object(newly_complete, group=dist_utils.gloo_group())
                                     if p.shard_trajectories and world_size > 1 else [newly_complete])
                for completed in completed_by_rank:
                    for tid, gi in completed:
                        complete_order[tid] = (wave, gi)
                cancelled: list[asyncio.Task] = []
                newly: list[_Traj] = []
                for gi in keep_per_group:
                    if gi in group_decided:
                        continue
                    done_ids = sorted((tid for tid, (_, group) in complete_order.items() if group == gi),
                                      key=lambda tid: (complete_order[tid][0], tid))
                    if len(done_ids) < keep_per_group[gi]:
                        continue
                    group_decided.add(gi)
                    keep = set(done_ids[:keep_per_group[gi]])
                    for t in by_group.get(gi, []):
                        if t.tid in keep:
                            continue
                        newly.append(t)
                        t.dropped, t.done = "extra_sample", True
                        if t.proc is not None and not t.proc.done():
                            t.proc.cancel()
                            cancelled.append(t.proc)
                if newly:
                    n_extra_dropped += len(newly)
                    pending_rids = {r.extras["req_id"] for r in pending}
                    for rid, t in list(req_meta.items()):
                        if t.dropped != "extra_sample":
                            continue
                        del req_meta[rid]
                        dropped_rids.add(rid)
                        t.inflight = False
                        if rid in pending_rids:
                            continue
                        abort_rids.append(rid)
                        outstanding -= 1
                    pending[:] = [r for r in pending if r.extras["req_id"] in req_meta]
                    if cancelled:
                        await asyncio.gather(*cancelled, return_exceptions=True)
                        for pr in cancelled:
                            procs.discard(pr)
                    await asyncio.gather(*(_close_env(t.env) for t in newly))
            local = float(outstanding + len(procs) + len(pending) + (1 if abort_rids else 0))
            remaining = (dist_utils.all_reduce_sum_list_gloo([local])[0] if world_size > 1 else local)
            report_progress()
            if remaining <= 0:
                break
            # 非 rank0 让出事件循环, 让工具/评测任务真的跑起来 (ARM 同款论证)
            if world_size > 1 and rank != 0:
                await asyncio.sleep(p.poll_seconds * 0.5)
            else:
                await asyncio.sleep(0)
        if keep_per_group:
            await asyncio.sleep(0.1)   # 让最后一波的中止送达 rank0

    t_wall0 = time.time()
    if isinstance(sampling_client, DistributedSamplingClient):
        await _run_independently()
    else:
        await _run_waves()
    t_wall = time.time() - t_wall0
    report_progress("complete", force=True)

    # ── 收尾: 终局奖励 -> 每个 datum ──
    datums, returns, groups, units, correctness = [], [], [], [], []
    n_infra = 0
    format_reward_total = 0.0
    for t in trajs:
        if t.infra and not t.dropped:
            t.dropped = "infra_error"
            n_infra += 1
        if t.dropped or not train:
            continue
        format_reward = 0.0
        if p.format_reward_weight:
            scores = [m["format_ok"] for m in t.turn_metrics if "format_ok" in m]
            if scores:
                format_reward = p.format_reward_weight * (sum(scores) / len(scores))
                format_reward_total += format_reward
        if not t.transitions:
            continue
        # 前缀递增的轮次合成一个 datum, 压缩断开处另起 (data_processing.trajectory_to_data)
        t.datums = [d for d in base_datums_for_response(Trajectory(
            transitions=t.transitions, final_ob=tinker.ModelInput.empty()))
            if float(d.loss_fn_inputs["mask"].to_torch().sum()) > 0]   # 全 mask 的段白占一次前向
        if not t.datums:
            continue
        datums.extend(t.datums)
        returns.extend([t.reward + format_reward - t.think_cost] * len(t.datums))
        groups.extend([(t.group,)] * len(t.datums))
        units.extend([(t.tid, 0)] * len(t.datums))
        # Use trajectory outcome labels; reward shaping must not change correctness.
        label = t.episode_metrics.get("correct")
        correctness.extend([None if label is None else bool(label)] * len(t.datums))

    metrics = _rollout_metrics(trajs, n_ctx_overflow, n_sample_fail, n_env_error,
                               n_extra_dropped, n_infra, t_wall, world_size)
    if train and p.format_reward_weight:
        metrics["rollout/format_reward_mean"] = format_reward_total / max(1, metrics["rollout/n_trajs"])
    if p.think_budget:
        kept_responses = [t for t in trajs if t.dropped != "extra_sample"]
        metrics["rollout/think_forced_rate"] = sum(t.n_think_forced for t in kept_responses) / max(1, sum(len(t.resp_lens) for t in kept_responses))
    if p.think_cost_lambda:
        kept = [t for t in trajs if not t.dropped]
        metrics["rollout/think_tokens_mean"] = sum(t.think_tokens for t in kept) / max(1, len(kept))
        metrics["rollout/think_cost_mean"] = sum(t.think_cost for t in kept) / max(1, len(kept))
    return RolloutResult(datums, returns, groups, units, trajs, metrics, correctness)


def _rollout_metrics(trajs: list[_Traj], n_ctx, n_sfail, n_enverr, n_extra, n_infra,
                     t_wall: float, world_size: int) -> dict[str, float]:
    """本 rank 的计数 + 全局截断率 (COLLECTIVE)。均值类在上层按轨迹数加权合并。"""
    kept = [t for t in trajs if t.dropped != "extra_sample"]
    judged = [t for t in kept if not t.infra and not t.dropped]
    n_resp = sum(len(t.resp_lens) for t in kept)
    n_trunc = sum(t.n_truncated for t in kept)
    tr = ([float(n_trunc), float(n_resp)] if world_size <= 1 else
          dist_utils.all_reduce_sum_list_gloo([float(n_trunc), float(n_resp)]))
    m: dict[str, float] = {
        "rollout/truncated_rate": tr[0] / tr[1] if tr[1] else 0.0,
        "rollout/n_trajs": float(len(judged)),
        "rollout/ctx_overflow_trajs": float(n_ctx),
        "rollout/sample_fail_trajs": float(n_sfail),
        "rollout/env_error_trajs": float(n_enverr),
        "rollout/extra_dropped_trajs": float(n_extra),
        "rollout/infra_dropped_trajs": float(n_infra),
        "rollout/invalid_trace_trajs": float(sum(t.dropped == 'invalid_trace' for t in kept)),
        "time/rollout_wall": t_wall,
    }
    if judged:
        rs = [t.reward for t in judged]
        m["rollout/reward_mean"] = sum(rs) / len(rs)
        m["rollout/correct_rate"] = sum(r >= 1 for r in rs) / len(rs)
        m["rollout/faster_eager_rate"] = sum(r >= 2 for r in rs) / len(rs)
        m["rollout/faster_compile_rate"] = sum(r >= 3 for r in rs) / len(rs)
        if any("resolved" in t.episode_metrics for t in judged):
            m["rollout/resolved_rate"] = sum(r >= 1 for r in rs) / len(rs)
        m["rollout/turns_per_traj"] = sum(len(t.resp_lens) for t in judged) / len(judged)
        lens = sorted(x for t in judged for x in t.resp_lens)
        if lens:
            m["rollout/response_len_mean"] = sum(lens) / len(lens)
            m["rollout/response_len_p95"] = float(lens[min(len(lens) - 1, int(0.95 * len(lens)))])
        pl = [x for t in judged for x in t.prompt_lens]
        if pl:
            m["rollout/prompt_len_max"] = float(max(pl))
        n_app = sum(t.n_appended for t in judged)
        n_rer = sum(t.n_rerendered for t in judged)
        if n_app + n_rer:
            m["rollout/append_rate"] = n_app / (n_app + n_rer)
        with_d = [t for t in judged if t.datums]
        if with_d:
            m["rollout/datums_per_traj"] = sum(len(t.datums) for t in with_d) / len(with_d)
            m["rollout/datum_len_max"] = float(max(d.model_input.length for t in with_d
                                                   for d in t.datums))
        tm = [x for t in judged for x in t.turn_metrics]
        if tm:
            for k in ("n_tool_calls", "n_invalid_calls", "format_ok"):
                vals = [float(x[k]) for x in tm if k in x]
                if vals:
                    m[f"rollout/{k}_per_turn"] = sum(vals) / len(vals)
        hc = [t.episode_metrics.get("episode/heavy_calls") for t in judged]
        hc = [x for x in hc if x is not None]
        if hc:
            m["rollout/eval_calls_per_traj"] = sum(hc) / len(hc)
    return m


class TurnLogger:
    """每条轨迹一行 jsonl.gz (rank 本地): 终局奖励 + 每轮回复原文与工具结果。"""

    def __init__(self, log_path: str, rank: int, enabled: bool):
        self.enabled = enabled
        self.path = os.path.join(log_path, "trajectories", f"rank{rank}.jsonl.gz")
        if enabled:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, batch: int, trajs: list[_Traj], tag: str = "train", identity: dict | None = None) -> None:
        if not self.enabled:
            return
        with gzip.open(self.path, "at") as f:
            for t in trajs:
                s = t.env.summary() if hasattr(t.env, "summary") else {}
                f.write(json.dumps({
                    "identity": identity or {},
                    "batch": batch, "tag": tag, "task": t.task_key, "tid": t.tid,
                    "reward": t.reward, "dropped": t.dropped, "turns": t.turn_log,
                    "summary": {k: v for k, v in s.items() if k != "final_files"},
                }, ensure_ascii=False, default=str) + "\n")


class BestSolutions:
    """每道题见过的最好解 (rank 本地 jsonl, 追加写; 只记比之前更好的)。"""

    def __init__(self, log_path: str, rank: int, enabled: bool):
        self.enabled = enabled
        self.path = os.path.join(log_path, "best_solutions", f"rank{rank}.jsonl")
        self.best: dict[str, float] = {}
        if enabled:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def update(self, batch: int, trajs: list[_Traj], tag: str = "train", identity: dict | None = None) -> None:
        if not self.enabled:
            return
        rows = []
        for t in trajs:
            if t.dropped or t.reward < 1 or not hasattr(t.env, "summary"):
                continue
            s = t.env.summary()
            score = (t.reward, float(s.get("speedup_vs_compile") or 0.0))
            best_key = (tag, json.dumps(identity or {}, sort_keys=True), t.task_key)
            prev = self.best.get(best_key)
            if prev is not None and tuple(prev) >= score:
                continue
            self.best[best_key] = score
            rows.append({"identity": identity or {}, "batch": batch, "tag": tag, "task": t.task_key, "reward": t.reward,
                         "speedup_vs_eager": s.get("speedup_vs_eager"),
                         "speedup_vs_compile": s.get("speedup_vs_compile"),
                         "files": s.get("best_files") or s.get("final_files")})
            if "model_patch" in s:
                rows[-1].update(instance_id=t.task_key, model_name_or_path="arl", model_patch=s["model_patch"])
        if rows:
            with open(self.path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
