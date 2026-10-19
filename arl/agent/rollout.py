"""agentic wave rollout: 多轮工具调用轨迹的异步采样 (移植自 ARM ppo_loop 的 overlap_turns 路径).

结构 (ARM ``_rollout_and_score`` 的拆轮间屏障路径, 换成消息层环境):

  * 每道题 G 条 (+extra_samples 条) 轨迹, 一条一个 env;
  * 一个**波次循环**: 每波 ``submit_and_poll`` 提交"上一波刚走完一步的轨迹的下一轮请求",
    同时收割**任意轨迹任意轮**已完成的生成; 收到的每条立刻起一个 ``_finish_turn`` 任务
    (env.step 跑工具 / 远端评测 -> 建 datum -> 排下一轮)。生成和评测在轨迹间完全重叠,
    没有轮间屏障 (agent 的轮数本来就各不相同, 屏障只会让所有轨迹陪最慢那条);
  * 退出条件是**全局**未完成数清零 (all_reduce): ``submit_and_poll`` 是 collective,
    谁先干完谁空手陪跑到全局收尾。

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
import time
from dataclasses import dataclass, field
from typing import Any

from arl.algorithms.completers import TokensWithLogprobs
from arl.algorithms.ppo_data import base_datums_for_response
from arl.algorithms.rl_types import Trajectory, Transition
from arl.models import tinker_compat as tinker
from arl.models.dist_sampling import SampleRequest
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


async def run_agent_rollout(tasks: list, source, sampling_client, renderer: AgentRenderer,
                            p: RolloutParams, *, train: bool = True, uid_prefix: str = "",
                            log_turns: bool = False) -> RolloutResult:
    """COLLECTIVE: 所有 rank 都要进 (本 rank 没有题也要进, 空手陪跑)。"""
    rank, world_size, _ = dist_utils.dist_info()
    stop = renderer.stop_token_ids()

    trajs: list[_Traj] = []
    keep_per_group: dict[int, int] = {}
    n_per = p.group_size + (p.extra_samples if train else 0)
    for gi, task in enumerate(tasks):
        key = source.task_key(task)
        if train and p.extra_samples:
            keep_per_group[gi] = p.group_size
        for j in range(n_per):
            env = source.make_env(task, uid=f"{uid_prefix}r{rank}-g{gi}-{j}")
            await env.reset()
            trajs.append(_Traj(env=env, group=gi, tid=len(trajs), task_key=key))

    pending: list[SampleRequest] = []
    req_meta: dict[str, _Traj] = {}
    procs: set[asyncio.Task] = set()
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

    async def _submit(t: _Traj) -> bool:
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
                return False
            t.n_rerendered += int(t.turns > 0)
        t.prompt_ids, t.prompt_text = ids, text
        rid = f"{t.tid}:{t.turns}"
        req_meta[rid] = t
        t.inflight = True
        pending.append(SampleRequest(
            prompt_ids=ids, num_samples=1, max_tokens=p.gen_cap,
            temperature=p.temperature, top_p=p.top_p, top_k=p.top_k,
            stop=stop, extras={"req_id": rid}))
        return True

    async def _finish_turn(t: _Traj, seq, prompt_ids: list[int]) -> None:
        nonlocal n_env_error
        try:
            resp = TokensWithLogprobs(tokens=seq.tokens, maybe_logprobs=seq.logprobs,
                                      stop_reason=seq.stop_reason,
                                      maybe_mask=getattr(seq, "loss_mask", None))
            truncated = resp.stop_reason == "length"
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

            t.last_resp_ids = list(resp.tokens)
            if train:
                if truncated and p.truncated_turn == "drop_traj":
                    t.dropped = "truncated"
                if truncated and p.truncated_turn == "mask":
                    # 留在序列里 (后面的轮次以它为前缀), 只是不进 loss
                    resp = TokensWithLogprobs(tokens=resp.tokens, maybe_logprobs=resp.maybe_logprobs,
                                              stop_reason=resp.stop_reason,
                                              maybe_mask=[0.0] * len(resp.tokens))
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
                await _submit(t)
        finally:
            t.proc = None

    by_group: dict[int, list[_Traj]] = {}
    for t in trajs:
        by_group.setdefault(t.group, []).append(t)
    group_decided: set[int] = set()
    complete_order: dict[int, int] = {}
    dropped_rids: set[str] = set()
    abort_rids: list[str] = []
    n_extra_dropped = 0
    wave = 0

    def _complete(t: _Traj) -> bool:
        return not t.inflight and (t.proc is None or t.proc.done()) and t.done

    t_wall0 = time.time()
    for t in trajs:
        await _submit(t)
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
                t.proc = asyncio.create_task(_end(t, "sampling failed"))
            else:
                t.proc = asyncio.create_task(_finish_turn(t, seq, t.prompt_ids or []))
            procs.add(t.proc)
        for pr in [pr for pr in procs if pr.done()]:
            procs.discard(pr)
            if not pr.cancelled():
                pr.result()
        # 组内超额采样: 某题凑够 G 条完成的就把该题其余轨迹丢掉 (rank 本地决定)
        if keep_per_group:
            for t in trajs:
                if t.tid not in complete_order and not t.dropped and _complete(t):
                    complete_order[t.tid] = wave
            cancelled: list[asyncio.Task] = []
            newly: list[_Traj] = []
            for gi, ts in by_group.items():
                if gi in group_decided:
                    continue
                done_ts = sorted((t for t in ts if t.tid in complete_order),
                                 key=lambda t: (complete_order[t.tid], t.tid))
                if len(done_ts) < keep_per_group[gi]:
                    continue
                group_decided.add(gi)
                keep = {t.tid for t in done_ts[:keep_per_group[gi]]}
                for t in ts:
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
        local = float(outstanding + len(procs) + len(pending) + (1 if abort_rids else 0))
        remaining = (dist_utils.all_reduce_sum_list_gloo([local])[0] if world_size > 1 else local)
        if remaining <= 0:
            break
        # 非 rank0 让出事件循环, 让工具/评测任务真的跑起来 (ARM 同款论证)
        if world_size > 1 and rank != 0:
            await asyncio.sleep(p.poll_seconds * 0.5)
        else:
            await asyncio.sleep(0)
    if keep_per_group:
        await asyncio.sleep(0.1)   # 让最后一波的中止送达 rank0
    t_wall = time.time() - t_wall0

    # ── 收尾: 终局奖励 -> 每个 datum ──
    datums, returns, groups, units = [], [], [], []
    n_infra = 0
    for t in trajs:
        if t.infra and not t.dropped:
            t.dropped = "infra_error"
            n_infra += 1
        if t.dropped or not train or not t.transitions:
            continue
        # 前缀递增的轮次合成一个 datum, 压缩断开处另起 (data_processing.trajectory_to_data)
        t.datums = [d for d in base_datums_for_response(Trajectory(
            transitions=t.transitions, final_ob=tinker.ModelInput.empty()))
            if float(d.loss_fn_inputs["mask"].to_torch().sum()) > 0]   # 全 mask 的段白占一次前向
        if not t.datums:
            continue
        datums.extend(t.datums)
        returns.extend([t.reward] * len(t.datums))
        groups.extend([(t.group,)] * len(t.datums))
        units.extend([(t.tid, 0)] * len(t.datums))

    metrics = _rollout_metrics(trajs, n_ctx_overflow, n_sample_fail, n_env_error,
                               n_extra_dropped, n_infra, t_wall, world_size)
    return RolloutResult(datums, returns, groups, units, trajs, metrics)


def _rollout_metrics(trajs: list[_Traj], n_ctx, n_sfail, n_enverr, n_extra, n_infra,
                     t_wall: float, world_size: int) -> dict[str, float]:
    """本 rank 的计数 + 全局截断率 (COLLECTIVE)。均值类在上层按轨迹数加权合并。"""
    kept = [t for t in trajs if t.dropped != "extra_sample"]
    judged = [t for t in kept if not t.infra and t.dropped != "env_error"]
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
        "time/rollout_wall": t_wall,
    }
    if judged:
        rs = [t.reward for t in judged]
        m["rollout/reward_mean"] = sum(rs) / len(rs)
        m["rollout/correct_rate"] = sum(r >= 1 for r in rs) / len(rs)
        m["rollout/faster_eager_rate"] = sum(r >= 2 for r in rs) / len(rs)
        m["rollout/faster_compile_rate"] = sum(r >= 3 for r in rs) / len(rs)
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

    def write(self, batch: int, trajs: list[_Traj], tag: str = "train") -> None:
        if not self.enabled:
            return
        with gzip.open(self.path, "at") as f:
            for t in trajs:
                s = t.env.summary() if hasattr(t.env, "summary") else {}
                f.write(json.dumps({
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

    def update(self, batch: int, trajs: list[_Traj], tag: str = "train") -> None:
        if not self.enabled:
            return
        rows = []
        for t in trajs:
            if t.dropped or t.reward < 1 or not hasattr(t.env, "summary"):
                continue
            s = t.env.summary()
            score = (t.reward, float(s.get("speedup_vs_compile") or 0.0))
            prev = self.best.get(t.task_key)
            if prev is not None and tuple(prev) >= score:
                continue
            self.best[t.task_key] = score
            rows.append({"batch": batch, "tag": tag, "task": t.task_key, "reward": t.reward,
                         "speedup_vs_eager": s.get("speedup_vs_eager"),
                         "speedup_vs_compile": s.get("speedup_vs_compile"),
                         "files": s.get("best_files") or s.get("final_files")})
        if rows:
            with open(self.path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
