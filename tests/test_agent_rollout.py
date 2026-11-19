"""单进程端到端: 假采样客户端 (按脚本吐 token) + 真 tokenizer 渲染 + 真 CudaWorkspaceEnv
(假评测服务) -> wave rollout -> datum -> Dr-GRPO advantage。不需要 GPU / 模型权重以外的东西。

    python3 tests/test_agent_rollout.py [--tokenizer /models/Qwen/Qwen3.8-27B]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "envs", "coding", "CudaEnv"))
sys.path.insert(0, os.path.join(ROOT, "envs", "coding", "CudaEnv", "tests"))

from cudaenv import CudaTask, CudaWorkspaceEnv, EnvConfig  # noqa: E402
from test_env_offline import FakeClient  # noqa: E402

from arl.agent import AgentRenderer, RolloutParams, run_agent_rollout  # noqa: E402
from arl.algorithms.ppo_data import finalize_ppo_data  # noqa: E402
from arl.models.types import SampledSequence  # noqa: E402


def call(name, **args) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args})}\n</tool_call>"


class ScriptedSampler:
    """submit_and_poll: 每个请求按"轨迹编号 + 轮号"挑一段脚本回复, 下一波返回 (模拟异步)。"""

    def __init__(self, tok, scripts):
        self.tok, self.scripts = tok, scripts
        self.queue: list = []
        self.n_prompts = 0
        self.max_prompt = 0

    async def submit_and_poll(self, requests, timeout=5.0, abort=None):
        out = list(self.queue)
        self.queue = []
        for r in requests:
            self.n_prompts += 1
            self.max_prompt = max(self.max_prompt, len(r.prompt_ids))
            tid, turn = map(int, r.extras["req_id"].split(":"))
            text, stop = self.scripts(tid, turn)
            # Qwen3.8 的生成提示以 "<think>\n" 结尾 (回复里只有闭标签); Qwen3 不带, 模型自己写开标签
            if not self.tok.decode(r.prompt_ids[-3:]).endswith("<think>\n") and not text.startswith("<think>"):
                text = "<think>\n" + text
            ids = self.tok.encode(text, add_special_tokens=False)
            if stop == "stop":
                ids = ids + [self.tok.convert_tokens_to_ids("<|im_end|>")]
            else:
                ids = ids[: r.max_tokens]
            self.queue.append((r.extras["req_id"], SampledSequence(
                tokens=ids, logprobs=[-0.5] * len(ids), stop_reason=stop)))
        return out


class Source:
    def __init__(self, client, max_turns=6):
        self.client, self.max_turns = client, max_turns

    def task_key(self, task):
        return task.instance_id

    def make_env(self, task, uid):
        return CudaWorkspaceEnv(task, self.client, EnvConfig(max_turns=self.max_turns))


def script(tid: int, turn: int):
    """偶数轨迹: 写对 + profile + 交卷 (里程碑 3); 奇数: 只 verify (假客户端的 verify 结果由
    model_new 内容决定) 然后交卷; tid=3 第 0 轮截断 (思考写爆)。"""
    if tid == 3 and turn == 0:
        return "<think>\n" + "very long reasoning " * 400, "length"
    if turn == 0:
        good = tid % 2 == 0
        return ("Let me write it.\n</think>\n\n"
                + call("write_file", path="model_new.py", content="GOOD" if good else "BAD")
                + call("write_file", path="kernels/k.cu", content="__global__ void k(){}")
                + call("profile" if good else "verify")), "stop"
    return "Done.\n</think>\n\nFinished.", "stop"


class ContentClient(FakeClient):
    async def evaluate(self, iid, files, tool, **kw):
        self.verify_ok = files.get("model_new.py") == "GOOD"
        return await super().evaluate(iid, files, tool, **kw)


async def main(tok_path: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path)
    renderer = AgentRenderer(tok)
    tasks = [CudaTask(instance_id=f"t{i}", ref_code="class Model: pass\n", registry="/x")
             for i in range(2)]
    client = ContentClient(speedups=(1.9, 1.3))
    sampler = ScriptedSampler(tok, script)
    p = RolloutParams(gen_cap=512, prompt_budget=12000, group_size=4, poll_seconds=0.01)
    res = await run_agent_rollout(tasks, Source(client), sampler, renderer, p, train=True)

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f" -- {detail}"))

    rewards = {t.tid: t.reward for t in res.trajs}
    check("8 trajectories, rewards by script",
          len(res.trajs) == 8 and rewards[0] == 3.0 and rewards[1] == -1.0, rewards)
    t3 = res.trajs[3]
    check("truncated turn kept in sequence but masked out of loss, trajectory continued",
          t3.n_truncated == 1 and t3.n_masked == 1 and len(t3.resp_lens) >= 2,
          (t3.n_truncated, t3.n_masked, t3.resp_lens))
    normal = [t for t in res.trajs if t.tid != 3]
    check("harness append-only: every later turn appended, one datum per trajectory",
          all(t.n_rerendered == 0 and t.n_appended == len(t.resp_lens) - 1 for t in normal)
          and all(len(t.datums) == 1 for t in normal),
          [(t.n_appended, t.n_rerendered, len(t.datums)) for t in res.trajs])
    # tid 3: 思考写到一半被截断, 模板会补闭标签并 trim -> 前缀对不上 -> 重渲染, 另起 datum
    # tid 3: 思考写到一半被截断。模板会补闭标签并 trim -- 文本前缀对得上就追加 (Qwen3),
    # 对不上就重渲染、另起一段 (Qwen3.8 + 这条脚本的前导 <think>); 全 mask 的那段不进训练
    check("mid-think truncation: appended or re-rendered, never an all-masked datum",
          t3.n_appended + t3.n_rerendered == len(t3.resp_lens) - 1 and len(t3.datums) == 1
          and all(float(d.loss_fn_inputs["mask"].to_torch().sum()) > 0 for d in t3.datums),
          (t3.n_appended, t3.n_rerendered, len(t3.datums)))
    check("returns/groups/units parallel to datums",
          len(res.datums) == len(res.returns) == len(res.groups) == len(res.units) == 8)
    check("terminal correctness stays aligned through truncation and merging",
          len(res.correctness) == len(res.datums)
          and all(c == (r >= 1) for c, r in zip(res.correctness, res.returns)))
    # 合成的 datum 里: 模型的思考原样在 (harness, 不剥), 且序列 = 最后一轮 prompt + 最后一轮回复
    t0 = res.trajs[0]
    seq = t0.datums[0].model_input.to_ints() + [int(t0.datums[0].loss_fn_inputs["target_tokens"].to_torch()[-1])]
    check("merged datum == last prompt + last response (token-exact)",
          seq == t0.prompt_ids + t0.last_resp_ids, (len(seq), len(t0.prompt_ids) + len(t0.last_resp_ids)))
    txt = tok.decode(t0.prompt_ids)
    check("history keeps reasoning, well-formed think blocks",
          "Let me write it." in txt and txt.count("</think>") <= txt.count("<think>") <= txt.count("</think>") + 1,
          txt[-400:])
    # 重渲染 (fallback) 和追加的文本一致: 模板对回放的历史逐字复现
    check("append path text == fresh render",
          t0.prompt_text == renderer.render_text(t0.env.messages(0), t0.env.tools) or t0.env.done)
    data, adv_m = finalize_ppo_data(res.datums, res.returns, whiten=True, groups=res.groups,
                                    units=res.units, group_adv_std=False, whiten_std=False,
                                    length_weighted_baseline=False)
    advs = []
    for d in data:
        a = d.loss_fn_inputs["advantages"].to_torch()
        m = d.loss_fn_inputs["mask"].to_torch()
        nz = a[m > 0]
        check_prompt = float(a[m == 0].abs().sum()) == 0.0
        if not check_prompt:
            check("prompt tokens carry zero advantage", False)
        advs.append(float(nz[0]) if len(nz) else None)   # None = 全 mask 的 datum (截断轮)
    pos = [a for a, r in zip(advs, res.returns) if r == 3.0 and a is not None]
    neg = [a for a, r in zip(advs, res.returns) if r == -1.0 and a is not None]
    check("Dr-GRPO advantages: winners > 0 > losers",
          pos and neg and min(pos) > 0 > max(neg), (pos[:3], neg[:3]))
    check("metrics present", res.metrics.get("rollout/correct_rate") == 0.5
          and "rollout/truncated_rate" in res.metrics, res.metrics)
    # Thinking cost changes training returns only, keeping task metrics and labels intact.
    p.think_cost_lambda = 8.192
    shaped = await run_agent_rollout(tasks, Source(client), ScriptedSampler(tok, script),
                                     renderer, p, train=True, log_turns=True)
    for t in shaped.trajs:
        check(f"trajectory {t.tid}: cost counts raw sampled thinking, including masked truncation",
              t.think_tokens > 0 and abs(t.think_cost - t.think_tokens * .001) < 1e-8
              and abs(sum(x["think_cost"] for x in t.turn_log) - t.think_cost) < 1e-8)
    expected_returns = [t.reward - t.think_cost for t in shaped.trajs for _ in t.datums]
    check("thinking cost is charged once per trajectory, shared by compressed segments",
          shaped.returns == expected_returns and shaped.correctness == res.correctness
          and shaped.metrics["rollout/correct_rate"] == .5)
    class SplitRenderer(AgentRenderer):
        def continue_prompt(self, *args, **kwargs):
            return None

        def fit(self, env, *args, **kwargs):
            ids, text, level = super().fit(env, *args, **kwargs)
            return [len(env.messages())] + ids, text, level

    split = await run_agent_rollout(tasks, Source(client), ScriptedSampler(tok, script),
                                    SplitRenderer(tok), p, train=True)
    check("broken prefixes split datums without duplicating cost or losing correctness",
          len(split.trajs[0].datums) > 1
          and split.returns == [t.reward - t.think_cost for t in split.trajs for _ in t.datums]
          and split.correctness == [t.reward >= 1 for t in split.trajs for _ in t.datums])
    p.truncated_turn = "drop_traj"
    dropped = await run_agent_rollout(tasks, Source(client), ScriptedSampler(tok, script),
                                      renderer, p, train=True)
    check("correctness alignment after dropping a truncated trajectory",
          len(dropped.correctness) == len(dropped.datums)
          and dropped.correctness == [t.reward >= 1 for t in dropped.trajs for _ in t.datums])
    print(f"    prompts={sampler.n_prompts} max_prompt_tokens={sampler.max_prompt} "
          f"datums={len(data)} adv_zero={adv_m.get('ppo/adv_zero_frac')}")
    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/models/Qwen/Qwen3.8-27B")
    a = ap.parse_args()
    random.seed(0)
    sys.exit(asyncio.run(main(a.tokenizer)))
