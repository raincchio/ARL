#!/usr/bin/env python3
"""真集群冒烟: 脚本化的 assistant 回复驱动 CudaWorkspaceEnv 走完
write_file -> compile -> verify -> profile [-> analyze] -> 作弊被拒 -> 交卷,
校验 /repo/evaluate 对接、反馈格式、里程碑奖励。不需要模型 / GPU。

    python3 scripts/smoke_live.py [--endpoint http://10.0.134.90:8000] [--analyze]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudaenv import CudaEvalClient, CudaWorkspaceEnv, EnvConfig, load_task_set  # noqa: E402
from cudaenv.client import check_health  # noqa: E402

GOLD = "/root/workspace/CudaTasks/examples/gold/toy-axpby"

CHEAT_MODEL_NEW = '''
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b
'''


def call(name, **args) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args})}\n</tool_call>"


def check(name, cond, detail="") -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f" -- {detail}"))
    return bool(cond)


def gold_files() -> dict[str, str]:
    files = {"model_new.py": open(os.path.join(GOLD, "model_new.py")).read()}
    for fn in sorted(os.listdir(os.path.join(GOLD, "kernels"))):
        files[f"kernels/{fn}"] = open(os.path.join(GOLD, "kernels", fn)).read()
    return files


async def main(endpoint: str | None, analyze: bool) -> int:
    print(">>> health:", check_health(endpoint))
    task = next(t for t in load_task_set("toy") if t.instance_id == "toy-axpby")
    client = CudaEvalClient(endpoint)
    env = CudaWorkspaceEnv(task, client, EnvConfig(max_turns=10, enable_analyze=analyze))
    await env.reset()
    ok = True

    t0 = time.time()
    writes = "".join(call("write_file", path=p, content=c) for p, c in gold_files().items())
    out = await env.step("<think>fuse</think>\n" + writes + call("compile"))
    ok &= check("compile", "Compile OK" in out.tool_results[-1][1], out.tool_results[-1][1][:400])

    out = await env.step(call("verify"))
    ok &= check("verify", "PASSED" in out.tool_results[-1][1], out.tool_results[-1][1][:400])

    out = await env.step(call("profile"))
    r = out.tool_results[-1][1]
    ok &= check("profile", "your extension" in r and "eager" in r, r[:600])
    print("    " + "\n    ".join(r.splitlines()[:8]))

    if analyze:
        out = await env.step(call("analyze"))
        r = out.tool_results[-1][1]
        ok &= check("analyze (acu)", "Roofline" in r or "no custom kernel" in r, r[:600])
        print("    " + "\n    ".join(r.splitlines()[:4]))

    best = env.st.best_milestone
    # 作弊: model_new.py 里直接用 torch 算 -> 运行期 dispatch 审计应当拒绝
    out = await env.step(call("write_file", path="model_new.py", content=CHEAT_MODEL_NEW)
                         + call("verify"))
    r = out.tool_results[-1][1]
    ok &= check("anti-cheat: torch fallback rejected",
                "REJECTED" in r or "STATIC CHECK FAILED" in r or "FAILED" in r, r[:400])
    print("    " + r.splitlines()[0])

    out = await env.step("Done.")
    ok &= check("episode done, best-of-trajectory reward kept",
                out.done and out.reward == max(best, 1.0) and not out.infra_error,
                f"done={out.done} reward={out.reward} best={best}")
    s = env.summary()
    print(f"    reward={out.reward} milestone={s['milestone']} x_eager={s['speedup_vs_eager']} "
          f"x_compile={s['speedup_vs_compile']} heavy_calls={s['heavy_calls']} "
          f"eval_s={s['eval_seconds']:.0f} wall={time.time() - t0:.0f}s")
    await client.close()
    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.endpoint, a.analyze)))
