"""离线单测: 假 CudaEval 客户端驱动环境走完各条路径 (不连网, 不需要 GPU).

    python3 -m pytest -q tests/        # 或 python3 tests/test_env_offline.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudaenv import CudaTask, CudaWorkspaceEnv, EnvConfig  # noqa: E402

REF = "class Model: ...\ndef get_inputs(): ...\ndef get_init_inputs(): ...\n"


def call(name, **args) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args})}\n</tool_call>"


def xml_call(name, **args) -> str:
    ps = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{ps}</function>\n</tool_call>"


class FakeClient:
    """按工具名回 /repo/evaluate 形状的响应。``script`` 可覆盖某次调用。"""

    def __init__(self, speedups=(1.9, 1.3), infra=False, verify_ok=True):
        self.speedups, self.infra, self.verify_ok = speedups, infra, verify_ok
        self.calls: list[tuple[str, dict]] = []

    async def evaluate(self, iid, files, tool, *, registry=None, timeout=600, cfg=None):
        self.calls.append((tool, dict(files)))
        if self.infra:
            return {"ok": False, "workspace": {}, "infra_error": True, "error_message": "down"}
        acts = {"compile": {"ok": True, "cached": False, "elapsed_s": 3.0}}
        ok = True
        if tool in ("verify", "profile", "analyze", "grade"):
            acts["verify"] = {"ok": self.verify_ok, "tests_passed": 5 if self.verify_ok else 2,
                              "tests_total": 5, "torch_fallback": {}}
            ok = self.verify_ok
        if ok and tool in ("profile", "grade"):
            e, c = self.speedups
            acts["profile"] = {"ok": True, "custom_ms": 1.0, "eager_ms": e, "compile_ms": c,
                               "speedup_vs_eager": e, "speedup_vs_compile": c,
                               "verified_in_call": True, "custom_kernels": [], "eager_kernels": []}
        if ok and tool == "analyze":
            acts["acu"] = {"ok": True, "kernels": [{"name": "k", "compute_pct": 20.0,
                                                    "memory_pct": 80.0}]}
        ws = {"ok": ok, "actions": acts, "static_violations": [],
              "error_message": None if ok else "mismatch", "infra_error": False}
        m = -1
        if ok and "profile" in acts:
            e, c = self.speedups
            m = 3 if (e >= 1.05 and c >= 1.05) else (2 if e >= 1.05 else 1)
        elif ok:
            m = 1
        return {"ok": ok, "workspace": ws, "milestone": m, "infra_error": False,
                "error_message": ws["error_message"], "resolved_correct": ok,
                "resolved_fast": m == 3}


def make_env(client, **kw):
    task = CudaTask(instance_id="t-1", ref_code=REF, registry="/nonexistent")
    return CudaWorkspaceEnv(task, client, EnvConfig(**kw))


def run(coro):
    return asyncio.run(coro)


def test_full_episode_milestone3():
    async def go():
        c = FakeClient()
        env = make_env(c)
        await env.reset()
        msgs = env.messages()
        assert msgs[0]["role"] == "system" and "model.py" in msgs[1]["content"]
        out = await env.step(call("write_file", path="model_new.py", content="x")
                             + xml_call("write_file", path="kernels/a.cu", content="y\nz")
                             + call("compile"))
        assert not out.done and "Compile OK" in out.tool_results[-1][1]
        out = await env.step(call("profile"))
        assert "x1.3" in out.tool_results[-1][1]
        out = await env.step("done, no more tools")
        assert out.done and out.reward == 3.0 and not out.infra_error
        s = env.summary()
        assert s["best_files"] == {"model_new.py": "x", "kernels/a.cu": "y\nz"}
        assert [t for t, _ in c.calls] == ["compile", "profile"]
        assert out.metrics["episode/faster_compile"] == 1.0
    run(go())


def test_never_verified_is_minus1():
    async def go():
        env = make_env(FakeClient(verify_ok=False))
        await env.reset()
        await env.step(call("write_file", path="model_new.py", content="x") + call("verify"))
        out = await env.step("give up")
        assert out.done and out.reward == -1.0
    run(go())


def test_all_infra_marks_infra_error():
    async def go():
        env = make_env(FakeClient(infra=True))
        await env.reset()
        out = await env.step(call("write_file", path="model_new.py", content="x") + call("verify"))
        assert "NOT a problem with your code" in out.tool_results[-1][1]
        out = await env.step("stop")
        assert out.done and out.infra_error and out.reward == 0.0
    run(go())


def test_illegal_path_bad_edit_unknown_tool():
    async def go():
        env = make_env(FakeClient())
        await env.reset()
        out = await env.step(call("write_file", path="utils/hack.py", content="x")
                             + call("edit_file", path="model_new.py", old_string="a", new_string="b")
                             + call("bash", cmd="ls"))
        texts = " ".join(r for _, r in out.tool_results)
        assert "unknown tool 'bash'" in texts and "Illegal path" in texts
        assert "does not exist" in texts and not out.done
        assert out.metrics["n_invalid_calls"] == 1.0
    run(go())


def test_truncated_without_call_is_not_submission():
    async def go():
        env = make_env(FakeClient(), max_turns=3)
        await env.reset()
        out = await env.step("<think>very long", truncated=True)
        assert not out.done and "token limit" in out.tool_results[-1][1]
        await env.step("<think>again", truncated=True)
        out = await env.step("<think>again", truncated=True)
        assert out.done and out.reward == -1.0      # 轮数用完
    run(go())


def test_compression_levels():
    async def go():
        env = make_env(FakeClient(), max_turns=20)
        await env.reset()
        # Qwen3.x 实际采样出来的形状: 没有开头的 <think>, 只有闭标签
        await env.step("old reasoning\n</think>\n\n" + call("read_file", path="model.py"))
        for _ in range(7):
            await env.step("r\n</think>\n\n" + call("read_file", path="model.py"))
        m0, m1, m2, m3 = (env.messages(a) for a in (0, 1, 2, 3))
        # 档 0: harness 式, 历史原样 (思考进 reasoning_content, content 里不留思考和标签)
        a0 = [m for m in m0 if m["role"] == "assistant"]
        assert len(a0) == 8 and a0[0]["reasoning_content"] == "old reasoning"
        assert all("</think>" not in m["content"] and "<tool_call>" in m["content"] for m in a0)
        # 档 1: 消息数不变, 只有老轮次 (最近 keep_full_turns=4 轮之外) 丢掉思考
        a1 = [m for m in m1 if m["role"] == "assistant"]
        assert len(m1) == len(m0) and "reasoning_content" not in a1[0]
        assert all("reasoning_content" in m for m in a1[-4:])
        # 档 2: 老轮次压成摘要; 档 3: 开始整段丢
        assert "details compressed" in [m for m in m2 if m["role"] == "assistant"][0]["content"]
        assert len(m3) < len(m2) and "omitted" in m3[2]["content"]
    run(go())


def test_analyze_tool_optional():
    async def go():
        env = make_env(FakeClient())
        assert "analyze" not in {t["function"]["name"] for t in env.tools}
        env2 = make_env(FakeClient(), enable_analyze=True)
        assert "analyze" in {t["function"]["name"] for t in env2.tools}
        assert "`analyze`" in env2.messages()[0]["content"]
        await env2.reset()
        out = await env2.step(call("write_file", path="model_new.py", content="x") + call("analyze"))
        assert "Roofline" in out.tool_results[-1][1]
    run(go())


def test_final_reward_mode():
    async def go():
        c = FakeClient(speedups=(1.5, 0.9))
        env = make_env(c, reward_mode="final")
        await env.reset()
        await env.step(call("write_file", path="model_new.py", content="x"))
        out = await env.step("submit")
        assert out.done and out.reward == 2.0 and c.calls[-1][0] == "grade"
        assert env.summary()["final_grade"]["resolved_correct"] is True
    run(go())


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print(f"[PASS] {f.__name__}")
    print(f"ALL PASS ({len(fns)})")
