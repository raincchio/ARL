#!/usr/bin/env python3
"""Real Docker/harness smoke; --gold validates the grader with a reference patch.

This is a trusted integration test, never a model prompt or training trajectory.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arl.envs.coding.swe import SweSourceConfig, SweTaskSource


async def run(args):
    source = SweTaskSource(SweSourceConfig(
        train_set=args.dataset, test_fraction=0, ssh_host=args.ssh_host, ssh_port=args.ssh_port,
        endpoint=args.endpoint, token_file=args.token_file,
        python=args.python, proxy=args.proxy, max_concurrency=1, eval_timeout=args.eval_timeout,
        work_dir=args.work_dir))
    env = None
    try:
        health = await source.check()
        if args.endpoint:
            health = {key: health.get(key) for key in
                      ("role", "reachable_workers", "max_workspaces", "max_graders")}
        print("worker:", health, flush=True)
        await source.initialize()
        task = next(t for t in source.train_tasks() if t.instance_id == args.instance_id)
        env = source.make_env(task, uid="smoke")
        await env.reset()
        output = await env.workspace.execute("pwd && git status --porcelain", timeout=30)
        if output["exit_code"] or not output["output"].startswith("/testbed\n"):
            raise RuntimeError(output)
        print("workspace:", task.instance_id, "clean checkout", flush=True)
        if args.gold:
            patch = task.record.get("patch")
            if not patch:
                raise RuntimeError("reference patch is absent")
            encoded = base64.b64encode(patch.encode()).decode()
            output = await env.workspace.execute(
                "printf %s " + shlex.quote(encoded) + " | base64 -d | git apply", timeout=30)
            if output["exit_code"]:
                raise RuntimeError(output)
            print("reference patch applied; grading in fresh container...", flush=True)
        if args.negative:
            output = await env.workspace.execute("printf 'integration smoke\\n' > arl_smoke_marker.txt", timeout=30)
            if output["exit_code"]:
                raise RuntimeError(output)
            print("non-fixing patch applied; official reward should be zero...", flush=True)
        if args.harness == 'mini_swe_agent':
            from arl.harnesses.mini_swe import MiniEnv, check_dependencies
            check_dependencies()
            env = MiniEnv(env)
            await env.reset()
            # Trusted fixture: make/inspect a patch, then emit it as required by
            # the upstream SWE prompt. Do not rely on automatic workspace diff.
            def call(command):
                return '</think>\n<tool_call>'+json.dumps({'name': 'bash', 'arguments': {'command': command}})+'</tool_call>'
            await env.step(call('git add -N -- . && git diff --binary > /tmp/arl-smoke-patch && cp /tmp/arl-smoke-patch patch.txt'))
            await env.step(call('cat patch.txt'))
            response = call('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt')
        elif args.harness == 'claude_code':
            from arl.harnesses.claude_code import ClaudeCodeEnv, TOOL
            env = ClaudeCodeEnv(env)
            await env.reset()
            await env.step('<tool_call>' + json.dumps({'name': TOOL, 'arguments': {'command': 'pwd'}}) + '</tool_call>')
            response = 'Finished the integration fixture; submit the current repository changes.'
        else:
            response = '</think>\n<tool_call>{"name":"submit","arguments":{}}</tool_call>'
        final = await env.step(response)
        summary = env.summary()
        print(json.dumps({"reward": final.reward, "infra_error": final.infra_error,
                          "patch_bytes": len(summary["model_patch"].encode()), "grade": summary["grade"]}), flush=True)
        if final.infra_error or (args.gold and final.reward != 1) or (args.negative and final.reward != 0):
            raise RuntimeError("SWE environment smoke failed")
    finally:
        try:
            if env is not None:
                await env.close()
        finally:
            await source.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="verified")
    p.add_argument("--instance-id", default="sympy__sympy-21847")
    p.add_argument("--ssh-host", default="")
    p.add_argument("--endpoint", default="")
    p.add_argument("--token-file", default="")
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--python", default="python3")
    p.add_argument("--proxy", default="")
    p.add_argument("--work-dir", default="/tmp/arl-swe-runs")
    p.add_argument("--eval-timeout", type=int, default=1800)
    p.add_argument('--harness', choices=['environment', 'mini_swe_agent', 'claude_code'], default='environment')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--gold", action="store_true")
    mode.add_argument("--negative", action="store_true", help="Grade a non-fixing patch; expect reward 0")
    asyncio.run(run(p.parse_args()))
