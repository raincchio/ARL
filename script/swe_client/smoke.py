#!/usr/bin/env python3
"""Live host-service isolation, command timeout and cleanup check (no GPU)."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arl.envs.coding.swe import SweSourceConfig, SweTaskSource


async def main(args):
    source = SweTaskSource(SweSourceConfig(endpoint=args.endpoint, token_file=args.token_file,
        train_set="remote:/mnt/ai_data/benchmarks/swebench_verified/verified_test.jsonl", test_fraction=0,
        max_concurrency=2))
    workspaces = []
    try:
        await source.initialize()
        task = next(t for t in source.train_tasks() if t.instance_id == "sympy__sympy-21847")
        a, b = [source._backend.workspace(task) for _ in range(2)]
        workspaces = [a, b]
        await asyncio.gather(a.start(), b.start())
        result = await a.execute("printf 'owned by A' > /tmp/swe-service-isolation", timeout=10)
        assert result["exit_code"] == 0, result
        result = await b.execute("test ! -e /tmp/swe-service-isolation", timeout=10)
        assert result["exit_code"] == 0, result
        result = await a.execute("sleep 20", timeout=1)
        assert result["exit_code"] in (124, 137), result
        result = await a.execute("printf alive", timeout=10)
        assert result["output"] == "alive", result
        running = asyncio.create_task(a.execute("sleep 120", timeout=180))
        await asyncio.sleep(.5)
        await asyncio.wait_for(a.close(), timeout=15)
        try:
            await running
            raise AssertionError("cancelled command unexpectedly completed")
        except source._ce.BackendError:
            pass
        names = [w.name for w in workspaces]
        await asyncio.gather(*(w.close() for w in workspaces))
        status = await source._backend.rpc("status")
        assert not ({e["name"] for e in status["active"]} & set(names)), status
        print(json.dumps(dict(isolated=True, timeout_enforced=True, workspace_survives_timeout=True,
                              cancel_running_command=True, cleanup=True, workspaces=names)))
    finally:
        await asyncio.gather(*(w.close() for w in workspaces))
        await source.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="http://10.0.134.70:18770")
    p.add_argument("--token-file", default="/root/ai_data/swe/service/token")
    asyncio.run(main(p.parse_args()))
