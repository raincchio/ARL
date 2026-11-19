#!/usr/bin/env python3
"""Two-worker real Docker/official-harness smoke; gold is a trusted fixture only."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arl.envs.coding.swe import _import_sweenv


async def main(args):
    ce = _import_sweenv()
    backend = ce.DockerBackend(ce.BackendConfig(endpoint=args.endpoint, token_file=args.token_file))
    workspaces = []
    try:
        health = await backend.health()
        assert health["reachable_workers"] >= 2, health
        rows = await backend.load_tasks('/mnt/ai_data/benchmarks/swebench_verified/verified_test.jsonl')
        record = next(r for r in rows if r['instance_id']=='sympy__sympy-21847')
        task = ce.SweTask.from_record(record)
        workspaces = [backend.workspace(task), backend.workspace(task)]
        await asyncio.gather(*(w.start() for w in workspaces))
        owners = [w.placement['worker_id'] for w in workspaces]
        assert len(set(owners)) == 2, owners
        a, b = workspaces
        for w in workspaces:
            out = await w.execute('pwd && git status --porcelain', timeout=30)
            assert out['exit_code']==0 and out['output']=='/testbed\n', out
            assert out['worker_id']==w.placement['worker_id'], out
        out = await a.execute('printf owner-a > /tmp/cluster-isolation', timeout=10)
        assert out['exit_code']==0, out
        out = await b.execute('test ! -e /tmp/cluster-isolation', timeout=10)
        assert out['exit_code']==0, out
        encoded = base64.b64encode(record['patch'].encode()).decode()
        out = await a.execute('printf %s '+shlex.quote(encoded)+' | base64 -d | git apply', timeout=30)
        assert out['exit_code']==0, out
        out = await b.execute("printf 'not a fix\\n' > cluster_smoke_marker.txt", timeout=10)
        assert out['exit_code']==0, out
        patches = await asyncio.gather(*(w.patch() for w in workspaces))
        print(json.dumps(dict(placements=owners, isolated=True, grading=True)), flush=True)
        grades = await asyncio.gather(*(w.grade(p['patch'], timeout=1800) for w,p in zip(workspaces,patches)))
        assert [g['resolved'] for g in grades]==[True,False], grades
        assert not any(g.get('infra_error') for g in grades), grades
        assert [g['worker_id'] for g in grades]==owners, grades
        await asyncio.gather(*(w.close() for w in workspaces))
        status = await backend.rpc('status')
        names = {w.name for w in workspaces}
        assert not any(e['name'] in names for e in status['active']), status
        result = dict(placements=owners, isolated=True, sticky_routing=True, cleanup=True,
                      official_rewards=[1,0], grades=grades)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        await asyncio.gather(*(w.close() for w in workspaces))
        await backend.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',default='http://10.0.134.70:18770')
    p.add_argument('--token-file',default='/root/ai_data/swe/service/token')
    p.add_argument('--output',default='')
    asyncio.run(main(p.parse_args()))
