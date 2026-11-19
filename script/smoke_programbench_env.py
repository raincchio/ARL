#!/usr/bin/env python3
"""CPU/Docker integration smoke, synthetic non-solution; no model is started."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arl.envs.coding.programbench import ProgramBenchSourceConfig, ProgramBenchTaskSource


def call(name, **arguments):
    return '</think>\n<tool_call>'+json.dumps({'name': name, 'arguments': arguments})+'</tool_call>'


async def run(args):
    source = ProgramBenchTaskSource(ProgramBenchSourceConfig(ssh_host=args.ssh_host, ssh_port=args.ssh_port,
        python=args.python, blob_dir=args.blob_dir, work_dir=args.work_dir,
        task_ids=args.instance_id, test_fraction=0, max_concurrency=1, max_turns=4, eval_timeout=600))
    try:
        await source.initialize()
        health = await source.check()
        env = source.make_env(source.train_tasks()[0], 'synthetic-smoke')
        if args.harness == 'mini_swe_agent':
            from arl.harnesses.mini_swe import MiniEnv, check_dependencies
            check_dependencies()
            env = MiniEnv(env)
        await env.reset()
        command = "printf '#!/bin/sh\\nexit 17\\n' > compile.sh && chmod +x compile.sh"
        if args.mode == 'stub':
            command = "printf 'int main(void){return 0;}\\n' > main.c && printf '#!/bin/sh\\ncc main.c -o executable\\n' > compile.sh && chmod +x compile.sh"
        executed = await env.step(call('bash', command=command))
        if executed.infra_error: raise RuntimeError(env.summary())
        out = await env.step(call('bash', command='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')
                             if args.harness == 'mini_swe_agent' else call('submit'))
        result = {'synthetic_non_solution': True, 'harness': args.harness, 'mode': args.mode, 'health': health,
                  'reward': out.reward, 'infra_error': out.infra_error, 'summary': env.summary()}
        if out.infra_error: raise RuntimeError(json.dumps(result))
        if args.mode == 'compile-failure' and out.reward != 0: raise RuntimeError('compile failure incorrectly rewarded')
        output = Path(args.output_dir) if args.output_dir else ROOT/'runs'/f'programbench_smoke_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}'
        output.mkdir(parents=True, exist_ok=False)
        (output/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'result': str(output/'result.json'), **result}, indent=2))
    finally:
        await source.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-host', default='root@10.0.134.90')
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--python', default='/mnt/ai_data/benchmarks/programbench/.venv/bin/python')
    parser.add_argument('--blob-dir', default='/mnt/ai_data/benchmarks/programbench/blobs')
    parser.add_argument('--work-dir', default='/mnt/ai_data/arl/programbench_runs')
    parser.add_argument('--instance-id', default='lh3__seqtk.94e7070')
    parser.add_argument('--mode', choices=['compile-failure', 'stub'], default='compile-failure')
    parser.add_argument('--output-dir')
    parser.add_argument('--harness', choices=['environment', 'mini_swe_agent'], default='mini_swe_agent')
    asyncio.run(run(parser.parse_args()))
