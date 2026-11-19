"""ProgramBench task inspection and official submission archive scoring."""
import asyncio
import json
from pathlib import Path

from arl.envs.coding.programbench import ProgramBenchSourceConfig, ProgramBenchTaskSource
from arl.harnesses import get_harness
from arl.envs import get_env_spec


def benchmark(**options):
    allowed = {'action', 'harness', 'submissions_dir', 'output_dir', 'train_set', 'task_ids', 'limit',
               'ssh_host', 'ssh_port', 'python', 'blob_dir', 'work_dir', 'eval_timeout', 'eval_cpus', 'branch_workers'}
    if set(options)-allowed: raise ValueError(f'unknown ProgramBench options: {sorted(set(options)-allowed)}')
    environment = get_env_spec('programbench')
    harness = get_harness(options.get('harness') or environment.default_harness)
    if harness.name not in environment.supported_harnesses:
        raise ValueError(f'incompatible ProgramBench harness: {harness.name}')
    provenance = {'env': environment.name, 'harness': harness.name, 'harness_status': harness.status,
                  'harness_execution': 'not_run_offline_grading'}
    action = options.get('action', 'evaluate' if options.get('submissions_dir') else 'inspect')
    if action not in ('inspect', 'check', 'evaluate'): raise ValueError('ProgramBench action must be inspect/check/evaluate')
    cfg = ProgramBenchSourceConfig(**{k:v for k,v in options.items() if k in ProgramBenchSourceConfig.__dataclass_fields__}, test_fraction=0)

    async def run():
        source = ProgramBenchTaskSource(cfg)
        try:
            await source.initialize()
            tasks = source.train_tasks()
            if action == 'inspect':
                from programbenchenv.backend import MANIFEST
                return {**provenance, 'tasks': len(tasks),
                    'instance_ids': [t.instance_id for t in tasks], 'training': 'ready',
                    'commit': MANIFEST['commit'], 'dataset_sha256': MANIFEST['dataset_sha256']}
            health = await source.check()
            if action == 'check': return health
            if not options.get('submissions_dir'): raise ValueError('ProgramBench evaluate requires submissions_dir on Docker host')
            records = []
            for task in tasks:
                try:
                    grade = await source.backend.evaluate(task, options['submissions_dir'], cfg.eval_timeout)
                    score, resolved = source.ce.classify_summary(grade['summary'])
                    records.append({'instance_id': task.instance_id, 'score': score, 'resolved': resolved, 'status': 'scored', **grade})
                except Exception as exc:
                    records.append({'instance_id': task.instance_id, 'score': None, 'resolved': None, 'status': 'infra_error', 'error': str(exc)})
            scored = [r for r in records if r['status']=='scored']
            result = {**provenance, 'provenance': health,
                'tasks': len(tasks), 'tasks_scored': len(scored), 'coverage': len(scored)/len(tasks),
                'complete': len(scored)==len(tasks),
                'mean_test_pass_fraction_scored': sum(r['score'] for r in scored)/len(scored) if scored else None,
                'resolved_rate_scored': sum(r['resolved'] for r in scored)/len(scored) if scored else None}
            if options.get('output_dir'):
                output = Path(options['output_dir']); output.mkdir(parents=True, exist_ok=False)
                (output/'results.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
                (output/'report.json').write_text(json.dumps(result, indent=2))
                result['report'] = str(output/'report.json')
            return result
        finally:
            await source.close()
    return asyncio.run(run())
