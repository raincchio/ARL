"""One SWE checkpoint per subprocess, driven by the ARM-derived eval service."""
import asyncio
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid
from types import SimpleNamespace

from arl.experiments.artifacts import write_json

logger = logging.getLogger(__name__)


def _wandb_run(spec, run):
    project = spec.get('wandb_project')
    if not project:
        return None
    try:
        import wandb
        name = f"{spec.get('wandb_name') or run.name}-eval"
        rid = 'aev-' + hashlib.sha1(str(run.resolve()).encode()).hexdigest()[:12]
        kwargs = dict(project=project, name=name, id=rid, resume='allow',
            dir=str(run/'async_eval'), config={'train_run': str(run.resolve()),
                'async_eval': True, 'args': spec.get('launch_args', {})})
        proxy = os.environ.get('KERNEL_RL_WANDB_PROXY', 'http://10.0.134.70:10809').strip()
        if proxy:
            kwargs['settings'] = wandb.Settings(http_proxy=proxy, https_proxy=proxy)
        return wandb.init(**kwargs)
    except Exception as exc:
        logger.warning('[async_eval] wandb init failed (%s); metrics.jsonl only', exc)
        return None


def upload_wandb_result(run, row):
    """ARM-style separate evaluation run; upload errors never invalidate scores."""
    spec = json.loads((run/'async_eval/spec.json').read_text())
    wb = _wandb_run(spec, run)
    if wb is None:
        return False
    uploaded = False
    try:
        if row['step'] < wb.step:
            logger.warning('[async_eval] wandb is at step %s > %s; re-evaluation stays in metrics.jsonl',
                           wb.step, row['step'])
            return False
        metrics = {k: v for k, v in row.items()
                   if k.startswith(('eval/', 'time/')) and k != 'eval/correct_rate'}
        metrics.update(checkpoint=row['checkpoint'], batch=row['batch'])
        wb.log(metrics, step=row['step'])
        uploaded = True
    except Exception as exc:
        logger.warning('[async_eval] wandb log failed (ignored): %s', exc)
    finally:
        try:
            wb.finish()
        except Exception as exc:
            logger.warning('[async_eval] wandb finish failed (ignored): %s', exc)
            uploaded = False
    return uploaded


def publish_spec(args, run, configs):
    from script.swe10 import write_fixed, task_selection
    out=run/'async_eval';out.mkdir(exist_ok=True)
    spec=dict(schema='arl.swe10.async_eval.v1',baseline_requested=args.eval_at_start or args.phase=='baseline',
        eval_every=args.eval_every,expected_batches=(task_selection(args)["count"]*args.epochs+args.batch_size-1)//args.batch_size,
        baseline=configs['baseline'],evaluation=configs['evaluate'],
        sampler_tp=args.eval_tp,
        wandb_project=configs['train']['train']['options'].get('wandb_project'),
        wandb_name=configs['train']['train']['options'].get('wandb_name'),
        launch_args=dict(model=args.model,harness=args.harness,eval_samples=args.eval_samples,eval_async=True),
        tasks_sha256=hashlib.sha256((run/'tasks.private.jsonl').read_bytes()).hexdigest(),
        evaluation_tasks_file=task_selection(args,True)['file'],
        evaluation_tasks_sha256=hashlib.sha256((run/task_selection(args,True)['file']).read_bytes()).hexdigest())
    # Freeze evaluation semantics, like ARM async_eval/spec.json. Device selection
    # belongs to the evaluator; no GPU startup happens on the training machine.
    write_fixed(out/'spec.json',spec)
    return spec


def read_checkpoint(run, name):
    records={}
    path=run/'train/checkpoints.jsonl'
    if path.exists():
        for line in path.read_text().splitlines(keepends=True):
            if line.endswith('\n') and line.strip():
                record=json.loads(line);records[record['name']]=record
    if name not in records:raise ValueError(f'checkpoint {name} has not been published')
    record=records[name]
    weights=Path(record['sampler_path'])
    if not weights.is_absolute():weights=Path(__file__).resolve().parents[2]/weights
    if not (weights/'config.json').is_file():raise ValueError(f'published checkpoint is not visible: {weights}')
    return str(weights),int(record['batch'])


def evaluation_config(run, name, spec, output):
    config=copy.deepcopy(spec['baseline'] if name=='base' else spec['evaluation'])
    batch=0
    if name!='base':
        weights,batch=read_checkpoint(run,name)
        config['models']['policy']['name']=weights
    config['evaluate']['output_dir']=str(output)
    # A token path may differ across machines; secret values never enter spec.
    if os.environ.get('SWE_EVAL_TOKEN_FILE'):
        config['evaluate']['options']['swe_token_file']=os.environ['SWE_EVAL_TOKEN_FILE']
    return config,batch


def finish_result(run,name,batch,output,report,elapsed):
    out=run/'async_eval'
    metrics_path=out/'metrics.jsonl'
    rows=[json.loads(line) for line in metrics_path.read_text().splitlines() if line] if metrics_path.exists() else []
    row=dict(checkpoint=name,batch=batch,step=max(0,batch-1),
        **{'eval/resolved_rate':report['resolved_rate'],
           'eval/pass_any_rate':report['tasks_solved_in_k']/len(report['per_task']),'time/eval':elapsed},
        evaluation_dir=str(output))
    rows=[r for r in rows if r['checkpoint']!=name]+[row]
    rows.sort(key=lambda r:r['batch'])
    temporary=metrics_path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows));temporary.replace(metrics_path)
    upload_wandb_result(run, row)
    (out/'done').mkdir(exist_ok=True)
    # Publish done only after artifacts and metrics have completed.
    write_json(out/'done'/name,row)
    spec=json.loads((out/'spec.json').read_text())
    if spec['baseline_requested'] and (out/'done/base').exists() and (out/'done/final').exists():
        from script.swe10 import compare
        compare(SimpleNamespace(**spec['launch_args']),run)
    return row


async def run_one(run, name):
    from arl.experiments.config import from_dict
    from arl.experiments.runner import run_native, validate_plan
    from script.swe10 import evaluation_report
    run=run.resolve();out=run/'async_eval'
    spec=json.loads((out/'spec.json').read_text())
    if spec.get('schema')!='arl.swe10.async_eval.v1':raise ValueError('unsupported async spec')
    if hashlib.sha256((run/'tasks.private.jsonl').read_bytes()).hexdigest()!=spec['tasks_sha256']:
        raise ValueError('frozen task records changed since training was prepared')
    eval_path=run/spec.get('evaluation_tasks_file','tasks.private.jsonl')
    if hashlib.sha256(eval_path.read_bytes()).hexdigest()!=spec.get('evaluation_tasks_sha256',spec['tasks_sha256']):
        raise ValueError('frozen evaluation task records changed since training was prepared')
    attempt=out/'results'/name/uuid.uuid4().hex
    attempt.parent.mkdir(parents=True,exist_ok=True)
    config,batch=evaluation_config(run,name,spec,attempt)
    options=config['evaluate']['options']
    if options.get('router_replay'):
        from arl.models.vllm_overlay import prepare_overlay
        overlay=prepare_overlay()
        if overlay:
            sys.path.insert(0,str(overlay));os.environ['PYTHONPATH']=str(overlay)+os.pathsep+os.environ.get('PYTHONPATH','')
    plan=from_dict(config).plan('evaluate');validate_plan(plan)
    (out/'configs').mkdir(exist_ok=True)
    write_json(out/'configs'/f'{name}-{attempt.name}.json',config)
    attempt.mkdir()
    (out/'progress').mkdir(exist_ok=True)
    start=time.monotonic()
    await run_native(plan,progress_path=out/'progress'/f'{name}.json')
    manifest=json.loads((run/'tasks.json').read_text())
    ids=manifest.get('evaluation_instance_ids',manifest['instance_ids'])
    report=evaluation_report(attempt,ids,spec['launch_args']['eval_samples'])
    if report['dropped']:raise RuntimeError(f"evaluation dropped {report['dropped']} trajectories; retrying is required")
    finish_result(run,name,batch,attempt,report,time.monotonic()-start)


def main():
    code=0
    try:
        if len(sys.argv)!=3:raise ValueError('usage: python -m arl.experiments.async_swe_eval RUN base|CHECKPOINT|final')
        asyncio.run(run_one(Path(sys.argv[1]),sys.argv[2]))
    except BaseException:
        import traceback
        traceback.print_exc();code=1
    finally:
        sys.stdout.flush();sys.stderr.flush()
        # ARM service cleans this process group before releasing the GPU lease.
        os._exit(code)


if __name__=='__main__':main()
