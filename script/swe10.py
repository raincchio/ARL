#!/usr/bin/env python3
"""SWE training and evaluation with frozen task selections. --check never starts GPUs or contacts SWE."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arl.experiments import eval_defaults
MODEL = '/models/Qwen/Qwen3.6-35B-A3B'
DATASET = '/mnt/ai_data/benchmarks/swebench_verified/verified_test.jsonl'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['all','prepare','baseline','train','evaluate','compare'], default='all')
    p.add_argument('--check', action='store_true', help='validate plans only; no network, GPU or output writes')
    p.add_argument('--run-dir', type=Path, help='explicit run directory; supports strftime dates; default model/date/vN')
    p.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                   help='upload training and separate asynchronous evaluation metrics to project ARL')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--harness', choices=['claude_code','mini_swe_agent'], default='claude_code',
                   help='same harness for baseline, training and final evaluation')
    p.add_argument('--train-gpus', default=os.environ.get('TRAIN_GPUS','all'))
    p.add_argument('--sampler-gpus', default=os.environ.get('SAMPLER_GPUS','all'))
    p.add_argument('--eval-gpus', help='default: same GPUs as training sampler')
    p.add_argument('--eval-tp', type=int, default=eval_defaults.SAMPLER_TP,
                   help='GPUs per standalone/asynchronous evaluation engine, independent of training DP')
    p.add_argument('--eval-util', type=float, default=eval_defaults.GPU_MEMORY_UTILIZATION)
    p.add_argument('--sp16', action=argparse.BooleanOptionalAction,
                   default=os.environ.get('SP16', os.environ.get('KERNEL_RL_SP16', '0')) == '1',
                   help='train one packed sequence across 16 Qwen3.5-MoE ranks (Qwen3.6-35B-A3B)')
    p.add_argument('--sampler-dp', type=int, default=4)
    p.add_argument('--colocate', action=argparse.BooleanOptionalAction, default=True, help='sampler GPUs must be a subset of training GPUs')
    p.add_argument('--sampling-util', type=float, default=.8)
    p.add_argument('--port', type=int, default=29524)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--train-group-size', '--group-size', dest='group_size', type=int, default=8,
                   help='target number of trajectories per training task, excluding extra samples')
    p.add_argument('--extra-samples', type=int, default=0,
                   help='extra training trajectories per task for first-completion selection')
    p.add_argument('--batch-size', type=int, default=10)
    p.add_argument('--temperature', type=float, default=1.0, help='sampling temperature for training and evaluation')
    p.add_argument('--top-p', type=float, default=.95, help='nucleus sampling threshold for training and evaluation')
    p.add_argument('--top-k', type=int, default=20, help='top-k sampling for training and evaluation; -1 disables it')
    p.add_argument('--eval-temperature', type=float, help='evaluation override for --temperature')
    p.add_argument('--eval-top-p', type=float, help='evaluation override for --top-p')
    p.add_argument('--eval-top-k', type=int, help='evaluation override for --top-k')
    p.add_argument('--eval-group-size', '--eval-samples', '--eval-num-samples', dest='eval_samples', type=int,
                   default=eval_defaults.SAMPLES_PER_TASK,
                   help=f'trajectories per evaluation task for baseline and checkpoint/final evaluation '
                        f'(default: {eval_defaults.SAMPLES_PER_TASK})')
    p.add_argument('--max-turns', type=int, default=250)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--max-model-len', type=int, default=0,
                   help='total input+output context; 0 reads the model config; no independent prompt cap')
    p.add_argument('--prefill-tokens', type=int, default=0, help='per-engine runtime scheduling token budget; 0 follows vLLM')
    p.add_argument('--prefill-batch-tokens', type=int, default=0, help='compilation/profiling token budget; with prefill_tokens defaults to at least 2048')
    p.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=None,
                   help='explicit cross-request prefix-cache switch; unset follows vLLM/model default')
    p.add_argument('--mamba-cache-mode', choices=['none','align','all'], default='none',
                   help='hybrid-state cache mode; Qwen supports align; align/all require prefix caching')
    p.add_argument('--learning-rate', type=float, default=1e-6)
    p.add_argument('--sampling-replay', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--router-replay', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--pack-tokens', type=int, default=int(os.environ.get('KERNEL_RL_PACK_TOKENS','32768')))
    p.add_argument('--head-chunks', type=int, default=int(os.environ.get('KERNEL_RL_HEAD_CHUNKS','0')))
    p.add_argument('--adv-estimator', choices=['grpo','tailrl'], default='grpo')
    p.add_argument('--warmup-steps', type=int, default=5)
    p.add_argument('--warmup-lr-scale', type=float, default=.5)
    p.add_argument('--think-budget', type=int, default=0)
    p.add_argument('--think-cost-lambda', type=float, default=0)
    p.add_argument('--format-reward-weight', type=float, default=0,
                   help='training bonus for the fraction of turns with valid tool-call format')
    p.add_argument('--think-cost-free-tokens', type=int, default=0)
    p.add_argument('--mask-positive-incorrect', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--mask-negative-correct', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--save-optimizer', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--save-trajectories', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-best-solutions', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--truncated-turn', choices=['mask','train','drop_traj'], default='mask')
    p.add_argument('--eval-async', action=argparse.BooleanOptionalAction, default=False,
                   help='publish checkpoints for a separate evaluation machine; do not wait for tests')
    p.add_argument('--eval-every', type=int, default=5, help='checkpoint/evaluation interval in async mode')
    p.add_argument('--eval-at-start', action=argparse.BooleanOptionalAction, default=True,
                   help='run the initial-model baseline (default); async mode queues it on the evaluator')
    p.add_argument('--endpoint', default='http://10.0.134.70:18770')
    p.add_argument('--token-file', default=os.environ.get('SWE_TOKEN_FILE',str(ROOT/'.secrets/swe_token')),
                   help='local SWE service token file; defaults to project .secrets/swe_token')
    p.add_argument('--dataset', default=DATASET, help='training dataset on the SWE service')
    p.add_argument('--task-limit', '--train-task-limit', dest='task_limit', type=int, default=10,
                   help='first N training tasks when no explicit IDs are given')
    p.add_argument('--train-instance-ids', help='comma-separated exact instance IDs, in selection order')
    p.add_argument('--train-ids-file', type=Path, help='JSON string list or one instance ID per line')
    p.add_argument('--eval-dataset', help='evaluation dataset on the SWE service; defaults to dataset')
    p.add_argument('--eval-task-limit', type=int, help='first N evaluation tasks; defaults to task_limit')
    p.add_argument('--eval-instance-ids', help='comma-separated exact evaluation instance IDs')
    p.add_argument('--eval-ids-file', type=Path, help='JSON string list or one evaluation ID per line')
    p.add_argument('--train-concurrency', '--concurrency', dest='concurrency', type=int,
                   default=os.environ.get('TRAIN_CONCURRENCY', '2'),
                   help='active SWE workspace limit per training rank; -1 for unlimited (default: TRAIN_CONCURRENCY or 2)')
    p.add_argument('--eval-concurrency', type=int, default=os.environ.get('EVAL_CONCURRENCY', eval_defaults.CONCURRENCY),
                   help='active SWE workspace limit in standalone evaluation; -1 for unlimited')
    return p


def normalize_key_values(argv):
    """Accept ARM-style key=value without shell evaluation or silent drops."""
    actions={action.dest:action for action in parser()._actions}
    aliases={'SP16':'sp16','POLICY':'model','TRAIN_GPUS':'train_gpus','SAMPLER_GPUS':'sampler_gpus',
        'EVAL_GPUS':'eval_gpus','EVAL_TP':'eval_tp','EVAL_UTIL':'eval_util','EVAL_SAMPLES':'eval_samples','SAMPLER_DP':'sampler_dp','COLOCATE':'colocate',
        'SAMPLING_UTIL':'sampling_util','PORT':'port','PREFILL_TOKENS':'prefill_tokens','PREFILL_BATCH_TOKENS':'prefill_batch_tokens','ENABLE_PREFIX_CACHING':'enable_prefix_caching','MAMBA_CACHE_MODE':'mamba_cache_mode','PACK_TOKENS':'pack_tokens','HEAD_CHUNKS':'head_chunks',
        'num_epochs':'epochs','eval_num_samples':'eval_samples','env_max_turns':'max_turns',
        'train_group_size':'group_size','eval_group_size':'eval_samples',
        'sampling_max_model_len':'max_model_len','eval_max_concurrency':'concurrency',
        'train_concurrency':'concurrency','TRAIN_CONCURRENCY':'concurrency','EVAL_CONCURRENCY':'eval_concurrency',
        'log_path':'run_dir','train_task_limit':'task_limit',
        'swe_endpoint':'endpoint','swe_token_file':'token_file'}
    result=[]
    for arg in argv:
        if '=' not in arg or arg.startswith('-'):
            result.append(arg);continue
        key,value=arg.split('=',1);dest=aliases.get(key,key)
        action=actions.get(dest)
        if action is None:
            raise ValueError(f'unsupported SWE10 parameter: {key}; Kevin/KernelBench environment options are not accepted')
        flag=next((f for f in action.option_strings if f.startswith('--') and not f.startswith('--no-')),None)
        if not flag:raise ValueError(f'unsupported SWE10 parameter: {key}')
        if isinstance(action,argparse.BooleanOptionalAction):
            values={'true':True,'false':False,'1':True,'0':False,'yes':True,'no':False}
            if value.lower() not in values:raise ValueError(f'{key} expects True/False or 1/0')
            result.append(flag if values[value.lower()] else '--no-'+flag[2:])
        elif isinstance(action,argparse._StoreTrueAction):
            if value.lower() not in ('true','false','1','0'):raise ValueError(f'{key} expects True/False')
            if value.lower() in ('true','1'):result.append(flag)
        else:result.extend([flag,value])
    return result


def task_selection(args, evaluation=False):
    separate = any(getattr(args, key) is not None for key in
                   ('eval_dataset', 'eval_task_limit', 'eval_instance_ids', 'eval_ids_file'))
    prefix = 'eval' if evaluation and separate else 'train'
    raw = getattr(args, prefix+'_instance_ids')
    path = getattr(args, prefix+'_ids_file')
    if raw is not None and path is not None:
        raise ValueError(f'{prefix}_instance_ids and {prefix}_ids_file are mutually exclusive')
    ids = None
    if raw is not None:
        ids = raw.split(',')
    elif path is not None:
        content = path.read_text(encoding='utf-8')
        ids = json.loads(content) if content.lstrip().startswith('[') else content.splitlines()
    if ids is not None:
        if (not isinstance(ids, list) or not ids or
            any(not isinstance(i, str) or not i or i != i.strip() for i in ids) or
            len(set(ids)) != len(ids)):
            raise ValueError(f'{prefix} IDs must be a nonempty list of unique exact strings without surrounding whitespace')
    limit = args.eval_task_limit if evaluation and separate and args.eval_task_limit is not None else args.task_limit
    if limit < 1:
        raise ValueError('task_limit and eval_task_limit must be positive')
    return dict(dataset=(args.eval_dataset or args.dataset) if evaluation and separate else args.dataset,
                ids=ids, count=len(ids) if ids is not None else limit,
                file='eval_tasks.private.jsonl' if evaluation and separate else 'tasks.private.jsonl')


def context_limits(args):
    """Only the total model window and per-response generation limit constrain input."""
    from arl.models.service import model_max_positions
    hard = model_max_positions(args.model)
    if args.max_model_len < 0:
        raise ValueError('max_model_len must be nonnegative; 0 reads the model config')
    window = args.max_model_len or hard
    if not window:
        raise ValueError('model context limit is unknown; specify max_model_len explicitly')
    if hard and window > hard:
        raise ValueError(f'max_model_len={window} exceeds model context limit {hard}')
    if window <= args.max_tokens:
        raise ValueError('max_model_len must exceed max_tokens to leave space for input')
    return window, window - args.max_tokens


def resolve_run_dir(args, *, reserve=False, now=None):
    """ARM-style date expansion and atomic version allocation; checks never write."""
    now = now or datetime.now().astimezone()
    if args.run_dir is not None:
        return Path(now.strftime(str(args.run_dir))).resolve()
    model = Path(args.model.rstrip('/')).name.lower()
    qwen = re.match(r'qwen\d+(?:\.\d+)?', model)
    model = (qwen.group() if qwen else model).replace('.', '')
    model = re.sub(r'[^a-z0-9]+', '_', model).strip('_') or 'model'
    parent = ROOT/'runs'
    stem = f'{model}_{now:%m%d}'
    if reserve:
        parent.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        path = parent/f'{stem}_v{version}'
        if reserve:
            try:
                path.mkdir()
                return path.resolve()
            except FileExistsError:
                pass
        elif not path.exists():
            return path.resolve()
        version += 1


def discover_gpu_ids():
    # Respect an explicit visibility mask; otherwise query physical device IDs.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and visible.lower() not in ('all','auto'):
        if not visible or visible=='-1':raise ValueError('CUDA_VISIBLE_DEVICES exposes no GPUs')
        return gpu_ids(visible)
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],
            capture_output=True,text=True,check=True,timeout=15)
        ids=[line.strip() for line in result.stdout.splitlines() if line.strip()]
        if ids and all(i.isdigit() for i in ids):return gpu_ids(','.join(ids))
    except (OSError,subprocess.SubprocessError):
        pass
    import torch
    count=torch.cuda.device_count()
    if not count:raise ValueError('No GPUs detected; specify TRAIN_GPUS and SAMPLER_GPUS explicitly for offline checks')
    return [str(i) for i in range(count)]


def gpu_ids(text):
    if text.lower() in ('all','auto'):return discover_gpu_ids()
    ids = text.split(',')
    if not ids or any(not g.strip() or g != g.strip() for g in ids) or len(ids) != len(set(ids)):
        raise ValueError('GPU lists must contain unique, nonempty device IDs')
    return ids


def validate_args(args):
    train, sampler = gpu_ids(args.train_gpus), gpu_ids(args.sampler_gpus)
    evaluation = gpu_ids(args.eval_gpus or args.sampler_gpus)
    args.train_gpus=','.join(train);args.sampler_gpus=','.join(sampler)
    if args.eval_gpus:args.eval_gpus=','.join(evaluation)
    if args.colocate:
        if not set(sampler) <= set(train):
            raise ValueError('--colocate requires sampler GPUs to be a subset of training GPUs')
    elif set(train) & set(sampler):
        raise ValueError('training and sampler GPUs overlap; use --colocate for shared GPUs')
    if not 0 < args.sampling_util < 1 or not 0 < args.eval_util < 1 or args.learning_rate <= 0:
        raise ValueError('sampling utilization must be in (0,1); learning rate must be positive')
    for name, limit in (('train_concurrency', args.concurrency), ('eval_concurrency', args.eval_concurrency)):
        if limit != -1 and limit < 1:
            raise ValueError(f'{name} must be positive or -1')
    for field in ('epochs','group_size','batch_size','eval_samples','max_turns','max_tokens'):
        if getattr(args,field) < 1:
            raise ValueError(f'{field} must be positive')
    if args.extra_samples < 0:
        raise ValueError('extra_samples must be nonnegative')
    for name in ('temperature', 'eval_temperature'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ('top_p', 'eval_top_p'):
        value = getattr(args, name)
        if value is not None and not 0 < value <= 1:
            raise ValueError(f'{name} must be in (0, 1]')
    task_selection(args)
    task_selection(args, True)
    if args.batch_size < len(train) and args.pack_tokens <= 0 and not args.sp16:
        raise ValueError('batch size below training rank count requires PACK_TOKENS>0 and rebalancing')
    if args.sampler_dp < 1 or len(sampler) % args.sampler_dp:
        raise ValueError('sampler GPU count must be divisible by sampler DP')
    if args.eval_tp < 1 or (not args.eval_async and len(evaluation) % args.eval_tp):
        raise ValueError('evaluation GPU count must be divisible by positive eval_tp')
    for field in ('pack_tokens','head_chunks','warmup_steps','think_budget','think_cost_free_tokens'):
        if getattr(args,field) < 0:
            raise ValueError(f'{field} must be nonnegative')
    if not 0 < args.warmup_lr_scale <= 1:
        raise ValueError('warmup LR scale must be in (0,1]')
    if args.think_budget and args.think_budget >= args.max_tokens:
        raise ValueError('think budget must leave generation space for the answer')
    if args.eval_every < 1:raise ValueError('eval_every must be positive')
    if not (Path(args.model) / 'config.json').is_file():
        raise ValueError(f'local model config not found: {args.model}')
    if args.sp16:
        from arl.models.sequence_parallel import validate_model
        validate_model(json.loads((Path(args.model) / 'config.json').read_text()), len(train))
    context_limits(args)
    from arl.models.prefill import engine_args as prefill_engine_args
    prefill_engine_args(args.prefill_tokens,args.prefill_batch_tokens)
    from arl.models.prefix_cache import engine_args as prefix_cache_engine_args
    prefix_cache_engine_args(args.enable_prefix_caching,args.mamba_cache_mode,
        json.loads((Path(args.model)/'config.json').read_text()))


def configurations(args, run):
    train_selection, eval_selection = task_selection(args), task_selection(args, True)
    window, input_capacity = context_limits(args)
    common = dict(test_fraction=0,
        swe_endpoint=args.endpoint, swe_token_file=args.token_file, swe_ssh_host='',
        swe_command_timeout=60, swe_eval_timeout=1800, swe_proxy='http://10.0.134.70:10809',
        eval_max_concurrency=args.concurrency, env_max_turns=args.max_turns,
        max_tokens=args.max_tokens, sampling_max_model_len=window,
        sampling_prefill_tokens=args.prefill_tokens, sampling_prefill_batch_tokens=args.prefill_batch_tokens,
        enable_prefix_caching=args.enable_prefix_caching, mamba_cache_mode=args.mamba_cache_mode,
        prompt_budget_tokens=input_capacity,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        enable_thinking=True, truncated_turn=args.truncated_turn,
        sampling_replay=args.sampling_replay, router_replay=args.router_replay,
        think_budget=args.think_budget, think_cost_lambda=0, think_cost_free_tokens=args.think_cost_free_tokens,
        save_trajectories=args.save_trajectories, save_best_solutions=args.save_best_solutions)
    def config(phase, model, output, options):
        selection = train_selection if phase == 'train' else eval_selection
        options = dict(train_set=str(run/selection['file']), task_limit=selection['count'], **options)
        if phase == 'evaluate':
            for key in ('temperature', 'top_p', 'top_k'):
                override = getattr(args, f'eval_{key}')
                if override is not None:
                    options[key] = override
            if options.get('temperature', args.temperature) == 0:
                # Greedy evaluation has no sampling distribution to replay.
                options['sampling_replay'] = False
            # Expert routing is replayed during training updates, not evaluation.
            options['router_replay'] = False
            options['batch_size'] = eval_selection['count']
            options['eval_max_concurrency'] = args.eval_concurrency
        return {'version':1, 'name':'swe_first10_training_set_diagnostic',
                'models':{'policy':{'backend':'local','name':model}},
                phase:{'env':'coding.swe','harness':args.harness,'model':'policy',
                       'split':'train','output_dir':str(output),'options':{**common,**options}}}
    baseline = config('evaluate', args.model, run/'baseline', dict(group_size=args.eval_samples))
    train = config('train', args.model, run/'train', dict(batch_size=args.batch_size,group_size=args.group_size,
        extra_samples=args.extra_samples,
        num_epochs=args.epochs, shuffle_seed=0, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, warmup_lr_scale=args.warmup_lr_scale, grad_clip=.05, num_substeps=2,
        clip_low=.2, clip_high=.28, group_adv_std=False, whiten_advantages=True, whiten_std=False,
        length_weighted_baseline=False, eval_every=0, eval_at_start=False,
        save_every=args.eval_every if args.eval_async else 0,
        save_optimizer=args.save_optimizer, adv_estimator=args.adv_estimator,
        mask_positive_incorrect=args.mask_positive_incorrect, mask_negative_correct=args.mask_negative_correct,
        think_cost_lambda=args.think_cost_lambda, format_reward_weight=args.format_reward_weight,
        wandb_project='ARL' if args.wandb else None, wandb_name=run.name))
    # The initial baseline is a separate evaluation, including when the launcher
    # receives eval_at_start=True. Keep internal evaluation off to avoid creating
    # a held-out split or starting a second evaluator on training GPUs.
    # Both evaluations use the same per-response limit and total model window.
    final = str(run/'train/checkpoints/final/policy')
    after = config('evaluate', final, run/'after', dict(group_size=args.eval_samples))
    return {'baseline':baseline, 'train':train, 'evaluate':after}


def write_fixed(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'existing configuration differs: {path}; choose a new run directory')
        return
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


async def freeze_tasks(args, run):
    from arl.envs.coding.swe import _import_sweenv
    ce = _import_sweenv()
    backend = ce.DockerBackend(ce.BackendConfig(endpoint=args.endpoint,token_file=args.token_file))
    cache = {}
    selections = [task_selection(args), task_selection(args, True)]
    frozen = {}
    try:
        for selection in selections:
            path = run/selection['file']
            if path.name in frozen:
                continue
            ids = selection['ids']
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            else:
                key = (selection['dataset'], 0 if ids is not None else selection['count'])
                if key not in cache:
                    cache[key] = await backend.load_tasks(*key)
                rows = cache[key]
                if ids is not None:
                    by_id = {}
                    for row in rows:
                        instance_id = ce.SweTask.from_record(row).instance_id
                        if instance_id in by_id:
                            raise ValueError(f'duplicate source instance ID: {instance_id}')
                        by_id[instance_id] = row
                    missing = [i for i in ids if i not in by_id]
                    if missing:
                        raise ValueError(f'instance IDs absent from {selection["dataset"]}: {missing}')
                    rows = [by_id[i] for i in ids]
            actual = [ce.SweTask.from_record(row).instance_id for row in rows]
            if len(actual) != selection['count'] or len(set(actual)) != len(actual):
                raise ValueError(f'expected {selection["count"]} unique tasks in {path.name}; got {len(actual)}')
            if ids is not None and actual != ids:
                raise ValueError(f'frozen IDs differ from requested selection: {path}; choose a new run directory')
            frozen[path.name] = actual
        train_ids, eval_ids = [frozen[x['file']] for x in selections]
        overlap = sorted(set(train_ids) & set(eval_ids))
        scope = ('training_set_diagnostic' if set(train_ids) == set(eval_ids) else
                 'held_out_evaluation' if not overlap else 'partially_overlapping_evaluation')
        # Validate selection identity before publishing any new records.
        manifest = dict(scope=scope, selections=selections, source_dataset=args.dataset,
            instance_ids=eval_ids, train_instance_ids=train_ids, evaluation_instance_ids=eval_ids,
            overlap_instance_ids=overlap, evaluation_split='train',
            note='Baseline and final use identical frozen evaluation tasks; inspect train/eval overlap.')
        old_path = run/'tasks.json'
        if old_path.exists():
            old = json.loads(old_path.read_text())
            legacy_default = ('selections' not in old and args.dataset == DATASET and
                              selections[0] == selections[1] and selections[0]['ids'] is None and
                              selections[0]['count'] == 10 and old.get('source_dataset') == args.dataset)
            if ((not legacy_default and old.get('selections') != selections) or
                old.get('train_instance_ids') != train_ids or old.get('evaluation_instance_ids') != eval_ids):
                raise ValueError('existing task selection differs; choose a new run directory')
        for selection in selections:
            path = run/selection['file']
            if not path.exists():
                key = (selection['dataset'], 0 if selection['ids'] is not None else selection['count'])
                by_id = {ce.SweTask.from_record(row).instance_id: row for row in cache[key]}
                with open(path,'x',encoding='utf-8',opener=lambda p,f:os.open(p,f,0o600)) as f:
                    for instance_id in frozen[path.name]:
                        f.write(json.dumps(by_id[instance_id],ensure_ascii=False)+'\n')
        manifest['sha256'] = hashlib.sha256((run/'tasks.private.jsonl').read_bytes()).hexdigest()
        manifest['evaluation_sha256'] = hashlib.sha256((run/selections[1]['file']).read_bytes()).hexdigest()
        # Existing default first-10 runs retain their original immutable manifest.
        if old_path.exists() and legacy_default:
            if old.get('sha256') != manifest['sha256']:
                raise ValueError('frozen task records changed; choose a new run directory')
            manifest = old
        write_fixed(old_path,manifest)
        union = list(dict.fromkeys(train_ids+eval_ids))
        ready = await backend.rpc('preflight',instance_ids=union)
        if not ready['ready']:
            raise RuntimeError(f"SWE images are not ready: {ready.get('missing')}")
        print(json.dumps({'prepared':True,'train_tasks':train_ids,'evaluation_tasks':eval_ids,'images_ready':True},ensure_ascii=False),flush=True)
    finally:
        await backend.close()


def runtime_env(args, train):
    env = os.environ.copy()
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    train_gpus=','.join(gpu_ids(args.train_gpus));sampler_gpus=','.join(gpu_ids(args.sampler_gpus))
    eval_gpus=','.join(gpu_ids(args.eval_gpus or args.sampler_gpus))
    visible = train_gpus if args.colocate else train_gpus+','+sampler_gpus
    env.update(PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        CUDA_VISIBLE_DEVICES=visible if train else eval_gpus,
        KERNEL_RL_SAMPLER_GPUS=sampler_gpus if train else eval_gpus,
        KERNEL_RL_SAMPLER_DP=str(args.sampler_dp if train else len(gpu_ids(eval_gpus))//args.eval_tp),
        KERNEL_RL_SAMPLING_GPU_UTIL=str(args.sampling_util if train else args.eval_util), KERNEL_RL_ZERO_OFFLOAD='optimizer',
        KERNEL_RL_SP16='1' if train and args.sp16 else '0',
        KERNEL_RL_MICRO_BSZ='1', KERNEL_RL_PACK_TOKENS=str(args.pack_tokens), KERNEL_RL_HEAD_CHUNKS=str(args.head_chunks),
        KERNEL_RL_REBALANCE='1', KERNEL_RL_COLOCATE='1' if train and args.colocate else '0',
        KERNEL_RL_COLOCATE_HEADROOM_GB='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        DEEPSPEED_TIMEOUT='90', NCCL_DEBUG='WARN')
    if not train:
        env.setdefault('NCCL_SHM_DISABLE','1')
        env.setdefault('NCCL_P2P_LEVEL','SYS')
    overlay=getattr(args,'vllm_overlay',None)
    if overlay:
        env['PYTHONPATH']=str(overlay)+os.pathsep+str(ROOT)
    return env


def command(args, run, phase):
    cmd = [sys.executable]
    if phase=='train' and len(gpu_ids(args.train_gpus))>1:
        cmd += ['-m','torch.distributed.run',f'--nproc_per_node={len(gpu_ids(args.train_gpus))}',f'--master_port={args.port}']
    cmd += ['-m','arl.experiments',str(run/f'{phase}.json'),'--phase','train' if phase=='train' else 'evaluate']
    return cmd


def execute(args, run, phase):
    output = run/('after' if phase=='evaluate' else phase)
    if output.exists():
        raise ValueError(f'{output} already exists; choose a new run directory or another --phase')
    if phase=='evaluate' and not (run/'train/checkpoints/final/policy/config.json').is_file():
        raise ValueError('final checkpoint is absent; finish the train phase first')
    cmd = command(args,run,phase)
    print(f'>>> {phase}: {cmd}',flush=True)
    with (run/f'{phase}.log').open('a') as log:
        proc = subprocess.Popen(cmd,cwd=ROOT,env=runtime_env(args,phase=='train'),
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        try:
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait():
                raise RuntimeError(f'{phase} failed with exit code {proc.returncode}; see {run/phase}.log')
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            raise


def evaluation_report(path, ids, samples):
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError(f'evaluation incomplete: {path}')
    rows=[]
    outcomes=path/'outcomes.jsonl'
    if outcomes.exists():
        rows=[json.loads(line) for line in outcomes.read_text().splitlines() if line]
    else:
        for file in sorted((path/'trajectories').glob('rank*.jsonl.gz')):
            with gzip.open(file,'rt') as f:rows.extend(json.loads(line) for line in f)
    if Counter(r['task'] for r in rows)!=Counter({i:samples for i in ids}):
        raise ValueError(f'evaluation task IDs/sample counts differ: {path}')
    scored=[r for r in rows if not r['dropped'] and not r['summary'].get('infra_error')]
    per_task={i:dict(attempts=samples,scored=sum(r['task']==i for r in scored),
        resolved=sum(r['task']==i and r['reward']>=1 for r in scored)) for i in ids}
    return dict(attempts=len(rows),scored=len(scored),dropped=len(rows)-len(scored),
        resolved=sum(r['reward']>=1 for r in scored),
        resolved_rate=sum(r['reward']>=1 for r in scored)/len(scored) if scored else None,
        tasks_solved_in_k=sum(v['resolved']>0 for v in per_task.values()),per_task=per_task)


def compare(args, run):
    manifest=json.loads((run/'tasks.json').read_text())
    ids=manifest.get('evaluation_instance_ids',manifest['instance_ids'])
    scope=manifest.get('scope','training_set_diagnostic')
    before_dir,after_dir=run/'baseline',run/'after'
    if args.eval_async:
        before_dir=Path(json.loads((run/'async_eval/done/base').read_text())['evaluation_dir'])
        after_dir=Path(json.loads((run/'async_eval/done/final').read_text())['evaluation_dir'])
    before=evaluation_report(before_dir,ids,args.eval_samples)
    after=evaluation_report(after_dir,ids,args.eval_samples)
    meta=json.loads((run/'train/checkpoints/final/policy/meta.json').read_text())
    updates=meta.get('policy_optimizer_steps')
    metrics_path=run/'train/metrics.jsonl'
    metrics=[json.loads(line) for line in metrics_path.read_text().splitlines() if line] if metrics_path.exists() else []
    signal_batches=sum(m.get('ppo/n_active_tokens',0)>0 and m.get('ppo/grad_norm',0)>0 for m in metrics)
    diagnostics=dict(logged_batches=len(metrics),batches_with_active_tokens_and_nonzero_gradient=signal_batches,
        mean_zero_advantage_fraction=sum(m.get('ppo/adv_zero_frac',0) for m in metrics)/len(metrics) if metrics else None)
    valid=not before['dropped'] and not after['dropped'] and isinstance(updates,int) and updates>0 and signal_batches>0
    delta=after['resolved_rate']-before['resolved_rate'] if before['resolved_rate'] is not None and after['resolved_rate'] is not None else None
    assessment=('inconclusive_infra_or_no_effective_update' if not valid else
                'improved_on_training_tasks' if delta>0 else 'no_improvement_observed_on_training_tasks')
    if valid and scope != 'training_set_diagnostic':
        assessment='improved_on_evaluation_tasks' if delta>0 else 'no_improvement_observed_on_evaluation_tasks'
    result=dict(scope=scope,initial_model=args.model,harness=args.harness,
        tasks=ids,evaluation_samples_per_task=args.eval_samples,baseline=before,after=after,
        resolved_rate_delta=delta,policy_optimizer_steps=updates,training_diagnostics=diagnostics,assessment=assessment,
        note=manifest.get('note','Sampling variation and small set size limit conclusions.'),
        train_tasks=manifest.get('train_instance_ids',ids),overlap_instance_ids=manifest.get('overlap_instance_ids',ids))
    (run/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','after','tasks')},ensure_ascii=False,indent=2))
    print(f"Resolved: {before['resolved']}/{before['scored']} -> {after['resolved']}/{after['scored']}; report: {run/'comparison.json'}")
    return result


def main(argv=None):
    args=parser().parse_args(normalize_key_values(sys.argv[1:] if argv is None else argv))
    validate_args(args)
    run=resolve_run_dir(args)
    configs=configurations(args,run)
    from arl.experiments.config import from_dict
    from arl.experiments.runner import validate_plan
    for phase,config in configs.items():
        validate_plan(from_dict(config).plan('train' if phase=='train' else 'evaluate'))
    if args.check:
        print(json.dumps(dict(validated=True,network_used=False,gpu_started=False,run_dir=str(run),
            training_ranks=len(gpu_ids(args.train_gpus)),training_gpus=args.train_gpus,sampler_gpus=args.sampler_gpus,
            sampler_dp=args.sampler_dp,sampler_tp=len(gpu_ids(args.sampler_gpus))//args.sampler_dp,
            sampler_replicas=[gpu_ids(args.sampler_gpus)[i:i+len(gpu_ids(args.sampler_gpus))//args.sampler_dp]
                for i in range(0,len(gpu_ids(args.sampler_gpus)),len(gpu_ids(args.sampler_gpus))//args.sampler_dp)],
            training_tasks=task_selection(args)['count'],
            evaluation_tasks=task_selection(args,True)['count'],task_selections=[task_selection(args),task_selection(args,True)],
            epochs=args.epochs,train_group_size=args.group_size,eval_group_size=args.eval_samples,
            group_size=args.group_size,extra_samples=args.extra_samples,eval_samples=args.eval_samples,
            train_concurrency=args.concurrency,
            train_workspace_limit=-1 if args.concurrency == -1 else args.concurrency*len(gpu_ids(args.train_gpus)),
            eval_tp=args.eval_tp,eval_concurrency=args.eval_concurrency,eval_util=args.eval_util,
            model=args.model,harness=args.harness,
            max_model_len=context_limits(args)[0],input_capacity=context_limits(args)[1],
            eval_async=args.eval_async,eval_at_start=args.eval_at_start,
            wandb_project='ARL' if args.wandb else None,wandb_name=run.name,
            configurations=configs,
            training_runtime={k:v for k,v in runtime_env(args,True).items() if k in (
                'KERNEL_RL_SP16','KERNEL_RL_PACK_TOKENS','KERNEL_RL_HEAD_CHUNKS','KERNEL_RL_COLOCATE','KERNEL_RL_SAMPLER_DP',
                'KERNEL_RL_ZERO3_REDUCE_BUCKET','KERNEL_RL_FREEZE_ROUTER','KERNEL_RL_FREEZE_SHARED_EXPERT')},
            commands={p:command(args,run,p) for p in configs}),ensure_ascii=False,indent=2))
        return
    if args.router_replay and args.phase!='compare':
        from arl.models.vllm_overlay import prepare_overlay
        args.vllm_overlay=prepare_overlay()
    if args.run_dir is None:
        # Reserve with mkdir so concurrent launches cannot share the same version.
        run=resolve_run_dir(args,reserve=True)
        configs=configurations(args,run)
    run.mkdir(parents=True,exist_ok=True)
    if getattr(args,'vllm_overlay',None):
        write_fixed(run/'vllm_overlay.json',json.loads((args.vllm_overlay/'overlay.json').read_text()))
    print('Run directory:',run,flush=True)
    print(f'W&B: project=ARL name={run.name}' if args.wandb else 'W&B disabled',flush=True)
    # Refuse identity/config changes before contacting the service or starting GPUs.
    for phase,config in configs.items():write_fixed(run/f'{phase}.json',config)
    if args.phase!='compare':asyncio.run(freeze_tasks(args,run))
    if args.eval_async and args.phase!='compare':
        from arl.experiments.async_swe_eval import publish_spec
        publish_spec(args,run,configs)
        print(f'Evaluation machine: ./script/eval_swe10_async.sh add {run} SAMPLING_UTIL={args.eval_util} # TP={args.eval_tp} from spec',flush=True)
    if args.phase=='prepare':return
    if args.phase=='all':
        if args.eval_async:
            execute(args,run,'train')
            report = (run/'comparison.json' if args.eval_at_start else run/'async_eval/metrics.jsonl')
            print(f'Training finished; asynchronous evaluator will produce {report}',flush=True)
        else:
            if args.eval_at_start:
                execute(args,run,'baseline')
            execute(args,run,'train')
            execute(args,run,'evaluate')
            if args.eval_at_start:
                compare(args,run)
    elif args.phase=='compare':compare(args,run)
    elif args.eval_async and args.phase in ('baseline','evaluate'):
        print('Evaluation is queued through async_eval/spec.json; start the evaluator on the other machine.',flush=True)
    else:execute(args,run,args.phase)


if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError) as exc:
        raise SystemExit(str(exc))
