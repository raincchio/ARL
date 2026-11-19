"""Execute a validated phase with independently selected env/harness/model."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import uuid

from .artifacts import append_datums, write_identity, write_json
from .config import PhasePlan

ROOT = Path(__file__).resolve().parents[2]

# Sampling/rendering and environment options only; optimizer settings have no
# effect in a standalone collect/evaluate process and therefore are rejected.
NATIVE_INFERENCE_OPTIONS = {
    'multicodeif_upstream', 'multicodeif_dependencies', 'multicodeif_suite', 'multicodeif_constraints',
    'multicodeif_languages', 'multicodeif_split_seed', 'multicodeif_max_rounds', 'multicodeif_reward_mode',
    'multicodeif_timeout', 'multicodeif_judge_base_url', 'multicodeif_judge_model', 'multicodeif_judge_api_key_env',
    'programbench_ssh_host', 'programbench_ssh_port', 'programbench_python', 'programbench_work_dir', 'programbench_blob_dir', 'programbench_memory', 'programbench_cpus', 'programbench_task_ids', 'programbench_split_seed', 'programbench_command_timeout', 'programbench_eval_timeout', 'programbench_eval_cpus', 'programbench_branch_workers', 'programbench_reward_mode',
    'train_set', 'test_set', 'test_fraction', 'task_limit', 'test_limit', 'env_max_turns',
    'env_enable_analyze', 'env_reward_mode', 'eval_endpoint', 'eval_max_concurrency',
    'swe_dataset_split', 'swe_endpoint', 'swe_token_file', 'swe_ssh_host', 'swe_ssh_port', 'swe_python', 'swe_worker_script', 'swe_work_dir',
    'swe_proxy', 'swe_memory', 'swe_cpus', 'swe_command_timeout', 'swe_eval_timeout',
    'ifeval_reward_mode', 'ifeval_split_seed', 'ifeval_python', 'ifeval_upstream', 'ifeval_timeout',
    'enable_thinking', 'reasoning_effort', 'batch_size', 'group_size', 'extra_samples',
    'temperature', 'top_p', 'top_k', 'max_tokens', 'prompt_budget_tokens',
    'sampling_prefill_tokens', 'sampling_prefill_batch_tokens', 'enable_prefix_caching', 'mamba_cache_mode',
    'sampling_max_model_len', 'truncated_turn', 'poll_seconds', 'think_cost_lambda',
    'think_cost_free_tokens', 'sampling_replay', 'router_replay', 'think_budget', 'think_force_text',
    'save_trajectories', 'save_best_solutions',
}


def prepare_native_config(plan):
    from arl.grpo import GRPOConfig
    options = {**plan.model.options, **plan.options}
    if plan.phase != 'train':
        unknown = set(options) - NATIVE_INFERENCE_OPTIONS
        if unknown:
            raise ValueError(f'options unused by native {plan.phase}: {sorted(unknown)}')
    else:
        # Independent evaluate phase is the default. Periodic GRPO evaluation can
        # still be explicitly enabled in train.options.
        options = {'eval_every': 0, 'eval_at_start': False, **options}
    try:
        cfg = GRPOConfig(model_name=plan.model.name, log_path=plan.output_dir or '/tmp/arl-check',
                         env=plan.env, harness=plan.harness,
                         environments=list(plan.environments) if plan.environments else None, **options)
    except TypeError as e:
        raise ValueError(f'invalid native options: {e}') from e
    from arl.algorithms.training_options import validate_training_options
    validate_training_options(cfg)
    if cfg.batch_size < 1 or cfg.group_size < 1 or cfg.max_tokens < 1 or cfg.prompt_budget_tokens < 1:
        raise ValueError('batch_size, group_size and token budgets must be positive')
    if plan.phase == 'train':
        from arl.grpo import _validate
        _validate(cfg, int(os.environ.get('WORLD_SIZE', '1')))
    else:
        from arl.algorithms.thinking_cost import validate_thinking_cost
        validate_thinking_cost(cfg)
        if cfg.truncated_turn not in ('mask', 'train', 'drop_traj'):
            raise ValueError('truncated_turn must be mask, train or drop_traj')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('standalone collect/evaluate currently use one process; configure sampler TP for multiple GPUs')
    return cfg


def validate_plan(plan):
    from arl.harnesses import resolve_pair
    if plan.environments:
        if plan.phase != 'train' or plan.model.backend != 'local':
            raise ValueError('mixed training requires a local model and train phase')
        prepare_native_config(plan)
        return plan
    pair = resolve_pair(plan.env, plan.harness, operation=plan.phase, model_backend=plan.model.backend)
    if pair.interface == 'agent_messages':
        prepare_native_config(plan)
    elif pair.interface == 'text_constraints':
        if plan.phase == 'collect' and plan.options.get('action', 'evaluate') != 'evaluate':
            raise ValueError('collect must generate/evaluate trajectories, not inspect/export task prompts')
        from arl.benchmarks.multicodeif import validate_options
        validate_options({**plan.model.options, **plan.options})
    else:
        raise ValueError(f'no phase runner for interface {pair.interface!r}')
    return plan


async def run_native(plan, *, source=None, sampling_client=None, tokenizer=None, progress_path=None):
    """Inference only: export sampled training datums or evaluate trajectories.

    Optional source/client/tokenizer injection exercises the complete data path
    without GPUs; production creates the existing local vLLM sampling runtime.
    """
    from arl.agent.rollout import BestSolutions, RolloutParams, TurnLogger
    from arl.harnesses import get_harness
    from arl.grpo import _make_source
    cfg = prepare_native_config(plan)
    from arl.algorithms.training_options import configure_training_runtime, thinking_sampling_options
    configure_training_runtime(cfg)
    output = Path(plan.output_dir)
    client = None
    source = source or _make_source(cfg)
    selected = get_harness(plan.harness)
    count = n_scored = n_passed = n_dropped = 0
    reward_sum = 0.
    manifest = None
    try:
        if hasattr(source, 'initialize'):
            await source.initialize()
        if hasattr(source, 'check'):
            health = source.check()
            if inspect.isawaitable(health):
                await health
        tasks = list(source.train_tasks() if plan.split == 'train' else source.test_tasks())
        if not tasks:
            raise ValueError(f'no tasks in {plan.split} split; set split explicitly or configure held-out data')
        if sampling_client is None:
            from arl.env import setup_environment
            from arl.models.service import LocalServiceClient, sampling_window
            from arl.models.dist_sampling import DistributedSamplingClient
            setup_environment()
            service = LocalServiceClient(inference_only=True, model_name=plan.model.name,
                max_tokens=cfg.max_tokens, enable_value_head=False,
                sampling_prefill_tokens=cfg.sampling_prefill_tokens,
                sampling_prefill_batch_tokens=cfg.sampling_prefill_batch_tokens,
                enable_prefix_caching=cfg.enable_prefix_caching, mamba_cache_mode=cfg.mamba_cache_mode,
                sampling_gpu_memory_utilization=float(os.environ.get('KERNEL_RL_SAMPLING_GPU_UTIL', '0.8')),
                sampling_max_model_len=sampling_window(plan.model.name, cfg.max_tokens,
                    cfg.sampling_max_model_len or cfg.prompt_budget_tokens + cfg.max_tokens,
                    prompt_floor=cfg.prompt_budget_tokens))
            client = await service.create_training_client_async(plan.model.name)
            tokenizer = client.get_tokenizer()
            sampling_client = DistributedSamplingClient(client._ensure_sampling_engine())
        if tokenizer is None:
            raise ValueError('native harness requires a tokenizer')
        renderer = selected.make_renderer(tokenizer, enable_thinking=cfg.enable_thinking,
                                          reasoning_effort=cfg.reasoning_effort or None)
        logger = TurnLogger(str(output), 0, cfg.save_trajectories)
        best = BestSolutions(str(output), 0, cfg.save_best_solutions)
        from arl.harnesses import resolve_pair
        pair = resolve_pair(plan.env, plan.harness, operation=plan.phase)
        manifest = {'schema': 'arl.collection.v1' if plan.phase == 'collect' else 'arl.evaluation.v1',
                    'identity': plan.identity, 'phase': plan.phase, 'split': plan.split, 'status': 'running',
                    'trace_fields': list(pair.trace_fields) if plan.phase == 'collect' else []}
        write_identity(output/'manifest.json', manifest)
        completed = turns_done = 0
        def progress(finished=False):
            if progress_path is not None:
                path = Path(progress_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, {'trajectories_done': completed, 'trajectories_total': len(tasks)*cfg.group_size,
                    'turns_done': turns_done, 'turns_total': len(tasks)*cfg.group_size*(cfg.env_max_turns or 250),
                    'finished': finished})
        progress()
        for batch, start in enumerate(range(0, len(tasks), cfg.batch_size)):
            params = RolloutParams(**thinking_sampling_options(cfg, tokenizer), gen_cap=cfg.max_tokens, prompt_budget=cfg.prompt_budget_tokens,
                temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                group_size=cfg.group_size, extra_samples=cfg.extra_samples,
                progress_path=str(output/'rollout_progress.rank0.json'),
                truncated_turn=cfg.truncated_turn, poll_seconds=cfg.poll_seconds,
                think_cost_lambda=cfg.think_cost_lambda if plan.phase == 'collect' else 0.,
                think_cost_free_tokens=cfg.think_cost_free_tokens)
            result = await selected.run(tasks=tasks[start:start+cfg.batch_size], source=source,
                sampling_client=sampling_client, renderer=renderer, params=params, operation=plan.phase, env_name=plan.env,
                uid_prefix=f'{plan.phase}-{batch}-', log_turns=cfg.save_trajectories)
            logger.write(batch, result.trajs, tag=plan.phase, identity=plan.identity)
            best.update(batch, result.trajs, tag=plan.phase, identity=plan.identity)
            completed += len(result.trajs)
            turns_done += sum(len(t.resp_lens) for t in result.trajs)
            progress()
            if plan.phase == 'evaluate':
                # Keep scalar outcomes for comparisons even when full traces are disabled.
                with (output/'outcomes.jsonl').open('a') as outcomes:
                    for traj in result.trajs:
                        outcomes.write(json.dumps({'task': traj.task_key, 'batch': batch, 'tid': traj.tid,
                            'reward': traj.reward, 'dropped': traj.dropped,
                            'summary': {'infra_error': traj.infra}}, allow_nan=False)+'\n')
            if plan.phase == 'collect':
                from arl.algorithms.rollout_data import validate_rollout
                validate_rollout(result)
                count += append_datums(output/'datums.jsonl', result, plan.identity, batch)
            for traj in result.trajs:
                if traj.dropped or traj.infra:
                    n_dropped += 1
                    continue
                n_scored += 1
                reward_sum += traj.reward
                n_passed += traj.reward >= 1
        summary = {'identity': plan.identity, 'phase': plan.phase, 'split': plan.split,
                   'datums': count, 'trajectories_scored': n_scored, 'trajectories_dropped': n_dropped,
                   'reward_mean': reward_sum/n_scored if n_scored else None,
                   'pass_rate': n_passed/n_scored if n_scored else None}
        if plan.phase == 'collect' and count == 0:
            raise ValueError('collection produced no trainable datums; see trajectories for drops')
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        manifest['status'] = 'complete'
        write_json(output/'manifest.json', manifest)
        progress(finished=True)
        return summary
    except BaseException:
        if manifest is not None:
            manifest['status'] = 'failed'
            write_json(output/'manifest.json', manifest)
        raise
    finally:
        try:
            if hasattr(source, 'close'):
                await source.close()
        finally:
            if client is not None:
                client.shutdown()


def run_text(plan):
    from arl.benchmarks import run_benchmark
    options = {**plan.model.options, **plan.options}
    options['harness'] = plan.harness
    options['task_split'] = plan.split
    if plan.model.backend == 'openai_compatible':
        options.update(model=plan.model.name, base_url=plan.model.base_url)
        if plan.model.api_key_env:
            options['api_key_env'] = plan.model.api_key_env
    options.setdefault('action', 'evaluate')
    options['output_dir'] = str(Path(plan.output_dir)/'benchmark')
    result = run_benchmark(plan.env, **options)
    root = Path(plan.output_dir)
    if plan.phase == 'collect':
        # Text feedback trajectories are useful for SFT/distillation. They do not
        # become PPO datums: this harness intentionally has no old-policy traces.
        report = result.get('report')
        if not report:
            raise ValueError('collect did not produce a benchmark report')
        source = Path(report).parent/'trajectories.jsonl'
        count = dropped = 0
        with (root/'samples.jsonl').open('w') as dst, source.open() as src:
            for line in src:
                item = json.loads(line)
                if not item.get('response') or item.get('generation_error'):
                    dropped += 1
                    continue
                count += 1
                item.update(identity=plan.identity, schema='arl.text_trajectory.v1')
                dst.write(json.dumps(item, ensure_ascii=False)+'\n')
        if count == 0:
            raise ValueError('collection produced no usable model responses')
        write_identity(root/'manifest.json', {'schema': 'arl.collection.v1', 'identity': plan.identity,
            'phase': 'collect', 'status': 'complete', 'trace_fields': [], 'format': 'text_feedback',
            'samples': count, 'dropped_generation_errors': dropped})
    return result


def run_phase(plan: PhasePlan, *, native_runtime=None):
    """Validate first; execution never silently falls back to another harness."""
    validate_plan(plan)
    if plan.phase == 'train' and int(os.environ.get('WORLD_SIZE', '1')) > 1 and not plan.output_dir:
        raise ValueError('distributed training requires an explicit shared output_dir')
    output = Path(plan.output_dir) if plan.output_dir else ROOT/'runs'/(
        f'experiment_{plan.phase}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}')
    if plan.phase == 'train':
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    plan = replace(plan, output_dir=str(output.resolve()))
    if int(os.environ.get('RANK', '0')) == 0:
        write_identity(output/'experiment.json', plan.identity)
        invocation = output/'invocations'
        invocation.mkdir(exist_ok=True)
        (invocation/(uuid.uuid4().hex+'.json')).write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    from arl.harnesses import resolve_pair
    interface = 'agent_messages' if plan.environments else resolve_pair(plan.env, plan.harness,
        operation=plan.phase, model_backend=plan.model.backend).interface
    if plan.phase == 'train':
        from arl.grpo import run_grpo
        import asyncio
        asyncio.run(run_grpo(prepare_native_config(plan)))
        return {'phase': 'train', 'output_dir': str(output), 'identity': plan.identity}
    if interface == 'agent_messages':
        import asyncio
        return asyncio.run(run_native(plan, **(native_runtime or {})))
    return run_text(plan)
