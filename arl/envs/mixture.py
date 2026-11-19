"""Validated environment mixtures and deterministic homogeneous batch schedules."""
from dataclasses import dataclass
import math
import random
import re

from arl.harnesses import resolve_pair

# Per-environment data, rollout and rendering settings. Optimizer/model/batch size
# stay shared by the run; each batch runs one member on every distributed rank.
MEMBER_OPTIONS = frozenset({
    'multicodeif_upstream', 'multicodeif_dependencies', 'multicodeif_suite', 'multicodeif_constraints',
    'multicodeif_languages', 'multicodeif_split_seed', 'multicodeif_max_rounds', 'multicodeif_reward_mode',
    'multicodeif_timeout', 'multicodeif_judge_base_url', 'multicodeif_judge_model', 'multicodeif_judge_api_key_env',
    'programbench_ssh_host', 'programbench_ssh_port', 'programbench_python', 'programbench_work_dir', 'programbench_blob_dir', 'programbench_memory', 'programbench_cpus', 'programbench_task_ids', 'programbench_split_seed', 'programbench_command_timeout', 'programbench_eval_timeout', 'programbench_eval_cpus', 'programbench_branch_workers', 'programbench_reward_mode',
    'train_set', 'test_set', 'test_fraction', 'task_limit', 'test_limit', 'env_max_turns',
    'env_enable_analyze', 'env_reward_mode', 'eval_endpoint', 'eval_max_concurrency',
    'swe_dataset_split', 'swe_endpoint', 'swe_token_file', 'swe_ssh_host', 'swe_ssh_port', 'swe_python', 'swe_worker_script', 'swe_work_dir',
    'swe_proxy', 'swe_memory', 'swe_cpus', 'swe_command_timeout', 'swe_eval_timeout',
    'ifeval_reward_mode', 'ifeval_split_seed', 'ifeval_python', 'ifeval_upstream', 'ifeval_timeout',
    'enable_thinking', 'reasoning_effort', 'group_size', 'extra_samples', 'temperature', 'top_p', 'top_k',
    'max_tokens', 'prompt_budget_tokens', 'truncated_turn', 'poll_seconds',
    'think_cost_lambda', 'think_cost_free_tokens', 'think_budget', 'think_force_text', 'eval_harness', 'eval_num_samples', 'eval_temperature', 'eval_top_p', 'eval_top_k',
})


def normalize_members(rows, *, harness=None, operation='train', model_backend='local'):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('environments must be a nonempty list')
    members, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row)-{'id', 'env', 'harness', 'weight', 'options'}:
            raise ValueError('each environment accepts id/env/harness/weight/options')
        if not isinstance(row.get('env'), str):
            raise ValueError('each environment requires env')
        pair = resolve_pair(row['env'], row.get('harness', harness), operation=operation, model_backend=model_backend)
        if pair.interface != 'agent_messages':
            raise ValueError('mixed training currently requires native agent_messages environments')
        member_id = row.get('id', pair.env)
        if not isinstance(member_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', member_id) or member_id in seen:
            raise ValueError('environment ids must be unique and contain only letters/digits/_.-')
        seen.add(member_id)
        weight = row.get('weight', 1.)
        if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError('environment weight must be finite and positive')
        options = row.get('options', {})
        if not isinstance(options, dict) or set(options)-MEMBER_OPTIONS:
            raise ValueError(f'invalid per-environment options; only data/render/rollout settings allowed: {sorted(set(options)-MEMBER_OPTIONS) if isinstance(options, dict) else options}')
        options = dict(options)
        if options.get('eval_harness'):
            options['eval_harness'] = resolve_pair(pair.env, options['eval_harness'], operation='evaluate', model_backend=model_backend).harness
        members.append(dict(id=member_id, env=pair.env, harness=pair.harness, weight=float(weight), options=options))
    # Normalize without overflowing when multiple large weights are supplied.
    scale = max(m['weight'] for m in members)
    total = sum(m['weight']/scale for m in members)
    for member in members:
        member['weight'] = (member['weight']/scale)/total
        if member['weight'] == 0:
            raise ValueError('environment weight ratio is too extreme')
    return tuple(members)


@dataclass(frozen=True)
class Member:
    id: str
    cfg: object
    weight: float


def expand_config(cfg):
    if cfg.environments is None:
        pair = resolve_pair(cfg.env, cfg.harness, operation='train', model_backend='local')
        import chz
        resolved = chz.replace(cfg, env=pair.env, harness=pair.harness)
        return (Member(pair.env, resolved, 1.),)
    rows = normalize_members(cfg.environments, harness=cfg.harness)
    import chz
    return tuple(Member(row['id'], chz.replace(cfg, environments=None,
        env=row['env'], harness=row['harness'], **row['options']), row['weight']) for row in rows)


def batch_schedule(members, sizes, *, batch_size, epochs, seed=0, steps_per_epoch=0):
    """One environment per global batch, weighted choice; task pools reshuffle on exhaustion.

    Reconstruct from the start and skip start_batch to resume the exact schedule.
    RNGs are local so source/runtime random calls cannot alter rank agreement.
    """
    if batch_size < 1 or epochs < 1 or steps_per_epoch < 0:
        raise ValueError('invalid mixture batch_size/epochs/steps_per_epoch')
    if len(sizes) != len(members) or any(size < 1 for size in sizes):
        raise ValueError('every mixture environment must have training tasks')
    steps = steps_per_epoch or math.ceil(sum(sizes)/batch_size)
    batches = []
    for epoch in range(epochs):
        chooser = random.Random(seed+epoch)
        rngs = [random.Random(f'{seed}:{epoch}:{member.id}') for member in members]
        pools, offsets = [[] for _ in members], [0]*len(members)
        for _ in range(steps):
            index = chooser.choices(range(len(members)), weights=[m.weight for m in members], k=1)[0]
            chosen = []
            while len(chosen) < batch_size:
                if offsets[index] == len(pools[index]):
                    pools[index] = list(range(sizes[index]))
                    rngs[index].shuffle(pools[index])
                    offsets[index] = 0
                take = min(batch_size-len(chosen), len(pools[index])-offsets[index])
                chosen.extend(pools[index][offsets[index]:offsets[index]+take])
                offsets[index] += take
            batches.append((index, chosen))
    return batches


class SourcePool:
    """Own every initialized member source, including partial-initialization failures."""
    def __init__(self, members):
        self.members, self.sources = members, []
        self.closed = False

    async def initialize(self):
        from arl.envs import make_task_source_from_config
        try:
            for member in self.members:
                source = make_task_source_from_config(member.cfg)
                self.sources.append(source)
                if hasattr(source, 'initialize'):
                    await source.initialize()
        except BaseException:
            await self.close()
            raise

    async def close(self):
        import asyncio
        if self.closed:
            return
        self.closed = True
        results = await asyncio.gather(*(source.close() for source in self.sources if hasattr(source, 'close')), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                import logging
                logging.getLogger(__name__).warning('environment close failed: %s', result)
