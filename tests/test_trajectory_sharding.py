"""CPU/Gloo regression: split siblings retain full-group GRPO semantics."""
import asyncio
from datetime import timedelta
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arl.agent.rollout import RolloutParams, run_agent_rollout, trajectory_slots
from arl.algorithms.ppo_data import finalize_ppo_data
from arl.algorithms.rollout_data import validate_rollout
from arl.models.dist_sampling import DistributedSamplingClient
from arl.models.types import Datum, ModelInput, SampledSequence, TensorData
from arl.utils import distributed as du
from test_swe_env import Source, Renderer, Sampler, ce, record


def datum(n):
    return Datum(ModelInput.from_ints([1] * (n+1)), {
        'mask': TensorData.from_torch(torch.tensor([0.] + [1.] * n)),
        'target_tokens': TensorData.from_torch(torch.ones(n+1, dtype=torch.long)),
        'logprobs': TensorData.from_torch(torch.zeros(n+1)),
    })


async def check_rollout(rank, world, root, group_size, extra):
    path = Path(root) / f'progress-{group_size}-{extra}-{rank}.json'
    class DelayedExtrasSampler(Sampler):
        def __init__(self):
            self.pending = {}
            self.aborted = []

        async def submit_and_poll(self, requests, *, abort=(), **kwargs):
            for rid in abort:
                self.pending.pop(rid, None)
                self.aborted.append(rid)
            for request in requests:
                self.pending[request.extras['req_id']] = request
            ready = [request for rid, request in self.pending.items()
                     if int(rid.split(':')[0]) < group_size]
            for request in ready:
                del self.pending[request.extras['req_id']]
            return await super().submit_and_poll(ready, **kwargs)

    sampler = DelayedExtrasSampler() if extra else Sampler()
    result = await run_agent_rollout([ce.SweTask.from_record(record())], Source(), sampler, Renderer(),
        RolloutParams(gen_cap=10, prompt_budget=100, group_size=group_size, extra_samples=extra,
                      shard_trajectories=True, poll_seconds=.001, progress_path=str(path)))
    validate_rollout(result)
    progress = json.loads(path.read_text())
    assert progress['total'] == len(result.trajs)
    assert progress['status'] == 'complete'
    assert all(t.env.workspace.closed for t in result.trajs)
    rows = du.all_gather_object([(t.tid, t.dropped) for t in result.trajs], group=du.gloo_group())
    flat = sorted(x for row in rows for x in row)
    assert [tid for tid, _ in flat] == list(range(group_size+extra))
    assert sum(not dropped for _, dropped in flat) == group_size
    assert sum(dropped == 'extra_sample' for _, dropped in flat) == extra
    if extra:
        assert [tid for tid, dropped in flat if not dropped] == list(range(group_size))
        assert not sampler.pending
        if rank >= group_size:
            assert sampler.aborted  # Unfinished generation was cancelled across ranks.
    # Ranks with no valid datum still take part in group statistics.
    finalized, _ = finalize_ppo_data(result.datums, result.returns, groups=result.groups,
        units=result.units, distributed_groups=True)
    assert len(finalized) == len(result.datums)


async def check_direct_rollout(rank, extra=0, insufficient=False):
    remote_second_turn = asyncio.Event()
    extra_started = asyncio.Event()
    extra_cancelled = asyncio.Event()

    class SamplerWithEngine(DistributedSamplingClient):
        async def _one(self, request):
            tid, turn = map(int, request.extras['req_id'].rsplit('.', 1)[-1].split(':'))
            if extra and tid >= 2:
                if tid == 2:
                    extra_started.set()
                    if not insufficient:
                        try:
                            await asyncio.Future()
                        finally:
                            extra_cancelled.set()
                return None
            if (tid, turn) == (0, 0):
                # Rank 1 must receive its first result and request its next turn
                # while rank 0's first request is still waiting.
                await asyncio.wait_for(remote_second_turn.wait(), timeout=10)
            if (tid, turn) == (1, 1):
                remote_second_turn.set()
                if extra:
                    await asyncio.wait_for(extra_started.wait(), timeout=10)
                if insufficient:
                    return None
            routing = np.full((len(request.prompt_ids), 2, 2), 270, dtype=np.int16)
            return SampledSequence([11 if turn else 10], [-.5], routed_experts=routing)

        async def submit_and_poll(self, *args, **kwargs):
            raise AssertionError('standard rollout should use direct requests')

    client = SamplerWithEngine(object() if rank == 0 else None)
    result = await run_agent_rollout([ce.SweTask.from_record(record())], Source(), client, Renderer(),
        RolloutParams(gen_cap=10, prompt_budget=100, group_size=2,
                      extra_samples=extra, shard_trajectories=True),
        uid_prefix='rpc-')
    validate_rollout(result)
    if extra and rank >= 2:
        assert len(result.trajs) == 1
        assert result.trajs[0].dropped == ('sampling_failed' if insufficient else 'extra_sample')
        assert result.trajs[0].env.workspace.closed
    elif insufficient and rank == 1:
        assert result.trajs[0].dropped == 'sampling_failed'
    elif rank < 2:
        assert len(result.trajs) == 1
        assert len(result.trajs[0].transitions) == 2
        for transition in result.trajs[0].transitions:
            trace = transition.ac.maybe_routing
            assert trace.dtype == np.int16 and np.all(trace == 270)
    else:
        assert not result.trajs
    if rank == 0 and extra and not insufficient:
        assert extra_cancelled.is_set()


def distributed_worker(rank, world, root):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                      MASTER_ADDR='127.0.0.1')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{root}/init', rank=rank,
                            world_size=world, timeout=timedelta(seconds=45))
    try:
        # Duplicate segments of one trajectory, two different tasks with signal,
        # and a constant-reward task which must stay zero after whitening.
        rewards = [0., 0., 2., 5., -1., 3., 7., 7.]
        groups = [0, 0, 0, 0, 1, 1, 2, 2]
        units = [0, 0, 1, 2, 3, 4, 5, 6]
        lengths = [1, 3, 2, 9, 5, 2, 4, 1]
        labels = [False, False, True, False, True, False, None, True]
        for empty_rank in (False, True):
            # Deliberately collide local unit IDs across ranks. Both segments
            # of a unit stay together, as they do in the real rollout.
            owners = [unit % (world-1 if empty_rank else world) for unit in units]
            indices = [i for i, owner in enumerate(owners) if owner == rank]
            local_units = [units[i] // (world-1 if empty_rank else world) for i in indices]
            for estimator, std, weighted, whiten_std in itertools.product(
                    ('grpo', 'tailrl'), (False, True), (False, True), (False, True)):
                options = dict(group_adv_std=std, length_weighted_baseline=weighted,
                    adv_estimator=estimator, whiten=True, whiten_std=whiten_std,
                    group_adv_std_floor=.5, mask_positive_incorrect=True, mask_negative_correct=True)
                reference, ref_metrics = finalize_ppo_data([datum(n) for n in lengths], rewards,
                    groups=groups, units=units, response_correctness=labels, **options)
                actual, metrics = finalize_ppo_data([datum(lengths[i]) for i in indices],
                    [rewards[i] for i in indices], groups=[groups[i] for i in indices],
                    units=local_units, response_correctness=[labels[i] for i in indices],
                    distributed_groups=True, **options)
                for local, i in zip(actual, indices):
                    for key in ('advantages', 'returns', 'mask'):
                        torch.testing.assert_close(local.loss_fn_inputs[key].to_torch(),
                            reference[i].loss_fn_inputs[key].to_torch(), rtol=1e-5, atol=1e-6)
                assert metrics['ppo/adv_zero_frac'] == ref_metrics['ppo/adv_zero_frac']
        # Empty rollout ranks and global oversampling/cancellation both finish.
        asyncio.run(check_rollout(rank, world, root, 2, 0))
        asyncio.run(check_rollout(rank, world, root, 2, 2))
        asyncio.run(check_direct_rollout(rank))
        asyncio.run(check_direct_rollout(rank, extra=2))
        asyncio.run(check_direct_rollout(rank, extra=2, insufficient=True))
    finally:
        dist.destroy_process_group()


class TrajectoryShardingTests(unittest.TestCase):
    def test_training_validation_counts_trajectories_instead_of_tasks(self):
        from arl.grpo import GRPOConfig, _validate
        with patch.dict(os.environ, {'KERNEL_RL_PACK_TOKENS': '0'}):
            _validate(GRPOConfig(model_name='test', log_path='/tmp/test',
                                 batch_size=8, group_size=8), 16)
            with self.assertRaisesRegex(SystemExit, '2 trajectories'):
                _validate(GRPOConfig(model_name='test', log_path='/tmp/test',
                                     batch_size=1, group_size=2), 16)

    def test_eight_tasks_eight_siblings_over_sixteen_ranks(self):
        assignments = [trajectory_slots(8, 8, rank, 16, True) for rank in range(16)]
        self.assertEqual([len(row) for row in assignments], [4]*16)
        self.assertEqual(sorted(slot for row in assignments for slot in row),
                         list(itertools.product(range(8), range(8))))

    def test_partial_batch_and_empty_ranks(self):
        for n_tasks, n_per, world in ((1, 2, 16), (3, 9, 16), (0, 8, 16)):
            rows = [trajectory_slots(n_tasks, n_per, r, world, True) for r in range(world)]
            self.assertEqual(sorted(x for row in rows for x in row),
                             list(itertools.product(range(n_tasks), range(n_per))))
            self.assertLessEqual(max(map(len, rows))-min(map(len, rows)), 1)

    def test_real_gloo_split_groups_match_unsplit_reference(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(distributed_worker, args=(4, root), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
