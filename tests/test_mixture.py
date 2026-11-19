"""Weighted schedules, phase configuration and mixed CUDA/IFEval GRPO driver."""
import asyncio
from collections import Counter
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from arl.envs.mixture import normalize_members, expand_config, batch_schedule, SourcePool
from arl.experiments.config import from_dict
from arl.experiments.runner import prepare_native_config
from arl.grpo import GRPOConfig, _validate, _batches, run_grpo
from arl.harnesses import HarnessPendingError
from arl.benchmarks import BenchmarkPendingError
from arl.models.types import SampledSequence
from test_swe_env import Source, Renderer as SweRenderer, ce as swe, record
from test_ifeval import ce as ifeval, Renderer as IFRenderer, row


def rows():
    return [{'id': 'cuda', 'env': 'cuda', 'weight': 7, 'options': {'enable_thinking': True, 'max_tokens': 10, 'prompt_budget_tokens': 100}},
            {'id': 'if', 'env': 'ifeval', 'weight': 3, 'options': {'enable_thinking': False, 'max_tokens': 20, 'prompt_budget_tokens': 200}}]


class ConfigurationTests(unittest.TestCase):
    def test_stage_independent_members_canonicalize_paths_and_shared_optimizer(self):
        with tempfile.TemporaryDirectory() as root:
            data = rows(); data[1]['options']['train_set'] = 'if.jsonl'
            exp = from_dict({'models': {'p': {'name': 'weights'}},
                'train': {'model': 'p', 'environments': data, 'options': {'learning_rate': .00002, 'mixture_steps_per_epoch': 10}},
                'evaluate': {'model': 'p', 'env': 'ifeval'}}, base_dir=root)
            plan = exp.plan('train')
            self.assertEqual(plan.env, 'mixture')
            self.assertEqual([m['weight'] for m in plan.environments], [.7, .3])
            self.assertEqual(plan.environments[1]['options']['train_set'], str(Path(root)/'if.jsonl'))
            cfg = prepare_native_config(plan)
            members = expand_config(cfg)
            self.assertEqual([m.cfg.env for m in members], ['coding.cuda', 'instruction_following.ifeval'])
            self.assertEqual([m.cfg.learning_rate for m in members], [.00002, .00002])
            self.assertEqual([m.cfg.enable_thinking for m in members], [True, False])
            self.assertEqual(exp.plan('evaluate').env, 'instruction_following.ifeval')

    def test_pending_and_bad_member_configuration_fails_before_runtime(self):
        for bad in ([], [{'env': 'cuda', 'weight': 0}], [{'env': 'cuda', 'weight': float('nan')}],
                    [{'env': 'cuda', 'weight': True}], [{'env': 'cuda', 'options': {'learning_rate': 1}}],
                    [{'id': 'same', 'env': 'cuda'}, {'id': 'same', 'env': 'ifeval'}],
                    [{'id': '../escape', 'env': 'cuda'}]):
            with self.assertRaises(ValueError): normalize_members(bad)
        with self.assertRaises(HarnessPendingError): normalize_members([{'env': 'cuda', 'harness': 'codex'}])
        self.assertEqual(normalize_members([{'env': 'multicodeif'}])[0]['harness'], 'multicodeif_self_repair')
        with self.assertRaises(ValueError): normalize_members([{'env': 'ifeval'}], model_backend='openai_compatible')
        with self.assertRaises(ValueError):
            _validate(GRPOConfig(model_name='p', log_path='/tmp/test', environments=[{'env': 'ifeval', 'options': {'ifeval_reward_mode': 'bad'}}]), 1)
        raw = {'models': {'p': {'name': 'p'}}, 'train': {'env': 'cuda', 'model': 'p', 'environments': rows()}}
        with self.assertRaisesRegex(ValueError, 'not both'): from_dict(raw).plan('train')

    def test_weighted_schedule_is_deterministic_sharded_and_resumable(self):
        members = expand_config(GRPOConfig(model_name='p', log_path='/tmp/test', environments=rows()))
        args = dict(batch_size=8, epochs=2, seed=7, steps_per_epoch=500)
        plan = batch_schedule(members, [3, 19], **args)
        self.assertEqual(plan, batch_schedule(members, [3, 19], **args))
        counts = Counter(index for index, _ in plan)
        self.assertLess(abs(counts[0]/len(plan)-.7), .05)
        self.assertEqual(len(plan), 1000)
        from arl.utils.distributed import shard_list
        for index, indices in plan:
            self.assertEqual(len(indices), 8)
            self.assertTrue(all(0 <= i < [3, 19][index] for i in indices))
            self.assertEqual([value for pair in zip(shard_list(indices, 0, 2), shard_list(indices, 1, 2)) for value in pair], indices)
        self.assertEqual(plan[23:], batch_schedule(members, [3, 19], **args)[23:])
        default = batch_schedule(members, [3, 19], batch_size=8, epochs=1)
        self.assertEqual(len(default), 3)
        with self.assertRaises(ValueError): batch_schedule(members, [0, 19], **args)

    def test_single_environment_batch_schedule_is_unchanged(self):
        cfg = GRPOConfig(model_name='p', log_path='/tmp/test', batch_size=3, num_epochs=2)
        self.assertEqual(len(expand_config(cfg)), 1)
        self.assertEqual([len(batch) for batch in _batches(7, cfg)], [3, 3, 3, 3, 2])


class ToolSource(Source):
    closed = False
    def train_tasks(self): return [swe.SweTask.from_record(record(1))]
    def test_tasks(self): return [swe.SweTask.from_record(record(2))]
    async def close(self): self.closed = True


class IFSource:
    closed = False
    def train_tasks(self): return [ifeval.Task.from_record(row())]
    def test_tasks(self): return [ifeval.Task.from_record(row(2, 'No commas please.'))]
    def task_key(self, task): return task.key
    def make_env(self, task, uid): return ifeval.IFEvalEnv(task, self)
    async def score(self, task, response):
        passed = ',' not in response
        return {mode: {'all': passed, 'checks': [passed]} for mode in ('strict', 'loose')}
    async def close(self): self.closed = True


class Sampler:
    async def submit_and_poll(self, requests, **kwargs):
        await asyncio.sleep(0)
        result = []
        for i, r in enumerate(requests):
            if r.prompt_ids[-1] == 101:
                token = 10 if i % 2 == 0 else 11
            else:
                token = 10 if r.extras['req_id'].endswith(':0') else 11
            result.append((r.extras['req_id'], SampledSequence(tokens=[token], logprobs=[-.5], stop_reason='stop')))
        return result


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_pool_closes_previous_and_partially_initialized_source(self):
        members = expand_config(GRPOConfig(model_name='p', log_path='/tmp/test', environments=rows()))
        first, second = MagicMock(), MagicMock()
        first.initialize, first.close = AsyncMock(), AsyncMock()
        second.initialize, second.close = AsyncMock(side_effect=RuntimeError('initialization failed')), AsyncMock()
        pool = SourcePool(members)
        with patch('arl.envs.make_task_source_from_config', side_effect=[first, second]):
            with self.assertRaisesRegex(RuntimeError, 'initialization failed'): await pool.initialize()
        first.close.assert_awaited_once(); second.close.assert_awaited_once()
        await pool.close()
        first.close.assert_awaited_once()

    async def drive(self, root, start_batch=0, fail=False, use_mini=False):
        sources = {'coding.cuda': ToolSource(), 'instruction_following.ifeval': IFSource()}
        members = rows()
        if use_mini:
            from test_mini_swe import MiniSource, MiniRenderer
            mini = MiniSource()
            mini.closed = False
            mini.train_tasks = lambda: [swe.SweTask.from_record(record(1))]
            mini.test_tasks = lambda: [swe.SweTask.from_record(record(2))]
            async def close(): mini.closed = True
            mini.close = close
            sources = {'coding.swe': mini, 'instruction_following.ifeval': IFSource()}
            members[0].update(id='swe', env='swe', harness='mini_swe_agent')
        client = MagicMock()
        service = MagicMock()
        service.create_training_client_async = AsyncMock(return_value=client)
        sampler, logger = Sampler(), MagicMock()
        client.get_tokenizer.return_value = object()
        cfg = GRPOConfig(model_name='fake-policy', log_path=str(root), environments=members,
            batch_size=1, group_size=2, mixture_steps_per_epoch=6, start_batch=start_batch,
            eval_every=2, eval_at_start=True, eval_num_samples=2, save_every=0,
            whiten_advantages=False, max_tokens=10, prompt_budget_tokens=100)
        async def update(data, selected_client, *args):
            self.assertIs(selected_client, client)
            self.assertTrue(any(bool((d.loss_fn_inputs['advantages'].to_torch()>0).any()) for d in data))
            self.assertTrue(any(bool((d.loss_fn_inputs['advantages'].to_torch()<0).any()) for d in data))
            if fail: raise RuntimeError('optimizer unavailable')
            return None, {'ppo/substeps': 1., 'ppo/n_active_tokens': 2.}
        renderer = lambda tok, **kw: (MiniRenderer() if use_mini else SweRenderer()) if kw['enable_thinking'] else IFRenderer()
        with patch('arl.envs.make_task_source_from_config', side_effect=lambda c: sources[c.env]), \
             patch('arl.grpo.setup_environment'), patch('arl.grpo.ml_log.setup_logging', return_value=logger), \
             patch('arl.grpo.sampling_window', side_effect=lambda model, peak, window: window), \
             patch('arl.grpo.tinker.ServiceClient', return_value=service) as factory, \
             patch('arl.grpo.get_updated_sampling_client', new=AsyncMock(return_value=sampler)) as sync, \
             patch('arl.harnesses.registry.HarnessSpec.make_renderer', side_effect=renderer), \
             patch('arl.grpo.train_step', new=AsyncMock(side_effect=update)) as train, \
             patch('arl.grpo.checkpoint_utils.save_checkpoint_async', new=AsyncMock()), \
             patch('arl.grpo.dist_utils.destroy'), patch('arl.grpo.dist_utils.barrier'):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'optimizer unavailable'): await run_grpo(cfg)
            else:
                await run_grpo(cfg)
                self.assertEqual(train.await_count, 6-start_batch)
                self.assertEqual(sync.await_count, 1+6-start_batch)
            factory.assert_called_once()
            self.assertEqual(factory.call_args.kwargs['sampling_max_model_len'], 220)
        self.assertTrue(all(source.closed for source in sources.values()))
        client.shutdown.assert_called_once()
        return logger

    async def test_actual_mixed_driver_shares_client_and_separates_eval_and_trace_identity(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            logger = await self.drive(root)
            with gzip.open(root/'trajectories/rank0.jsonl.gz', 'rt') as f:
                trajectories = [json.loads(line) for line in f]
            train = [row for row in trajectories if row['tag'] != 'eval']
            self.assertEqual({row['identity']['environment_id'] for row in train}, {'cuda', 'if'})
            metric_calls = [call.args[0] for call in logger.log_metrics.call_args_list]
            self.assertTrue(any('eval/cuda/pass_rate' in metrics and 'eval/if/pass_rate' in metrics for metrics in metric_calls))
            self.assertTrue(any('env/cuda/rollout/reward_mean' in metrics for metrics in metric_calls))
            self.assertTrue(any('env/if/rollout/reward_mean' in metrics for metrics in metric_calls))
            self.assertEqual(json.loads((root/'run_identity.json').read_text())['env'], 'mixture')

    async def test_resume_skips_schedule_prefix_and_optimizer_error_closes_all_sources(self):
        with tempfile.TemporaryDirectory() as root:
            await self.drive(Path(root), start_batch=3)
        with tempfile.TemporaryDirectory() as root:
            await self.drive(Path(root), fail=True)

    async def test_mini_swe_and_direct_share_optimizer_and_sync_after_each_update(self):
        with tempfile.TemporaryDirectory() as root:
            await self.drive(Path(root), use_mini=True)
            with gzip.open(Path(root)/'trajectories/rank0.jsonl.gz', 'rt') as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual({r['identity']['harness'] for r in rows}, {'mini_swe_agent', 'direct'})


if __name__ == '__main__': unittest.main()
