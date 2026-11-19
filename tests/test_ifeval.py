"""IFEval split, official grading and token-level GRPO integration."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from arl.envs.instruction_following.ifeval import IFEvalSourceConfig, IFEvalTaskSource, _import_ifevalenv
from arl.experiments.config import from_dict
from arl.experiments.runner import run_phase
from arl.experiments.artifacts import validate_collection
from arl.agent.rollout import RolloutParams, run_agent_rollout
from arl.algorithms.ppo_data import finalize_ppo_data
from arl.models.types import SampledSequence

ce = _import_ifevalenv()


def row(key=1, prompt='Say hello without commas.'):
    return dict(key=key, prompt=prompt, instruction_id_list=['punctuation:no_comma'], kwargs=[{}])


class DataTests(unittest.IsolatedAsyncioTestCase):
    async def test_disjoint_stable_split_limits_after_split(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'tasks.jsonl'
            records = [row(i, f'Task {i}') for i in range(20)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            source = IFEvalTaskSource(IFEvalSourceConfig(train_set=str(path), test_fraction=.2))
            await source.initialize()
            train, test = source.train_tasks(), source.test_tasks()
            self.assertEqual((len(train), len(test)), (16, 4))
            self.assertFalse({t.key for t in train} & {t.key for t in test})
            reversed_tasks = [ce.Task.from_record(r) for r in reversed(records)]
            _, reversed_test = ce.split_tasks(reversed_tasks, .2)
            self.assertEqual({t.key for t in test}, {t.key for t in reversed_test})
            limited = IFEvalTaskSource(IFEvalSourceConfig(train_set=str(path), test_fraction=.2, limit=2, test_limit=2))
            await limited.initialize()
            self.assertEqual(limited.train_tasks(), train[:2])
            self.assertEqual(limited.test_tasks(), test[:2])
            await source.close(); await limited.close()

    async def test_explicit_overlap_duplicate_and_invalid_records_fail(self):
        with tempfile.TemporaryDirectory() as root:
            train, test = Path(root)/'train.jsonl', Path(root)/'test.jsonl'
            train.write_text(json.dumps(row()))
            # Catch prompt leakage even if task IDs differ.
            test.write_text(json.dumps(row(2)))
            source = IFEvalTaskSource(IFEvalSourceConfig(train_set=str(train), test_set=str(test)))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                await source.initialize()
            train.write_text(json.dumps(row())+'\n'+json.dumps(row()))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                ce.load_tasks(train)
            for invalid in ({'kwargs': []}, {'instruction_id_list': []}, {'prompt': ''}):
                with self.assertRaises(ValueError):
                    ce.Task.from_record({**row(), **invalid})
            await source.close()

    async def test_factory_and_capability_registration(self):
        from arl.envs import get_env_spec, make_task_source_from_config
        from arl.grpo import GRPOConfig
        from arl.harnesses import resolve_pair
        pair = resolve_pair('ifeval', operation='train', model_backend='local')
        self.assertEqual((pair.env, pair.harness), ('instruction_following.ifeval', 'direct'))
        self.assertEqual(get_env_spec('ifeval').category, 'instruction_following')
        cfg = GRPOConfig(model_name='test', log_path='/tmp/test', env='ifeval', ifeval_reward_mode='fraction', ifeval_split_seed=7)
        source = make_task_source_from_config(cfg)
        self.assertEqual(source.cfg.train_set, 'ifeval')
        self.assertEqual(source.cfg.reward_mode, 'fraction')
        self.assertEqual(source.cfg.split_seed, 7)
        await source.close()


class EnvTests(unittest.IsolatedAsyncioTestCase):
    async def test_reward_modes_reasoning_and_private_labels(self):
        task = ce.Task.from_record({**row(), 'instruction_id_list': ['punctuation:no_comma', 'detectable_format:json_format'], 'kwargs': [{}, {}]})
        for mode, expected in (('strict', 0.), ('fraction', .5)):
            grader = AsyncMock()
            grader.score.return_value = {'strict': {'all': False, 'checks': [True, False]}, 'loose': {'all': False, 'checks': [True, False]}}
            env = ce.IFEvalEnv(task, grader, reward_mode=mode)
            await env.reset()
            self.assertEqual(env.messages(), [{'role': 'user', 'content': task.prompt}])
            out = await env.step('reasoning, hidden</think>hello')
            self.assertEqual(out.reward, expected)
            self.assertFalse(out.metrics['correct'])
            grader.score.assert_awaited_once_with(task, 'hello')
            self.assertTrue(out.done)
            with self.assertRaises(RuntimeError): await env.step('again')

    async def test_truncation_is_zero_and_checker_failure_is_infra(self):
        grader = AsyncMock()
        task = ce.Task.from_record(row())
        out = await ce.IFEvalEnv(task, grader).step('good prefix', truncated=True)
        self.assertEqual(out.reward, 0.)
        self.assertFalse(out.infra_error)
        grader.score.assert_not_awaited()
        grader.score.side_effect = RuntimeError('checker unavailable')
        out = await ce.IFEvalEnv(task, grader).step('hello')
        self.assertTrue(out.infra_error)
        self.assertIsNone(out.metrics['correct'])
        out = await ce.IFEvalEnv(task, grader).finish('context overflow')
        self.assertEqual(out.reward, 0.)
        self.assertEqual(ce.final_answer('<think>unfinished'), '')


class Renderer:
    tokenizer = None
    def stop_token_ids(self): return []
    def fit(self, env, *args): return [100, 101], '', 0
    def continue_prompt(self, *args): return None
    def decode(self, ids): return 'Hello world.' if ids[0] == 10 else 'Hello, world.'


class Sampler:
    async def submit_and_poll(self, requests, **kwargs):
        await asyncio.sleep(0)
        return [(r.extras['req_id'], SampledSequence(
            tokens=[10 if i % 2 == 0 else 11], logprobs=[-.5], stop_reason='stop')) for i, r in enumerate(requests)]


@unittest.skipUnless(ce.PYTHON.exists() and (ce.NLTK_DATA/'tokenizers/punkt_tab').exists(), 'run script/setup_ifeval.sh')
class OfficialTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_official_data_preflight_and_real_positive_negative_checks(self):
        source = IFEvalTaskSource(IFEvalSourceConfig())
        try:
            await source.initialize()
            self.assertEqual((len(source.train_tasks()), len(source.test_tasks())), (487, 54))
            health = await source.check()
            self.assertTrue(health['ready'])
            task = ce.Task.from_record(row())
            self.assertTrue((await source.grader.score(task, 'Hello world.'))['strict']['all'])
            self.assertFalse((await source.grader.score(task, 'Hello, world.'))['strict']['all'])
            json_task = ce.Task.from_record({**row(prompt='Return JSON'), 'instruction_id_list': ['detectable_format:json_format']})
            scored = await source.grader.score(json_task, '```json\n{"ok": true}\n```')
            self.assertIn('loose', scored)
        finally:
            await source.close()

    async def test_hash_validation_rejects_changed_official_code(self):
        import shutil
        with tempfile.TemporaryDirectory() as root:
            shutil.copytree(ce.UPSTREAM, root, dirs_exist_ok=True)
            ce.verify_upstream(root)
            (Path(root)/'instruction_following_eval/evaluation_lib.py').write_text('# changed')
            with self.assertRaisesRegex(ValueError, 'missing/modified'):
                ce.verify_upstream(root)

    async def test_actual_token_rollout_reaches_grpo_advantages(self):
        source = IFEvalTaskSource(IFEvalSourceConfig())
        task = ce.Task.from_record(row())
        try:
            result = await run_agent_rollout([task], source, Sampler(), Renderer(),
                RolloutParams(gen_cap=10, prompt_budget=100, group_size=2))
            self.assertEqual(result.returns, [1., 0.])
            self.assertEqual(result.correctness, [True, False])
            data, _ = finalize_ppo_data(result.datums, result.returns, groups=result.groups, units=result.units,
                whiten=False, group_adv_std=False, length_weighted_baseline=False,
                response_correctness=result.correctness)
            advantages = [d.loss_fn_inputs['advantages'].to_torch() for d in data]
            self.assertTrue(bool((advantages[0]>0).any()))
            self.assertTrue(bool((advantages[1]<0).any()))
        finally:
            await source.close()

    async def test_checker_failure_is_dropped_from_training(self):
        source = IFEvalTaskSource(IFEvalSourceConfig())
        source.grader.score = AsyncMock(side_effect=RuntimeError('checker failed'))
        try:
            result = await run_agent_rollout([ce.Task.from_record(row())], source, Sampler(), Renderer(),
                RolloutParams(gen_cap=10, prompt_budget=100, group_size=2))
            self.assertEqual(result.datums, [])
            self.assertTrue(all(t.infra for t in result.trajs))
        finally:
            await source.close()


@unittest.skipUnless(ce.PYTHON.exists() and (ce.NLTK_DATA/'tokenizers/punkt_tab').exists(), 'run script/setup_ifeval.sh')
class ExperimentTests(unittest.TestCase):
    def test_offline_benchmark_uses_official_strict_loose_scoring(self):
        from arl.benchmarks import run_benchmark
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            tasks, predictions = root/'tasks.jsonl', root/'preds.jsonl'
            tasks.write_text(json.dumps(row()))
            predictions.write_text(json.dumps({'key': 1, 'response': 'Hello world.'}))
            result = run_benchmark('ifeval', train_set=str(tasks), predictions=str(predictions), output_dir=str(root/'report'))
            self.assertEqual(result['strict']['prompt_accuracy'], 1.)
            self.assertEqual(result['loose']['instruction_accuracy'], 1.)
            self.assertTrue(Path(result['report']).exists())
            predictions.write_text(json.dumps({'key': 2, 'response': 'Hello world.'}))
            with self.assertRaisesRegex(ValueError, 'missing prediction'):
                run_benchmark('ifeval', train_set=str(tasks), predictions=str(predictions))

    def test_real_official_grader_to_exported_collection(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'train.jsonl').write_text(json.dumps(row()))
            (root/'test.jsonl').write_text(json.dumps(row(2, 'Say hi without commas.')))
            exp = from_dict({'models': {'p': {'name': 'fake-sampler'}},
                'defaults': {'env': 'ifeval', 'model': 'p', 'options': {
                    'train_set': str(root/'train.jsonl'), 'test_set': str(root/'test.jsonl'),
                    'max_tokens': 10, 'prompt_budget_tokens': 100, 'group_size': 2}},
                'collect': {'output_dir': 'out'}}, base_dir=root)
            with patch('arl.harnesses.registry.HarnessSpec.make_renderer', return_value=Renderer()):
                summary = run_phase(exp.plan('collect'), native_runtime={'sampling_client': Sampler(), 'tokenizer': object()})
            self.assertEqual(summary['pass_rate'], .5)
            self.assertEqual(validate_collection(root/'out', env='ifeval', harness='direct', require_training_trace=True)['datums_validated'], 2)


if __name__ == '__main__':
    unittest.main()
