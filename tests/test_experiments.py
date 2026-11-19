"""Phase routing and real collection paths, with CPU sampling and fake workspaces."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from arl.experiments.config import from_dict, ModelConfig
from arl.experiments.runner import run_phase, validate_plan
from arl.experiments.artifacts import validate_collection, validate_datum
from arl.harnesses import resolve_pair, HarnessCompatibilityError, HarnessPendingError
from arl.benchmarks import BenchmarkPendingError
from arl.benchmarks import multicodeif as mc
from test_swe_env import Source, Sampler, Renderer, ce, record


def config(root):
    return from_dict({'models': {'policy': {'name': 'test-policy'}, 'judge': {
        'name': 'test-api', 'backend': 'openai_compatible', 'base_url': 'http://localhost:8200/v1'}},
        'defaults': {'env': 'cuda', 'model': 'policy', 'options': {'max_tokens': 10,
            'prompt_budget_tokens': 100, 'batch_size': 1, 'group_size': 2}},
        'collect': {'output_dir': 'collect'}, 'train': {'output_dir': 'train'},
        'evaluate': {'output_dir': 'evaluate'}}, base_dir=root)


class PhaseTests(unittest.TestCase):
    def test_aliases_defaults_and_independent_phase_selection(self):
        raw = {'models': {'local': {'name': 'weights'}, 'api': {'name': 'api-model',
            'backend': 'openai_compatible', 'base_url': 'http://localhost/v1'}},
            'defaults': {'env': 'cuda', 'model': 'local'}, 'collect': {}, 'train': {},
            'evaluate': {'env': 'multicodeif', 'model': 'api'}}
        exp = from_dict(raw)
        self.assertEqual((exp.plan('collect').env, exp.plan('collect').harness), ('coding.cuda', 'arl_agent'))
        self.assertEqual(exp.plan('train').split, 'train')
        evaluation = exp.plan('evaluate')
        self.assertEqual((evaluation.harness, evaluation.model.name, evaluation.split),
                         ('multicodeif_self_repair', 'api-model', 'test'))

    def test_fail_before_backend_on_pending_incompatible_and_trace(self):
        with self.assertRaises(HarnessPendingError):
            resolve_pair('cuda', 'codex', operation='collect')
        with self.assertRaises(HarnessCompatibilityError):
            resolve_pair('cuda', 'multicodeif_self_repair', operation='evaluate')
        with self.assertRaises(HarnessCompatibilityError):
            resolve_pair('cuda', operation='evaluate', model_backend='openai_compatible')
        self.assertEqual(resolve_pair('multicodeif', operation='train', model_backend='local').harness, 'multicodeif_self_repair')
        with self.assertRaises(HarnessCompatibilityError):
            resolve_pair('multicodeif', operation='train', model_backend='openai_compatible')
        from dataclasses import replace
        from arl.harnesses import get_harness
        spec = replace(get_harness('arl_agent'), trace_fields=())
        with patch('arl.harnesses.compatibility.get_harness', return_value=spec):
            with self.assertRaisesRegex(HarnessCompatibilityError, 'missing'):
                resolve_pair('cuda', operation='train')

    def test_no_silent_model_or_data_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            exp = config(root)
            from dataclasses import replace
            with self.assertRaisesRegex(ValueError, 'unused'):
                validate_plan(replace(exp.plan('collect'), options={'learning_rate': .1}))
            for key in ('harness', 'dataset', 'task_split'):
                raw = {'models': {'p': {'name': 'x'}}, 'collect': {'env': 'cuda', 'model': 'p', 'options': {key: 'x'}}}
                with self.assertRaises(ValueError):
                    from_dict(raw).plan('collect')
        with self.assertRaises(ValueError):
            ModelConfig('x', backend='openai_compatible', base_url='http://key:secret@localhost/v1')

    def test_datum_validation_rejects_missing_misaligned_nonfinite_trace(self):
        valid = dict(token_ids=[1, 2], target_tokens=[2, 3], logprobs=[-.1, -.2], loss_mask=[0, 1])
        validate_datum(valid)
        for invalid in ({'logprobs': [0]}, {'logprobs': [0, float('nan')]},
                        {'loss_mask': [0, .5]}, {'logprobs': ['bad', 0]}, {'token_ids': [True, 1]}):
            with self.assertRaises(ValueError):
                validate_datum({**valid, **invalid})


class CollectionSource(Source):
    closed = False
    def train_tasks(self): return [ce.SweTask.from_record(record(1))]
    def test_tasks(self): return [ce.SweTask.from_record(record(2))]
    async def close(self): self.closed = True


class NativeExecutionTests(unittest.TestCase):
    def test_collection_and_evaluation_use_selected_harness_and_split(self):
        with tempfile.TemporaryDirectory() as root:
            exp = config(root)
            for phase in ('collect', 'evaluate'):
                source = CollectionSource()
                with patch('arl.harnesses.registry.HarnessSpec.make_renderer', return_value=Renderer()):
                    summary = run_phase(exp.plan(phase), native_runtime={
                        'source': source, 'sampling_client': Sampler(), 'tokenizer': object()})
                self.assertEqual(summary['pass_rate'], .5)
                self.assertTrue(source.closed)
                self.assertTrue(all(env.workspace.closed for env in source.envs))
                output = Path(root)/phase
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'complete')
                self.assertEqual(manifest['identity']['harness'], 'arl_agent')
                if phase == 'collect':
                    checked = validate_collection(output, env='cuda', harness='native', require_training_trace=True)
                    self.assertEqual(checked['datums_validated'], 4)
                    rows = [json.loads(line) for line in (output/'datums.jsonl').read_text().splitlines()]
                    self.assertEqual({row['task'] for row in rows}, {'example__repo-1'})
                    self.assertEqual([row['reward'] for row in rows], [1., 1., 0., 0.])
                    with self.assertRaisesRegex(ValueError, 'model'):
                        validate_collection(output, env='cuda', harness='native', model={'name': 'other'})
                else:
                    self.assertFalse((output/'datums.jsonl').exists())

    def test_failed_collection_marks_manifest_and_closes_workspaces(self):
        with tempfile.TemporaryDirectory() as root:
            source, sampler = CollectionSource(), Sampler()
            sampler.submit_and_poll = AsyncMock(side_effect=RuntimeError('sampler failed'))
            with patch('arl.harnesses.registry.HarnessSpec.make_renderer', return_value=Renderer()):
                with self.assertRaisesRegex(RuntimeError, 'sampler failed'):
                    run_phase(config(root).plan('collect'), native_runtime={
                        'source': source, 'sampling_client': sampler, 'tokenizer': object()})
            self.assertTrue(source.closed)
            self.assertTrue(all(env.workspace.closed for env in source.envs))
            output = Path(root)/'collect'
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'failed')
            with self.assertRaisesRegex(ValueError, 'not complete'):
                validate_collection(output, env='cuda', harness='native', require_training_trace=True)


class TextExecutionTests(unittest.TestCase):
    def test_offline_collection_preserves_text_trace_and_rejects_grpo_import(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'preds.jsonl').write_text('{}\n')
            exp = from_dict({'models': {'p': {'name': 'archived-model', 'backend': 'offline'}},
                'collect': {'env': 'multicodeif', 'model': 'p', 'split': 'all',
                    'options': {'predictions': 'preds.jsonl'}, 'output_dir': 'output'}}, base_dir=root)
            def benchmark(name, **kwargs):
                self.assertEqual(kwargs['harness'], 'multicodeif_self_repair')
                self.assertEqual(kwargs['task_split'], 'all')
                out = Path(kwargs['output_dir']); out.mkdir()
                (out/'trajectories.jsonl').write_text(json.dumps({'response': 'code',
                    'messages': [{'role': 'user', 'content': 'task'}], 'generation_source': 'offline_predictions'})+'\n')
                return {'report': str(out/'report.json')}
            with patch('arl.benchmarks.run_benchmark', side_effect=benchmark):
                run_phase(exp.plan('collect'))
            rows = [json.loads(line) for line in (root/'output/samples.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['identity']['model']['name'], 'archived-model')
            self.assertEqual(rows[0]['messages'][0]['content'], 'task')
            with self.assertRaisesRegex(ValueError, 'no token'):
                validate_collection(root/'output', env='multicodeif', harness='multicodeif_self_repair', require_training_trace=True)

    def test_multilevel_split_keeps_constraint_chain_together(self):
        with tempfile.TemporaryDirectory() as root:
            data = Path(root)/'dataset/instructions/multi_level'; data.mkdir(parents=True)
            rows = [dict(id=i*4+level, level=level, prompt='task', answer='python;list',
                category='data_structure', constraint='type') for i in range(5) for level in range(4)]
            (data/'sample.json').write_text(json.dumps(rows))
            train = mc.load_tasks(root, suite='multi_level', task_split='train', test_fraction=.4)
            test = mc.load_tasks(root, suite='multi_level', task_split='test', test_fraction=.4)
            group = lambda tasks: {task.checks[0]['id'] for task in tasks}
            self.assertFalse(group(train) & group(test))
            self.assertEqual((len(train), len(test)), (12, 8))
            self.assertEqual([t.task_id for t in test], [t.task_id for t in mc.load_tasks(
                root, suite='multi_level', task_split='test', test_fraction=.4)])


if __name__ == '__main__':
    unittest.main()
