"""MultiCodeIF task alignment, unavailable scoring, repair and official-rule smoke."""
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arl.benchmarks import get_benchmark
from arl.benchmarks import multicodeif as mc


def row(i=1, **kwargs):
    return dict(id=i, category='data_structure', constraint='type',
                prompt='Build a Python list', answer='python;list', **kwargs)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for suite in mc.SUITES:
            (self.root / 'dataset/instructions' / suite).mkdir(parents=True)

    def write(self, suite, name, rows):
        path = self.root / 'dataset/instructions' / suite / (name + '.json')
        path.write_text(json.dumps(rows))
        return path

    def test_export_has_no_answers_or_references(self):
        self.write('single_level', 'sample', [row()])
        task = mc.load_tasks(self.root)[0]
        self.assertEqual(task.task_id, 'single_level/sample:1')
        self.assertEqual(set(task.public()), {'task_id', 'prompt'})
        self.assertNotIn('python;list', json.dumps(task.public()))

    def test_multilevel_selection_retains_all_earlier_constraints(self):
        self.write('multi_level', 'sample', [row(1, level=0), row(2, level=1), row(3, level=2)])
        tasks = mc.load_tasks(self.root, suite='multi_level', limit=2)
        self.assertEqual([len(t.checks) for t in tasks], [1, 2])
        self.assertEqual(tasks[1].checks[0]['id'], 1)
        self.write('multi_level', 'sample', [row(1, level=0), row(2, level=2)])
        with self.assertRaisesRegex(ValueError, 'invalid multilevel chain|broken multilevel chain'):
            mc.load_tasks(self.root, suite='multi_level')

    def test_duplicate_tasks_and_unknown_filters_fail(self):
        self.write('single_level', 'sample', [row(), row()])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            mc.load_tasks(self.root)
        with self.assertRaisesRegex(ValueError, 'unknown constraint'):
            mc.load_tasks(self.root, constraints=['missing'])

    def test_native_predictions_are_aligned_by_task_id_not_position(self):
        self.write('single_level', 'sample', [row(1), row(2)])
        tasks = mc.load_tasks(self.root)
        p = self.root / 'preds.jsonl'
        p.write_text('\n'.join(json.dumps({'task_id': t.task_id, 'responses': ['bad', 'fixed']}) for t in reversed(tasks)))
        self.assertEqual(mc.load_predictions(p, tasks)[tasks[0].task_id], ['bad', 'fixed'])
        p.write_text(json.dumps({'task_id': tasks[0].task_id, 'response': 'ok'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'missing 1 predictions'):
            mc.load_predictions(p, tasks)

    def test_official_generations_and_duplicate_ids(self):
        self.write('single_level', 'sample', [row()])
        tasks = mc.load_tasks(self.root)
        d = self.root / 'preds'; d.mkdir()
        p = d / 'sample_model_timestamp.json'
        p.write_text(json.dumps([{'id': 1, 'response': [['prompt', 'code']]}]))
        self.assertEqual(mc.load_predictions(d, tasks), {tasks[0].task_id: ['code']})
        p.write_text(json.dumps([{'id': 1, 'response': [['p', 'c']]}] * 2))
        with self.assertRaisesRegex(ValueError, 'duplicate prediction'):
            mc.load_predictions(d, tasks)

    def test_unavailable_check_never_becomes_a_model_failure(self):
        results = [dict(status='incomplete', passed=None, checks=[
            dict(status='scored', passed=True), dict(status='unavailable', passed=None)])]
        summary = mc.summarize(results)
        self.assertEqual(summary['coverage'], .5)
        self.assertEqual(summary['constraint_accuracy_scored'], 1.)
        self.assertEqual(summary['tasks_scored'], 0)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['functional_correctness'], 'not_evaluated')


class AdapterTests(unittest.TestCase):
    def test_api_uses_custom_endpoint_and_separate_judge_and_rejects_truncation(self):
        config = dict(base_url='http://local/v1', model='target', judge_base_url='http://judge/v1',
                      judge_model='external-judge', chat_template_kwargs={'enable_thinking': False})
        content = {'choices': [{'message': {'content': 'YES'}, 'finish_reason': 'stop'}]}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(content).encode())) as call:
            self.assertEqual(mc.chat(config, [{'role': 'user', 'content': 'p'}], judge=True), 'YES')
            self.assertEqual(call.call_args.args[0].full_url, 'http://judge/v1/chat/completions')
            self.assertEqual(json.loads(call.call_args.args[0].data)['model'], 'external-judge')
        content['choices'][0]['finish_reason'] = 'length'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(content).encode())):
            with self.assertRaisesRegex(RuntimeError, 'truncated'):
                mc.chat(config, [{'role': 'user', 'content': 'p'}])
        content['choices'][0].update(finish_reason='stop', message={'content': 'Maybe yes and no'})
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(content).encode())):
            with self.assertRaisesRegex(RuntimeError, 'exactly YES or NO'):
                mc.chat(config, [], judge=True)

    def test_generic_cli_passes_config_and_prints_result(self):
        from arl.benchmarks.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'; path.write_text('{"action":"inspect"}')
            out = io.StringIO()
            with patch('arl.benchmarks.__main__.run_benchmark', return_value={'tasks': 2}) as run, redirect_stdout(out):
                self.assertEqual(main(['multicodeif', '--config', str(path)]), 0)
            run.assert_called_once_with('multicodeif', mode='benchmark', action='inspect')
            self.assertEqual(json.loads(out.getvalue()), {'tasks': 2})


@unittest.skipUnless((mc.CACHE / 'MultiCodeIF/.git').exists() and
                     (mc.CACHE / 'multicodeif-deps/tree_sitter').exists(), 'run script/setup_multicodeif.sh for official integration tests')
class OfficialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_official_data_inventory_and_constraint_only_inspection(self):
        result = get_benchmark('multicodeif').run()
        self.assertEqual(result['datasets']['single_level']['tasks'], 1526)
        self.assertEqual(result['datasets']['multi_level']['tasks'], 660)
        self.assertEqual(result['training'], 'local_on_policy_grpo')

    def test_offline_repair_changes_failed_check_to_passed_check(self):
        tasks = mc.load_tasks(mc.CACHE/'MultiCodeIF', constraints=['code_convention_naming_convention'], limit=1)
        p = self.root/'preds.jsonl'
        p.write_text(json.dumps({'task_id': tasks[0].task_id, 'responses': [
            'class database_connector:\n    def ExecuteQuery(self): pass',
            'class DatabaseConnector:\n    def execute_query(self): pass']}) + '\n')
        result = mc.benchmark(predictions=str(p), constraints=['code_convention_naming_convention'], limit=1,
                              max_rounds=2, output_dir=str(self.root/'run'))
        self.assertEqual([r['checks_passed'] for r in result['rounds']], [0, 1])
        self.assertTrue(result['final']['complete'])
        report = json.loads(Path(result['report']).read_text())
        self.assertIn('previous code', report['results'][0]['repair_template'])

    def test_generation_keeps_answers_private_until_failed_feedback(self):
        replies = ['class database_connector:\n    def ExecuteQuery(self): pass',
                   'class DatabaseConnector:\n    def execute_query(self): pass']
        with patch.object(mc, 'chat', side_effect=replies) as model:
            result = mc.benchmark(base_url='http://mock/v1', model='target', concurrency=1,
                constraints=['code_convention_naming_convention'], limit=1, max_rounds=2,
                output_dir=str(self.root/'generated'))
        initial = model.call_args_list[0].args[1][0]['content']
        repair = model.call_args_list[1].args[1][0]['content']
        self.assertNotIn('python;class:PascalCase', initial)
        self.assertIn('python;class:PascalCase', repair)
        self.assertIn('previous code', repair)
        self.assertEqual([r['checks_passed'] for r in result['rounds']], [0, 1])

    def test_api_failure_is_not_counted_as_model_failure(self):
        with patch.object(mc, 'chat', side_effect=RuntimeError('API unavailable')):
            result = mc.benchmark(base_url='http://mock/v1', model='target',
                constraints=['code_convention_naming_convention'], limit=1,
                output_dir=str(self.root/'failed_generation'))
        self.assertEqual(result['final']['checks_scored'], 0)
        self.assertIsNone(result['final']['constraint_accuracy_scored'])
        self.assertEqual(result['final']['unavailable_or_error'], {'generation_error': 1})

    def test_missing_judge_is_reported_unavailable_not_false(self):
        task = mc.load_tasks(mc.CACHE/'MultiCodeIF', constraints=['algorithm_type'], limit=1)[0]
        p = self.root/'preds.jsonl'
        p.write_text(json.dumps({'task_id': task.task_id, 'response': 'def solve(): pass'}) + '\n')
        result = mc.benchmark(predictions=str(p), constraints=['algorithm_type'], limit=1,
                              output_dir=str(self.root/'run'))
        self.assertEqual(result['final']['checks_scored'], 0)
        self.assertEqual(result['final']['unavailable_or_error'], {'unavailable': 1})
        self.assertIsNone(result['final']['constraint_accuracy_scored'])

    def test_multilevel_uses_all_prefix_checks(self):
        tasks = mc.load_tasks(mc.CACHE/'MultiCodeIF', suite='multi_level', limit=4)
        p = self.root/'preds.jsonl'
        code = ('use std::collections::BinaryHeap;\nfn main() { let mut scores: BinaryHeap<i32> = BinaryHeap::new(); '
                'for i in 0..500 { scores.push(i); } scores.peek(); scores.pop(); }')
        p.write_text('\n'.join(json.dumps({'task_id': task.task_id, 'response': code}) for task in tasks))
        result = mc.benchmark(suite='multi_level', limit=4, predictions=str(p), output_dir=str(self.root/'run'))
        self.assertEqual(result['final']['checks_requested'], 10)
        self.assertEqual(result['final']['checks_scored'], 10)
        # Preserve upstream's current Rust-size semantics (case mismatch), rather
        # than silently correcting the benchmark to make this fixture pass.
        self.assertEqual(result['final']['checks_passed'], 6)


if __name__ == '__main__':
    unittest.main()
