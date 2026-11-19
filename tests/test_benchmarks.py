"""Benchmark inventory completeness and safe default/pending dispatch."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arl.benchmarks import (BenchmarkPendingError, get_benchmark, list_benchmarks,
                            run_benchmark)
from arl.benchmarks.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'coding': {
        'swe_bench_verified', 'swe_bench_pro', 'swe_bench_pro_v2', 'swe_bench_multilingual',
        'deepswe_v1_1', 'frontierswe', 'frontiercode_v1_1', 'cursorbench_4_0',
        'terminal_bench_2_1', 'terminal_bench_4_0', 'nl2repo_bench', 'programbench',
        'livecodebench_v6', 'codeforces_rating', 'kernelbench',
    },
    'instruction_following': {
        'ifeval', 'ifbench', 'followbench', 'complexbench', 'multi_if', 'iheval',
        'ifstruct', 'agentif', 'ifevalcode', 'codeif', 'multicodeif',
    },
}


class BenchmarkTests(unittest.TestCase):
    def test_inventory_covers_requested_names_and_versions(self):
        self.assertEqual(len(list_benchmarks()), 26)
        for category, names in EXPECTED.items():
            self.assertEqual({spec.name.split('.', 1)[1] for spec in list_benchmarks(category)}, names)
        self.assertEqual(len(list_benchmarks('coding', 'repository_development')), 8)
        self.assertEqual(len(list_benchmarks('instruction_following', 'code')), 3)

    def test_entries_have_distinct_categorized_directories_and_pending_status(self):
        paths = set()
        for spec in list_benchmarks():
            with self.subTest(name=spec.name):
                self.assertTrue(spec.path.is_relative_to(ROOT / 'envs' / spec.category / 'benchmarks' / spec.group))
                self.assertTrue((spec.path / 'entry.py').is_file())
                self.assertTrue((spec.path / 'README.md').is_file())
                self.assertNotIn(spec.path, paths)
                paths.add(spec.path)
                expected = 'partial' if spec.name in ('instruction_following.multicodeif', 'instruction_following.ifeval', 'coding.programbench') else 'pending'
                self.assertEqual((spec.benchmark_status, spec.training_status), (expected, 'ready' if spec.name in ('instruction_following.ifeval', 'instruction_following.multicodeif') else 'pending'))

    def test_default_mode_reports_benchmark_pending_for_every_entry(self):
        for spec in list_benchmarks():
            if spec.benchmark_status != 'pending':
                continue
            with self.subTest(name=spec.name), self.assertRaises(BenchmarkPendingError) as error:
                run_benchmark(spec.name)
            self.assertEqual(error.exception.mode, 'benchmark')
            self.assertIn('待完成', str(error.exception))

    def test_unimplemented_training_remains_pending(self):
        for spec in list_benchmarks():
            if spec.training_status == 'ready':
                continue
            with self.subTest(name=spec.name), self.assertRaises(BenchmarkPendingError) as error:
                run_benchmark(spec.name, mode='train')
            self.assertEqual(error.exception.mode, 'train')
            self.assertIn('训练接入', str(error.exception))

    def test_real_entry_dispatch_defaults_to_benchmark_not_train(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'entry.py').write_text(
                'def benchmark(**kwargs): return ("benchmark", kwargs)\n'
                'def train(**kwargs): return ("train", kwargs)\n')
            spec = replace(get_benchmark('ifeval'), directory=directory)
            with patch('arl.benchmarks.registry.get_benchmark', return_value=spec):
                self.assertEqual(run_benchmark('ifeval', sample=3), ('benchmark', {'sample': 3}))
                self.assertEqual(run_benchmark('ifeval', mode='train'), ('train', {}))
            with self.assertRaisesRegex(ValueError, 'unknown benchmark mode'):
                spec.run(mode='invalid')

    def test_short_names_resolve_and_unknown_names_fail(self):
        self.assertIs(get_benchmark('ifeval'), get_benchmark('instruction_following.ifeval'))
        with self.assertRaisesRegex(ValueError, 'unknown benchmark'):
            get_benchmark('unknown')
        with self.assertRaisesRegex(ValueError, 'unknown benchmark category'):
            list_benchmarks('unknown')

    def test_cli_defaults_and_train_messages_without_tracebacks(self):
        for argv, expected in ((['ifbench'], 'benchmark 评测接入'),
                               (['ifbench', '--mode', 'train'], '训练接入')):
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(argv)
            self.assertEqual(code, 2)
            self.assertIn(expected, out.getvalue())
            self.assertNotIn('Traceback', out.getvalue())

    def test_json_listing_filters_and_status(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(['--list', '--category', 'coding', '--group', 'terminal_tasks', '--json'])
        self.assertEqual(code, 0)
        rows = json.loads(out.getvalue())
        self.assertEqual({row['display_name'] for row in rows}, {'Terminal-Bench 2.1', 'Terminal-Bench 4.0'})
        self.assertTrue(all(row['training_status'] == 'pending' for row in rows))

    def test_benchmark_names_cannot_accidentally_construct_training_backends(self):
        from arl.envs import make_task_source, make_task_source_from_config
        from types import SimpleNamespace
        from arl.envs import find_env_spec
        for spec in list_benchmarks():
            if find_env_spec(spec.name) is not None:
                continue
            for factory, arg in ((make_task_source, spec.name),
                                 (make_task_source_from_config, SimpleNamespace(env=spec.name))):
                with self.subTest(name=spec.name), self.assertRaises(BenchmarkPendingError) as error:
                    factory(arg)
                self.assertEqual(error.exception.mode, 'train')

    def test_grpo_validation_reports_pending_before_model_initialization(self):
        from arl.grpo import _validate
        from types import SimpleNamespace
        # No model/optimizer fields: pending training must fail before needing any.
        with self.assertRaisesRegex(BenchmarkPendingError, '训练接入'):
            _validate(SimpleNamespace(env='instruction_following.ifbench'), 1)

    def test_cli_does_not_import_the_training_or_docker_stack(self):
        code = '''
import sys
from arl.benchmarks.__main__ import main
try:
    main(['ifeval', '--mode', 'train'])
except SystemExit as exc:
    assert exc.code == 2
else:
    raise AssertionError('missing IFEval training config should fail')
for name in ('torch', 'deepspeed', 'vllm', 'docker', 'swebench'):
    assert name not in sys.modules, name
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, timeout=15, capture_output=True)


if __name__ == '__main__':
    unittest.main()
