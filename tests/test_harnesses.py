"""Harness selection stays separate from benchmark and model execution."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from arl.harnesses import HarnessPendingError, get_harness, list_harnesses, run_harness


class HarnessTests(unittest.TestCase):
    def test_three_requested_harnesses_have_distinct_entries(self):
        specs = list_harnesses()
        self.assertTrue({'codex', 'claude_code', 'mini_swe_agent'}.issubset({s.name for s in specs}))
        self.assertEqual(len({s.path for s in specs}), len(specs))
        for spec in specs:
            self.assertIn(spec.status, ('pending', 'ready'))
            self.assertTrue((spec.path/'entry.py').is_file())
            self.assertTrue((spec.path/'README.md').is_file())

    def test_aliases_and_invalid_names(self):
        self.assertIs(get_harness('claude-code'), get_harness('claude_code'))
        self.assertIs(get_harness('mini-swe-agent'), get_harness('mini_swe_agent'))
        with self.assertRaisesRegex(ValueError, 'unknown harness'):
            get_harness('missing')

    def test_unfinished_harnesses_never_start_agent(self):
        for spec in list_harnesses():
            if spec.status != 'pending':
                continue
            with self.subTest(name=spec.name), self.assertRaisesRegex(HarnessPendingError, '待完成'):
                run_harness(spec.name)

    def test_arl_agent_is_restricted_to_cuda_kernelbench(self):
        from arl.harnesses import resolve_pair, HarnessCompatibilityError
        self.assertEqual(resolve_pair('cuda', operation='train').harness, 'arl_agent')
        self.assertEqual(resolve_pair('ifeval', operation='train').harness, 'direct')
        for env in ('swe', 'programbench', 'ifeval'):
            for operation in ('collect', 'train', 'evaluate'):
                with self.subTest(env=env, operation=operation), self.assertRaises(HarnessCompatibilityError):
                    resolve_pair(env, 'arl_agent', operation=operation)

    def test_task_model_options_are_forwarded_to_selected_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'entry.py').write_text('def run(**kwargs): return kwargs\n')
            spec = replace(get_harness('codex'), directory=directory, status='ready')
            self.assertEqual(spec.run(task='sample', model='target'), {'task':'sample','model':'target'})


if __name__ == '__main__':
    unittest.main()
