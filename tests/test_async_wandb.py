"""Asynchronous W&B resume, steps, and failure handling without network calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from arl.experiments import async_swe_eval as evaluation
from script import swe10


class AsyncWandbTests(unittest.TestCase):
    def fixture(self, root):
        (root/'async_eval').mkdir()
        spec = {'wandb_project': 'ARL', 'wandb_name': 'training', 'launch_args': {}}
        (root/'async_eval/spec.json').write_text(json.dumps(spec))
        sdk = MagicMock()
        sdk.init.return_value.step = 0
        return spec, sdk

    def test_checkpoints_resume_same_eval_run_at_training_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sdk = self.fixture(root)
            with patch.dict('sys.modules', {'wandb': sdk}):
                for name, batch in [('base', 0), ('000005', 5)]:
                    self.assertTrue(evaluation.upload_wandb_result(root, {
                        'checkpoint': name, 'batch': batch, 'step': max(0, batch-1),
                        'eval/resolved_rate': .5, 'eval/correct_rate': .5,
                        'time/eval': 60., 'evaluation_dir': 'local'}))
            calls = sdk.init.call_args_list
            self.assertEqual(calls[0].kwargs['id'], calls[1].kwargs['id'])
            self.assertEqual(calls[0].kwargs['id'], 'aev-'+hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12])
            self.assertEqual(calls[0].kwargs['name'], 'training-eval')
            self.assertEqual(calls[0].kwargs['resume'], 'allow')
            self.assertEqual(calls[0].kwargs['project'], 'ARL')
            logs = sdk.init.return_value.log.call_args_list
            self.assertEqual([c.kwargs['step'] for c in logs], [0, 4])
            self.assertNotIn('evaluation_dir', logs[0].args[0])
            self.assertNotIn('eval/correct_rate', logs[0].args[0])
            self.assertEqual(logs[0].args[0]['eval/resolved_rate'], .5)
            self.assertEqual(sdk.init.return_value.finish.call_count, 2)

    def test_old_step_is_not_reuploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, sdk = self.fixture(root)
            sdk.init.return_value.step = 5
            with patch.dict('sys.modules', {'wandb': sdk}):
                self.assertFalse(evaluation.upload_wandb_result(root, {'step': 0}))
            sdk.init.return_value.log.assert_not_called()
            sdk.init.return_value.finish.assert_called_once()

    def test_disabled_and_empty_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, sdk = self.fixture(root)
            with patch.dict('sys.modules', {'wandb': sdk}), patch.dict('os.environ', {'KERNEL_RL_WANDB_PROXY': ''}):
                self.assertIsNone(evaluation._wandb_run({**spec, 'wandb_project': None}, root))
                sdk.init.assert_not_called()
                evaluation._wandb_run(spec, root)
                self.assertNotIn('settings', sdk.init.call_args.kwargs)

    def test_upload_failure_preserves_local_result_and_done(self):
        for failure in ('init', 'log', 'finish'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, sdk = self.fixture(root)
                target = sdk.init if failure == 'init' else getattr(sdk.init.return_value, failure)
                target.side_effect = RuntimeError('unavailable')
                with patch.dict('sys.modules', {'wandb': sdk}):
                    row = evaluation.finish_result(root, 'base', 0, root/'attempt',
                        {'resolved_rate': .5, 'tasks_solved_in_k': 1, 'per_task': {'a': {}, 'b': {}}}, 10.)
                self.assertEqual(json.loads((root/'async_eval/metrics.jsonl').read_text()), row)
                self.assertEqual(row['eval/resolved_rate'], .5)
                self.assertNotIn('eval/correct_rate', row)
                self.assertTrue((root/'async_eval/done/base').exists())

    def test_spec_inherits_training_wandb_preferences(self):
        for enabled in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'tasks.private.jsonl').write_text('tasks')
                args = swe10.parser().parse_args(['--wandb' if enabled else '--no-wandb'])
                configs = swe10.configurations(args, root)
                spec = evaluation.publish_spec(args, root, configs)
                self.assertEqual(spec['wandb_project'], 'ARL' if enabled else None)
                self.assertEqual(spec['wandb_name'], root.name)
