"""Exact task selection and independent frozen train/eval data; no GPUs."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from script import swe10
from arl.experiments.async_swe_eval import publish_spec, run_one
from test_swe_env import ce, record


class TaskSelectionTests(unittest.TestCase):
    def args(self, *values):
        return swe10.parser().parse_args(swe10.normalize_key_values(values))

    def backend(self):
        backend=type('Backend', (), {})()
        async def load(dataset, limit):
            rows=[record(i) for i in range(20)]
            return rows[:limit] if limit else rows
        backend.load_tasks=AsyncMock(side_effect=load)
        backend.rpc=AsyncMock(return_value={'ready':True})
        backend.close=AsyncMock()
        return backend

    def freeze(self,args,root,backend):
        with patch.object(ce,'DockerBackend',return_value=backend), contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(swe10.freeze_tasks(args,root))
        return json.loads((root/'tasks.json').read_text())

    def test_count_and_default_evaluation_inheritance(self):
        args=self.args('train_task_limit=12','batch_size=12')
        configs=swe10.configurations(args,Path('/tmp/select'))
        self.assertEqual(configs['train']['train']['options']['task_limit'],12)
        self.assertEqual(configs['baseline']['evaluate']['options']['task_limit'],12)
        args=self.args('train_instance_ids=example__repo-3,example__repo-1')
        self.assertEqual(swe10.task_selection(args),swe10.task_selection(args,True))

    def test_independent_ordered_ids_freeze_resume_and_async(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);backend=self.backend()
            args=self.args('train_instance_ids=example__repo-7,example__repo-2,example__repo-0',
                           'eval_instance_ids=example__repo-9,example__repo-5','batch_size=2','num_epochs=3')
            manifest=self.freeze(args,root,backend)
            self.assertEqual(manifest['scope'],'held_out_evaluation')
            self.assertEqual(manifest['train_instance_ids'],['example__repo-7','example__repo-2','example__repo-0'])
            self.assertEqual(manifest['evaluation_instance_ids'],['example__repo-9','example__repo-5'])
            backend.load_tasks.assert_awaited_once_with(args.dataset,0)
            backend.rpc.assert_awaited_with('preflight',instance_ids=manifest['train_instance_ids']+manifest['evaluation_instance_ids'])
            for name in ('tasks.private.jsonl','eval_tasks.private.jsonl'):
                self.assertEqual((root/name).stat().st_mode & 0o777,0o600)
            self.assertNotIn('SECRET',(root/'tasks.json').read_text())
            self.freeze(args,root,backend)
            self.assertEqual(backend.load_tasks.await_count,1)
            configs=swe10.configurations(args,root)
            self.assertEqual(configs['baseline']['evaluate']['options']['train_set'],str(root/'eval_tasks.private.jsonl'))
            self.assertEqual(configs['evaluate']['evaluate']['options'],configs['baseline']['evaluate']['options'])
            spec=publish_spec(args,root,configs)
            self.assertEqual(spec['expected_batches'],5)  # GRPO flattens epochs before batching.
            (root/'eval_tasks.private.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'evaluation task records changed'):
                asyncio.run(run_one(root,'base'))

    def test_first_n_train_and_separate_eval_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);backend=self.backend()
            args=self.args('task_limit=4','eval_task_limit=2','eval_dataset=/other.jsonl')
            manifest=self.freeze(args,root,backend)
            self.assertEqual(len(manifest['train_instance_ids']),4)
            self.assertEqual(len(manifest['evaluation_instance_ids']),2)
            self.assertEqual(manifest['scope'],'partially_overlapping_evaluation')
            self.assertEqual(backend.load_tasks.await_args_list[1].args,('/other.jsonl',2))

    def test_files_exact_ids_and_invalid_selectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ids.txt'
            for content in ('example__repo-4\nexample__repo-2\n','["example__repo-4", "example__repo-2"]'):
                path.write_text(content)
                args=self.args(f'train_ids_file={path}')
                self.assertEqual(swe10.task_selection(args)['ids'],['example__repo-4','example__repo-2'])
            for values in (('train_instance_ids=a,a',),('train_instance_ids= a',),
                           ('train_instance_ids=',),('task_limit=0',),('eval_task_limit=-1',),
                           (f'train_ids_file={path}','train_instance_ids=a')):
                with self.subTest(values=values),self.assertRaises(ValueError):
                    swe10.task_selection(self.args(*values),'eval_task_limit=-1' in values)

    def test_missing_ids_fail_without_publishing_and_changed_ids_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);backend=self.backend()
            with self.assertRaisesRegex(ValueError,'absent'):
                self.freeze(self.args('train_instance_ids=example__repo-999'),root,backend)
            self.assertFalse((root/'tasks.json').exists())
            self.freeze(self.args('train_instance_ids=example__repo-1'),root,backend)
            with self.assertRaisesRegex(ValueError,'frozen IDs differ'):
                self.freeze(self.args('train_instance_ids=example__repo-2'),root,backend)
