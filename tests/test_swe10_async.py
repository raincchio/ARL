"""SWE asynchronous specs/checkpoint publication/remote outcomes, no GPUs."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from script import swe10
from arl.experiments.async_swe_eval import publish_spec, read_checkpoint, evaluation_config, finish_result, run_one
from arl.experiments import eval_service as service
from test_swe10 import RecipeTests


class AsyncSweTests(unittest.TestCase):
    def fixture(self,root,*extra_args):
        args=swe10.parser().parse_args(swe10.normalize_key_values([
            'eval_async=True','eval_every=5','batch_size=8','num_epochs=10','harness=mini_swe_agent','wandb=False',
            'eval_num_samples=2',f'run_dir={root}',*extra_args]))
        (root/'tasks.private.jsonl').write_text('frozen records')
        (root/'tasks.json').write_text(json.dumps({'instance_ids':[str(i) for i in range(10)]}))
        configs=swe10.configurations(args,root)
        spec=publish_spec(args,root,configs)
        return args,configs,spec

    def publish(self,root,name,batch,partial=False):
        weights=root/'train/checkpoints'/name/'policy';weights.mkdir(parents=True)
        (weights/'config.json').write_text('{}')
        with (root/'train/checkpoints.jsonl').open('a') as f:
            f.write(json.dumps({'name':name,'batch':batch,'sampler_path':str(weights)})+('' if partial else '\n'))
        return weights

    def test_arm_key_value_aliases_preserve_order_and_reject_kernel_options(self):
        args=swe10.parser().parse_args(swe10.normalize_key_values([
            'COLOCATE=1','SAMPLING_UTIL=.7','PACK_TOKENS=32768','HEAD_CHUNKS=8',
            'sampling_replay=True','router_replay=True','save_optimizer=False','save_optimizer=False',
            'harness=claude_code','num_epochs=10','eval_async=True','eval_at_start=False',
            'save_best_solutions=False','SAMPLER_DP=2','num_epochs=3']))
        self.assertTrue(args.colocate);self.assertTrue(args.sampling_replay);self.assertTrue(args.router_replay)
        self.assertFalse(args.save_optimizer);self.assertFalse(args.save_best_solutions)
        self.assertEqual(args.epochs,3);self.assertEqual(args.pack_tokens,32768);self.assertEqual(args.sampler_dp,2)
        for option in ('gamma=.4','eval_profiling=True','adv_unit=turn','curriculum=True','save_best_kernels=False'):
            with self.assertRaises(ValueError):swe10.normalize_key_values([option])

    def test_publication_is_immutable_and_only_complete_index_lines_are_scheduled(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,configs,spec=self.fixture(root)
            self.assertEqual(configs['train']['train']['options']['save_every'],5)
            self.assertEqual(spec['expected_batches'],13)
            self.assertEqual(spec['sampler_tp'],4)
            args.phase='prepare';self.assertEqual(publish_spec(args,root,configs),spec)
            run=service.load_run(root,2,8)
            self.assertIsInstance(run,service.Run)
            self.assertEqual(service.eval_points(run),['base'])
            self.publish(root,'000005',5);self.publish(root,'000020',20);self.publish(root,'final',20)
            self.assertEqual(service.eval_points(run),['base','000005','final'])
            self.assertEqual(read_checkpoint(root,'000005')[1],5)
            self.publish(root,'000015',15,partial=True)
            self.assertNotIn('000015',service._checkpoints(run))
            with self.assertRaises(ValueError):read_checkpoint(root,'000015')

    def test_remote_checkpoint_plan_keeps_frozen_harness_tasks_and_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,configs,spec=self.fixture(root)
            weights=self.publish(root,'000005',5)
            plan,batch=evaluation_config(root,'000005',spec,root/'attempt')
            self.assertEqual(batch,5)
            self.assertEqual(plan['models']['policy']['name'],str(weights))
            self.assertEqual(plan['evaluate']['options'],configs['evaluate']['evaluate']['options'])
            self.assertEqual(plan['evaluate']['harness'],'mini_swe_agent')
            env=service.child_env(service.load_run(root,2,8),['0','1','2','3'],.7)
            self.assertEqual(env['KERNEL_RL_SAMPLER_DP'],'2')
            self.assertEqual(env['KERNEL_RL_COLOCATE'],'0')
            self.assertNotIn('WORLD_SIZE',env)

    def test_unlimited_workspace_limit_reaches_all_eval_phases(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, configs, spec = self.fixture(root, 'train_concurrency=3', 'eval_concurrency=-1')
            self.assertEqual(configs['train']['train']['options']['eval_max_concurrency'], 3)
            for name in ('base', '000005', 'final'):
                if name != 'base':
                    self.publish(root, name, 5)
                config, _ = evaluation_config(root, name, spec, root/'attempt')
                self.assertEqual(config['evaluate']['options']['eval_max_concurrency'], -1)

    def test_eval_sampling_overrides_reach_checkpoint_and_inherit_unset_fields(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, configs, spec = self.fixture(root, 'sampling_replay=True', 'temperature=0.8',
                'top_p=0.9', 'top_k=40', 'eval_temperature=0', 'eval_top_k=-1')
            train = configs['train']['train']['options']
            self.assertEqual((train['temperature'], train['top_p'], train['top_k']), (.8, .9, 40))
            self.assertTrue(train['sampling_replay'])
            self.publish(root, '000005', 5)
            for name in ('base', '000005'):
                config, _ = evaluation_config(root, name, spec, root/'attempt')
                options = config['evaluate']['options']
                self.assertEqual((options['temperature'], options['top_p'], options['top_k']), (0., .9, -1))
                self.assertFalse(options['sampling_replay'])

    def test_spec_pins_tp_across_evaluator_restarts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,spec=self.fixture(root)
            spec['sampler_tp']=4
            (root/'async_eval/spec.json').write_text(json.dumps(spec))
            for ngpus,expected_dp in ((8,2),(16,4)):
                run=service.load_run(root,None,ngpus)
                self.assertEqual(run.dp,expected_dp)
                self.assertEqual(run.dp_src,'spec sampler_tp')
            self.assertEqual(service.load_run(root,1,8).dp,1)
            self.assertEqual(service.load_run(root,None,8,8).dp,1)
            spec.pop('sampler_tp')
            (root/'async_eval/spec.json').write_text(json.dumps(spec))
            self.assertEqual(service.load_run(root,None,8).dp,2)
            self.assertEqual(service.load_run(root,None,16).dp,4)
            for invalid in (True,0,3,'4'):
                spec['sampler_tp']=invalid
                (root/'async_eval/spec.json').write_text(json.dumps(spec))
                self.assertIsInstance(service.load_run(root,None,8),str)

    def test_base_and_final_scalar_results_trigger_comparison_and_done_markers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,configs,spec=self.fixture(root)
            fixture=RecipeTests()
            fixture.fixture(root,1,'base-results');fixture.fixture(root,3,'final-results')
            meta=self.publish(root,'final',20)/'meta.json'
            meta.write_text(json.dumps({'policy_optimizer_steps':40}))
            (root/'train/metrics.jsonl').write_text(json.dumps({'ppo/n_active_tokens':8,'ppo/grad_norm':.1})+'\n')
            ids=[str(i) for i in range(10)]
            with contextlib.redirect_stdout(io.StringIO()):
                finish_result(root,'base',0,root/'base-results',swe10.evaluation_report(root/'base-results',ids,2),1.)
                self.assertFalse((root/'comparison.json').exists())
                finish_result(root,'final',20,root/'final-results',swe10.evaluation_report(root/'final-results',ids,2),2.)
            result=json.loads((root/'comparison.json').read_text())
            self.assertAlmostEqual(result['resolved_rate_delta'],.2)
            self.assertEqual(service.next_ckpt(service.load_run(root,2,8),2),'FINISHED')

    def test_exhausted_baseline_is_reported_even_after_final_is_done(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            self.publish(root, 'final', 13)
            run = service.load_run(root, 2, 8)
            (run.out / 'failed').mkdir()
            (run.out / 'failed/base').write_text('2')
            (run.out / 'done').mkdir()
            (run.out / 'done/final').touch()
            self.assertIsNone(service.next_ckpt(run, 2))
            daemon = service.Daemon(SimpleNamespace(gpus='0,1,2,3,4,5,6,7', ngpus=8, retries=2))
            daemon.runs[root] = run
            with patch.object(daemon, 'sync_queue'), patch.object(daemon, 'write_status'):
                daemon.tick()
            self.assertFalse(run.finished)
            self.assertIn('retry limit reached for base', run.state)

    def test_unreachable_swe_master_waits_before_leasing_gpus(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, spec = self.fixture(root)
            run = service.load_run(root, 2, 8)
            daemon = service.Daemon(SimpleNamespace(gpus='0,1,2,3,4,5,6,7', ngpus=8, retries=2))
            daemon.runs[root] = run
            with patch.object(daemon, 'sync_queue'), patch.object(daemon, 'write_status'), \
                 patch.object(daemon, 'pick') as pick, patch.object(service, 'build_opener') as opener:
                opener.return_value.open.side_effect = ConnectionRefusedError
                daemon.tick()
                endpoint = spec['baseline']['evaluate']['options']['swe_endpoint']
                opener.return_value.open.assert_called_once_with(f'{endpoint}/healthz', timeout=5)
                pick.assert_not_called()
            self.assertIn('evaluation service unreachable', run.state)
            self.assertFalse((run.out / 'failed/base').exists())

    def test_async_all_does_not_launch_or_wait_for_local_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            def freeze(args,run):
                (run/'tasks.private.jsonl').write_text('frozen')
            with patch.object(swe10,'freeze_tasks',new=AsyncMock(side_effect=freeze)), \
                 patch.object(swe10,'execute') as execute,contextlib.redirect_stdout(io.StringIO()):
                swe10.main([f'run_dir={root}','eval_async=True','eval_at_start=True'])
            self.assertEqual([c.args[2] for c in execute.call_args_list],['train'])
            self.assertTrue((root/'async_eval/spec.json').exists())
            spec=json.loads((root/'async_eval/spec.json').read_text())
            self.assertTrue(spec['baseline_requested'])
            self.assertFalse(json.loads((root/'train.json').read_text())['train']['options']['eval_at_start'])
            self.assertEqual(service.eval_points(service.load_run(root,4,4)),['base'])
            self.assertFalse((root/'baseline').exists())

    def test_disabled_initial_evaluation_keeps_final_evaluation(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous), tempfile.TemporaryDirectory() as d:
                root=Path(d)/'run'

                def freeze(args,run):
                    (run/'tasks.private.jsonl').write_text('frozen')

                with patch.object(swe10,'freeze_tasks',new=AsyncMock(side_effect=freeze)), \
                     patch.object(swe10,'execute') as execute, patch.object(swe10,'compare') as compare, \
                     contextlib.redirect_stdout(io.StringIO()):
                    swe10.main([f'run_dir={root}',f'eval_async={asynchronous}',
                                'eval_at_start=False','num_epochs=10'])
                phases=[call.args[2] for call in execute.call_args_list]
                self.assertEqual(phases,['train'] if asynchronous else ['train','evaluate'])
                compare.assert_not_called()
                if asynchronous:
                    spec=json.loads((root/'async_eval/spec.json').read_text())
                    self.assertFalse(spec['baseline_requested'])
                    run=service.load_run(root,4,4)
                    self.assertEqual(service.eval_points(run),[])
                    self.publish(root,'000005',5)
                    self.publish(root,'final',10)
                    self.assertEqual(service.eval_points(run),['000005','final'])

    def test_bad_frozen_task_hash_is_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'tasks.private.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                asyncio.run(run_one(root,'base'))


if __name__=='__main__':unittest.main()
