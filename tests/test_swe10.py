"""SWE first-10 recipe: frozen tasks, matching evaluation, honest comparison."""
import asyncio
import gzip
import json
from pathlib import Path
import tempfile
import contextlib
import io
import os
import ast
import unittest
from unittest.mock import AsyncMock, patch

from script import swe10
from test_swe_env import record, ce


class RecipeTests(unittest.TestCase):
    def args(self,root):
        return swe10.parser().parse_args(['--run-dir',str(root),'--train-gpus','0,1,2,3,8,9,10,11','--sampler-gpus','4,5,6,7,12,13,14,15','--no-colocate'])

    def test_both_evaluations_use_all_training_tasks_and_same_budgets(self):
        root=Path('/tmp/swe10-plan')
        args=self.args(root)
        configs=swe10.configurations(args,root)
        before=configs['baseline']['evaluate'];after=configs['evaluate']['evaluate']
        train=configs['train']['train']
        self.assertEqual(before['options'],after['options'])
        self.assertEqual(before['options']['batch_size'],10)
        self.assertEqual(before['options']['eval_max_concurrency'],64)
        self.assertEqual(before['options']['group_size'],8)
        self.assertEqual(train['options']['eval_max_concurrency'],2)
        for plan in (before,after,train):
            self.assertEqual(plan['split'],'train')
            self.assertEqual(plan['harness'],'claude_code')
            self.assertEqual(plan['options']['test_fraction'],0)
            self.assertEqual(plan['options']['task_limit'],10)
            self.assertEqual(plan['options']['train_set'],str(root/'tasks.private.jsonl'))
        self.assertEqual(configs['baseline']['models']['policy']['name'],swe10.MODEL)
        self.assertEqual(configs['evaluate']['models']['policy']['name'],str(root/'train/checkpoints/final/policy'))
        self.assertEqual(train['options']['group_size'],8)
        self.assertEqual(train['options']['num_epochs'],5)
        self.assertEqual(train['options']['think_cost_lambda'],0)
        self.assertFalse(train['options']['save_optimizer'])

    def test_evaluation_queues_selected_tasks_with_independent_concurrency(self):
        args=swe10.parser().parse_args(swe10.normalize_key_values([
            'eval_task_limit=7','eval_concurrency=12','concurrency=3','batch_size=4','extra_samples=2']))
        configs=swe10.configurations(args,Path('/tmp/swe-eval-queue'))
        for name in ('baseline','evaluate'):
            options=configs[name]['evaluate']['options']
            self.assertEqual(options['batch_size'],7)
            self.assertEqual(options['eval_max_concurrency'],12)
            self.assertNotIn('extra_samples',options)
        self.assertEqual(configs['train']['train']['options']['batch_size'],4)
        self.assertEqual(configs['train']['train']['options']['eval_max_concurrency'],3)
        self.assertEqual(configs['train']['train']['options']['extra_samples'],2)

    def test_gpu_layout_and_rank_launch(self):
        root=Path('/tmp/swe10-plan');args=self.args(root)
        self.assertIn('--nproc_per_node=8',swe10.command(args,root,'train'))
        env=swe10.runtime_env(args,True)
        self.assertEqual(len(env['CUDA_VISIBLE_DEVICES'].split(',')),16)
        self.assertEqual(len(set(env['CUDA_VISIBLE_DEVICES'].split(','))),16)
        eval_env=swe10.runtime_env(args,False)
        self.assertEqual(eval_env['CUDA_VISIBLE_DEVICES'],args.sampler_gpus)
        self.assertEqual(eval_env['KERNEL_RL_SAMPLER_DP'],'2')
        self.assertEqual(eval_env['KERNEL_RL_SAMPLING_GPU_UTIL'],'0.85')
        self.assertEqual(eval_env['NCCL_SHM_DISABLE'],'1')
        self.assertEqual(eval_env['NCCL_P2P_LEVEL'],'SYS')
        self.assertNotIn('WORLD_SIZE',eval_env)
        args.colocate=True;args.sampler_gpus='0,1,2,3'
        self.assertEqual(swe10.runtime_env(args,True)['CUDA_VISIBLE_DEVICES'],args.train_gpus)

    def test_harness_selection_applies_to_all_three_phases(self):
        from arl.experiments.config import from_dict
        from arl.experiments.runner import validate_plan
        for harness in ('claude_code','mini_swe_agent'):
            args=swe10.parser().parse_args(['--harness',harness])
            configs=swe10.configurations(args,Path('/tmp/swe10-harness-selection'))
            for name,config in configs.items():
                phase='train' if name=='train' else 'evaluate'
                self.assertEqual(config[phase]['harness'],harness)
                validate_plan(from_dict(config).plan(phase))

    def test_prepare_freezes_service_order_and_keeps_gold_private(self):
        backend=type('Backend',(),{})()
        backend.load_tasks=AsyncMock(return_value=[record(i) for i in range(10)])
        backend.rpc=AsyncMock(return_value={'ready':True});backend.close=AsyncMock()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.args(root)
            with patch.object(ce,'DockerBackend',return_value=backend):
                asyncio.run(swe10.freeze_tasks(args,root))
                asyncio.run(swe10.freeze_tasks(args,root))
            backend.load_tasks.assert_awaited_once_with(args.dataset,10)
            public=(root/'tasks.json').read_text()
            self.assertNotIn('SECRET',public)
            manifest=json.loads(public)
            self.assertEqual(manifest['instance_ids'],[record(i)['instance_id'] for i in range(10)])
            self.assertEqual(manifest['train_instance_ids'],manifest['evaluation_instance_ids'])
            self.assertEqual((root/'tasks.private.jsonl').stat().st_mode & 0o777,0o600)
            self.assertIn('SECRET_GOLD',(root/'tasks.private.jsonl').read_text())

    def test_user_arm_training_options_reach_configs_and_runtime(self):
        with patch.dict(os.environ, {'KERNEL_RL_ZERO3_REDUCE_BUCKET':'100000000',
                'KERNEL_RL_FREEZE_ROUTER':'0','KERNEL_RL_FREEZE_SHARED_EXPERT':'1'}):
            args=swe10.parser().parse_args([
                '--sampling-replay','--router-replay','--pack-tokens','32768','--head-chunks','8',
                '--batch-size','8','--group-size','8','--epochs','10','--eval-samples','16',
                '--max-turns','4','--max-tokens','40960',
                '--think-budget','32768','--warmup-steps','30','--warmup-lr-scale','.5',
                '--think-cost-lambda','.005','--think-cost-free-tokens','8192','--format-reward-weight','.05',
                '--mask-positive-incorrect','--mask-negative-correct',
                '--no-save-trajectories','--no-save-optimizer','--no-save-best-solutions'])
            configs=swe10.configurations(args,Path('/tmp/aligned'))
            train=configs['train']['train']['options']
            expected={'sampling_replay':True,'router_replay':True,'warmup_steps':30,
                'think_budget':32768,'think_cost_lambda':.005,'think_cost_free_tokens':8192,'format_reward_weight':.05,
                'mask_positive_incorrect':True,'mask_negative_correct':True,
                'save_optimizer':False,'save_trajectories':False,'save_best_solutions':False}
            for key,value in expected.items():self.assertEqual(train[key],value,key)
            for phase in ('baseline','evaluate'):
                options=configs[phase]['evaluate']['options']
                self.assertFalse(options['router_replay'])
                self.assertEqual(options['think_cost_lambda'],0)
                self.assertNotIn('format_reward_weight', options)
                self.assertEqual(options['think_budget'],32768)
                self.assertFalse(options['save_trajectories'])
            env=swe10.runtime_env(args,True)
            self.assertEqual(env['KERNEL_RL_PACK_TOKENS'],'32768')
            self.assertEqual(env['KERNEL_RL_HEAD_CHUNKS'],'8')
            self.assertEqual(env['KERNEL_RL_ZERO3_REDUCE_BUCKET'],'100000000')
            self.assertEqual(env['KERNEL_RL_FREEZE_ROUTER'],'0')
            self.assertEqual(env['KERNEL_RL_FREEZE_SHARED_EXPERT'],'1')

    def test_scalar_outcomes_support_comparison_without_full_trajectories(self):
        from arl.experiments.config import from_dict
        from arl.experiments.runner import run_phase
        from test_mini_swe import MiniSource, MiniRenderer, Sampler
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config={'models':{'p':{'name':'scripted-policy'}}, 'evaluate':{
                'env':'swe','harness':'mini_swe_agent','model':'p','split':'train','output_dir':'eval',
                'options':{'group_size':2,'batch_size':1,'max_tokens':10,'prompt_budget_tokens':100,
                           'save_trajectories':False,'save_best_solutions':False}}}
            source=MiniSource()
            with patch('arl.harnesses.registry.HarnessSpec.make_renderer',return_value=MiniRenderer()):
                run_phase(from_dict(config,base_dir=root).plan('evaluate'),native_runtime={
                    'source':source,'sampling_client':Sampler(),'tokenizer':object()})
            self.assertFalse((root/'eval/trajectories').exists())
            self.assertFalse((root/'eval/best_solutions').exists())
            ids=[source.task_key(t) for t in source.train_tasks()]
            report=swe10.evaluation_report(root/'eval',ids,2)
            self.assertEqual(report['scored'],2)
            self.assertEqual(report['resolved'],1)
            outcomes=(root/'eval/outcomes.jsonl').read_text()
            self.assertNotIn('model_patch',outcomes)
            self.assertNotIn('SECRET',outcomes)

    def test_router_check_does_not_create_overlay_or_start_work(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'new'
            with patch('arl.models.vllm_overlay.prepare_overlay') as overlay, \
                 patch.object(swe10,'freeze_tasks') as freeze, patch.object(swe10,'execute') as execute, \
                 contextlib.redirect_stdout(io.StringIO()):
                swe10.main(['--run-dir',str(run),'--router-replay','--check'])
            overlay.assert_not_called();freeze.assert_not_called();execute.assert_not_called()
            self.assertFalse(run.exists())

    def fixture(self,root,good,phase,drop=False):
        path=root/phase
        (path/'trajectories').mkdir(parents=True)
        (path/'manifest.json').write_text(json.dumps({'status':'complete'}))
        with gzip.open(path/'trajectories/rank0.jsonl.gz','wt') as f:
            for i in range(10):
                for j in range(2):
                    f.write(json.dumps({'task':str(i),'reward':int(i<good),'dropped':'infra_error' if drop and i==0 else '',
                                        'summary':{'infra_error':drop and i==0}})+'\n')

    def test_comparison_requires_real_optimizer_updates_and_reports_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.args(root);args.eval_samples=2
            (root/'tasks.json').write_text(json.dumps({'instance_ids':[str(i) for i in range(10)]}))
            self.fixture(root,1,'baseline');self.fixture(root,3,'after')
            meta=root/'train/checkpoints/final/policy/meta.json';meta.parent.mkdir(parents=True)
            meta.write_text(json.dumps({'policy_optimizer_steps':4}))
            (root/'train/metrics.jsonl').write_text(json.dumps({'ppo/n_active_tokens':8,'ppo/grad_norm':.1})+'\n')
            report=swe10.compare(args,root)
            self.assertAlmostEqual(report['resolved_rate_delta'],.2)
            self.assertEqual(report['assessment'],'improved_on_training_tasks')
            meta.write_text(json.dumps({'policy_optimizer_steps':0}))
            self.assertEqual(swe10.compare(args,root)['assessment'],'inconclusive_infra_or_no_effective_update')
            rows=root/'after/trajectories/rank0.jsonl.gz'
            with gzip.open(rows,'rt') as f:values=[json.loads(line) for line in f]
            values[0]['dropped']='infra_error'
            with gzip.open(rows,'wt') as f:
                for item in values:f.write(json.dumps(item)+'\n')
            meta.write_text(json.dumps({'policy_optimizer_steps':4}))
            self.assertEqual(swe10.compare(args,root)['after']['dropped'],1)

    def test_missing_tasks_cannot_be_silently_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,1,'baseline')
            with self.assertRaises(ValueError):
                swe10.evaluation_report(root/'baseline',[str(i) for i in range(10)],4)


if __name__=='__main__':unittest.main()
