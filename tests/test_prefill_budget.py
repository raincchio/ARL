"""Persistent scheduler budgets and 4xTP4 inference layout without GPU startup."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from arl.models.prefill import CONFIG_KEY, engine_args
from script import swe10


class PrefillBudgetTests(unittest.TestCase):
    def test_engine_config_preserves_norm_settings_and_profile_budget(self):
        self.assertEqual(engine_args(),{})
        norm={'arm_fixed_rmsnorm':{'version':1,'hidden_size':5120}}
        result=engine_args(1056,2048,norm)
        self.assertEqual(result['max_num_batched_tokens'],2048)
        self.assertTrue(result['async_scheduling'])
        self.assertTrue(result['enable_chunked_prefill'])
        self.assertEqual(result['additional_config']['arm_fixed_rmsnorm'],norm['arm_fixed_rmsnorm'])
        self.assertNotIn(CONFIG_KEY,norm)
        self.assertEqual(result['additional_config'][CONFIG_KEY]['tokens'],1056)
        self.assertEqual(engine_args(1056)['max_num_batched_tokens'],2048)
        self.assertEqual(engine_args(4096)['max_num_batched_tokens'],4096)
        for values in ((-1,0),(1,-1),(2048,1056),(True,0),(1.5,2048)):
            with self.subTest(values=values),self.assertRaises(ValueError):engine_args(*values)

    def test_budget_changes_runtime_only_and_align_validation_uses_actual_block(self):
        from arl.models.prefill_scheduler import BudgetAsyncScheduler
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.config.scheduler import SchedulerConfig
        config=SimpleNamespace(additional_config={CONFIG_KEY:{'version':1,'tokens':1056}})
        def initialize(instance,*args,**kwargs):
            instance.scheduler_config=SimpleNamespace(max_num_batched_tokens=2048)
            instance.cache_config=SimpleNamespace(mamba_cache_mode='align')
            instance.block_size=528
            instance.max_num_scheduled_tokens=2048
        with patch.object(AsyncScheduler,'__init__',initialize):
            scheduler=BudgetAsyncScheduler(config)
            self.assertEqual(scheduler.max_num_scheduled_tokens,1056)
            self.assertEqual(scheduler.scheduler_config.max_num_batched_tokens,2048)
            self.assertEqual(scheduler.arl_prefill_status['block_size'],528)
            config.additional_config[CONFIG_KEY]['tokens']=1024
            with self.assertRaisesRegex(ValueError,'block_size=528'):BudgetAsyncScheduler(config)
        real=SchedulerConfig(max_model_len=262144,is_encoder_decoder=False,max_num_batched_tokens=2048,max_num_seqs=256,
            enable_chunked_prefill=True,async_scheduling=True,
            scheduler_cls='arl.models.prefill_scheduler.BudgetAsyncScheduler')
        self.assertIs(real.get_scheduler_cls(),BudgetAsyncScheduler)

    def test_launcher_freezes_budget_for_training_and_both_evaluations(self):
        args=swe10.parser().parse_args(swe10.normalize_key_values(['prefill_tokens=1056','PREFILL_BATCH_TOKENS=2048']))
        configs=swe10.configurations(args,Path('/tmp/prefill-budget'))
        for name,cfg in configs.items():
            phase='train' if name=='train' else 'evaluate'
            self.assertEqual(cfg[phase]['options']['sampling_prefill_tokens'],1056)
            self.assertEqual(cfg[phase]['options']['sampling_prefill_batch_tokens'],2048)
            from arl.experiments.config import from_dict
            from arl.experiments.runner import prepare_native_config
            result=prepare_native_config(from_dict(cfg).plan(phase))
            self.assertEqual(result.sampling_prefill_tokens,1056)
            self.assertEqual(result.sampling_prefill_batch_tokens,2048)

    def client(self,inference):
        import torch
        from arl.models.training_client import LocalTrainingClient
        client=object.__new__(LocalTrainingClient)
        client._sampling_engine=None
        client._inference_only=inference
        client._sampling_cfg=dict(enable_sleep_mode=False,enforce_eager=False,
            prefill_tokens=1056,prefill_batch_tokens=2048,enable_prefix_caching=True,mamba_cache_mode='align')
        client.model_name='policy';client.dtype=torch.bfloat16
        return client

    def test_training_and_single_process_inference_both_construct_four_tp4_engines(self):
        for inference in (False,True):
            client=self.client(inference);calls=[]
            def build(*args,**kwargs):
                calls.append((os.environ['CUDA_VISIBLE_DEVICES'],kwargs))
                return Mock()
            env={'WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0','CUDA_VISIBLE_DEVICES':','.join(map(str,range(16))),
                'KERNEL_RL_SAMPLER_GPUS':','.join(map(str,range(16))),'KERNEL_RL_SAMPLER_DP':'4'}
            with self.subTest(inference=inference),patch.dict(os.environ,env),patch('arl.models.sampling.SamplingEngine',side_effect=build):
                result=client._ensure_sampling_engine()
                self.assertEqual(len(result.engines),4)
                self.assertEqual([c[0] for c in calls],['0,1,2,3','4,5,6,7','8,9,10,11','12,13,14,15'])
                self.assertTrue(all(c[1]['tensor_parallel_size']==4 for c in calls))
                self.assertTrue(all(c[1]['prefill_tokens']==1056 and c[1]['prefill_batch_tokens']==2048 for c in calls))
                self.assertTrue(all(c[1]['enable_prefix_caching'] and c[1]['mamba_cache_mode']=='align' for c in calls))
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],env['CUDA_VISIBLE_DEVICES'])

    def test_partial_replica_startup_failure_cleans_existing_engines(self):
        client=self.client(True)
        first=Mock()
        env={'WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0','CUDA_VISIBLE_DEVICES':'0,1,2,3',
             'KERNEL_RL_SAMPLER_GPUS':'0,1,2,3','KERNEL_RL_SAMPLER_DP':'2'}
        with patch.dict(os.environ,env),patch('arl.models.sampling.SamplingEngine',side_effect=[first,RuntimeError('replica failed')]):
            with self.assertRaisesRegex(RuntimeError,'replica failed'):client._ensure_sampling_engine()
            first.shutdown.assert_called_once()
            self.assertIsNone(client._sampling_engine)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],env['CUDA_VISIBLE_DEVICES'])

    def test_legacy_rank_local_inference_keeps_tp_selection(self):
        client=self.client(True)
        env={k:v for k,v in os.environ.items() if k!='KERNEL_RL_SAMPLER_GPUS'}
        env.update(WORLD_SIZE='1',LOCAL_RANK='0',CUDA_VISIBLE_DEVICES='13,14',KERNEL_RL_SAMPLER_TP='2')
        with patch.dict(os.environ,env,clear=True),patch('arl.models.sampling.SamplingEngine',return_value=Mock()) as build:
            client._ensure_sampling_engine()
            self.assertEqual(build.call_count,1)
            self.assertEqual(build.call_args.kwargs['tensor_parallel_size'],2)
