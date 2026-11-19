"""Prefix-cache switches flow to all native phases and actual vLLM arguments."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from arl.models.prefix_cache import engine_args
from script import swe10


class PrefixCacheOptionsTests(unittest.TestCase):
    def test_default_and_explicit_switch_validation(self):
        self.assertEqual(engine_args(),{})
        self.assertEqual(engine_args(False),{'enable_prefix_caching':False,'mamba_cache_mode':'none'})
        options=engine_args(True,'align')
        self.assertTrue(options['enable_prefix_caching'])
        self.assertEqual(options['mamba_cache_mode'],'align')
        self.assertTrue(options['enable_chunked_prefill'])
        for values in ((False,'align'),(None,'align'),(True,'bad'),('True','none')):
            with self.subTest(values=values),self.assertRaises(ValueError):engine_args(*values)

    def test_qwen_all_mode_is_rejected_before_loading_weights(self):
        for model_type in ('qwen3_5','qwen3_5_text','qwen3_5_moe_text'):
            config={'text_config':{'model_type':model_type}}
            with self.subTest(model_type=model_type),self.assertRaisesRegex(ValueError,'use align'):
                engine_args(True,'all',config)
        self.assertEqual(engine_args(True,'all',{'model_type':'mamba'})['mamba_cache_mode'],'all')

    def args(self,root):
        return swe10.parser().parse_args(swe10.normalize_key_values([f'run_dir={root}',
            'enable_prefix_caching=True','mamba_cache_mode=align','prefill_tokens=1056','prefill_batch_tokens=2048']))

    def test_launcher_and_native_plans_keep_the_same_explicit_switches(self):
        from arl.experiments.config import from_dict
        from arl.experiments.runner import prepare_native_config
        root=Path('/tmp/prefix-config');args=self.args(root)
        for name,config in swe10.configurations(args,root).items():
            phase='train' if name=='train' else 'evaluate'
            cfg=prepare_native_config(from_dict(config).plan(phase))
            self.assertTrue(cfg.enable_prefix_caching)
            self.assertEqual(cfg.mamba_cache_mode,'align')
            self.assertEqual(cfg.sampling_prefill_tokens,1056)
        defaults=swe10.parser().parse_args([])
        self.assertIsNone(defaults.enable_prefix_caching)
        self.assertEqual(defaults.mamba_cache_mode,'none')
        disabled=swe10.parser().parse_args(swe10.normalize_key_values(['enable_prefix_caching=False']))
        self.assertFalse(disabled.enable_prefix_caching)

    def test_training_client_keeps_switches_for_every_sampler_replica(self):
        from arl.models.training_client import LocalTrainingClient
        with patch.dict(os.environ,{'WORLD_SIZE':'1'}),patch('arl.models.training_client.ensure_distributed_env'), \
             patch('arl.models.tokenizer.get_tokenizer',return_value=Mock()), \
             patch.object(LocalTrainingClient,'_ensure_sampling_engine',return_value=Mock()):
            client=LocalTrainingClient('policy',inference_only=True,enable_prefix_caching=True,
                mamba_cache_mode='align',sampling_prefill_tokens=1056,sampling_prefill_batch_tokens=2048)
        self.assertTrue(client._sampling_cfg['enable_prefix_caching'])
        self.assertEqual(client._sampling_cfg['mamba_cache_mode'],'align')
        self.assertEqual(client._sampling_cfg['prefill_tokens'],1056)

    def test_actual_vllm_engine_arguments_combine_prefix_and_prefill_controls(self):
        from arl.models.sampling import SamplingEngine
        with patch('vllm.AsyncLLMEngine.from_engine_args',return_value=Mock()) as build, \
             patch('arl.models.sampling.router_replay.enabled',return_value=False):
            engine=SamplingEngine(swe10.MODEL,tensor_parallel_size=4,max_model_len=262144,
                enable_prefix_caching=True,mamba_cache_mode='align',prefill_tokens=1056,prefill_batch_tokens=2048)
            args=build.call_args.args[0]
            self.assertTrue(engine._require_prefix_cache_reset)
        self.assertTrue(args.enable_prefix_caching)
        self.assertEqual(args.mamba_cache_mode,'align')
        self.assertEqual(args.tensor_parallel_size,4)
        self.assertEqual(args.max_num_batched_tokens,2048)
        self.assertEqual(args.additional_config['arl_prefill_budget']['tokens'],1056)

    def test_async_spec_freezes_prefix_configuration(self):
        from arl.experiments.async_swe_eval import publish_spec,evaluation_config
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.args(root)
            (root/'tasks.private.jsonl').write_text('frozen records')
            spec=publish_spec(args,root,swe10.configurations(args,root))
            frozen=json.loads((root/'async_eval/spec.json').read_text())
            for key in ('baseline','evaluation'):
                options=frozen[key]['evaluate']['options']
                self.assertTrue(options['enable_prefix_caching'])
                self.assertEqual(options['mamba_cache_mode'],'align')
            plan,_=evaluation_config(root,'base',spec,root/'attempt')
            self.assertTrue(plan['evaluate']['options']['enable_prefix_caching'])
