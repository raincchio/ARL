"""Training-only ARM integration: replay probabilities, routing and colocate phases."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch

from arl.grpo import GRPOConfig
from arl.algorithms.loss import SamplingReplay, gather_target_logprobs
from arl.algorithms.training_options import configure_training_runtime, training_loss_config, validate_training_options, training_phase, thinking_sampling_options
from arl.models import router_replay as rr
from arl.models.fixed_rmsnorm import CONFIG, CONFIG_KEY, engine_args
from arl.models.training_client import LocalTrainingClient
from arl.models.types import Datum, ModelInput, TensorData


class TrainingOptionsTests(unittest.TestCase):
    def cfg(self, **kw):
        return GRPOConfig(model_name='local-policy', log_path='/tmp/test-training-sync', **kw)

    def test_processed_sampler_logprobs_and_trainer_distribution_switch_together(self):
        cfg = self.cfg(sampling_replay=True, router_replay=True, temperature=.7, top_k=5, top_p=.9)
        with patch.dict(os.environ):
            configure_training_runtime(cfg)
            self.assertEqual(os.environ['KERNEL_RL_SAMPLER_LOGPROBS_MODE'], 'processed_logprobs')
            self.assertEqual(os.environ['KERNEL_RL_ROUTER_REPLAY'], '1')
            config = training_loss_config(cfg, 8192)
            self.assertEqual([config[k] for k in ('replay_temperature','replay_top_k','replay_top_p')], [.7,5,.9])
            self.assertEqual(config['dr_grpo_const_len'], 8192)
            configure_training_runtime(self.cfg())
            self.assertEqual(os.environ['KERNEL_RL_SAMPLER_LOGPROBS_MODE'], 'raw_logprobs')
            self.assertEqual(os.environ['KERNEL_RL_ROUTER_REPLAY'], '0')
            self.assertNotIn('replay_temperature', training_loss_config(self.cfg(), 2048))

    def test_invalid_replay_configuration_fails_before_model_load(self):
        for kw in ({'sampling_replay':True,'temperature':0}, {'sampling_replay':True,'top_p':0},
                   {'sampling_replay':True,'loss_fn_config':{'kl_beta':.1}}, {'router_replay':'yes'}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                validate_training_options(self.cfg(**kw))

    def test_thinking_budget_encodes_force_tokens_and_can_be_disabled(self):
        class Tokenizer:
            def encode(self,text,**kw):return [90,91] if text=='</think>' else [8,90,91]
        options=thinking_sampling_options(self.cfg(think_budget=4),Tokenizer())
        self.assertEqual(options,dict(think_budget=4,think_force_ids=[8,90,91],think_end_ids=[90,91]))
        self.assertEqual(thinking_sampling_options(self.cfg(),None),{})
        with self.assertRaises(ValueError):
            thinking_sampling_options(self.cfg(think_budget=4),NS(encode=lambda text,**kw: [1] if text=='</think>' else [2]))

    def test_fixed_rmsnorm_config_selects_supported_models_and_validates_override(self):
        model={'text_config':{'model_type':'qwen3_5_text','hidden_size':5120}}
        with patch.dict(os.environ, {'KERNEL_RL_FIXED_RMSNORM':''}):
            args=engine_args(model)
            self.assertEqual(args['additional_config'][CONFIG_KEY], CONFIG)
            self.assertEqual(args['worker_cls'], 'arl.models.fixed_rmsnorm_worker.FixedRMSNormWorker')
            self.assertEqual(engine_args({}), {})
        with patch.dict(os.environ, {'KERNEL_RL_FIXED_RMSNORM':'0'}):
            self.assertEqual(engine_args(model), {})
        with patch.dict(os.environ, {'KERNEL_RL_FIXED_RMSNORM':'1'}):
            with self.assertRaises(ValueError): engine_args({})

    def test_launcher_gpu_layout_has_no_duplicate_devices_and_requires_subset(self):
        for train,sampler,mode,expected in [('0,1','2,3','0','0,1,2,3'),('0,1','1','1','0,1')]:
            out=subprocess.check_output(['bash','-c','source script/training_runtime.sh; arl_training_gpu_layout "$1" "$2" "$3" && echo "$ARL_TRAIN_VISIBLE_GPUS"', 'test',train,sampler,mode],text=True)
            self.assertEqual(out.strip(),expected)
        for train,sampler,mode in [('0,1','1','0'),('0,1','2','1')]:
            result=subprocess.run(['bash','-c','source script/training_runtime.sh; arl_training_gpu_layout "$1" "$2" "$3"','test',train,sampler,mode],capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def test_replay_logprobs_and_gradients_match_truncated_reference(self):
        torch.manual_seed(17)
        for temp,k,p in [(1.,5,.9),(.7,8,.95),(1.,-1,1.)]:
            logits=(torch.randn(9,20)*2).requires_grad_()
            z=logits.detach().double()/temp
            sorted_z,idx=z.sort(dim=-1)
            if k>0: sorted_z=sorted_z.masked_fill(sorted_z<sorted_z[:,-k,None],-float('inf'))
            remove=sorted_z.softmax(-1).cumsum(-1)<=1-p
            remove[:,-1]=False
            sorted_z=sorted_z.masked_fill(remove,-float('inf'))
            filtered=torch.full_like(z,-float('inf')).scatter(-1,idx,sorted_z)
            target=filtered.argmax(-1)
            actual=gather_target_logprobs(logits,target,SamplingReplay(temp,k,p))
            ref=(logits/temp).masked_fill(~torch.isfinite(filtered),-float('inf')).log_softmax(-1).gather(1,target[:,None]).squeeze(1)
            self.assertTrue(torch.allclose(actual,ref,atol=1e-6))
            ga=torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0]
            gr=torch.autograd.grad(ref.sum(),logits)[0]
            self.assertTrue(torch.allclose(ga,gr,atol=1e-6))


class RouterTests(unittest.TestCase):
    def tearDown(self):
        rr.end();rr.pop_stats()

    def test_forced_routes_keep_selected_expert_gradients_and_mask_missing_rows(self):
        class Router(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.eye(4));self.hidden_dim=4;self.top_k=2
        gate=Router()
        model=NS(named_modules=lambda:[('model.layers.0.mlp.gate',gate)])
        rr.install(model)
        x=torch.tensor([[4.,3.,2.,1.],[1.,2.,3.,4.]],requires_grad=True)
        forced=torch.tensor([[[3,2]],[[-1,-1]]],dtype=torch.int16)
        with rr.replaying(forced):
            _,weight,ids=gate(x)
            self.assertEqual(ids[0].tolist(),[3,2])
            self.assertEqual(ids[1].tolist(),[3,2])
            weight[:,0].sum().backward()
            stats=rr.pop_stats()
        self.assertEqual(stats['ppo/router_replay_frac'],.5)
        self.assertTrue(torch.isfinite(gate.weight.grad).all())
        self.assertGreater(gate.weight.grad.abs().sum(),0)
        with rr.replaying(torch.zeros(1,1,2,dtype=torch.int16)):
            gate(x)
        self.assertEqual(rr.pop_stats()['ppo/router_shape_miss'],1)

    def test_packed_and_padded_route_layout(self):
        a=torch.tensor([[[1,2]],[[3,4]]]);b=torch.tensor([[[5,6]]])
        packed=rr.build_index([a,None,b],[2,1,1],'cpu')
        self.assertEqual(packed[:,0].tolist(),[[1,2],[3,4],[-1,-1],[5,6]])
        padded=rr.build_index([a,b],[2,1],'cpu',pad_to=2)
        self.assertEqual(padded[:,0].tolist(),[[1,2],[3,4],[5,6],[-1,-1]])


class ColocateTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_switch_and_initial_bootstrap(self):
        client=NS(colocate_phase=AsyncMock())
        with patch.dict(os.environ,{'KERNEL_RL_COLOCATE':'0'}):
            await training_phase(client,'train')
            client.colocate_phase.assert_not_awaited()
        with patch.dict(os.environ,{'KERNEL_RL_COLOCATE':'1'}):
            await training_phase(client,'sample'); await training_phase(client,'train')
            self.assertEqual([call.args[0] for call in client.colocate_phase.await_args_list],['sample','train'])
        from arl.colocate.controller import Colocator
        colo=object.__new__(Colocator)
        colo.rank=0;colo.phase='sample';colo._sampler_sleep=AsyncMock()
        with patch('arl.colocate.controller.dist_utils.barrier'):
            await colo.enter_train(); await colo.enter_train()
        colo._sampler_sleep.assert_awaited_once()
        self.assertEqual(colo.phase,'train')

    async def test_two_stage_generation_masks_injected_tokens_and_keeps_final_routes(self):
        from arl.models.sampling import SamplingClient
        from arl.models.types import SamplingParams
        def output(tokens,reason,n_input):
            return NS(outputs=[NS(token_ids=tokens,finish_reason=reason,
                logprobs=[{t:NS(logprob=-.5)} for t in tokens],
                routed_experts=np.ones((n_input,2,1),dtype=np.uint8))])
        engine=NS(generate=AsyncMock(side_effect=[output([1,2],'length',3),output([3,4],'stop',7)]))
        client=SamplingClient(engine)
        seq=await client._sample_with_think_budget([10,11],SamplingParams(max_tokens=10,think_budget=2,
            think_force_ids=[90,91],think_end_ids=[90,91]),lambda n:n)
        self.assertEqual(seq.tokens,[1,2,90,91,3,4])
        self.assertEqual(seq.loss_mask,[1,1,0,0,1,1])
        self.assertTrue(seq.think_forced)
        self.assertEqual(seq.routed_experts.shape,(7,2,1))
        self.assertEqual(engine.generate.await_args_list[1].args,([10,11,1,2,90,91],6))

    async def test_ipc_sync_replaces_cross_gpu_nccl_for_colocated_client(self):
        client=object.__new__(LocalTrainingClient)
        client._sync_counter=0; client._inference_only=False;client._has_value_head=False
        client._colocate=NS(sync_weights=AsyncMock())
        client.engine=NS(module=torch.nn.Linear(2,2))
        client._ensure_sampling_engine=Mock(return_value=object())
        with patch('arl.models.dist_sampling.DistributedSamplingClient',return_value='sampler'):
            self.assertEqual(await client.save_weights_and_get_sampling_client_async(),'sampler')
        self.assertEqual([n for n,_ in client._colocate.sync_weights.await_args.args[0]],['weight','bias'])


class TrainingClientReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_forward_backward_uses_replayed_distribution_on_cpu(self):
        # Exercise LocalTrainingClient, not a replacement loss implementation.
        class LM(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.randn(12,12))
            def forward(self,input_ids,**kw):return NS(logits=self.weight[input_ids])
        class Engine:
            def __init__(self):self.module=LM()
            def __call__(self,**kw):return self.module(**kw)
            def set_gradient_accumulation_boundary(self,*a):pass
            def backward(self,loss):loss.backward()
        torch.manual_seed(3)
        client=object.__new__(LocalTrainingClient)
        client.engine=Engine();client.device=torch.device('cpu');client._require_engine=Mock()
        client._dr_grpo_const_len=4.;client._const_len_pinned=False
        client._tokenizer=NS(pad_token_id=0);client._has_value_head=False
        ids=torch.tensor([1,2,3,4]);target=client.engine.module.weight[ids].argmax(-1)
        old=gather_target_logprobs(client.engine.module.weight[ids],target,SamplingReplay(.8,4,.9)).detach()
        values={k:TensorData.from_torch(v) for k,v in {'target_tokens':target,'logprobs':old,'advantages':torch.ones(4),'mask':torch.ones(4)}.items()}
        data=[Datum(ModelInput.from_ints(ids.tolist()),values)]
        with patch.dict(os.environ,{'KERNEL_RL_PACK_TOKENS':'0','KERNEL_RL_HEAD_CHUNKS':'0','KERNEL_RL_ROUTER_REPLAY':'0'}):
            result=await (await client.forward_backward_async(data,loss_fn_config={'replay_temperature':.8,'replay_top_k':4,'replay_top_p':.9})).result_async()
        self.assertAlmostEqual(result.metrics['ppo/mean_ratio'],1.,places=5)
        self.assertEqual(result.metrics['ppo/replay_outside_frac'],0)
        self.assertTrue(torch.isfinite(client.engine.module.weight.grad).all())
        self.assertGreater(client.engine.module.weight.grad.abs().sum(),0)


class RolloutTrainingMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_reply_free_allowance_and_routing_reach_training_datums(self):
        from test_swe_env import Source, Renderer, ce, record, call
        from test_thinking_cost import Tokenizer
        from arl.agent.rollout import RolloutParams, run_agent_rollout
        from arl.models.types import SampledSequence
        class Render(Renderer):
            tokenizer=Tokenizer()
            def decode(self,ids):
                return call('bash',command='pwd') if ids[-1]==10 else call('submit')
        class Sampler:
            async def submit_and_poll(self, requests, **kw):
                result=[]
                for req in requests:
                    self_test.assertEqual(req.think_budget,4)
                    self_test.assertEqual(req.think_end_ids,[90,91])
                    first=req.extras['req_id'].endswith(':0')
                    ids=([1,2,3,99,90,91,10] if first else [1,90,91,11])
                    mask=([1,1,1,0,0,0,1] if first else [1,0,0,1])
                    routing=np.full((len(req.prompt_ids)+len(ids)-1,2,1),7,dtype=np.uint8)
                    result.append((req.extras['req_id'],SampledSequence(tokens=ids,logprobs=[-.5]*len(ids),routed_experts=routing,loss_mask=mask,think_forced=True)))
                return result
        self_test=self
        result=await run_agent_rollout([ce.SweTask.from_record(record())],Source(),Sampler(),Render(),
            RolloutParams(gen_cap=10,prompt_budget=100,group_size=2,think_cost_lambda=.2,think_cost_free_tokens=2,think_budget=4,think_force_ids=[90,91],think_end_ids=[90,91]))
        self.assertEqual(result.correctness,[True,True,False,False])
        self.assertAlmostEqual(result.returns[0],1-.2/8192)
        self.assertAlmostEqual(result.returns[-1],-.2/8192)
        self.assertAlmostEqual(result.metrics['rollout/think_cost_mean'],.2/8192)
        self.assertEqual(result.metrics['rollout/think_tokens_mean'],4)
        self.assertEqual(result.metrics['rollout/think_forced_rate'],1)
        for datum in result.datums:
            routes=datum.loss_fn_inputs['routed_experts'].to_torch()
            self.assertEqual(tuple(routes.shape),(datum.model_input.length,2,1))
            self.assertTrue((routes==7).all())


if __name__=='__main__':unittest.main()
