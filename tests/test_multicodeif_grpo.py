"""On-policy self-repair, official scoring and real CPU GRPO optimizer updates."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from arl.agent.rollout import RolloutParams
from arl.algorithms.rollout_data import validate_rollout
from arl.algorithms.ppo_data import finalize_ppo_data
from arl.algorithms.loss import ppo_surrogate
from arl.envs.instruction_following.multicodeif import MultiCodeIFSourceConfig, MultiCodeIFTaskSource
from arl.harnesses import get_harness, resolve_pair, HarnessCompatibilityError
from arl.models.types import SampledSequence, TensorData, ForwardBackwardOutput
from arl.benchmarks import multicodeif as mc
from test_swe_env import Renderer, Sampler

BAD = 'class database_connector:\n    def ExecuteQuery(self): pass'
GOOD = 'class DatabaseConnector:\n    def execute_query(self): pass'


class CodeRenderer(Renderer):
    def decode(self, ids): return GOOD if ids[0] == 11 else BAD


class RepairSampler(Sampler):
    async def submit_and_poll(self, requests, **kwargs):
        await asyncio.sleep(0)
        out = []
        for request in requests:
            tid, turn = map(int, request.extras['req_id'].split(':'))
            token = 11 if tid == 0 and turn else 10
            out.append((request.extras['req_id'], SampledSequence([token], [-.5], 'stop')))
        return out


@unittest.skipUnless((mc.CACHE/'MultiCodeIF/.git').exists() and (mc.CACHE/'multicodeif-deps/tree_sitter').exists(),
                     'run script/setup_multicodeif.sh')
class OfficialLocalTests(unittest.IsolatedAsyncioTestCase):
    def source(self, **kwargs):
        return MultiCodeIFTaskSource(MultiCodeIFSourceConfig(
            constraints=['code_convention_naming_convention'], languages=['python'], limit=1, **kwargs))

    async def test_two_round_official_feedback_to_trajectory_grpo(self):
        source = self.source(max_rounds=2)
        try:
            await source.initialize()
            self.assertTrue((await source.check())['ready'])
            result = await get_harness('multicodeif_self_repair').run(tasks=source.train_tasks(), source=source,
                sampling_client=RepairSampler(), renderer=CodeRenderer(),
                params=RolloutParams(gen_cap=10, prompt_budget=100, group_size=2),
                operation='train', env_name='instruction_following.multicodeif', log_turns=True)
            validate_rollout(result)
            self.assertEqual(result.returns, [1., 1., 0., 0.])
            self.assertEqual(result.correctness, [True, True, False, False])
            self.assertEqual(result.units, [(0,0), (0,0), (1,0), (1,0)])
            data, _ = finalize_ppo_data(result.datums, result.returns, groups=result.groups, units=result.units,
                whiten=False, group_adv_std=False, length_weighted_baseline=False)
            self.assertTrue((data[0].loss_fn_inputs['advantages'].to_torch()>0).any())
            self.assertTrue((data[-1].loss_fn_inputs['advantages'].to_torch()<0).any())
        finally: await source.close()

    async def test_unavailable_grading_and_truncation_never_enter_training(self):
        source = self.source(max_rounds=2)
        try:
            await source.initialize()
            env = source.make_env(source.train_tasks()[0], 'privacy'); await env.reset()
            self.assertNotIn('python;class:PascalCase', json.dumps(env.messages()))
            self.assertFalse((await env.step(BAD)).done)
            self.assertIn('python;class:PascalCase', json.dumps(env.messages()))
            env = source.make_env(source.train_tasks()[0], 'failure'); await env.reset()
            source.score = AsyncMock(return_value={'task_id': env.task.task_id, 'status': 'incomplete', 'passed': None,
                'checks': [{'status': 'unavailable', 'passed': None}]})
            out = await env.step(BAD)
            self.assertTrue(out.infra_error)
            self.assertIsNone(out.metrics['correct'])
            env = source.make_env(source.train_tasks()[0], 'truncated'); await env.reset()
            out = await env.step(BAD, truncated=True)
            self.assertTrue(out.done)
            self.assertEqual(out.reward, 0)
            self.assertFalse(out.infra_error)
        finally: await source.close()

    async def test_local_collection_exports_tokens_not_api_text(self):
        from arl.experiments.config import from_dict
        from arl.experiments.runner import run_phase
        from arl.experiments.artifacts import validate_collection
        source = self.source(max_rounds=2)
        with tempfile.TemporaryDirectory() as directory:
            plan = from_dict({'models': {'p': {'name': 'cpu-test'}}, 'collect': {'env': 'multicodeif',
                'model': 'p', 'output_dir': 'collect', 'options': {'batch_size': 1, 'group_size': 2,
                    'max_tokens': 10, 'prompt_budget_tokens': 100}}}, base_dir=directory).plan('collect')
            with patch('arl.harnesses.registry.HarnessSpec.make_renderer', return_value=CodeRenderer()):
                summary = await asyncio.to_thread(run_phase, plan, native_runtime={
                    'source': source, 'sampling_client': RepairSampler(), 'tokenizer': object()})
            self.assertEqual(summary['pass_rate'], .5)
            check = validate_collection(directory+'/collect', env='multicodeif', harness='multicodeif_self_repair', require_training_trace=True)
            self.assertEqual(check['datums_validated'], 4)

    async def test_multilevel_chains_stay_in_one_split_and_overlap_is_rejected(self):
        source = MultiCodeIFTaskSource(MultiCodeIFSourceConfig(suite='multi_level'))
        try:
            await source.initialize()
            group = lambda t: (t.file, t.checks[0]['id'])
            self.assertFalse({group(t) for t in source.train_tasks()} & {group(t) for t in source.test_tasks()})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'ids.json'
                path.write_text(json.dumps([source.train_tasks()[0].task_id]))
                overlap = MultiCodeIFTaskSource(MultiCodeIFSourceConfig(suite='multi_level', test_set=str(path)))
                with self.assertRaisesRegex(ValueError, 'overlap'): await overlap.initialize()
                await overlap.close()
        finally: await source.close()

    async def test_actual_grpo_driver_updates_cpu_weights_and_resamples_updated_policy(self):
        from arl.grpo import GRPOConfig, run_grpo
        client = CPUTrainingClient()
        source = self.source(max_rounds=1)
        service = SimpleNamespace(create_training_client_async=AsyncMock(return_value=client))
        with tempfile.TemporaryDirectory() as directory:
            cfg = GRPOConfig(model_name='cpu-policy', log_path=directory, env='multicodeif',
                multicodeif_constraints=['code_convention_naming_convention'], multicodeif_languages=['python'],
                multicodeif_max_rounds=1, task_limit=1, batch_size=1, group_size=8, num_epochs=2,
                learning_rate=.05, num_substeps=1, max_tokens=10, prompt_budget_tokens=100,
                eval_at_start=False, eval_every=0, save_every=0, whiten_advantages=False)
            with patch('arl.grpo.setup_environment'), patch('arl.grpo.ml_log.setup_logging', return_value=MagicMock()), \
                 patch('arl.envs.make_task_source_from_config', return_value=source), \
                 patch('arl.grpo.tinker.ServiceClient', return_value=service), \
                 patch('arl.grpo.sampling_window', return_value=110), \
                 patch('arl.harnesses.registry.HarnessSpec.make_renderer', return_value=CodeRenderer()), \
                 patch('arl.grpo.checkpoint_utils.save_checkpoint_async', new=AsyncMock()), \
                 patch('arl.grpo.dist_utils.destroy'), patch('arl.grpo.dist_utils.barrier'):
                await run_grpo(cfg)
        self.assertEqual(client.updates, 2)
        self.assertEqual(len(client.snapshots), 3)  # initial + after both real Adam steps
        self.assertFalse(torch.equal(client.snapshots[0], client.snapshots[1]))
        self.assertTrue(torch.equal(client.snapshots[-1], client.logits.detach()))
        self.assertGreater(client.logits[11].item(), client.snapshots[0][11].item())
        self.assertEqual(client.sampled_versions, [0]*8+[1]*8)
        self.assertTrue(client.closed)


class Future:
    def __init__(self, value): self.value=value
    async def result_async(self): return self.value


class CPUTrainingClient:
    """Small categorical policy; actual autograd/Adam and production PPO loss.

    The production GRPO driver, advantage calculation, train_step and weight
    sync are unpatched. Only model/service hardware is replaced for CPU tests.
    """
    def __init__(self):
        initial=torch.full((128,), -20.)
        initial[10:13]=0
        self.logits=torch.nn.Parameter(initial)
        self.optimizer=torch.optim.Adam([self.logits],lr=.05)
        self.snapshots=[]; self.sampled_versions=[]; self.updates=0; self.closed=False
        self.generator=torch.Generator().manual_seed(7)

    def get_tokenizer(self): return object()
    def shutdown(self): self.closed=True

    async def save_weights_and_get_sampling_client_async(self):
        snapshot=self.logits.detach().clone(); self.snapshots.append(snapshot)
        version=len(self.snapshots)-1
        owner=self
        class Sampler:
            async def submit_and_poll(self, requests, **kwargs):
                await asyncio.sleep(0)
                result=[]
                logprobs=torch.log_softmax(snapshot,dim=0)
                for request in requests:
                    token=int(torch.multinomial(logprobs.exp(),1,generator=owner.generator))
                    owner.sampled_versions.append(version)
                    result.append((request.extras['req_id'], SampledSequence([token],[logprobs[token].item()],'stop')))
                return result
        return Sampler()

    async def forward_backward_async(self, data, *, loss_fn, loss_fn_config):
        self.optimizer.zero_grad()
        losses=[]; outputs=[]; active=0
        for datum in data:
            values=datum.loss_fn_inputs
            targets=values['target_tokens'].to_torch().long()
            lp=torch.log_softmax(self.logits,dim=0)[targets]
            old=values['logprobs'].to_torch(); adv=values['advantages'].to_torch()
            surrogate=ppo_surrogate(lp, old, adv, loss_fn_config['clip_low_threshold'], loss_fn_config['clip_high_threshold'])
            losses.append(-surrogate.sum()/loss_fn_config['dr_grpo_const_len'])
            active+=int((adv!=0).sum())
            outputs.append({'logprobs':TensorData.from_torch(lp.detach())})
        loss=torch.stack(losses).mean(); loss.backward()
        return Future(ForwardBackwardOutput(outputs, {'loss':loss.item(),'ppo/n_active_tokens':float(active)}))

    async def optim_step_async(self, params, skip=False):
        if not skip:
            for group in self.optimizer.param_groups: group['lr']=params.learning_rate
            self.optimizer.step(); self.updates+=1
        return Future({'learning_rate':params.learning_rate})


class RoutingTests(unittest.TestCase):
    def test_only_local_on_policy_path_can_train(self):
        self.assertEqual(resolve_pair('multicodeif', operation='train', model_backend='local').interface,'agent_messages')
        for backend in ('openai_compatible','offline'):
            pair=resolve_pair('multicodeif',operation='collect',model_backend=backend)
            self.assertEqual(pair.trace_fields,())
            self.assertEqual(pair.interface,'text_constraints')
            with self.assertRaises(HarnessCompatibilityError): resolve_pair('multicodeif',operation='train',model_backend=backend)
