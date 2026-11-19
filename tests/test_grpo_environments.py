"""All five real environments/harnesses feed one GRPO policy/optimizer on CPU.

Docker/CudaEval/IFEval services are replaced by deterministic test backends;
MultiCodeIF uses the pinned official checker. Optimizer/loss/sampling sync run.
"""
import gzip
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from arl.grpo import GRPOConfig, run_grpo
from arl.envs.instruction_following.multicodeif import MultiCodeIFTaskSource, MultiCodeIFSourceConfig
from arl.benchmarks import multicodeif as mc
from test_multicodeif_grpo import CPUTrainingClient, CodeRenderer
from test_swe_env import ce as swe, record, Workspace as SweWorkspace, call
from test_programbench import ce as pb, ProgramTask, Workspace as PBWorkspace, score, IDS
from test_mixture import IFSource
from test_agent_rollout import CudaTask, CudaWorkspaceEnv, EnvConfig as CudaConfig, ContentClient


class Source:
    def __init__(self, task, factory): self.task, self.factory, self.closed = task, factory, False
    def train_tasks(self): return [self.task]
    def test_tasks(self): return []
    def task_key(self, task): return task.instance_id
    def make_env(self, task, uid): return self.factory(task)
    async def close(self): self.closed=True


class Renderer(CodeRenderer):
    def __init__(self, env): self.env=env
    def decode(self, tokens):
        good=tokens[0]==11
        if self.env == 'coding.cuda':
            return call('write_file',path='model_new.py',content='GOOD' if good else 'BAD')+call('profile')
        if self.env == 'coding.swe':
            return call('mcp__swe__bash', command='echo '+('GOOD' if good else 'BAD'))
        if self.env == 'coding.programbench':
            return call('bash',command='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT '+('GOOD' if good else 'BAD'))
        if self.env == 'instruction_following.ifeval': return 'ok' if good else 'bad, comma'
        return super().decode(tokens)


@unittest.skipUnless((mc.CACHE/'MultiCodeIF/.git').exists() and (mc.CACHE/'multicodeif-deps/tree_sitter').exists(),
                     'run script/setup_multicodeif.sh')
class AllEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_policy_real_updates_across_every_environment(self):
        cuda_client=ContentClient()
        cuda=Source(CudaTask(instance_id='cuda-cpu',ref_code='class Model: pass',registry='/fixture'),
            lambda task: CudaWorkspaceEnv(task,cuda_client,CudaConfig(max_turns=1)))
        def swe_factory(task):
            workspace=SweWorkspace()
            async def execute(command, **kwargs):
                workspace.resolved='GOOD' in command
                return {'exit_code':0,'output':'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nfixture patch\n'}
            workspace.execute=execute
            return swe.SweWorkspaceEnv(task,workspace,swe.EnvConfig(max_turns=1))
        def pb_factory(task):
            workspace=PBWorkspace()
            async def execute(command, **kwargs):
                workspace.summary=score(1. if 'GOOD' in command else .2)
                return {'exit_code':0,'output':'ok' if command.startswith('git config') else 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'}
            workspace.execute=execute
            return pb.ProgramBenchEnv(task,workspace,swe.EnvConfig(max_turns=1))
        sources={'coding.cuda':cuda, 'coding.swe':Source(swe.SweTask.from_record(record()),swe_factory),
            'coding.programbench':Source(ProgramTask(IDS[0]),pb_factory), 'instruction_following.ifeval':IFSource(),
            'instruction_following.multicodeif':MultiCodeIFTaskSource(MultiCodeIFSourceConfig(
                constraints=['code_convention_naming_convention'],languages=['python'],limit=1,max_rounds=1))}
        members=[{'id':name.split('.')[-1],'env':name,'weight':1.,'options':{'enable_thinking':name != 'instruction_following.ifeval'}} for name in sources]
        client=CPUTrainingClient()
        client.colocate_phase=AsyncMock()
        service=SimpleNamespace(create_training_client_async=AsyncMock(return_value=client))
        renderers=iter([Renderer(name) for name in sources])
        with tempfile.TemporaryDirectory() as directory:
            cfg=GRPOConfig(model_name='shared-cpu-policy',log_path=directory,environments=members,
                batch_size=1,group_size=8,mixture_steps_per_epoch=20,shuffle_seed=1,learning_rate=.05,num_substeps=1,
                max_tokens=10,prompt_budget_tokens=100,eval_at_start=False,eval_every=0,save_every=0,
                whiten_advantages=False)
            with patch.dict(os.environ, {'KERNEL_RL_COLOCATE':'1'}), patch('arl.grpo.setup_environment'), patch('arl.grpo.ml_log.setup_logging',return_value=MagicMock()), \
                 patch('arl.envs.make_task_source_from_config',side_effect=lambda cfg:sources[cfg.env]), \
                 patch('arl.grpo.tinker.ServiceClient',return_value=service), \
                 patch('arl.grpo.sampling_window',return_value=110), \
                 patch('arl.harnesses.registry.HarnessSpec.make_renderer',side_effect=lambda *args,**kwargs:next(renderers)), \
                 patch('arl.grpo.checkpoint_utils.save_checkpoint_async',new=AsyncMock()), \
                 patch('arl.grpo.dist_utils.destroy'), patch('arl.grpo.dist_utils.barrier'):
                await run_grpo(cfg)
            with gzip.open(Path(directory)/'trajectories/rank0.jsonl.gz','rt') as f:
                trajectories=[json.loads(line) for line in f]
            self.assertEqual({t['identity']['env'] for t in trajectories},set(sources))
            self.assertEqual({t['identity']['harness'] for t in trajectories},
                {'arl_agent','claude_code','mini_swe_agent','direct','multicodeif_self_repair'})
            self.assertTrue(all(not t['dropped'] for t in trajectories))
        self.assertEqual([c.args[0] for c in client.colocate_phase.await_args_list], ["sample", "train"]*20)
        self.assertEqual(client.updates,20)
        self.assertEqual(len(client.snapshots),21)
        self.assertTrue(torch.equal(client.logits.detach(),client.snapshots[-1]))
        self.assertGreater(client.logits[11].item(),client.snapshots[0][11].item())
        self.assertEqual(client.sampled_versions,[version for version in range(20) for _ in range(8)])
