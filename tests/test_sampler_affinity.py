"""Trajectory-level routing survives load changes, thinking stages and cleanup."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from arl.models.sampling import MultiSamplingEngine, SamplingClient
from arl.models.dist_sampling import DistributedSamplingClient, SampleRequest
from arl.models.types import ModelInput, SamplingParams


def response(tokens,reason='stop'):
    return NS(outputs=[NS(token_ids=tokens,finish_reason=reason,
        logprobs=[{t:NS(logprob=-.5)} for t in tokens])])


class Engine:
    def __init__(self,index):
        self.index=index;self.n_inflight=0;self.calls=[]
        self.block=None;self.cancelled=False
    async def generate(self,prompt,params):
        self.calls.append((list(prompt),params.max_tokens))
        self.n_inflight+=1
        try:
            if self.block is not None:await self.block.wait()
            await asyncio.sleep(0)
            return response([10 if prompt[-1]==2 else 11])
        except asyncio.CancelledError:
            self.cancelled=True;raise
        finally:self.n_inflight-=1


def _distributed_affinity_worker(rank,directory):
    from datetime import timedelta
    import torch.distributed as dist
    os.environ.update(RANK=str(rank),WORLD_SIZE='4',LOCAL_RANK=str(rank),GLOO_SOCKET_IFNAME='lo',CUDA_VISIBLE_DEVICES='')
    dist.init_process_group('gloo',rank=rank,world_size=4,
        init_method='file://'+str(Path(directory)/'gloo'),timeout=timedelta(seconds=30))
    try:
        async def run():
            from test_swe_env import Source,Renderer,ce,record
            from arl.agent.rollout import run_agent_rollout,RolloutParams
            class TracedSource(Source):
                def make_env(self,task,uid):
                    env=super().make_env(task,uid)
                    env.affinity_test_id=rank*4+len(self.envs)-1
                    return env
            class TracedRenderer(Renderer):
                def fit(self,env,*args):return [env.affinity_test_id,len(env.messages())],'',0
            pool=MultiSamplingEngine([Engine(i) for i in range(4)]) if rank==0 else None
            client=DistributedSamplingClient(pool)
            tasks=[ce.SweTask.from_record(record(rank))] if rank<2 else []
            result=await run_agent_rollout(tasks,TracedSource(),client,TracedRenderer(),
                RolloutParams(gen_cap=10,prompt_budget=100,group_size=4,poll_seconds=.001),train=False)
            assert len(result.trajs)==(4 if rank<2 else 0)
            if rank==0:
                seen={i:[] for i in range(8)}
                for engine in pool.engines:
                    for prompt,_ in engine.calls:seen[prompt[0]].append(engine.index)
                assert all(len(ids)==2 and len(set(ids))==1 for ids in seen.values()),seen
                assert [len(e.calls) for e in pool.engines]==[4]*4,seen
                assert not pool._affinity_replicas and not client._inflight and not client._scope_affinities
                Path(directory,'result.json').write_text(json.dumps(seen))
        asyncio.run(run())
    finally:dist.destroy_process_group()


class AffinityTests(unittest.IsolatedAsyncioTestCase):
    def pool(self):return MultiSamplingEngine([Engine(i) for i in range(4)])

    def request(self,scope,trajectory='0',rid='0:0'):
        return SampleRequest([100,2],1,10,1.,extras={'req_id':rid,
            'affinity_scope':scope,'affinity_key':f'{scope}:{trajectory}'})

    async def test_new_trajectories_balance_and_later_turns_ignore_load_changes(self):
        pool=self.pool()
        selected=[pool.select_replica(str(i)) for i in range(8)]
        self.assertEqual([e.index for e in selected],[0,1,2,3,0,1,2,3])
        for i,e in enumerate(selected):
            e.n_inflight=100+i
            self.assertIs(pool.select_replica(str(i)),e)
        pool.release_affinity('0');pool.release_affinity('0')
        self.assertEqual(sum(pool._affinity_load),7)

    async def test_same_named_trajectory_from_different_ranks_is_isolated(self):
        pool=self.pool();client=DistributedSamplingClient(pool)
        scope=client.begin_affinity_scope();req=self.request(scope)
        a=client._with_affinity(req,0);b=client._with_affinity(req,1)
        self.assertNotEqual(a.extras['affinity_key'],b.extras['affinity_key'])
        self.assertNotEqual(a.extras['affinity_key'],req.extras['affinity_key'])
        self.assertIsNot(pool.select_replica(a.extras['affinity_key']),pool.select_replica(b.extras['affinity_key']))
        await client.end_affinity_scope(scope)
        self.assertEqual(pool._affinity_replicas,{})
        other=DistributedSamplingClient(pool)._with_affinity(req,0)
        self.assertNotEqual(a.extras['affinity_key'],other.extras['affinity_key'])

    async def test_two_stage_thinking_stays_on_one_replica_even_without_key(self):
        pool=self.pool();chosen=pool.engines[0]
        async def generate(prompt,params):
            chosen.calls.append((list(prompt),params.max_tokens))
            if len(chosen.calls)==1:
                chosen.n_inflight=100  # Force a second load-based pick to choose another engine.
                return response([1,2],'length')
            return response([3])
        chosen.generate=generate
        for key in (None,'trajectory'):
            chosen.n_inflight=0;chosen.calls=[]
            seq=await SamplingClient(pool).sample_async(ModelInput.from_ints([10,11]),1,
                SamplingParams(max_tokens=10,think_budget=2,think_force_ids=[90,91],think_end_ids=[90,91],affinity_key=key))
            self.assertEqual(len(chosen.calls),2)
            self.assertEqual(seq.sequences[0].tokens,[1,2,90,91,3])
            self.assertTrue(all(not e.calls for e in pool.engines[1:]))

    async def test_scope_cancellation_aborts_only_its_work_and_releases_bindings(self):
        pool=self.pool();client=DistributedSamplingClient(pool)
        scope1,scope2=client.begin_affinity_scope(),client.begin_affinity_scope()
        for engine in pool.engines:engine.block=asyncio.Event()
        client._start([self.request(scope1,rid='1.0:0')],0)
        client._start([self.request(scope2,rid='2.0:0')],0)
        for _ in range(5):await asyncio.sleep(0)
        self.assertEqual(len(pool._affinity_replicas),2)
        await client.end_affinity_scope(scope1)
        self.assertTrue(pool.engines[0].cancelled)
        self.assertEqual(len(pool._affinity_replicas),1)
        self.assertEqual(len(client._inflight),1)
        await client.end_affinity_scope(scope2)
        self.assertFalse(client._inflight)
        self.assertFalse(client._request_scopes)
        self.assertFalse(client._scope_affinities)
        self.assertFalse(pool._affinity_replicas)
        self.assertEqual(pool._affinity_load,[0]*4)

    async def test_real_two_turn_rollout_routes_eight_trajectories_and_cleans_up(self):
        from test_swe_env import Source,Renderer,ce,record
        from arl.agent.rollout import run_agent_rollout,RolloutParams
        class TracedSource(Source):
            def make_env(self,task,uid):
                env=super().make_env(task,uid)
                env.affinity_test_id=len(self.envs)-1
                return env
        class TracedRenderer(Renderer):
            def fit(self,env,*args):return [env.affinity_test_id,len(env.messages())],'',0
        pool=self.pool();client=DistributedSamplingClient(pool)
        result=await run_agent_rollout([ce.SweTask.from_record(record())],TracedSource(),client,TracedRenderer(),
            RolloutParams(gen_cap=10,prompt_budget=100,group_size=8,poll_seconds=.001),train=False)
        seen={i:[] for i in range(8)}
        for engine in pool.engines:
            for prompt,_ in engine.calls:seen[prompt[0]].append(engine.index)
        self.assertEqual(len(result.trajs),8)
        self.assertTrue(all(len(indices)==2 and len(set(indices))==1 for indices in seen.values()),seen)
        self.assertEqual([len(e.calls) for e in pool.engines],[4]*4)
        self.assertFalse(pool._affinity_replicas)
        self.assertFalse(client._scope_affinities)
        self.assertFalse(client._inflight)

    async def test_rollout_exception_releases_scope_without_extra_collectives(self):
        from arl.agent.rollout import run_agent_rollout,RolloutParams
        from test_swe_env import Source,Renderer,ce,record
        class BrokenSource(Source):
            def make_env(self,*args,**kwargs):raise RuntimeError('workspace failed')
        pool=self.pool();client=DistributedSamplingClient(pool)
        with self.assertRaisesRegex(RuntimeError,'workspace failed'):
            await run_agent_rollout([ce.SweTask.from_record(record())],BrokenSource(),client,Renderer(),
                RolloutParams(gen_cap=10,prompt_budget=100,group_size=1),train=False)
        self.assertFalse(client._scope_affinities)
        self.assertFalse(pool._affinity_replicas)


class DistributedAffinityTests(unittest.TestCase):
    def test_four_real_cpu_ranks_including_empty_ranks_keep_ids_isolated(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            context=mp.spawn(_distributed_affinity_worker,args=(directory,),nprocs=4,join=False)
            deadline=time.monotonic()+50
            try:
                while not context.join(timeout=1):
                    if time.monotonic()>deadline:raise TimeoutError('Distributed affinity test timed out')
            finally:
                for process in context.processes:
                    if process.is_alive():process.terminate()
                for process in context.processes:process.join(timeout=5)
            result=json.loads(Path(directory,'result.json').read_text())
            self.assertEqual(len(result),8)
