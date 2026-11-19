"""Reject invalid on-policy traces and never learn from sampler outages."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from arl.agent.rollout import RolloutParams, run_agent_rollout
from arl.algorithms.rollout_data import validate_rollout, validate_sample
from arl.models.types import SampledSequence
from test_swe_env import Source, Renderer, Sampler, ce, record


class TraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_trace_is_dropped_before_any_tool_runs(self):
        for lp,mask in ((None,None),([],None),([float('nan')],None),([-.5],[.5])):
            source=Source()
            class InvalidSampler:
                async def submit_and_poll(self, requests, **kwargs):
                    await asyncio.sleep(0)
                    return [(r.extras['req_id'],SampledSequence([10],lp,'stop',mask)) for r in requests]
            result=await run_agent_rollout([ce.SweTask.from_record(record())],source,InvalidSampler(),Renderer(),
                RolloutParams(gen_cap=10,prompt_budget=100,group_size=2))
            self.assertEqual(result.datums,[])
            self.assertTrue(all(t.dropped=='invalid_trace' for t in result.trajs))
            self.assertTrue(all(env.workspace.closed and not env.workspace.commands for env in source.envs))

    async def test_failure_after_one_turn_discards_earlier_segments_and_grade(self):
        class FailingSampler(Sampler):
            async def submit_and_poll(self, requests, **kwargs):
                good=await super().submit_and_poll(requests,**kwargs)
                return [(rid,seq if rid.endswith(':0') else None) for rid,seq in good]
        result=await run_agent_rollout([ce.SweTask.from_record(record())],Source(),FailingSampler(),Renderer(),
            RolloutParams(gen_cap=10,prompt_budget=100,group_size=2))
        self.assertEqual(result.datums,[])
        self.assertEqual(result.metrics['rollout/n_trajs'],0)
        self.assertTrue(all(t.dropped=='sampling_failed' for t in result.trajs))

    async def test_algorithm_boundary_rejects_broken_groups_and_arrays(self):
        result=await run_agent_rollout([ce.SweTask.from_record(record())],Source(),Sampler(),Renderer(),
            RolloutParams(gen_cap=10,prompt_budget=100,group_size=2))
        validate_rollout(result)
        with self.assertRaisesRegex(ValueError,'alignment'):
            validate_rollout(replace(result,returns=[]))
        with self.assertRaisesRegex(ValueError,'group'):
            validate_rollout(replace(result,groups=[(99,)]*len(result.groups)))
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            validate_rollout(replace(result,returns=[.4,*result.returns[1:]]))
