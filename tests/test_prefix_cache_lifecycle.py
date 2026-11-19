"""No silent stale-policy cache; phase changes clear caches before unmapping."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

from arl.models.sampling import SamplingEngine, MultiSamplingEngine


class PrefixCacheLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def sampler(self,explicit,reset_ok):
        sampler=object.__new__(SamplingEngine)
        sampler._require_prefix_cache_reset=explicit
        sampler.engine=NS(reset_prefix_cache=AsyncMock(return_value=reset_ok))
        return sampler

    async def test_explicit_cache_reset_failure_stops_instead_of_continuing(self):
        sampler=self.sampler(True,False)
        with self.assertRaisesRegex(RuntimeError,'stale policy KV'):
            await sampler.reset_prefix_cache()
        sampler.engine.reset_prefix_cache.assert_awaited_once()
        await self.sampler(True,True).reset_prefix_cache()

    async def test_legacy_unspecified_cache_retains_warning_behavior(self):
        with self.assertLogs('arl.models.sampling',level='WARNING'):
            await self.sampler(False,False).reset_prefix_cache()

    async def test_any_replica_reset_failure_propagates(self):
        good,bad=self.sampler(True,True),self.sampler(True,False)
        pool=MultiSamplingEngine([good,bad])
        with self.assertRaisesRegex(RuntimeError,'stale policy KV'):
            await pool.reset_prefix_cache()
        good.engine.reset_prefix_cache.assert_awaited_once()
        bad.engine.reset_prefix_cache.assert_awaited_once()

    async def test_colocate_all_replicas_pause_clear_unmap_then_map_resume_reset(self):
        from arl.colocate.controller import Colocator
        events=[];engines=[]
        for index in range(4):
            async def pause(*,mode,clear_cache,i=index):
                self.assertEqual(mode,'abort');self.assertTrue(clear_cache)
                events.append(('pause_clear',i))
            async def resume(i=index):events.append(('resume',i))
            async def reset(i=index):events.append(('reset',i))
            async def rpc(fn,args=(),i=index):
                events.append((fn.__name__,i))
                return [(f'GPU-{i}',1024)] if fn.__name__=='kv_sleep' else [True]
            engines.append(NS(n_inflight=0,engine=NS(pause_generation=AsyncMock(side_effect=pause),
                resume_generation=AsyncMock(side_effect=resume),collective_rpc=AsyncMock(side_effect=rpc)),
                reset_prefix_cache=AsyncMock(side_effect=reset)))
        colo=object.__new__(Colocator)
        colo.client=NS(_sampling_engine=NS(engines=engines))
        colo._kv_bytes={}
        await colo._sampler_sleep()
        self.assertEqual(len(colo._kv_bytes),4)
        await colo._sampler_wake()
        positions=lambda kind:[i for i,event in enumerate(events) if event[0]==kind]
        for kind in ('pause_clear','kv_sleep','kv_wake','resume','reset'):
            self.assertEqual(len(positions(kind)),4,events)
        self.assertLess(max(positions('pause_clear')),min(positions('kv_sleep')))
        self.assertLess(max(positions('kv_sleep')),min(positions('kv_wake')))
        self.assertLess(max(positions('kv_wake')),min(positions('resume')))
        self.assertLess(max(positions('resume')),min(positions('reset')))
