"""First single-rank NCCL sync must initialize the lazily-created sampler."""
import unittest
from unittest.mock import AsyncMock, Mock, patch

from arl.models.training_client import LocalTrainingClient


class SamplerStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampler_exists_before_initial_weight_sync(self):
        client = object.__new__(LocalTrainingClient)
        client._sync_counter = 0
        client._inference_only = False
        client._nccl_sync = True
        client._sampling_engine = None
        engine = Mock()
        def ensure():
            client._sampling_engine = engine
            return engine
        client._ensure_sampling_engine = Mock(side_effect=ensure)
        async def sync():
            self.assertIs(client._sampling_engine, engine)
        client._sync_to_sampler_nccl = AsyncMock(side_effect=sync)
        with patch('arl.models.dist_sampling.DistributedSamplingClient', return_value='client'):
            self.assertEqual(await client.save_weights_and_get_sampling_client_async(), 'client')
        client._sync_to_sampler_nccl.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
