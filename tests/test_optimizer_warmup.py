"""Exercise real optimizer-step warmup control flow with a fake engine."""

import asyncio
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from arl.models.service import LocalServiceClient
from arl.models.training_client import LocalTrainingClient
from arl.models.types import AdamParams


class OptimizerWarmupTest(unittest.TestCase):
    def client(self, steps=5, scale=0.5):
        client = object.__new__(LocalTrainingClient)
        client._warmup_steps = steps
        client._warmup_lr_scale = scale
        client._policy_optimizer_steps = 0
        client._has_value_head = False
        client._require_engine = Mock()
        client._grad_has_inf_or_nan = Mock(return_value=False)
        client._save_value_head = Mock()
        client._load_value_head = Mock()
        client.model_name = "test-model"
        client.engine = Mock()
        client.engine.optimizer = SimpleNamespace(param_groups=[{}, {}])
        client.engine.get_global_grad_norm.return_value = 0.1
        client.engine.skipped_steps = 0
        return client

    def step(self, client, lr=1e-6, skip=False):
        async def run():
            future = await client.optim_step_async(AdamParams(learning_rate=lr), skip=skip)
            return await future.result_async()
        return asyncio.run(run())

    def test_fifth_and_sixth_update_across_batch_boundaries(self):
        client = self.client()
        rates = []
        for _batch in range(3):
            for _substep in range(2):
                rates.append(self.step(client)["learning_rate"])
        self.assertEqual(rates, [5e-7] * 5 + [1e-6])
        self.assertEqual(client._policy_optimizer_steps, 6)
        for group in client.engine.optimizer.param_groups:
            self.assertEqual(group["lr"], 1e-6)

    def test_disabled_and_custom_warmup(self):
        self.assertEqual(self.step(self.client(steps=0))["learning_rate"], 1e-6)
        client = self.client(steps=1, scale=0.25)
        self.assertEqual(self.step(client)["learning_rate"], 2.5e-7)
        self.assertEqual(self.step(client)["learning_rate"], 1e-6)

    def test_skips_do_not_consume_warmup(self):
        client = self.client(steps=1)
        self.step(client, skip=True)
        client._grad_has_inf_or_nan.return_value = True
        self.step(client)
        self.assertEqual(client._policy_optimizer_steps, 0)
        client._grad_has_inf_or_nan.return_value = False
        self.assertEqual(self.step(client)["learning_rate"], 5e-7)
        self.assertEqual(self.step(client)["learning_rate"], 1e-6)

    def test_critic_only_steps_do_not_consume_policy_warmup(self):
        client = self.client(steps=1)
        client._has_value_head = True
        client._value_head_step = Mock()
        self.assertEqual(self.step(client, lr=0)["learning_rate"], 0)
        self.assertEqual(client._policy_optimizer_steps, 0)
        client._value_head_step.assert_called_once_with(skip=False)
        self.assertEqual(self.step(client)["learning_rate"], 5e-7)

    def test_optimizer_checkpoint_restores_warmup_progress(self):
        client = self.client()
        for _ in range(4):
            self.step(client)
        with tempfile.TemporaryDirectory() as directory:
            client.save_optimizer_state(directory)
            state = client.engine.save_checkpoint.call_args.kwargs["client_state"]
            restored = self.client()
            restored.engine.load_checkpoint.return_value = (directory, state)
            restored.load_state(directory)
            self.assertEqual(self.step(restored)["learning_rate"], 5e-7)
            self.assertEqual(self.step(restored)["learning_rate"], 1e-6)

    def test_weights_metadata_carries_warmup_progress(self):
        client = self.client()
        self.step(client)
        with tempfile.TemporaryDirectory() as directory:
            client._write_meta(directory)
            meta = LocalServiceClient()._read_meta(directory)
        self.assertEqual(meta["policy_optimizer_steps"], 1)


if __name__ == "__main__":
    unittest.main()
