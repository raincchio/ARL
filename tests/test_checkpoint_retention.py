"""Checkpoint retention and publication tests without GPUs or DeepSpeed."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from arl.models.service import LocalServiceClient
from arl.utils import checkpoint_utils as checkpoints


class FakeTrainer:
    def __init__(self):
        self.pending = None
        self.fail_policy = False

    def wait_pending_save(self):
        if self.pending:
            callback, self.pending = self.pending, None
            callback()

    def save_optimizer_state(self, out_dir):
        path = Path(out_dir)
        (path / "ds_state").mkdir(exist_ok=True)
        (path / "ds_state" / "optimizer.pt").write_text("optimizer")
        (path / "ds_state" / "value_head.pt").write_text("critic + Adam")
        (path / "latest").write_text("ds_state")

    def save_policy_weights(self, out_dir, background=False, on_done=None):
        if self.fail_policy:
            raise OSError("disk write failed")
        path = Path(out_dir)
        path.mkdir(exist_ok=True)
        (path / "config.json").write_text("{}")
        (path.parent / "value_head.pt").write_text("critic weights")
        if background:
            self.pending = on_done


class CheckpointRetentionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rank = patch.dict(os.environ, {"RANK": "0"})
        self.rank.start()
        self.addCleanup(self.rank.stop)
        self.trainer = FakeTrainer()

    def save(self, name, batch, **kwargs):
        return asyncio.run(checkpoints.save_checkpoint_async(
            self.trainer, name, self.tmp.name, {"batch": batch}, **kwargs))

    def test_only_latest_optimizer_survives_including_final(self):
        first = self.save("000005", 5)
        second = self.save("000010", 10)
        final = self.save("final", 10)
        for saved in (first, second):
            root = Path(saved["state_path"])
            self.assertFalse((root / "ds_state").exists())
            self.assertFalse((root / "latest").exists())
            self.assertTrue((root / "value_head.pt").exists())
            self.assertTrue((Path(saved["sampler_path"]) / "config.json").exists())
        self.assertTrue((Path(final["state_path"]) / "ds_state" / "value_head.pt").is_file())
        records = checkpoints.load_checkpoints_file(self.tmp.name)
        self.assertEqual([r["state_path"] for r in records], ["", "", final["state_path"]])
        self.assertEqual(checkpoints.get_last_checkpoint(self.tmp.name)["name"], "final")

    def test_background_save_keeps_previous_until_published(self):
        first = self.save("000005", 5)
        self.save("000010", 10, background=True)
        self.assertTrue((Path(first["state_path"]) / "ds_state").is_dir())
        self.assertEqual(checkpoints.get_last_checkpoint(self.tmp.name)["batch"], 5)
        self.trainer.wait_pending_save()
        self.assertFalse((Path(first["state_path"]) / "ds_state").exists())
        self.assertEqual(checkpoints.get_last_checkpoint(self.tmp.name)["batch"], 10)

    def test_failed_save_preserves_previous_resume_point(self):
        first = self.save("000005", 5)
        self.trainer.fail_policy = True
        with self.assertRaisesRegex(OSError, "disk write failed"):
            self.save("000010", 10)
        self.assertTrue((Path(first["state_path"]) / "ds_state").is_dir())
        self.assertEqual(checkpoints.get_last_checkpoint(self.tmp.name)["batch"], 5)

    def test_next_save_joins_background_before_saving_optimizer(self):
        self.save("000005", 5, background=True)
        save_optimizer = self.trainer.save_optimizer_state

        def checked_save(path):
            self.assertIsNone(self.trainer.pending)
            self.assertEqual(checkpoints.get_last_checkpoint(self.tmp.name)["batch"], 5)
            save_optimizer(path)

        with patch.object(self.trainer, "save_optimizer_state", side_effect=checked_save):
            self.save("000010", 10)

    def test_sampler_only_save_preserves_optimizer(self):
        self.save("000005", 5)
        self.save("000010", 10, kind="sampler")
        self.assertEqual(checkpoints.get_last_checkpoint(self.tmp.name)["batch"], 5)

    def test_best_retention_keeps_latest_resume_point(self):
        first = self.save("000005", 5, score=10, keep_best_n=1)
        second = self.save("000010", 10, score=5, keep_best_n=1)
        self.save("000015", 15, score=1, keep_best_n=1)
        self.assertTrue(Path(first["sampler_path"]).is_dir())
        self.assertFalse(Path(second["sampler_path"]).exists())
        self.assertEqual(checkpoints.get_last_checkpoint(self.tmp.name)["batch"], 15)

    def test_nonzero_rank_does_not_publish_or_prune(self):
        first = self.save("000005", 5)
        with patch.dict(os.environ, {"RANK": "1"}):
            self.save("000010", 10)
        self.assertTrue((Path(first["state_path"]) / "ds_state").is_dir())
        self.assertEqual(len(checkpoints.load_checkpoints_file(self.tmp.name)), 1)

    def test_resume_loads_optimizer_when_present(self):
        saved = self.save("000005", 5)
        service = LocalServiceClient()
        client = Mock()
        with patch.object(service, "create_training_client_from_state_async",
                          new=AsyncMock(return_value=client)):
            restored = asyncio.run(service.create_training_client_from_state_with_optimizer_async(
                saved["state_path"]))
        self.assertIs(restored, client)
        client.load_state.assert_called_once_with(saved["state_path"])

    def test_weights_only_resume_does_not_load_optimizer(self):
        saved = self.save("000005", 5, kind="sampler")
        service = LocalServiceClient()
        client = Mock()
        with patch.object(service, "create_training_client_from_state_async",
                          new=AsyncMock(return_value=client)):
            asyncio.run(service.create_training_client_from_state_with_optimizer_async(
                str(Path(saved["sampler_path"]).parent)))
        client.load_state.assert_not_called()

    def test_policy_metadata_restores_model_and_warmup_without_model_override(self):
        saved = self.save("000005", 5, kind="sampler")
        policy = Path(saved["sampler_path"])
        (policy / "meta.json").write_text(json.dumps({
            "model_name": "original-model", "policy_optimizer_steps": 4}))
        client = Mock()
        client._has_value_head = False
        client.colocate_bootstrap = AsyncMock()
        with patch("arl.models.training_client.LocalTrainingClient", return_value=client) as factory:
            asyncio.run(LocalServiceClient().create_training_client_from_state_async(str(policy.parent)))
        factory.assert_called_once_with(model_name="original-model")
        client.colocate_bootstrap.assert_awaited_once()
        client.load_policy_weights.assert_called_once_with(str(policy))
        self.assertEqual(client._policy_optimizer_steps, 4)


if __name__ == "__main__":
    unittest.main()
