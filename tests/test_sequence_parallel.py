"""Packed input and autograd invariants for the SP training path."""
import asyncio
import math
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arl.models import sequence_parallel as sp
from script import swe10


def check_shards(rank, address):
    os.environ.update(GLOO_SOCKET_IFNAME='lo', WORLD_SIZE='2', RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=address, rank=rank, world_size=2)
    try:
        # Rank 1 owns both examples. Padding is kept out of both examples.
        pack = sp.Pack(1, [0, 1], [3, 2], (1, 1))
        per = []
        if rank == 1:
            for start, length in ((4, 3), (8, 2)):
                ids = torch.arange(start, start + length)
                per.append((ids, ids+1, None, None, length, None, None, None,
                            ids.to(torch.uint8).view(-1, 1, 1)))
        with patch.object(sp.router_replay, 'begin') as begin:
            args, targets, slots = sp.prepare(pack, per, 'cpu', 0, 0, True)
            assert slots == [(0, 0), (0, 3)]
            assert args['cu_seq_lens_q'].tolist() == [0, 3, 5, 6]
            assert args['position_ids'].tolist() == ([[0, 1, 2]], [[0, 1, 0]])[rank]
            assert targets.tolist() == ([5, 6, 7], [9, 10, 0])[rank]
            assert begin.call_args.args[0].flatten().tolist() == ([4, 5, 6], [8, 9, -1])[rank]
        # Cross-rank causal history, then a loss on the source rank only.
        local = (torch.arange(6).view(3, 2) + 6*rank).float().requires_grad_()
        full = sp._gather_positions(local[None])
        result = sp._scatter_positions(full.cumsum(dim=1))[0]
        actual = sp.gather_output(result, 5)
        ref = torch.arange(12).view(6, 2).float().requires_grad_()
        expected = ref.cumsum(0)[:5]
        torch.testing.assert_close(actual, expected)
        (actual.sum() if rank == 1 else actual[:0].sum()).backward()
        expected.sum().backward()
        torch.testing.assert_close(local.grad, ref.grad.chunk(2)[rank])
        check_owner_ppo(rank)
    finally:
        dist.destroy_process_group()


def check_owner_ppo(rank):
    from arl.models.training_client import LocalTrainingClient
    from arl.models.types import Datum, ModelInput, TensorData

    # The two-token example is owned by rank 1. Rank 0 must join the same
    # backward even though its own forward_backward_async receives an empty list.
    table = torch.arange(16, dtype=torch.float32).reshape(4, 4).requires_grad_()
    engine = Mock(side_effect=lambda input_ids, **kw: SimpleNamespace(logits=table[input_ids]))
    engine.backward.side_effect = lambda loss: loss.backward()
    client = object.__new__(LocalTrainingClient)
    client._inference_only = False
    client._sp16 = True
    client._has_value_head = False
    client._const_len_pinned = True
    client._dr_grpo_const_len = 2
    client._tokenizer = SimpleNamespace(pad_token_id=0)
    client.engine, client.device = engine, torch.device('cpu')
    ids, targets = torch.tensor([1, 2]), torch.tensor([2, 3])
    expected = table[ids].log_softmax(-1).gather(1, targets[:, None]).flatten()
    inputs = dict(target_tokens=targets, advantages=torch.ones(2), logprobs=expected.detach())
    datum = Datum(ModelInput.from_ints(ids.tolist()), {k: TensorData.from_torch(v) for k, v in inputs.items()})
    with patch.dict(os.environ, {'KERNEL_RL_HEAD_CHUNKS': '0', 'KERNEL_RL_ROUTER_REPLAY': '0'}):
        output = asyncio.run(client.forward_backward_async([datum] if rank == 1 else [])).result()
    if rank == 1:
        torch.testing.assert_close(output.loss_fn_outputs[0]['logprobs'].to_torch(), expected)
        assert math.isclose(output.metrics['loss'], -1., abs_tol=1e-6), output.metrics
    else:
        assert not output.loss_fn_outputs and output.metrics['loss'] == 0
    dist.all_reduce(table.grad)
    torch.testing.assert_close(table.grad, torch.autograd.grad(-expected.mean(), table)[0])


class SequenceParallelTests(unittest.TestCase):
    def test_owner_loss_and_packed_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(check_shards, args=(Path(directory, 'store').as_uri(),), nprocs=2)

    def test_training_only_switch(self):
        cards = ','.join(map(str, range(16)))
        with patch.dict(os.environ, {'KERNEL_RL_SP16': '0'}):
            parser = swe10.parser()
            args = parser.parse_args(['--train-gpus', cards, '--sampler-gpus', cards])
            self.assertEqual(swe10.runtime_env(args, True)['KERNEL_RL_SP16'], '0')
            args = parser.parse_args(swe10.normalize_key_values(['SP16=1']), namespace=args)
            self.assertEqual(swe10.runtime_env(args, True)['KERNEL_RL_SP16'], '1')
            self.assertEqual(swe10.runtime_env(args, False)['KERNEL_RL_SP16'], '0')
            args = parser.parse_args(['--no-sp16'], namespace=args)
            self.assertFalse(args.sp16)
        with self.assertRaisesRegex(ValueError, '16 training ranks'):
            sp.validate_model({}, 8)
