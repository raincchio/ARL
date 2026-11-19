"""Rebalance planning and preservation of per-rank response normalization (CPU)."""
import unittest

import torch

from arl.algorithms.loop import _with_resp_weight, plan_rebalance
from arl.models.types import Datum, ModelInput, TensorData


def datum(length, active=True):
    return Datum(ModelInput.from_ints([1] * length), {
        "advantages": TensorData.from_torch(torch.ones(length) if active else torch.zeros(length))})


class RebalanceTests(unittest.TestCase):
    def test_every_datum_assigned_once_and_load_reduced(self):
        costs = [[900, 850, 800], [400, 380, 360], [250, 240, 230], [120, 110, 100]]
        assign = plan_rebalance(costs)
        self.assertEqual(sorted(x for xs in assign for x in xs),
                         [(r, i) for r, xs in enumerate(costs) for i in range(len(xs))])
        self.assertLess(max(sum(costs[r][i] for r, i in xs) for xs in assign),
                        max(map(sum, costs)))
        self.assertEqual(assign, plan_rebalance(costs))

    def test_original_response_weights_preserve_objective_after_moving(self):
        batches = [[datum(100), datum(80), datum(60)], [datum(10), datum(10, False)]]
        weighted = [_with_resp_weight(b) for b in batches]
        assign = plan_rebalance([[d.model_input.length for d in b] for b in batches])
        contributions = [[float(d.loss_fn_inputs['advantages'].to_torch().sum()) for d in b]
                         for b in batches]
        original = sum(sum(xs) / sum(bool(x) for x in xs) for xs in contributions)
        moved = sum(contributions[r][i] * float(weighted[r][i].loss_fn_inputs['resp_weight'].to_torch()[0])
                    for xs in assign for r, i in xs)
        self.assertAlmostEqual(original, moved)
        naive = sum(sum(contributions[r][i] for r, i in xs) /
                    max(1, sum(bool(contributions[r][i]) for r, i in xs)) for xs in assign)
        self.assertNotAlmostEqual(original, naive)

    def test_empty_and_zero_signal_ranks(self):
        self.assertEqual(plan_rebalance([[], []]), [[], []])
        self.assertEqual(_with_resp_weight([]), [])
        d = _with_resp_weight([datum(3, False)])[0]
        self.assertEqual(d.loss_fn_inputs['resp_weight'].to_torch().item(), 1.)


if __name__ == '__main__':
    unittest.main()
