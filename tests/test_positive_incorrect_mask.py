"""CPU tests: GRPO correctness gating after whitening, and rollout alignment."""
import asyncio
import unittest

import torch

from arl.models.types import Datum, ModelInput, TensorData
from arl.algorithms.loss import ppo_surrogate
from arl.algorithms.ppo_data import finalize_ppo_data


def datum(n=2):
    return Datum(model_input=ModelInput.from_ints([1] * (n + 1)), loss_fn_inputs={
        'mask': TensorData.from_torch(torch.tensor([0.] + [1.] * n)),
        'target_tokens': TensorData.from_torch(torch.ones(n + 1, dtype=torch.long)),
        'logprobs': TensorData.from_torch(torch.zeros(n + 1)),
    })


def run(rewards, labels, enabled=True, lengths=None, **kw):
    data = [datum(n) for n in (lengths or [2] * len(rewards))]
    opts = dict(groups=[0] * len(data), whiten=False, group_adv_std=False,
                length_weighted_baseline=False, mask_positive_incorrect=enabled,
                response_correctness=labels)
    opts.update(kw)
    return finalize_ppo_data(data, rewards, **opts)


class MaskTests(unittest.TestCase):
    def test_gates_positive_wrong_only_and_gradient(self):
        rewards=[0., 2., 3., 4.]
        labels=[False, False, True, False]
        baseline, _=run(rewards, labels, enabled=False)
        result, metrics=run(rewards, labels)
        for i in range(3):
            self.assertTrue(torch.equal(result[i].loss_fn_inputs['advantages'].to_torch(),
                                        baseline[i].loss_fn_inputs['advantages'].to_torch()))
        a=result[3].loss_fn_inputs['advantages'].to_torch()
        self.assertEqual(a.count_nonzero(), 0)
        self.assertEqual(result[3].loss_fn_inputs['mask'].to_torch().count_nonzero(), 0)
        lp=torch.zeros_like(a, requires_grad=True)
        ppo_surrogate(lp, torch.zeros_like(a), a).sum().backward()
        self.assertEqual(lp.grad.count_nonzero(), 0)
        self.assertEqual(metrics['ppo/positive_incorrect_masked_frac'], .25)

    def test_whitening_can_flip_negative_to_positive(self):
        # Raw middle advantage = -1; long negative response shifts the token mean.
        result,_=run([0.,2.,7.], [True,False,True], lengths=[100,2,2],
                     whiten=True, whiten_std=False)
        self.assertEqual(result[1].loss_fn_inputs['advantages'].to_torch().count_nonzero(), 0)

    def test_whitening_can_flip_positive_to_negative(self):
        result,_=run([0.,5.,7.], [True,False,True], lengths=[2,2,100],
                     whiten=True, whiten_std=False)
        self.assertLess(result[1].loss_fn_inputs['advantages'].to_torch()[1], 0)
        self.assertEqual(result[1].loss_fn_inputs['mask'].to_torch().sum(), 2)

    def test_unknown_zero_and_disabled(self):
        result,_=run([0.,1.], [False,None])
        self.assertGreater(result[1].loss_fn_inputs['advantages'].to_torch()[1], 0)
        result,_=run([1.,1.], [False,False])
        self.assertEqual(result[0].loss_fn_inputs['mask'].to_torch().sum(), 2)
        result,metrics=run([0.,1.], [False,False], enabled=False)
        self.assertGreater(result[1].loss_fn_inputs['advantages'].to_torch()[1], 0)
        self.assertNotIn('ppo/positive_incorrect_masked_frac', metrics)

    def test_tailrl_and_shared_trajectory_advantage(self):
        result,_=run([0.,0.,2.,2.], [False,False,False,True],
                     units=[0,0,1,1], adv_estimator='tailrl')
        self.assertEqual(result[2].loss_fn_inputs['advantages'].to_torch().count_nonzero(), 0)
        self.assertGreater(result[3].loss_fn_inputs['advantages'].to_torch()[1], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): run([0.,1.], [False])
        with self.assertRaises(ValueError): run([0.,1.], [False,False], groups=None)

    def test_negative_correct_gate_and_gradient(self):
        rewards = [0., 1., 2., 4.]
        labels = [True, False, True, False]
        baseline, default_metrics = run(rewards, labels, enabled=False)
        self.assertLess(baseline[0].loss_fn_inputs['advantages'].to_torch()[1], 0)
        self.assertNotIn('ppo/negative_correct_masked_frac', default_metrics)
        result, metrics = run(rewards, labels, enabled=False, mask_negative_correct=True)
        for i in range(1, 4):
            for key in ('advantages', 'mask', 'returns'):
                self.assertTrue(torch.equal(result[i].loss_fn_inputs[key].to_torch(),
                                            baseline[i].loss_fn_inputs[key].to_torch()))
        a = result[0].loss_fn_inputs['advantages'].to_torch()
        self.assertEqual(a.count_nonzero(), 0)
        self.assertEqual(result[0].loss_fn_inputs['mask'].to_torch().count_nonzero(), 0)
        lp = torch.zeros_like(a, requires_grad=True)
        ppo_surrogate(lp, torch.zeros_like(a), a).sum().backward()
        self.assertEqual(lp.grad.count_nonzero(), 0)
        self.assertEqual(metrics['ppo/negative_correct_masked_frac'], .25)
        self.assertEqual(metrics['ppo/negative_correct_masked_token_frac'], .25)

    def test_both_gates_unknown_and_zero(self):
        result, metrics = run([0., 0., 1., 2.], [True, None, True, False],
                              mask_negative_correct=True)
        for i in (0, 3):
            self.assertEqual(result[i].loss_fn_inputs['mask'].to_torch().sum(), 0)
        self.assertLess(result[1].loss_fn_inputs['advantages'].to_torch()[1], 0)
        self.assertGreater(result[2].loss_fn_inputs['advantages'].to_torch()[1], 0)
        self.assertEqual(metrics['ppo/negative_correct_masked_frac'], .25)
        self.assertEqual(metrics['ppo/positive_incorrect_masked_frac'], .25)
        result, _ = run([1., 1.], [True, True], mask_negative_correct=True)
        self.assertEqual(result[0].loss_fn_inputs['mask'].to_torch().sum(), 2)

    def test_negative_gate_uses_post_whitening_sign(self):
        for rewards, lengths, blocked in (([0., 5., 7.], [2, 2, 100], True),
                                          ([0., 2., 7.], [100, 2, 2], False)):
            result, _ = run(rewards, [False, True, False], enabled=False,
                            mask_negative_correct=True, lengths=lengths,
                            whiten=True, whiten_std=False)
            self.assertEqual(result[1].loss_fn_inputs['mask'].to_torch().sum(),
                             0 if blocked else 2)

    def test_negative_gate_invalid_inputs(self):
        for labels, kw in (([True], {}), (None, {}), ([True, False], {'groups': None})):
            with self.assertRaisesRegex(ValueError, 'mask_negative_correct requires'):
                run([0., 1.], labels, enabled=False, mask_negative_correct=True, **kw)

if __name__ == '__main__':
    unittest.main()
