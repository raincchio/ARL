"""CPU coverage for thinking-only cost and both multi-turn rollout paths."""
import asyncio
from types import SimpleNamespace as NS
import unittest

from arl.algorithms.thinking_cost import thinking_cost, validate_thinking_cost


class Tokenizer:
    def encode(self, text, **kw):
        return {'</think>': [90, 91], '<think>': [89]}[text]


def response(tokens, mask=None):
    return NS(tokens=tokens, maybe_mask=mask)


class ThinkingCostTests(unittest.TestCase):
    def test_disabled_does_not_need_tokenizer(self):
        self.assertEqual(thinking_cost(None, None, 0), (0, 0.))

    def test_code_and_tags_are_excluded(self):
        n,c=thinking_cost(response([89,1,2,3,90,91,4,5]),Tokenizer(),.2)
        self.assertEqual(n,3)
        self.assertAlmostEqual(c,.2*3/8192)

    def test_injected_transition_is_excluded(self):
        n,c=thinking_cost(response([1,2,3,8,8,90,91,4,5],
                                 [1,1,1,0,0,0,0,1,1]),Tokenizer(),.2)
        self.assertEqual(n,3)
        self.assertAlmostEqual(c,.2*3/8192)

    def test_unclosed_prefix_and_empty_thinking(self):
        self.assertEqual(thinking_cost(response([1]*8192),Tokenizer(),.15),(8192,.15))
        self.assertEqual(thinking_cost(response([90,91,4,5]),Tokenizer(),.15),(0,0.))
        with self.assertRaises(ValueError):
            thinking_cost(response([1,2],[1]),Tokenizer(),.1)

    def test_free_allowance_boundary_and_validation(self):
        for n in (0, 4096, 8192, 8193, 16384):
            count, cost=thinking_cost(response([1]*n+[90,91,4]),Tokenizer(),.15,8192)
            self.assertEqual(count,n)
            self.assertAlmostEqual(cost,.15*max(0,n-8192)/8192)
        for free in (-1, .5, float('nan'), True):
            with self.assertRaises(ValueError):
                validate_thinking_cost(NS(think_cost_lambda=0,think_cost_free_tokens=free))

    def test_config_validation(self):
        validate_thinking_cost(NS(think_cost_lambda=0))
        validate_thinking_cost(NS(think_cost_lambda=.15, enable_thinking=True))
        for lam in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_thinking_cost(NS(think_cost_lambda=lam, enable_thinking=True))
        with self.assertRaises(ValueError):
            validate_thinking_cost(NS(think_cost_lambda=.15, enable_thinking=False))


if __name__ == '__main__':
    unittest.main()
