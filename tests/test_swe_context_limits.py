"""SWE total-context budgeting with fixed per-response limits; no GPUs."""
import json
from pathlib import Path
import tempfile
import unittest
from script import swe10
from arl.agent.render import AgentRenderer


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ''.join(m['content'] for m in messages)
    def encode(self,text,**kwargs): return list(map(ord,text))
    def decode(self,ids,**kwargs): return ''.join(map(chr,ids))


class History:
    tools=[]
    def __init__(self,text): self.text=text
    def messages(self,aggressiveness):
        return [{'role':'user','content':self.text if aggressiveness == 0 else 'compacted'}]


class ContextLimitTests(unittest.TestCase):
    def args(self,root,*values):
        model=root/'model';model.mkdir(exist_ok=True)
        (model/'config.json').write_text(json.dumps({'text_config':{'max_position_embeddings':262144}}))
        return swe10.parser().parse_args(swe10.normalize_key_values([f'POLICY={model}','max_tokens=40960',*values]))

    def test_default_uses_full_model_window_for_both_harnesses_and_all_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.args(root)
            self.assertEqual(swe10.context_limits(args),(262144,221184))
            for harness in ('claude_code','mini_swe_agent'):
                args.harness=harness
                configs=swe10.configurations(args,root/'run')
                for name in ('train','baseline','evaluate'):
                    options=configs[name]['train' if name=='train' else 'evaluate']['options']
                    self.assertEqual(options['sampling_max_model_len'],262144)
                    self.assertEqual(options['prompt_budget_tokens'],221184)
                    self.assertEqual(options['max_tokens'],40960)
                    self.assertNotIn('max_tokens_late',options)

    def test_total_window_override_and_invalid_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=self.args(root,'max_model_len=131072')
            self.assertEqual(swe10.context_limits(args),(131072,90112))
            args=self.args(root,'sampling_max_model_len=65536')
            self.assertEqual(swe10.context_limits(args),(65536,24576))
            for limit in (-1,40960,300000):
                with self.subTest(limit=limit),self.assertRaises(ValueError):
                    swe10.context_limits(self.args(root,f'max_model_len={limit}'))
            args=self.args(root)
            (Path(args.model)/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unknown'):
                swe10.context_limits(args)

    def test_long_multiturn_history_stays_uncompressed_until_total_window_requires_it(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.args(Path(directory));window,budget=swe10.context_limits(args)
            renderer=AgentRenderer(CharacterTokenizer())
            previous='p'*32000;response=[ord('r')]*40960
            history=History(previous+'r'*40960+'tool output')
            continued=renderer.continue_prompt(history,[ord('p')]*32000,previous,response,budget)
            self.assertIsNotNone(continued)
            self.assertGreater(len(continued[0]),32768)
            self.assertLessEqual(len(continued[0])+args.max_tokens,window)
            ids,text,level=renderer.fit(history,budget)
            self.assertEqual(level,0)
            self.assertEqual(text,history.text)
            _,_,level=renderer.fit(History('x'*(budget+1)),budget)
            self.assertGreater(level,0)

    def test_removed_prompt_and_late_generation_options_fail_explicitly(self):
        for option in ('prompt_budget=32768','prompt_budget_tokens=32768',
                       'max_tokens_late=40960','max_tokens_switch_trunc=.08','max_tokens_switch_patience=1'):
            with self.subTest(option=option),self.assertRaises(ValueError):
                swe10.normalize_key_values([option])
        from arl.grpo import GRPOConfig
        self.assertNotIn('max_tokens_late',GRPOConfig.__annotations__)
