"""Actual Claude Code subprocess + local token bridge; fake SWE workspaces."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

from arl.harnesses.claude_code import (
    ClaudeCodeEnv, TOOL, VERSION, check_dependencies, translate_request, translate_response,
)
from arl.harnesses import get_harness, resolve_pair, HarnessCompatibilityError
from arl.agent.rollout import RolloutParams
from arl.algorithms.ppo_data import finalize_ppo_data
from test_swe_env import ce, record, Workspace, Source, Renderer, Sampler


def call(command):
    return '<tool_call>' + json.dumps({'name': TOOL, 'arguments': {'command': command}}) + '</tool_call>'


class TranslationTests(unittest.TestCase):
    def test_cli_messages_and_tools_preserve_tool_identity(self):
        messages, tools = translate_request({'system': [{'type': 'text', 'text': 'SYSTEM'}], 'tools': [
            {'name': TOOL, 'description': 'remote', 'input_schema': {'type': 'object'}}], 'messages': [
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': TOOL, 'input': {'command': 'pwd'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': [{'type': 'text', 'text': '/testbed'}]}]}]})
        self.assertEqual(messages[-1], {'role': 'tool', 'tool_call_id': 't1', 'content': '/testbed'})
        self.assertEqual(messages[-2]['tool_calls'][0]['id'], 't1')
        self.assertEqual(tools[0]['function']['name'], TOOL)

    def test_qwen_xml_hermes_and_plain_final_response(self):
        xml = f'<tool_call><function={TOOL}><parameter=command>\npwd\n</parameter></function></tool_call>'
        for text in (call('pwd'), xml, '<think>private</think>' + xml):
            self.assertEqual(translate_response(text, {TOOL})[-1]['input'], {'command': 'pwd'})
        self.assertEqual(translate_response('Done.', {TOOL}), [{'type': 'text', 'text': 'Done.'}])

    def test_invalid_calls_and_nonworkspace_tools_fail_closed(self):
        for text in ('<tool_call>{', '<think>partial', call(' '), '<tool_call>{"name":"Bash","arguments":{"command":"pwd"}}</tool_call>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                translate_response(text, {TOOL})
        with self.assertRaises(ValueError):
            translate_request({'messages': [], 'tools': [{'name': 'Read', 'input_schema': {}}]})
        with self.assertRaises(ValueError):
            translate_request({'messages': [{'role': 'user', 'content': [{'type': 'image'}]}]})


class ClaudeTests(unittest.IsolatedAsyncioTestCase):
    def env(self, workspace=None, **kwargs):
        return ClaudeCodeEnv(ce.SweWorkspaceEnv(ce.SweTask.from_record(record()), workspace or Workspace(), ce.EnvConfig(**kwargs)))

    async def test_real_cli_exec_feedback_final_grade_and_cleanup(self):
        env = self.env()
        try:
            await env.reset()
            self.assertEqual([t['function']['name'] for t in env.tools], [TOOL])
            for secret in ('SECRET_GOLD', 'SECRET_TEST', 'SECRET_HINT'):
                self.assertNotIn(secret, json.dumps(env.messages()))
            out = await env.step(call('pwd'))
            self.assertFalse(out.done)
            self.assertEqual(env.env.workspace.commands, ['pwd'])
            self.assertEqual(env.messages()[-1]['role'], 'tool')
            self.assertIn('exit_code=0', env.messages()[-1]['content'])
            out = await env.step('Finished.')
            self.assertEqual((out.reward, out.infra_error), (1., False))
            self.assertEqual(env.summary()['claude_code_version'], VERSION)
            self.assertIsNotNone(env._process.returncode)
            self.assertTrue(env.env.workspace.closed)
        finally:
            await env.close()

    async def test_truncated_and_invalid_calls_never_execute(self):
        env = self.env()
        try:
            await env.reset()
            self.assertFalse((await env.step(call('should-not-run'), truncated=True)).done)
            self.assertFalse((await env.step('<tool_call>{')).done)
            self.assertEqual(env.env.workspace.commands, [])
            self.assertTrue((await env.step('Done.')).done)
        finally:
            await env.close()

    async def test_limit_and_cancellation_stop_cli_and_close_workspace(self):
        env = self.env(max_turns=1)
        try:
            await env.reset()
            out = await env.step(call('pwd'))
            self.assertTrue(out.done)
            self.assertIsNotNone(env._process.returncode)
        finally:
            await env.close()
        env = self.env()
        try:
            await env.reset()
            await env.close()
            self.assertIsNotNone(env._process.returncode)
            self.assertTrue(env.env.workspace.closed)
        finally:
            await env.close()

    async def test_cancel_during_remote_command_reclaims_cli_and_workspace(self):
        workspace = Workspace()
        workspace.block = asyncio.Event()
        env = self.env(workspace)
        try:
            await env.reset()
            step = asyncio.create_task(env.step(call('blocked')))
            while not workspace.commands:
                await asyncio.sleep(.01)
            step.cancel()
            await asyncio.gather(step, return_exceptions=True)
            await asyncio.wait_for(env.close(), 5)
            self.assertTrue(workspace.closed)
            self.assertIsNotNone(env._process.returncode)
        finally:
            await env.close()

    async def test_workspace_failure_marks_infra_and_drops_reward(self):
        w = Workspace()
        w.execute = AsyncMock(side_effect=ce.BackendError('disconnected'))
        env = self.env(w)
        try:
            await env.reset()
            out = await env.step(call('pwd'))
            self.assertTrue(out.infra_error)
            self.assertIsNone(out.metrics['correct'])
            self.assertEqual(w.grade_calls, 0)
        finally:
            await env.close()

    async def test_real_cli_rollout_retains_tokens_logprobs_and_grpo_units(self):
        class ClaudeRenderer(Renderer):
            def decode(self, ids):
                return call('pwd') if ids[0] == 10 else 'Finished.'
        source = Source()
        result = await get_harness('claude_code').run(tasks=[ce.SweTask.from_record(record())],
            source=source, sampling_client=Sampler(), renderer=ClaudeRenderer(),
            params=RolloutParams(gen_cap=10, prompt_budget=100, group_size=2),
            operation='train', env_name='coding.swe', log_turns=True)
        self.assertEqual(result.returns, [1., 1., 0., 0.])
        self.assertEqual(result.correctness, [True, True, False, False])
        from arl.experiments.artifacts import append_datums, validate_collection
        with tempfile.TemporaryDirectory() as directory:
            identity = {'env': 'coding.swe', 'harness': 'claude_code', 'model': 'local-policy'}
            manifest = {'status': 'complete', 'identity': identity,
                        'trace_fields': ['token_ids', 'logprobs', 'loss_mask']}
            Path(directory, 'manifest.json').write_text(json.dumps(manifest))
            count = append_datums(Path(directory, 'datums.jsonl'), result, identity)
            checked = validate_collection(directory, env='swe', harness='claude_code', require_training_trace=True)
            self.assertEqual(checked['datums_validated'], count)
            self.assertGreater(count, 0)
        self.assertEqual(len(set(result.units)), 2)
        self.assertTrue(all(-.5 in d.loss_fn_inputs['logprobs'].tolist() for d in result.datums))
        data, _ = finalize_ppo_data(result.datums, result.returns, groups=result.groups, units=result.units,
            whiten=False, group_adv_std=False, length_weighted_baseline=False)
        self.assertTrue((data[0].loss_fn_inputs['advantages'].to_torch() > 0).any())
        self.assertTrue((data[-1].loss_fn_inputs['advantages'].to_torch() < 0).any())
        self.assertTrue(all(e.workspace.closed for e in source.envs))

    async def test_pairing_local_swe_only(self):
        self.assertEqual(resolve_pair('swe', operation='train', model_backend='local').harness, 'claude_code')
        for env, backend in [('programbench', 'local'), ('swe', 'openai_compatible')]:
            with self.assertRaises(HarnessCompatibilityError):
                resolve_pair(env, 'claude_code', operation='collect', model_backend=backend)


if __name__ == '__main__':
    unittest.main()
