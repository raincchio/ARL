"""SWE AgentEnv/TaskSource/rollout contract tests, no Docker or datasets download."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from arl.envs.coding.swe import SweSourceConfig, SweTaskSource, _import_sweenv
ce = _import_sweenv()
from sweenv.tools import parse_tool_calls
from arl.agent.rollout import RolloutParams, run_agent_rollout
from arl.algorithms.ppo_data import finalize_ppo_data
from arl.models.types import SampledSequence


def record(i=0):
    return {"instance_id": f"example__repo-{i}", "repo": "example/repo", "base_commit": "a" * 40,
            "problem_statement": f"Fix issue {i}", "patch": "SECRET_GOLD", "test_patch": "SECRET_TEST",
            "FAIL_TO_PASS": '["SECRET_TEST_NAME"]', "hints_text": "SECRET_HINT"}


def call(name, **args):
    return '</think>\n<tool_call>' + json.dumps({"name": name, "arguments": args}) + '</tool_call>'


class Workspace:
    def __init__(self, resolved=True, patch_text="diff --git a/a.py b/a.py\n", infra=False):
        self.patch_text, self.resolved, self.infra = patch_text, resolved, infra
        self.commands = []
        self.closed = False
        self.grade_calls = 0
        self.output = {"exit_code": 0, "output": "ok"}
        self.block = None

    async def execute(self, command, **kwargs):
        self.commands.append(command)
        if self.block is not None:
            await self.block.wait()
        return self.output

    async def patch(self):
        return {"patch": self.patch_text}

    async def grade(self, patch, timeout):
        self.grade_calls += 1
        return {"resolved": self.resolved, "infra_error": self.infra}

    async def close(self):
        self.closed = True


class EnvTests(unittest.IsolatedAsyncioTestCase):
    def env(self, workspace=None, **kwargs):
        return ce.SweWorkspaceEnv(ce.SweTask.from_record(record()), workspace or Workspace(), ce.EnvConfig(**kwargs))

    async def test_persistent_commands_and_final_official_reward(self):
        w = Workspace()
        env = self.env(w)
        await env.reset()
        out = await env.step(call("bash", command="sed -n '1,30p' a.py"))
        self.assertFalse(out.done)
        self.assertEqual(w.commands, ["sed -n '1,30p' a.py"])
        self.assertEqual(env.messages()[-1]['role'], 'tool')
        out = await env.step(call("submit"))
        self.assertEqual((out.reward, out.done, out.infra_error), (1., True, False))
        self.assertEqual(out.metrics['correct'], 1.)
        self.assertEqual(w.grade_calls, 1)
        self.assertTrue(w.closed)
        with self.assertRaises(RuntimeError):
            await env.step(call("submit"))

    async def test_gold_and_hidden_tests_never_in_messages_or_summary(self):
        env = self.env()
        text = json.dumps([env.messages(a) for a in range(5)]) + json.dumps(env.summary())
        for secret in ('SECRET_GOLD', 'SECRET_TEST', 'SECRET_TEST_NAME', 'SECRET_HINT'):
            self.assertNotIn(secret, text)

    async def test_empty_patch_zero_without_grader(self):
        w = Workspace(patch_text='')
        out = await self.env(w).step('</think>Done.')
        self.assertEqual((out.reward, out.done, out.infra_error), (0., True, False))
        self.assertEqual(w.grade_calls, 0)
        self.assertTrue(w.closed)

    async def test_failed_tests_are_zero_infra_is_dropped(self):
        for infra in (False, True):
            env = self.env(Workspace(resolved=False, infra=infra))
            out = await env.step(call('submit'))
            self.assertEqual(out.reward, 0.)
            self.assertEqual(out.infra_error, infra)
            self.assertEqual(out.metrics['correct'], None if infra else 0.)

    async def test_command_nonzero_and_timeout_are_model_feedback(self):
        for code in (1, 124, 137):
            w = Workspace()
            w.output = {'exit_code': code, 'output': 'failed or timed out'}
            out = await self.env(w).step(call('bash', command='false'))
            self.assertFalse(out.done)
            self.assertFalse(out.infra_error)
            self.assertIn(str(code), out.tool_results[0][1])

    async def test_transport_failure_is_infra_and_closes(self):
        w = Workspace()
        w.execute = AsyncMock(side_effect=ce.BackendError('disconnected'))
        out = await self.env(w).step(call('bash', command='pwd'))
        self.assertTrue(out.infra_error)
        self.assertTrue(w.closed)
        self.assertEqual(w.grade_calls, 0)

    async def test_truncated_or_incomplete_call_never_runs(self):
        w = Workspace()
        env = self.env(w)
        out = await env.step(call('bash', command='echo should-not-run'), truncated=True)
        self.assertFalse(out.done)
        self.assertEqual(w.commands, [])
        out = await env.step('</think><tool_call>{"name":"bash"')
        self.assertFalse(out.done)
        self.assertEqual(w.commands, [])

    async def test_max_turns_and_external_finish_submit(self):
        for external in (False, True):
            w = Workspace()
            env = self.env(w, max_turns=1 if not external else 64)
            out = await env.finish('context overflow') if external else await env.step(call('bash', command='pwd'))
            self.assertTrue(out.done)
            self.assertEqual(w.grade_calls, 1)
            self.assertTrue(w.closed)

    async def test_call_limit_and_submit_short_circuit(self):
        w = Workspace()
        env = self.env(w, max_calls_per_turn=1)
        out = await env.step(call('bash', command='one') + call('bash', command='two'))
        self.assertFalse(out.done)
        self.assertEqual(w.commands, [])
        await self.env(w).step(call('submit') + call('bash', command='too late'))
        self.assertEqual(w.commands, [])

    async def test_compression_preserves_recent_history_and_issue(self):
        env = self.env()
        for i in range(8):
            await env.step(call('bash', command=f'echo {i}'))
        self.assertIn('Fix issue 0', json.dumps(env.messages(4)))
        self.assertIn('echo 7', json.dumps(env.messages(4)))
        self.assertNotIn('echo 0', json.dumps(env.messages(4)))
        self.assertIn('echo 0', json.dumps(env.messages(0)))


class ParserTests(unittest.TestCase):
    def test_xml_json_and_non_thinking_calls(self):
        xml = '<tool_call><function=bash><parameter=command>\necho hi\n</parameter></function></tool_call>'
        for text in (xml, '</think>' + xml, call('bash', command='echo hi')):
            calls, errors = parse_tool_calls(text)
            self.assertEqual(calls, [{'name': 'bash', 'arguments': {'command': 'echo hi'}}])
            self.assertEqual(errors, [])

    def test_invalid_unknown_and_reasoning_calls(self):
        for text in (call('unknown'), call('bash', command=1), call('submit', x=1),
                     '</think><tool_call>[]</tool_call>'):
            calls, errors = parse_tool_calls(text)
            self.assertEqual(calls, [])
            self.assertTrue(errors)
        calls, _ = parse_tool_calls('<think>' + call('bash', command='secret').replace('</think>', '') + '</think>done')
        self.assertEqual(calls, [])


class SourceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'tasks.jsonl'
        self.path.write_text('\n'.join(json.dumps(record(i)) for i in range(10)))

    async def test_local_partition_deterministic_and_disjoint(self):
        cfg = SweSourceConfig(train_set=str(self.path), test_fraction=.2)
        a, b = SweTaskSource(cfg), SweTaskSource(cfg)
        await a.initialize()
        await b.initialize()
        self.assertEqual(len(a.train_tasks()), 8)
        self.assertEqual([t.instance_id for t in a.test_tasks()], [t.instance_id for t in b.test_tasks()])
        self.assertFalse({t.instance_id for t in a.train_tasks()} & {t.instance_id for t in a.test_tasks()})
        env = a.make_env(a.train_tasks()[0], uid='test')
        self.assertFalse(env.workspace._started)
        await a.close()
        self.assertTrue(env.workspace._closed)

    async def test_explicit_test_overlap_rejected(self):
        source = SweTaskSource(SweSourceConfig(train_set=str(self.path), test_set=str(self.path)))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            await source.initialize()

    async def test_remote_dataset_and_limits(self):
        source = SweTaskSource(SweSourceConfig(train_set='remote:/host/data.jsonl', limit=2, test_fraction=0))
        source._backend.load_tasks = AsyncMock(return_value=[record(1), record(2)])
        await source.initialize()
        self.assertEqual(len(source.train_tasks()), 2)
        source._backend.load_tasks.assert_awaited_once_with('/host/data.jsonl', 2)

    async def test_invalid_or_duplicate_records_rejected(self):
        for rows in ([record(), record()], [{**record(), 'base_commit': 'main;rm'}]):
            self.path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                await SweTaskSource(SweSourceConfig(train_set=str(self.path))).initialize()

    async def test_factory_grpo_routing(self):
        from arl.grpo import GRPOConfig, _make_source
        source = _make_source(GRPOConfig(model_name='test', log_path='/tmp/test', env='swe',
            train_set=str(self.path), swe_ssh_host='example', swe_command_timeout=15))
        self.assertIsInstance(source, SweTaskSource)
        self.assertEqual(source.cfg.ssh_host, 'example')
        self.assertEqual(source._env_cfg.command_timeout, 15)
        await source.close()


class Renderer:
    tokenizer = None
    def stop_token_ids(self): return []
    def fit(self, env, *args): return [100, len(env.messages())], '', 0
    def continue_prompt(self, *args): return None
    def decode(self, ids): return call('bash', command='echo hi') if ids[0] == 10 else call('submit')


class Source:
    def __init__(self): self.envs = []
    def task_key(self, task): return task.instance_id
    def make_env(self, task, uid):
        w = Workspace(resolved=len(self.envs) % 2 == 0)
        env = ce.SweWorkspaceEnv(task, w)
        self.envs.append(env)
        return env


class Sampler:
    async def submit_and_poll(self, requests, **kw):
        await asyncio.sleep(0)
        return [(r.extras['req_id'], SampledSequence(
            tokens=[10 if r.extras['req_id'].endswith(':0') else 11], logprobs=[-.5], stop_reason='stop'))
                for r in requests]


class RolloutTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_progress_counts_finished_trajectories_before_batch_ends(self):
        release = asyncio.Event()

        class BlockingSource(Source):
            def make_env(self, task, uid):
                env = super().make_env(task, uid)
                if len(self.envs) == 2:
                    env.workspace.block = release
                return env

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'rollout_progress.rank0.json'
            work = asyncio.create_task(run_agent_rollout(
                [ce.SweTask.from_record(record())], BlockingSource(), Sampler(), Renderer(),
                RolloutParams(gen_cap=10, prompt_budget=100, group_size=2, progress_path=str(path))))
            try:
                async def partial_progress():
                    while True:
                        if path.exists():
                            progress = json.loads(path.read_text())
                            if progress['completed'] == 1:
                                return progress
                        await asyncio.sleep(.01)

                progress = await asyncio.wait_for(partial_progress(), 5)
                self.assertFalse(work.done())
                self.assertEqual(progress['total'], 2)
                self.assertEqual(progress['environment'], 1)
                self.assertEqual(progress['status'], 'running')
                release.set()
                await asyncio.wait_for(work, 2)
                final = json.loads(path.read_text())
                self.assertEqual((final['completed'], final['total'], final['status']), (2, 2, 'complete'))
                self.assertEqual(final['turns_completed'], 4)
            finally:
                release.set()
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)

    async def test_two_turn_swe_rollout_to_grpo_and_labels(self):
        source = Source()
        res = await run_agent_rollout([ce.SweTask.from_record(record())], source, Sampler(), Renderer(),
                                     RolloutParams(gen_cap=10, prompt_budget=100, group_size=2))
        self.assertEqual(res.returns, [1., 1., 0., 0.])
        self.assertEqual(res.correctness, [True, True, False, False])
        self.assertEqual(res.metrics['rollout/resolved_rate'], .5)
        data, _ = finalize_ppo_data(res.datums, res.returns, groups=res.groups, units=res.units,
            whiten=False, group_adv_std=False, length_weighted_baseline=False,
            response_correctness=res.correctness, mask_positive_incorrect=True)
        for d, correct in zip(data, res.correctness):
            a = d.loss_fn_inputs['advantages'].to_torch()
            self.assertTrue(bool((a > 0).any()) if correct else bool((a < 0).any()))
        self.assertTrue(all(e.workspace.closed for e in source.envs))

    async def test_sampler_exception_closes_all_created_environments(self):
        source = Source()
        sampler = Sampler()
        sampler.submit_and_poll = AsyncMock(side_effect=RuntimeError('sampler unavailable'))
        with self.assertRaisesRegex(RuntimeError, 'sampler unavailable'):
            await run_agent_rollout([ce.SweTask.from_record(record())], source, sampler, Renderer(),
                                    RolloutParams(gen_cap=10, prompt_budget=100, group_size=2))
        self.assertTrue(all(e.workspace.closed for e in source.envs))

    async def test_extra_sample_cancellation_closes_its_workspace(self):
        source = Source()
        class ExtraSampler(Sampler):
            async def submit_and_poll(self, requests, **kw):
                await asyncio.sleep(0)
                results = []
                for r in requests:
                    tid = int(r.extras['req_id'].split(':')[0])
                    if tid == 2:
                        source.envs[2].workspace.block = asyncio.Event()
                    results.append((r.extras['req_id'], SampledSequence(
                        tokens=[10 if tid == 2 else 11], logprobs=[-.5], stop_reason='stop')))
                return results
        res = await asyncio.wait_for(run_agent_rollout([ce.SweTask.from_record(record())], source,
            ExtraSampler(), Renderer(), RolloutParams(gen_cap=10, prompt_budget=100, group_size=2, extra_samples=1)), 5)
        self.assertEqual(res.metrics['rollout/extra_dropped_trajs'], 1)
        self.assertTrue(source.envs[2].workspace.closed)


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_rpc_progresses_when_agents_fill_default_executor(self):
        from concurrent.futures import ThreadPoolExecutor
        import contextvars
        import threading

        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        with patch.dict('os.environ', {'SWE_WORKER_TOKEN': 'test-token'}):
            backend = ce.DockerBackend(ce.BackendConfig(endpoint='http://unused'))
        context = contextvars.ContextVar('rpc_context', default='missing')
        context.set('preserved')
        barrier = threading.Barrier(2)
        pending = []

        def service_rpc(action, timeout, payload, cancelled):
            return {'command': payload['command'], 'context': context.get()}

        def agent(index):
            # Both default workers must be occupied before either RPC is sent.
            barrier.wait(timeout=2)
            future = asyncio.run_coroutine_threadsafe(
                backend.rpc('exec', command=str(index)), loop)
            pending.append(future)
            return future.result(timeout=2)

        try:
            with patch.object(backend, '_service_rpc', side_effect=service_rpc):
                results = await asyncio.wait_for(asyncio.gather(
                    *(asyncio.to_thread(agent, i) for i in range(2))), timeout=4)
            self.assertEqual(results, [
                {'command': str(i), 'context': 'preserved'} for i in range(2)])
        finally:
            for future in pending:
                future.cancel()
            await backend.close()

    async def test_service_rpc_cancellation_signals_http_worker(self):
        import threading

        with patch.dict('os.environ', {'SWE_WORKER_TOKEN': 'test-token'}):
            backend = ce.DockerBackend(ce.BackendConfig(endpoint='http://unused'))
        started = asyncio.Event()
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        cancellation_seen = threading.Event()

        def service_rpc(action, timeout, payload, cancelled):
            loop.call_soon_threadsafe(started.set)
            if cancelled.wait(2):
                cancellation_seen.set()
            loop.call_soon_threadsafe(stopped.set)
            return {}

        task = None
        try:
            with patch.object(backend, '_service_rpc', side_effect=service_rpc):
                task = asyncio.create_task(backend.rpc('exec', command='blocked'))
                await asyncio.wait_for(started.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(stopped.wait(), 2)
                self.assertTrue(cancellation_seen.is_set())
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await backend.close()

    async def test_workspace_command_and_grade_deadlines_stay_distinct_from_transport(self):
        backend = ce.DockerBackend(ce.BackendConfig())
        backend.rpc = AsyncMock(return_value={'exit_code': 0, 'output': ''})
        w = backend.workspace(ce.SweTask.from_record(record()))
        await w.execute('echo hi', timeout=17, output_limit=500)
        self.assertEqual(backend.rpc.call_args.kwargs['command_timeout'], 17)
        self.assertEqual(backend.rpc.call_args.kwargs['timeout'], 77)
        await w.grade('patch', 42)
        self.assertEqual(backend.rpc.call_args.kwargs['grade_timeout'], 42)
        self.assertEqual(backend.rpc.call_args.kwargs['timeout'], 702)
        await w.close()

    async def test_failed_start_releases_slot_and_attempts_cleanup(self):
        backend = ce.DockerBackend(ce.BackendConfig(max_concurrency=1))
        backend.rpc = AsyncMock(side_effect=[ce.BackendError('start failed'), {'closed': True}])
        w = backend.workspace(ce.SweTask.from_record(record()))
        with self.assertRaises(ce.BackendError):
            await w.start()
        self.assertEqual(backend._slots._value, 1)
        self.assertEqual(backend.rpc.call_args.args, ('close',))

    async def test_cancelled_command_workspace_can_be_closed(self):
        backend = ce.DockerBackend(ce.BackendConfig(max_concurrency=1))
        backend.rpc = AsyncMock(return_value={})
        w = backend.workspace(ce.SweTask.from_record(record()))
        await w.start()
        backend.rpc = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await w.execute('sleep 999')
        backend.rpc = AsyncMock(return_value={})
        await w.close()
        self.assertEqual(backend._slots._value, 1)
        self.assertTrue(w._closed)

    async def test_ssh_argv_calls_host_owned_script_and_sends_payload_on_stdin(self):
        import shlex
        backend = ce.DockerBackend(ce.BackendConfig(ssh_host='host', python='/path with space/python'))
        argv = backend._argv()
        self.assertEqual(shlex.split(argv[-1]), ['/path with space/python', backend.cfg.worker_script])
        from unittest.mock import Mock
        proc = Mock(returncode=0)
        proc.communicate = AsyncMock(return_value=(b'{"exit_code":0,"output":"ok"}', b''))
        with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=proc)):
            await backend.rpc('exec', command='echo `host-secret`', command_timeout=10)
        payload = json.loads(proc.communicate.call_args.args[0])
        self.assertEqual(payload['command'], 'echo `host-secret`')
        self.assertNotIn('host-secret', ''.join(argv))
        await backend.close()

    async def test_endpoint_client_does_not_read_host_service_source(self):
        with patch.dict('os.environ', {'SWE_WORKER_TOKEN': 'a' * 48}), \
                patch('pathlib.Path.read_text', side_effect=FileNotFoundError('no server source in training container')):
            backend = ce.DockerBackend(ce.BackendConfig(endpoint='http://127.0.0.1:18770'))
        self.assertIsNone(backend._worker)
        await backend.close()

    async def test_explicit_source_ip_429_exec_is_retried_but_transport_error_is_not(self):
        import io
        import threading
        import urllib.error
        from unittest.mock import MagicMock, Mock
        with patch.dict('os.environ', {'SWE_WORKER_TOKEN': 'a' * 48}):
            backend = ce.DockerBackend(ce.BackendConfig(endpoint='http://127.0.0.1:18770'))
        response = MagicMock()
        response.__enter__.return_value = io.BytesIO(b'{"exit_code":0,"output":"ok"}')
        opener = Mock()
        opener.open.side_effect = [urllib.error.HTTPError('http://worker/rpc',429,'queued',{},io.BytesIO(b'queued')), response]
        cancelled = threading.Event()
        with patch('urllib.request.build_opener',return_value=opener), patch.object(cancelled,'wait',return_value=False):
            result = backend._service_rpc('exec',10,{'command':'pwd'},cancelled)
        self.assertEqual(result['exit_code'],0)
        self.assertEqual(opener.open.call_count,2)
        opener.open.reset_mock()
        opener.open.side_effect = urllib.error.URLError('response lost')
        with patch('urllib.request.build_opener',return_value=opener):
            with self.assertRaises(ce.BackendError):
                backend._service_rpc('exec',10,{'command':'pwd'},cancelled)
        self.assertEqual(opener.open.call_count,1)
        await backend.close()

    async def test_worker_infra_and_invalid_json_are_errors(self):
        from unittest.mock import Mock
        for output in (b'{"infra_error":true,"error":"down"}', b'not JSON'):
            proc = Mock(returncode=0)
            proc.communicate = AsyncMock(return_value=(output, b''))
            with patch('asyncio.create_subprocess_exec', new=AsyncMock(return_value=proc)):
                with self.assertRaises(ce.BackendError):
                    await ce.DockerBackend(ce.BackendConfig()).rpc('health')


if __name__ == '__main__':
    unittest.main()
