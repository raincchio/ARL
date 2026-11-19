"""Real upstream mini agent, scripted token sampler, isolated fake workspaces."""
import asyncio
import unittest
import tempfile
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from arl.harnesses import get_harness, resolve_pair
from arl.harnesses.mini_swe import check_dependencies, MiniEnv, TokenModel, swe_config, SWE_CONFIG_SHA256, programbench_config, PROGRAMBENCH_CONFIG_SHA256
from arl.agent.rollout import RolloutParams
from arl.algorithms.ppo_data import finalize_ppo_data
from test_swe_env import Source, Workspace, Renderer, Sampler, call, ce, record


class MiniRenderer(Renderer):
    def decode(self, ids):
        return call('bash', command='echo hi' if ids[0] == 10 else 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')


class MiniSource(Source):
    closed = False
    def train_tasks(self): return [ce.SweTask.from_record(record(1))]
    def test_tasks(self): return [ce.SweTask.from_record(record(2))]
    async def close(self): self.closed = True
    def make_env(self, task, uid):
        env = super().make_env(task, uid)
        async def execute(command, **kwargs):
            env.workspace.commands.append(command)
            return {'exit_code': 0, 'output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'+env.workspace.patch_text
                    if command.startswith('echo COMPLETE') else 'ok'}
        env.workspace.execute = execute
        return env


class MiniTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        check_dependencies()

    async def test_split_siblings_size_agent_pool_for_actual_local_trajectories(self):
        with patch('arl.utils.distributed.dist_info', return_value=(1, 4, 1)):
            result = await get_harness('mini_swe_agent').run(
                tasks=[ce.SweTask.from_record(record(i)) for i in range(2)],
                source=MiniSource(), sampling_client=Sampler(), renderer=MiniRenderer(),
                params=RolloutParams(gen_cap=10, prompt_budget=100, group_size=8,
                                     shard_trajectories=True, poll_seconds=.001),
                operation='train', env_name='coding.swe')
        self.assertEqual(len(result.trajs), 4)
        self.assertEqual(result.trajs[0].env.agent_executor._max_workers, 4)
        self.assertEqual([t.tid for t in result.trajs], [1, 5, 9, 13])

    async def test_agent_workers_follow_local_batch_and_group_without_using_default_pool(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from arl.harnesses.mini_swe import WorkspaceBridge

        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        for operation, expected in (('evaluate', 4), ('train', 6), ('collect', 6)):
            with self.subTest(operation=operation):
                source = MiniSource()
                barrier = threading.Barrier(expected)
                execute = WorkspaceBridge.execute

                def concurrent_execute(bridge, action):
                    if action['command'] == 'echo hi':
                        # Every trajectory must have a worker before any can
                        # continue. HTTP-like work still uses the default pool.
                        barrier.wait(timeout=3)
                    return execute(bridge, action)

                make_env = source.make_env

                def make_threaded_env(task, uid):
                    env = make_env(task, uid)
                    workspace_execute = env.workspace.execute

                    async def http_execute(command, **kwargs):
                        await asyncio.to_thread(lambda: None)
                        return await workspace_execute(command, **kwargs)

                    env.workspace.execute = http_execute
                    return env

                source.make_env = make_threaded_env
                with patch.object(WorkspaceBridge, 'execute', concurrent_execute):
                    result = await asyncio.wait_for(get_harness('mini_swe_agent').run(
                        tasks=[ce.SweTask.from_record(record(i)) for i in range(2)],
                        source=source, sampling_client=Sampler(), renderer=MiniRenderer(),
                        params=RolloutParams(gen_cap=10, prompt_budget=100, group_size=2, extra_samples=1),
                        operation=operation, env_name='coding.swe'), timeout=6)
                self.assertEqual(len(result.trajs), expected)
                pool = result.trajs[0].env.agent_executor
                self.assertEqual(pool._max_workers, expected)
                self.assertTrue(all(t.env.agent_executor is pool for t in result.trajs))
                self.assertTrue(all(env.workspace.closed for env in source.envs))
                with self.assertRaises(RuntimeError):
                    pool.submit(lambda: None)

    async def test_swe_initial_messages_equal_upstream_run_without_arl_additions(self):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.exceptions import Submitted
        cfg = swe_config()
        class StopModel(TokenModel):
            def query(self, messages):
                raise Submitted(dict(role='exit', content='', extra={'submission': ''}))
        reference = DefaultAgent(StopModel(cfg['model']), SimpleNamespace(
            get_template_vars=lambda: {}, serialize=lambda: {}), **cfg['agent'])
        task = ce.SweTask.from_record(record())
        reference.run(task.problem_statement)
        env = MiniEnv(ce.SweWorkspaceEnv(task, Workspace(), ce.EnvConfig(max_turns=250, command_timeout=60)))
        await env.reset()
        self.assertEqual(env.messages(), reference.messages[:2])
        self.assertEqual(env.agent.config.system_template, cfg['agent']['system_template'])
        self.assertEqual(env.agent.config.instance_template, cfg['agent']['instance_template'])
        self.assertEqual(env.agent.config.step_limit, 250)
        self.assertEqual(env.summary()['upstream_config_sha256'], SWE_CONFIG_SHA256)
        await env.close()

    async def test_official_tool_feedback_long_output_and_command_environment(self):
        workspace = Workspace()
        workspace.execute = AsyncMock(return_value={'exit_code': 0, 'output': 'HEAD'+('x'*20000)+'TAIL'})
        env = MiniEnv(ce.SweWorkspaceEnv(ce.SweTask.from_record(record()), workspace))
        await env.reset()
        await env.step(call('bash', command='cat large-file'))
        messages = env.messages()
        self.assertEqual(messages[-1]['role'], 'tool')
        self.assertEqual(messages[-1]['tool_call_id'], messages[-2]['tool_calls'][0]['id'])
        self.assertNotIn('<tool_call>', messages[-2]['content'])
        self.assertEqual(messages[-2]['tool_calls'][0]['function']['arguments'], {'command': 'cat large-file'})
        self.assertIsInstance(env.agent.messages[-2]['tool_calls'][0]['function']['arguments'], str)
        self.assertIn('<output_head>', messages[-1]['content'])
        self.assertIn('<output_tail>', messages[-1]['content'])
        self.assertIn('HEAD', messages[-1]['content'])
        self.assertIn('TAIL', messages[-1]['content'])
        kwargs = workspace.execute.call_args.kwargs
        self.assertIsNone(kwargs['output_limit'])
        self.assertEqual(kwargs['environment']['BASH_ENV'], '/root/.bashrc')
        self.assertEqual(kwargs['interpreter'], ['bash', '-c'])
        await env.close()

    async def test_only_emitted_submission_is_graded_and_limits_do_not_auto_submit(self):
        submitted = 'diff --git a/selected.py b/selected.py\nSELECTED PATCH\n'
        for output, expected in (('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'+submitted, submitted),
                                 ('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', ''), ('ok', '')):
            w = Workspace(patch_text='UNSUBMITTED WORKSPACE DIFF')
            w.patch = AsyncMock(side_effect=AssertionError('must not auto-read workspace diff'))
            w.grade = AsyncMock(return_value={'resolved': True, 'infra_error': False})
            w.execute = AsyncMock(return_value={'exit_code': 0, 'output': output})
            env = MiniEnv(ce.SweWorkspaceEnv(ce.SweTask.from_record(record()), w, ce.EnvConfig(max_turns=1)))
            await env.reset()
            out = await env.step(call('bash', command='submit-or-hit-limit'))
            self.assertTrue(out.done)
            self.assertEqual(env.summary()['model_patch'], expected)
            self.assertEqual(out.reward, float(bool(expected)))
            if expected: self.assertEqual(w.grade.call_args.args[0], submitted)
            else: w.grade.assert_not_awaited()
            if output == 'ok': self.assertEqual(env.summary()['exit_status'], 'LimitsExceeded')
            w.patch.assert_not_awaited()
            await env.close()

    async def test_official_format_error_keeps_response_in_metadata_only(self):
        env = MiniEnv(ce.SweWorkspaceEnv(ce.SweTask.from_record(record()), Workspace()))
        await env.reset()
        await env.step('</think>missing-tool')
        self.assertEqual([m['role'] for m in env.messages()], ['system', 'user', 'user'])
        self.assertIn('Tool call error:', env.messages()[-1]['content'])
        self.assertEqual(env.agent.messages[-1]['extra']['response']['content'], 'missing-tool')
        await env.close()

    async def test_upstream_agent_two_turns_tokens_rewards_and_advantages(self):
        source = MiniSource()
        result = await get_harness('mini_swe_agent').run(tasks=[ce.SweTask.from_record(record())],
            source=source, sampling_client=Sampler(), renderer=MiniRenderer(),
            params=RolloutParams(gen_cap=10, prompt_budget=100, group_size=2),
            operation='train', env_name='coding.swe', log_turns=True)
        self.assertEqual(result.returns, [1., 1., 0., 0.])
        self.assertEqual(result.correctness, [True, True, False, False])
        self.assertEqual(result.trajs[0].env.agent.__class__.__module__, 'minisweagent.agents.default')
        self.assertEqual(result.trajs[0].env.summary()['exit_status'], 'Submitted')
        self.assertEqual(result.trajs[0].env.agent.n_calls, 2)
        data, _ = finalize_ppo_data(result.datums, result.returns, groups=result.groups, units=result.units,
            whiten=False, group_adv_std=False, length_weighted_baseline=False)
        self.assertTrue((data[0].loss_fn_inputs['advantages'].to_torch() > 0).any())
        self.assertTrue((data[-1].loss_fn_inputs['advantages'].to_torch() < 0).any())
        self.assertTrue(all(e.workspace.closed for e in source.envs))
        for d in result.datums:
            self.assertIn(-.5, d.loss_fn_inputs['logprobs'].to_torch().tolist())

    async def test_invalid_truncated_and_plain_text_do_not_execute_or_submit(self):
        env = MiniEnv(ce.SweWorkspaceEnv(ce.SweTask.from_record(record()), Workspace()))
        await env.reset()
        try:
            self.assertNotIn('SECRET_GOLD', str(env.messages()))
            self.assertFalse((await env.step(call('bash', command='bad'), truncated=True)).done)
            self.assertFalse((await env.step('done')).done)
            self.assertTrue((await env.step(call('submit'))).done)
            self.assertEqual(env.env.workspace.commands, [])
            self.assertEqual(env.summary()['exit_status'], 'RepeatedFormatError')
        finally:
            await env.close()

    async def test_cancellation_interrupts_blocked_command_and_closes(self):
        workspace = Workspace()
        workspace.block = asyncio.Event()
        env = MiniEnv(ce.SweWorkspaceEnv(ce.SweTask.from_record(record()), workspace))
        await env.reset()
        task = asyncio.create_task(env.step(call('bash', command='blocked')))
        while not workspace.commands:
            await asyncio.sleep(.001)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(env.close(), 2)
        self.assertTrue(workspace.closed)

    async def test_programbench_uses_official_submission_reward(self):
        from test_programbench import ce as pb, Workspace as PBWorkspace, IDS, score
        from arl.envs.coding.programbench import ProgramTask
        workspace = PBWorkspace(score(.2))
        workspace.execute = AsyncMock(side_effect=lambda command, **kwargs: {'exit_code': 0,
            'output': '' if command.startswith('git config') else 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'})
        env = MiniEnv(pb.ProgramBenchEnv(ProgramTask(IDS[0]), workspace, ce.EnvConfig()))
        await env.reset()
        out = await env.step(call('bash', command='echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'))
        self.assertEqual((out.reward, out.infra_error, out.metrics['correct']), (.2, False, False))
        self.assertTrue(workspace.closed)
        self.assertEqual(env.summary()['mini_version'], '2.4.6')

    async def test_pairing_and_evaluation(self):
        for env in ('swe', 'programbench'):
            for operation in ('collect', 'train', 'evaluate'):
                self.assertEqual(resolve_pair(env, 'mini_swe_agent', operation=operation).harness, 'mini_swe_agent')
        source = MiniSource()
        result = await get_harness('mini_swe_agent').run(tasks=[ce.SweTask.from_record(record())],
            source=source, sampling_client=Sampler(), renderer=MiniRenderer(),
            params=RolloutParams(gen_cap=10, prompt_budget=100, group_size=1),
            operation='evaluate', env_name='coding.swe')
        self.assertEqual(result.datums, [])
        self.assertEqual(result.trajs[0].reward, 1.)


class PhaseTests(unittest.TestCase):
    def test_collection_exports_valid_token_trace_with_actual_harness_identity(self):
        from arl.experiments.config import from_dict
        from arl.experiments.runner import run_phase
        from arl.experiments.artifacts import validate_collection
        with tempfile.TemporaryDirectory() as root:
            exp = from_dict({'models': {'p': {'name': 'scripted-policy'}}, 'collect': {
                'env': 'swe', 'harness': 'mini_swe_agent', 'model': 'p', 'output_dir': 'collect', 'options': {
                    'batch_size': 1, 'group_size': 2, 'max_tokens': 10, 'prompt_budget_tokens': 100}}}, base_dir=root)
            source = MiniSource()
            with patch('arl.harnesses.registry.HarnessSpec.make_renderer', return_value=MiniRenderer()):
                summary = run_phase(exp.plan('collect'), native_runtime={
                    'source': source, 'sampling_client': Sampler(), 'tokenizer': object()})
            checked = validate_collection(root+'/collect', env='swe', harness='mini_swe_agent', require_training_trace=True)
            self.assertEqual(checked['datums_validated'], 4)
            self.assertEqual(summary['pass_rate'], .5)
            self.assertTrue(source.closed)


class ProgramBenchOfficialTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        check_dependencies()

    def make_env(self, steps=1000):
        from test_programbench import ce as pb, Workspace as PBWorkspace, IDS, score
        from arl.envs.coding.programbench import ProgramTask
        workspace = PBWorkspace(score(.2))
        env = MiniEnv(pb.ProgramBenchEnv(ProgramTask(IDS[0]), workspace,
            ce.EnvConfig(max_turns=steps, command_timeout=180)))
        return env, workspace

    async def test_prompt_exactly_matches_upstream_with_platform_and_git_identity(self):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
        from minisweagent.exceptions import Submitted
        cfg = programbench_config()
        class StopModel(TokenModel):
            def query(self, messages):
                raise Submitted(dict(role='exit', content='', extra={'submission': ''}))
        # A real upstream environment config/template renderer; no Docker start.
        docker = object.__new__(DockerEnvironment)
        docker.container_id = None
        docker.config = DockerEnvironmentConfig(image='programbench/test:task_cleanroom_v6', **cfg['environment'])
        reference = DefaultAgent(StopModel(cfg['model']), docker, **cfg['agent'])
        reference.run()
        env, workspace = self.make_env()
        try:
            await env.reset()
            self.assertEqual(env.messages(), reference.messages[:2])
            self.assertEqual(env.agent.config.step_limit, 1000)
            self.assertEqual(env.agent.config.wall_time_limit_seconds, 21600)
            self.assertEqual(workspace.commands, ['git config user.name "mini-swe-agent" && git config user.email "mini-swe-agent@proton.me"'])
            summary = env.summary()
            self.assertEqual(summary['harness_profile'], 'upstream_programbench')
            self.assertEqual(summary['upstream_config_sha256'], PROGRAMBENCH_CONFIG_SHA256)
        finally:
            await env.close()

    async def test_structured_feedback_includes_step_time_and_output_warnings(self):
        env, workspace = self.make_env(steps=3)
        try:
            await env.reset()
            workspace.execute = AsyncMock(return_value={'exit_code': 0, 'output': 'HEAD'+('x'*20000)+'TAIL'})
            env.agent._start_time -= 21100
            out = await env.step(call('bash', command='inspect'))
            self.assertFalse(out.done)
            message = env.messages()[-1]
            self.assertEqual(message['role'], 'tool')
            self.assertIn('2 steps away', message['content'])
            self.assertIn('running low on time', message['content'])
            self.assertIn('<output_head>', message['content'])
            self.assertIn('TAIL', message['content'])
            self.assertEqual(workspace.execute.call_args.kwargs['interpreter'], ['bash', '-lc'])
            self.assertEqual(workspace.execute.call_args.kwargs['environment'], programbench_config()['environment']['env'])
            self.assertIsNone(workspace.execute.call_args.kwargs['output_limit'])
            self.assertNotIn('raw_output', env.summary()['agent']['messages'][-1]['extra'])
        finally:
            await env.close()

    async def test_wall_time_and_step_limits_archive_workspace(self):
        for timed_out in (False, True):
            env, workspace = self.make_env(steps=1 if not timed_out else 1000)
            try:
                await env.reset()
                workspace.submit = AsyncMock(wraps=workspace.submit)
                if timed_out: env.agent._start_time -= 21601
                out = await env.step(call('bash', command='inspect'))
                self.assertTrue(out.done)
                self.assertEqual(out.reward, .2)
                workspace.submit.assert_awaited_once()
                self.assertEqual(env.summary()['exit_status'], 'TimeExceeded' if timed_out else 'LimitsExceeded')
                self.assertEqual(len(workspace.commands), 1 if timed_out else 2)  # startup + optional model command
            finally:
                await env.close()

    async def test_official_programbench_collect_preserves_ppo_trace(self):
        from test_programbench import Workspace as PBWorkspace, score, IDS
        from arl.envs.coding.programbench import ProgramBenchSourceConfig, ProgramBenchTaskSource, ProgramTask
        source = ProgramBenchTaskSource(ProgramBenchSourceConfig())
        workspaces = []
        def make_workspace(task):
            workspace = PBWorkspace(score(1. if len(workspaces) == 0 else .2))
            async def execute(command, **kwargs):
                workspace.commands.append(command)
                return {'exit_code': 0, 'output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n' if command.startswith('echo COMPLETE') else 'ok'}
            workspace.execute = execute
            workspaces.append(workspace)
            return workspace
        source.backend.workspace = make_workspace
        try:
            result = await get_harness('mini_swe_agent').run(tasks=[ProgramTask(IDS[0])], source=source,
                sampling_client=Sampler(), renderer=MiniRenderer(),
                params=RolloutParams(gen_cap=10, prompt_budget=100, group_size=2),
                operation='collect', env_name='coding.programbench')
            self.assertEqual(result.returns, [1., 1., .2, .2])
            self.assertEqual(result.correctness, [True, True, False, False])
            self.assertTrue(all(w.closed for w in workspaces))
            for datum in result.datums:
                self.assertEqual(int(datum.loss_fn_inputs['mask'].to_torch().sum()), 1)
                self.assertIn(-.5, datum.loss_fn_inputs['logprobs'].to_torch().tolist())
        finally:
            await source.close()
