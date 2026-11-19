"""ProgramBench isolation, official score semantics, source splits and RL data."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from arl.envs.coding.programbench import ProgramBenchSourceConfig, ProgramBenchTaskSource, ProgramTask, _import_programbenchenv
from arl.envs import get_env_spec
from arl.harnesses import resolve_pair
from arl.grpo import GRPOConfig, _make_source, _validate
from arl.agent.rollout import RolloutParams, run_agent_rollout
from arl.algorithms.ppo_data import finalize_ppo_data
from test_swe_env import call, Renderer, Sampler

ce = _import_programbenchenv()
from sweenv.env import EnvConfig
IDS = ['lh3__seqtk.94e7070', 'tomnomnom__gron.88a6234', 'cmatsuoka__figlet.202a0a8', 'nuta__nsh.bdd0702']


def score(value=.5, **kwargs):
    return dict(score=value, n_resolved=round(value*10), n_tests=10, error_code=None,
                test_branch_errors={}, n_system_errors=0, **kwargs)


class Workspace:
    def __init__(self, summary=None): self.summary, self.closed, self.commands = summary or score(), False, []
    async def execute(self, command, **kwargs):
        self.commands.append(command)
        return {'exit_code': 0, 'output': 'ok'}
    async def submit(self, timeout): return {'summary': self.summary, 'archive': '/host/submission.tar.gz'}
    async def close(self): self.closed = True


class EnvTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_archive_reward_and_no_hidden_data_in_prompt(self):
        workspace = Workspace()
        env = ce.ProgramBenchEnv(ProgramTask(IDS[0]), workspace, EnvConfig())
        await env.reset()
        prompt = json.dumps(env.messages())
        self.assertIn('/workspace', prompt)
        self.assertNotIn('/testbed', prompt)
        self.assertNotIn('test_branch', prompt)
        self.assertNotIn('tests.json', prompt)
        self.assertIn('/workspace', env.tools[0]['function']['description'])
        out = await env.step(call('bash', command='printf source > main.c'))
        self.assertFalse(out.done)
        out = await env.step(call('submit'))
        self.assertEqual(out.reward, .5)
        self.assertFalse(out.metrics['correct'])
        self.assertFalse(out.infra_error)
        self.assertTrue(workspace.closed)
        self.assertIn('source', workspace.commands[0])

    async def test_strict_reward_truncation_and_error_labels(self):
        for summary, infra, reward in ((score(1.), False, 1.), (score(.5), False, 0.),
            ({**score(0.), 'error_code': 'compile_failed'}, False, 0.),
            ({**score(), 'test_branch_errors': {'branch': ['results_missing']}}, True, 0.)):
            workspace = Workspace(summary)
            env = ce.ProgramBenchEnv(ProgramTask(IDS[0]), workspace, EnvConfig(), reward_mode='strict')
            await env.step(call('bash', command='should-not-run'), truncated=True)
            self.assertEqual(workspace.commands, [])
            out = await env.finish('budget')
            self.assertEqual((out.infra_error, out.reward), (infra, reward))
            self.assertEqual(out.metrics['correct'], None if infra else reward == 1.)
            self.assertTrue(workspace.closed)
        for invalid in ({**score(), 'n_tests': 0}, {**score(), 'score': float('nan')},
                        {**score(), 'score': .9}, {**score(), 'n_system_errors': 1}):
            with self.assertRaises(ce.BackendError): ce.classify_summary(invalid)


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def source(self, **kwargs):
        source = ProgramBenchTaskSource(ProgramBenchSourceConfig(**kwargs))
        source.backend.inventory = AsyncMock(return_value=[{'instance_id': iid} for iid in IDS])
        return source

    async def test_stable_disjoint_split_and_limits_apply_after_split(self):
        full = self.source(test_fraction=.25)
        limited = self.source(test_fraction=.25, limit=1)
        try:
            await full.initialize(); await limited.initialize()
            self.assertEqual((len(full.train_tasks()), len(full.test_tasks())), (3, 1))
            self.assertEqual(limited.test_tasks(), full.test_tasks())
            self.assertEqual(limited.train_tasks(), full.train_tasks()[:1])
            self.assertFalse(set(full.train_tasks()) & set(full.test_tasks()))
            full.backend.health = AsyncMock(return_value={'ready': True})
            await full.check()
            self.assertEqual(set(full.backend.health.call_args.args[0]), set(IDS))
        finally:
            await full.close(); await limited.close()

    async def test_explicit_ids_overlap_unknown_and_factory(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'test.json'; path.write_text(json.dumps([IDS[0]]))
            source = self.source(test_set=str(path))
            with self.assertRaisesRegex(ValueError, 'overlap'): await source.initialize()
            await source.close()
        source = self.source(task_ids='missing__repo.abcdef0')
        with self.assertRaisesRegex(ValueError, 'unknown'): await source.initialize()
        await source.close()
        self.assertEqual(get_env_spec('programbench').default_harness, 'mini_swe_agent')
        self.assertEqual(resolve_pair('programbench', operation='evaluate').harness, 'mini_swe_agent')
        from arl.harnesses import HarnessCompatibilityError
        with self.assertRaises(HarnessCompatibilityError):
            resolve_pair('programbench', 'arl_agent', operation='train')
        self.assertEqual(get_env_spec('programbench').category, 'coding')
        cfg = GRPOConfig(model_name='test', log_path='/tmp/test', env='programbench', harness='arl_agent', programbench_reward_mode='strict')
        with self.assertRaises(HarnessCompatibilityError): _validate(cfg, 1)
        source = _make_source(cfg)
        self.assertEqual(source.cfg.train_set, 'programbench')
        self.assertEqual(source.cfg.reward_mode, 'strict')
        await source.close()

    async def test_native_token_rollout_to_grpo_and_correctness(self):
        source = self.source()
        workspaces = []
        def workspace(task):
            w = Workspace(score(1. if len(workspaces)%2==0 else .2))
            workspaces.append(w)
            return w
        source.backend.workspace = workspace
        try:
            result = await run_agent_rollout([ProgramTask(IDS[0])], source, Sampler(), Renderer(),
                RolloutParams(gen_cap=10, prompt_budget=100, group_size=2))
            self.assertEqual(result.returns, [1., 1., .2, .2])
            self.assertEqual(result.correctness, [True, True, False, False])
            data, _ = finalize_ppo_data(result.datums, result.returns, groups=result.groups, units=result.units,
                whiten=False, group_adv_std=False, length_weighted_baseline=False, response_correctness=result.correctness)
            self.assertTrue(bool((data[0].loss_fn_inputs['advantages'].to_torch()>0).any()))
            self.assertTrue(bool((data[-1].loss_fn_inputs['advantages'].to_torch()<0).any()))
            self.assertTrue(all(w.closed for w in workspaces))
        finally:
            await source.close()


class BenchmarkTests(unittest.TestCase):
    def test_default_mini_is_ready_but_inventory_does_not_execute_an_agent(self):
        from arl.benchmarks import run_benchmark, get_benchmark
        from arl.harnesses import HarnessPendingError
        from arl.experiments.config import load_experiment
        self.assertEqual(get_benchmark('programbench').default_harness, 'mini_swe_agent')
        cfg = GRPOConfig(model_name='test', log_path='/tmp/test', env='programbench')
        _validate(cfg, 1)
        with patch.object(ce.ProgramBenchBackend, 'inventory', new=AsyncMock(return_value=[{'instance_id': IDS[0]}])):
            result = run_benchmark('programbench')
        self.assertEqual(result['harness'], 'mini_swe_agent')
        self.assertEqual(result['harness_status'], 'ready')
        self.assertEqual(result['harness_execution'], 'not_run_offline_grading')
        path = Path(__file__).resolve().parents[1]/'experiments/programbench.yaml'
        exp = load_experiment(path)
        self.assertEqual(exp.plan('train').harness, 'mini_swe_agent')
        self.assertEqual(exp.plan('evaluate').harness, 'mini_swe_agent')

    def test_offline_evaluation_separates_partial_score_and_infrastructure_coverage(self):
        from arl.benchmarks import run_benchmark
        with tempfile.TemporaryDirectory() as root:
            with patch.object(ce.ProgramBenchBackend, 'inventory', new=AsyncMock(return_value=[{'instance_id': iid} for iid in IDS[:2]])), \
                 patch.object(ce.ProgramBenchBackend, 'health', new=AsyncMock(return_value={'version': '1.2.5'})), \
                 patch.object(ce.ProgramBenchBackend, 'evaluate', new=AsyncMock(side_effect=[
                     {'summary': score(.5)}, ce.BackendError('test image unavailable')])):
                result = run_benchmark('programbench', submissions_dir='/host/submissions', output_dir=str(Path(root)/'report'))
            self.assertEqual(result['tasks_scored'], 1)
            self.assertEqual(result['coverage'], .5)
            self.assertEqual(result['mean_test_pass_fraction_scored'], .5)
            self.assertEqual(result['resolved_rate_scored'], 0.)
            self.assertFalse(result['complete'])
            rows = [json.loads(line) for line in (Path(root)/'report/results.jsonl').read_text().splitlines()]
            self.assertEqual([row['status'] for row in rows], ['scored', 'infra_error'])


class WorkerTests(unittest.TestCase):
    def worker(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1]/'envs/coding/ProgramBenchEnv/worker.py'
        spec = importlib.util.spec_from_file_location('pb_worker_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_cleanroom_command_is_offline_unprivileged_and_workspace_paths_correct(self):
        worker = self.worker()
        name = 'arl-pb-'+'a'*32
        manifest = {'image_tag': 'task_cleanroom_v6'}
        with patch.object(worker, 'instances', return_value={IDS[0]: {}}), patch.object(worker, 'run', return_value={'exit_code': 0, 'output': 'id'}) as run:
            worker.dispatch(dict(action='start', name=name, instance_id=IDS[0], memory='4g', cpus=2, manifest=manifest))
            command = run.call_args.args[0]
            self.assertIn('--pull=never', command)
            self.assertEqual(command[command.index('--network')+1], 'none')
            self.assertEqual(command[command.index('--user')+1], 'agent')
            self.assertNotIn('/var/run/docker.sock', command)
        with self.assertRaisesRegex(ValueError, 'invalid owned'):
            worker.dispatch(dict(action='close', name='another-users-container'))

    def test_official_command_environment_preserves_full_output_and_timeout_feedback(self):
        from unittest.mock import Mock
        import subprocess
        worker = self.worker()
        request = dict(action='exec', name='arl-pb-'+'a'*32, command='inspect', command_timeout=180,
            output_limit=None, environment={'LESS': '-R', 'TQDM_DISABLE': '1'}, interpreter=['bash', '-lc'])
        output = 'HEAD'+('x'*20000)+'TAIL'
        with patch.object(worker.subprocess, 'run', return_value=Mock(returncode=0, stdout=output)) as run:
            result = worker.dispatch(request)
            self.assertEqual(result['output'], output)
            self.assertEqual(run.call_args.args[0][-3:], ['bash', '-lc', 'inspect'])
            self.assertIn('LESS=-R', run.call_args.args[0])
            self.assertEqual(run.call_args.kwargs['timeout'], 180)
        with patch.object(worker.subprocess, 'run', side_effect=subprocess.TimeoutExpired('inspect', 180, output=b'partial')):
            result = worker.dispatch(request)
            self.assertEqual(result['exit_code'], -1)
            self.assertEqual(result['output'], 'partial')
            self.assertIn('timed out', result['exception_info'])


if __name__ == '__main__': unittest.main()
