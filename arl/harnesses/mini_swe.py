"""Upstream DefaultAgent.step with ARL token sampling and official env rewards."""
import asyncio
from concurrent.futures import CancelledError, ThreadPoolExecutor
import contextvars
import importlib
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import re
from types import SimpleNamespace

VERSION = '2.4.6'
ROOT = Path(__file__).resolve().parents[2]
SWE_CONFIG_SHA256 = '9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e'
PROGRAMBENCH_CONFIG_SHA256 = '9c2cc36b69c53e83cb4f60b0aa809768996373f4c3dddb3a202b78e93899bf8e'


def _official_config(name, expected_sha256):
    """Read the pinned release's actual config; never maintain a prompt copy."""
    import minisweagent
    import yaml
    path = Path(minisweagent.__file__).parent / f'config/benchmarks/{name}.yaml'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f'mini_swe_agent official {name}.yaml hash mismatch; reinstall pinned 2.4.6')
    return yaml.safe_load(raw)


def swe_config():
    return _official_config('swebench', SWE_CONFIG_SHA256)


def programbench_config():
    return _official_config('programbench', PROGRAMBENCH_CONFIG_SHA256)


def check_dependencies():
    cache = ROOT / '.cache/benchmarks/mini-swe-agent'
    if cache.is_dir() and str(cache) not in sys.path:
        sys.path.insert(0, str(cache))
    os.environ.setdefault('MSWEA_SILENT_STARTUP', '1')
    os.environ.setdefault('MSWEA_GLOBAL_CONFIG_DIR', str(ROOT / '.cache/benchmarks/mini-swe-config'))
    try:
        upstream = importlib.import_module('minisweagent')
        importlib.import_module('minisweagent.agents.default')
    except ImportError as exc:
        raise RuntimeError('mini_swe_agent dependencies missing; run bash script/setup_mini_swe_agent.sh') from exc
    if upstream.__version__ != VERSION:
        raise RuntimeError(f'mini_swe_agent requires {VERSION}, found {upstream.__version__}')
    swe_config()
    programbench_config()


class TokenModel:
    """Supply the sampled response to upstream; retain raw text for token replay.

    Both benchmarks use upstream structured tool messages and templates. Sampled token
    IDs and old-policy logprobs stay in the shared rollout, without re-encoding.
    """
    text = ''
    truncated = False

    def __init__(self, official=None):
        self.official = official

    def format_message(self, **kwargs): return kwargs

    def query(self, messages):
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.actions_toolcall import parse_toolcall_actions
        from arl.envs.coding.swe import _import_sweenv
        _import_sweenv()
        from sweenv.tools import parse_tool_calls, split_reasoning
        reasoning, content = split_reasoning(self.text)
        response = dict(role='assistant', content=content, reasoning_content=reasoning)
        calls, errors = parse_tool_calls(self.text)
        if self.truncated:
            errors = ['Response truncated. Emit a shorter, complete bash tool call.']
        if any(c['name'] != 'bash' or not isinstance(c['arguments'].get('command'), str) for c in calls):
            errors.append('Only bash with a string command is supported. Submit using echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.')
        try:
            if errors and self.official:
                from jinja2 import Template, StrictUndefined
                feedback = Template(self.official['format_error_template'], undefined=StrictUndefined).render(
                    error='\n'.join(errors), actions=[], has_tool_calls=bool(calls),
                    finish_reason='length' if self.truncated else 'stop')
                raise FormatError(dict(role='user', content=feedback, extra={'interrupt_type': 'FormatError'}))
            if errors:
                raise FormatError(dict(role='user', content='\n'.join(errors), extra={}))
            tool_calls = [SimpleNamespace(id=str(i), function=SimpleNamespace(
                name=c['name'], arguments=json.dumps(c['arguments']))) for i, c in enumerate(calls)]
            actions = parse_toolcall_actions(tool_calls,
                format_error_template=self.official['format_error_template'] if self.official else '{{ error }}',
                template_kwargs={'finish_reason': 'length' if self.truncated else 'stop'})
        except FormatError as exc:
            if self.official:
                # Upstream LitellmModel records a bad response in extra metadata,
                # and only appends the error observation to the model history.
                exc.messages[0].setdefault('extra', {}).update(response=response, cost=0.)
                raise
            raise FormatError(response, *exc.messages) from exc
        if self.official:
            response['content'] = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
            response['tool_calls'] = [{'id': str(i), 'type': 'function', 'function': {
                'name': c['name'], 'arguments': json.dumps(c['arguments'])}} for i, c in enumerate(calls)]
        return {**response, 'extra': {'actions': actions}}

    def format_observation_messages(self, message, outputs, template_vars=None):
        if self.official:
            from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages
            return format_toolcall_observation_messages(actions=message['extra']['actions'], outputs=outputs,
                observation_template=self.official['observation_template'], template_vars=template_vars)
        from minisweagent.models.utils.actions_text import format_observation_messages
        return format_observation_messages(outputs, observation_template=(
            '<returncode>{{ output.returncode }}</returncode>\n<output>{{ output.output }}</output>'))

    def get_template_vars(self): return self.official or {}
    def serialize(self):
        return {'info': {'config': {'model_bridge': 'arl.local_tokens',
            'model_templates': self.official or {}}}}


class WorkspaceBridge:
    def __init__(self, owner):
        self.owner = owner
        self.loop = asyncio.get_running_loop()
        self.pending = None
        self.closed = False

    def execute(self, action):
        if self.closed:
            raise CancelledError('workspace closed')
        env = self.owner.env
        options = dict(timeout=env.cfg.command_timeout, output_limit=env.cfg.max_tool_result_chars)
        if self.owner.official:
            options.update(output_limit=None, environment=self.owner.config['environment']['env'],
                           interpreter=self.owner.config['environment'].get('interpreter', ['bash', '-lc']))
        self.pending = asyncio.run_coroutine_threadsafe(env.workspace.execute(action['command'],
            **options), self.loop)
        if self.closed:
            self.pending.cancel()
        try:
            result = self.pending.result()
        finally:
            self.pending = None
        output = dict(output=result['output'], returncode=result['exit_code'], exception_info=result.get('exception_info', ''))
        from minisweagent.environments.docker import DockerEnvironment
        # Official submission semantics, without constructing a second container.
        DockerEnvironment._check_finished(self, output)
        return output

    def get_template_vars(self):
        # Same source as upstream DockerEnvironment.get_template_vars: the
        # controller's uname, not an invented task-specific system description.
        return {**(self.owner.config or {}).get('environment', {}), **platform.uname()._asdict()}
    def serialize(self): return {}


class MiniEnv:
    def __init__(self, env, *, official=None, agent_executor=None):
        self.env = env
        self.agent_executor = agent_executor
        self.programbench = env.workspace_dir == '/workspace'
        self.official = True if official is None else official
        self.worker = None
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
        self.tools = [BASH_TOOL]

    async def _run_agent(self, function, *args):
        if self.agent_executor is None:
            return await asyncio.to_thread(function, *args)
        return await asyncio.get_running_loop().run_in_executor(
            self.agent_executor, contextvars.copy_context().run, function, *args)

    async def reset(self):
        from minisweagent.agents.default import DefaultAgent
        await self.env.reset()
        self.config = (programbench_config() if self.programbench else swe_config()) if self.official else None
        self.model = TokenModel(self.config['model'] if self.official else None)
        self.bridge = WorkspaceBridge(self)
        if self.official:
            if self.programbench:
                # Official runner initializes git identity before agent.run().
                self.worker = asyncio.create_task(self._run_agent(self.bridge.execute, {'command':
                    'git config user.name "mini-swe-agent" && git config user.email "mini-swe-agent@proton.me"'}))
                await asyncio.shield(self.worker)
            self.agent = DefaultAgent(self.model, self.bridge, **{
                **self.config['agent'], 'step_limit': self.env.cfg.max_turns,
                # A local on-policy sampler has no API dollar billing. Preserve
                # and disclose this override rather than inventing model cost.
                'cost_limit': 0})
            self.agent.extra_template_vars = ({'task': '', 'instance': {
                'instance_id': self.env.task.instance_id,
                'image_name': 'programbench/'+self.env.task.instance_id.replace('__', '_1776_')}}
                if self.programbench else {'task': self.env.task.problem_statement})
            self.agent.add_messages(
                self.model.format_message(role='system', content=self.agent._render_template(self.agent.config.system_template)),
                self.model.format_message(role='user', content=self.agent._render_template(self.agent.config.instance_template)))
            return
        self.agent = DefaultAgent(self.model, self.bridge, system_template='{{ system_prompt }}',
            instance_template='{{ task }}', step_limit=self.env.cfg.max_turns, cost_limit=0)
        initial = self.env.initial_messages()
        system = initial[0]['content'].replace(
            'Call submit when finished; a final response without tool calls also submits the current changes.', '').replace(
            'Call submit when finished; a final answer without tools also submits.', '')
        system += ('\nEvery response must include a bash tool call. To submit, run the command '
                   '`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` by itself. Plain final text does not submit.')
        self.agent.extra_template_vars = {'system_prompt': system, 'task': initial[1]['content']}
        self.agent.add_messages(dict(role='system', content=self.agent._render_template(self.agent.config.system_template)),
                                dict(role='user', content=self.agent._render_template(self.agent.config.instance_template)))

    def messages(self, aggressiveness=0):
        # Full upstream history; context overflow ends the episode.
        result = []
        for original in self.agent.messages:
            if original['role'] == 'exit':
                continue
            message = {k: v for k, v in original.items() if k != 'extra'}
            if self.official and message.get('tool_calls'):
                # HF/Qwen templates expect argument mappings; the upstream/API
                # history retains JSON strings. Convert only the rendering view.
                message['tool_calls'] = [{**call, 'function': {**call['function'],
                    'arguments': json.loads(call['function']['arguments'])}}
                    for call in message['tool_calls']]
            result.append(message)
        return result

    async def step(self, text, *, truncated=False):
        from minisweagent.exceptions import FormatError, InterruptAgentFlow
        from arl.envs.coding.swe import _import_sweenv
        StepOutput = _import_sweenv().StepOutput
        self.model.text, self.model.truncated = text, truncated
        start = len(self.agent.messages)
        invalid = False
        self.worker = asyncio.create_task(self._run_agent(self.agent.step))
        try:
            await asyncio.shield(self.worker)
            self.agent.n_consecutive_format_errors = 0
        except FormatError as exc:
            invalid = True
            self.agent.n_consecutive_format_errors += 1
            self.agent.add_messages(*exc.messages)
            if self.agent.n_consecutive_format_errors >= self.agent.config.max_consecutive_format_errors:
                self.agent.add_messages(dict(role='exit', content='RepeatedFormatError',
                    extra={'exit_status': 'RepeatedFormatError'}))
        except InterruptAgentFlow as exc:
            self.agent.add_messages(*exc.messages)
        self.env._turns = self.agent.n_calls
        feedback = [(m['role'], m.get('content', '')) for m in self.agent.messages[start:] if m['role'] != 'assistant']
        if self.official and self.agent.messages[-1]['role'] != 'exit' and self.agent.n_calls >= self.agent.config.step_limit:
            # Match the next upstream query's LimitsExceeded exit without an
            # unnecessary extra sampler request.
            from minisweagent.exceptions import LimitsExceeded
            try:
                self.agent.query()
            except LimitsExceeded as exc:
                self.agent.add_messages(*exc.messages)
        terminal = self.agent.messages[-1]['role'] == 'exit' or self.agent.n_calls >= self.env.cfg.max_turns
        out = await self.finish('agent stopped') if terminal else StepOutput(0., False, [], {})
        out.tool_results = feedback
        out.metrics.update(n_invalid_calls=int(invalid), format_ok=float(not invalid))
        return out

    async def finish(self, reason=''):
        self.exit_reason = self.agent.messages[-1].get('extra', {}).get('exit_status', reason)
        if self.official and not self.programbench:
            submission = self.agent.messages[-1].get('extra', {}).get('submission', '')
            return await self.env.finish_submission(submission)
        return await self.env.finish(reason)

    def summary(self):
        agent_data = self.agent.serialize()
        if self.programbench:
            # Match upstream ProgramBenchAgent's trajectory size reduction.
            for message in agent_data.get('messages', []):
                extra = message.get('extra', {})
                extra.pop('raw_output', None)
                for observation in extra.get('observations', []):
                    observation.pop('raw_output', None)
        backend_config = getattr(getattr(self.env.workspace, 'backend', None), 'cfg', None)
        return {**self.env.summary(), 'harness': 'mini_swe_agent', 'mini_version': VERSION,
                'exit_status': getattr(self, 'exit_reason', ''), 'agent': agent_data,
                'harness_profile': ('upstream_programbench' if self.programbench else 'upstream_swebench') if self.official else 'arl_legacy_bridge',
                'upstream_config_sha256': (PROGRAMBENCH_CONFIG_SHA256 if self.programbench else SWE_CONFIG_SHA256) if self.official else None,
                'runtime_overrides': {'step_limit': self.agent.config.step_limit,
                    'command_timeout': self.env.cfg.command_timeout, 'cost_limit': self.agent.config.cost_limit,
                    'wall_time_limit_seconds': self.agent.config.wall_time_limit_seconds,
                    'memory': getattr(backend_config, 'memory', None), 'cpus': getattr(backend_config, 'cpus', None),
                    'transport': 'arl.local_tokens',
                    'container_isolation': 'arl.programbench' if self.programbench else 'arl.swe'} if self.official else {}}

    async def close(self):
        if hasattr(self, 'bridge'):
            self.bridge.closed = True
            if self.bridge.pending is not None:
                self.bridge.pending.cancel()
        if self.worker is not None:
            await asyncio.gather(self.worker, return_exceptions=True)
        await self.env.close()


async def run(*, tasks, source, sampling_client, renderer, params, operation,
              env_name=None, uid_prefix='', log_turns=False):
    if env_name not in ('coding.swe', 'coding.programbench'):
        raise ValueError('mini_swe_agent supports coding.swe and coding.programbench only')
    if operation not in ('collect', 'train', 'evaluate'):
        raise ValueError(f'unsupported mini_swe_agent operation {operation}')
    check_dependencies()
    training = operation in ('collect', 'train')
    from arl.agent.rollout import trajectory_slots
    from arl.utils.distributed import dist_info
    rank, world_size, _ = dist_info()
    agent_workers = max(1, len(trajectory_slots(len(tasks), params.group_size +
        (params.extra_samples if training else 0), rank, world_size, params.shard_trajectories)))
    agent_executor = ThreadPoolExecutor(max_workers=agent_workers, thread_name_prefix='mini-agent')
    class Source:
        def task_key(self, task): return source.task_key(task)
        def make_env(self, task, uid):
            return MiniEnv(source.make_env(task, uid), agent_executor=agent_executor)
    from arl.agent.rollout import run_agent_rollout
    try:
        return await run_agent_rollout(tasks, Source(), sampling_client, renderer, params,
            train=training, uid_prefix=uid_prefix, log_turns=log_turns)
    finally:
        # Rollout cleanup cancels pending commands and joins agent tasks first.
        agent_executor.shutdown(wait=False, cancel_futures=True)
