"""Real Claude Code CLI with a suspended Anthropic endpoint for on-policy sampling.

The CLI owns the agent loop. ARL supplies each model response from its token
sampler, and the sole MCP tool executes in the existing remote SWE workspace.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
import uuid

VERSION = '2.1.226'
TOOL = 'mcp__swe__bash'


@lru_cache(maxsize=1)
def check_dependencies():
    import aiohttp  # noqa: F401
    binary = os.environ.get('ARL_CLAUDE_CODE_BIN') or shutil.which('claude')
    if not binary:
        candidate = Path.home() / '.local/bin/claude'
        binary = str(candidate) if candidate.is_file() else None
    if not binary:
        raise RuntimeError(f'Claude Code {VERSION} required; set ARL_CLAUDE_CODE_BIN')
    result = subprocess.run([binary, '--version'], capture_output=True, text=True, timeout=15, check=True)
    if result.stdout.split()[0] != VERSION:
        raise RuntimeError(f'Claude Code requires {VERSION}, found {result.stdout.strip()}')
    return binary


def _text(blocks):
    if isinstance(blocks, str):
        return blocks
    result = []
    for block in blocks:
        if block['type'] != 'text':
            raise ValueError(f"unsupported Claude content block: {block['type']}")
        result.append(block['text'])
    return '\n'.join(result)


def translate_request(body):
    """Render the actual CLI request as Qwen messages and function schemas."""
    messages = []
    if body.get('system'):
        messages.append({'role': 'system', 'content': _text(body['system'])})
    for message in body['messages']:
        content = message['content']
        if isinstance(content, str):
            messages.append({'role': message['role'], 'content': content})
            continue
        text, calls = [], []
        for block in content:
            kind = block['type']
            if kind == 'tool_use':
                calls.append({'id': block['id'], 'type': 'function', 'function': {
                    'name': block['name'], 'arguments': block['input']}})
            elif kind == 'tool_result':
                if text:
                    messages.append({'role': message['role'], 'content': '\n'.join(text)})
                    text = []
                messages.append({'role': 'tool', 'tool_call_id': block['tool_use_id'],
                                 'content': _text(block['content'])})
            elif kind == 'text':
                text.append(block['text'])
            elif kind == 'thinking':
                # CLI stores the provider's thinking separately from visible text.
                continue
            else:
                raise ValueError(f'unsupported Claude content block: {kind}')
        if text or calls:
            converted = {'role': message['role'], 'content': '\n'.join(text)}
            if calls:
                converted['tool_calls'] = calls
            messages.append(converted)
    tools = [{'type': 'function', 'function': {'name': tool['name'],
              'description': tool.get('description', ''), 'parameters': tool['input_schema']}}
             for tool in body.get('tools', [])]
    if any(t['function']['name'] != TOOL for t in tools):
        raise ValueError('Claude Code exposed a tool outside the SWE workspace')
    return messages, tools


def translate_response(text, tool_names):
    """Qwen XML/Hermes calls -> Anthropic blocks, without re-encoding tokens."""
    visible = text.split('</think>', 1)[-1] if '</think>' in text else text
    if '<think>' in visible:
        raise ValueError('incomplete reasoning block')
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', visible, re.S)
    if visible.count('<tool_call>') != len(blocks):
        raise ValueError('incomplete tool call')
    result = []
    prose = re.sub(r'<tool_call>.*?</tool_call>', '', visible, flags=re.S).strip()
    if prose:
        result.append({'type': 'text', 'text': prose})
    for block in blocks:
        if block.lstrip().startswith('{'):
            call = json.loads(block)
            name, arguments = call['name'], call.get('arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        else:
            match = re.fullmatch(r'<function=([\w.-]+)>\s*(.*?)\s*</function>', block, re.S)
            if not match:
                raise ValueError('invalid function call')
            name = match[1]
            arguments = {key: value for key, value in re.findall(
                r'<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>', match[2], re.S)}
        if name not in tool_names or name != TOOL:
            raise ValueError(f'unknown workspace tool: {name}')
        if not isinstance(arguments, dict) or set(arguments) != {'command'} or not isinstance(arguments['command'], str) or not arguments['command'].strip():
            raise ValueError('workspace bash requires one nonempty command string')
        result.append({'type': 'tool_use', 'id': 'toolu_' + uuid.uuid4().hex,
                       'name': name, 'input': arguments})
    return result or [{'type': 'text', 'text': '(No response)'}]


def model_response(blocks, model):
    return {'id': 'msg_' + uuid.uuid4().hex, 'type': 'message', 'role': 'assistant',
            'model': model, 'content': blocks, 'stop_reason':
            'tool_use' if any(b['type'] == 'tool_use' for b in blocks) else 'end_turn',
            'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}


class ClaudeCodeEnv:
    def __init__(self, env):
        self.env = env
        self.done = False
        self.tools = []
        self._messages = []
        self._queue = asyncio.Queue()
        self._pending = None
        self._runner = self._process = self._completion = self._tmp = None
        self._turns = 0
        self._infra = False
        self._closed = False
        self._feedback = []
        self._tool_error = ''
        self._secret = secrets.token_urlsafe(32)

    async def reset(self):
        from aiohttp import web
        await self.env.reset()
        binary = check_dependencies()
        self._tmp = tempfile.TemporaryDirectory(prefix='arl-claude-')
        app = web.Application(client_max_size=16 * 1024**2)
        app.router.add_post('/v1/messages', self._model)
        app.router.add_post('/v1/messages/count_tokens', self._count_tokens)
        app.router.add_post('/mcp', self._mcp)
        self._runner = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=1)
        await self._runner.setup()
        site = web.TCPSite(self._runner, '127.0.0.1', 0)
        await site.start()
        url = 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])
        config = {'mcpServers': {'swe': {'type': 'http', 'url': url + '/mcp',
                   'headers': {'Authorization': 'Bearer ' + self._secret}}}}
        # No inherited credentials, proxy, hooks, project config, plugins or local
        # tools. The only execution capability is the remote workspace MCP tool.
        child_env = {k: v for k, v in os.environ.items() if k in
                     ('PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH')}
        child_env.update(HOME=self._tmp.name, CLAUDE_CONFIG_DIR=self._tmp.name + '/config',
            ANTHROPIC_BASE_URL=url, ANTHROPIC_API_KEY=self._secret,
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', DISABLE_AUTOUPDATER='1',
            ENABLE_TOOL_SEARCH='false', CLAUDE_CODE_DISABLE_AUTO_MEMORY='1',
            NO_PROXY='127.0.0.1,localhost')
        prompt = (f'Fix the issue in repository {self.env.task.repo}. '
            'All repository files are in /testbed on the remote SWE workspace. '
            'Use mcp__swe__bash to inspect, edit and test there; each call uses a fresh shell. '
            'Do not modify tests or commit changes. When finished, respond with a final summary; '
            'the current repository diff will be submitted for official grading.\n\n'
            + self.env.task.problem_statement)
        self._process = await asyncio.create_subprocess_exec(binary, '--bare', '-p', prompt,
            '--model', 'arl-qwen', '--tools', '', '--strict-mcp-config',
            '--mcp-config', json.dumps(config), '--allowedTools', TOOL,
            '--permission-mode', 'dontAsk', '--setting-sources', '',
            '--disable-slash-commands', '--no-session-persistence', '--output-format', 'json',
            cwd=self._tmp.name, env=child_env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self._completion = asyncio.create_task(self._process.communicate())
        await self._advance()
        if self._pending is None:
            raise RuntimeError('Claude Code exited without requesting local model inference')

    def _authorized(self, request, mcp=False):
        return request.headers.get('Authorization' if mcp else 'x-api-key') == (
            'Bearer ' + self._secret if mcp else self._secret)

    async def _count_tokens(self, request):
        from aiohttp import web
        if not self._authorized(request):
            raise web.HTTPUnauthorized()
        body = await request.json()
        return web.json_response({'input_tokens': max(1, len(json.dumps(body)) // 3)})

    async def _model(self, request):
        from aiohttp import web
        if not self._authorized(request):
            raise web.HTTPUnauthorized()
        body = await request.json()
        future = asyncio.get_running_loop().create_future()
        await self._queue.put((body, future))
        response = await future
        if not body.get('stream'):
            return web.json_response(response)
        events = [('message_start', {'type': 'message_start', 'message': {
            **response, 'content': [], 'stop_reason': None}})]
        for i, block in enumerate(response['content']):
            start = {**block, **({'text': ''} if block['type'] == 'text' else {'input': {}})}
            delta = ({'type': 'text_delta', 'text': block['text']} if block['type'] == 'text' else
                     {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])})
            events.extend([('content_block_start', {'type': 'content_block_start', 'index': i, 'content_block': start}),
                ('content_block_delta', {'type': 'content_block_delta', 'index': i, 'delta': delta}),
                ('content_block_stop', {'type': 'content_block_stop', 'index': i})])
        events.extend([('message_delta', {'type': 'message_delta', 'delta': {
            'stop_reason': response['stop_reason'], 'stop_sequence': None}, 'usage': {'output_tokens': 0}}),
            ('message_stop', {'type': 'message_stop'})])
        return web.Response(text=''.join(f'event: {name}\ndata: {json.dumps(data)}\n\n' for name, data in events),
                            content_type='text/event-stream')

    async def _mcp(self, request):
        from aiohttp import web
        if not self._authorized(request, mcp=True):
            raise web.HTTPUnauthorized()
        body = await request.json()
        method = body['method']
        if 'id' not in body:
            return web.Response(status=202)
        if method == 'initialize':
            result = {'protocolVersion': body['params']['protocolVersion'], 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'arl-swe', 'version': '1.0'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': 'bash', 'description': 'Execute bash in the persistent remote /testbed repository.',
                'inputSchema': {'type': 'object', 'properties': {'command': {'type': 'string'}},
                                'required': ['command'], 'additionalProperties': False}}]}
        elif method == 'tools/call':
            params = body['params']
            try:
                if params['name'] != 'bash':
                    raise ValueError('unknown tool')
                args = params['arguments']
                if set(args) != {'command'} or not isinstance(args['command'], str) or not args['command'].strip():
                    raise ValueError('expected one nonempty command')
                output = await self.env.workspace.execute(args['command'], timeout=self.env.cfg.command_timeout,
                                                         output_limit=self.env.cfg.max_tool_result_chars)
                text = f"exit_code={output['exit_code']}\n{output['output']}"
                self._feedback.append((TOOL, text))
                result = {'content': [{'type': 'text', 'text': text}], 'isError': False}
            except Exception as exc:
                self._tool_error = str(exc)
                result = {'content': [{'type': 'text', 'text': 'Workspace infrastructure failed'}], 'isError': True}
        elif method == 'ping':
            result = {}
        else:
            return web.json_response({'jsonrpc': '2.0', 'id': body['id'], 'error': {
                'code': -32601, 'message': 'Method not found'}})
        return web.json_response({'jsonrpc': '2.0', 'id': body['id'], 'result': result})

    async def _advance(self):
        request = asyncio.create_task(self._queue.get())
        try:
            ready, _ = await asyncio.wait([request, self._completion], timeout=self.env.cfg.max_calls_per_turn * self.env.cfg.command_timeout + 120,
                                          return_when=asyncio.FIRST_COMPLETED)
            if request in ready:
                self._body, self._pending = request.result()
                self._messages, self.tools = translate_request(self._body)
                return
            if self._completion in ready:
                stdout, stderr = self._completion.result()
                if self._process.returncode:
                    raise RuntimeError('Claude Code exited: ' + stderr.decode(errors='replace')[-1500:])
                result = json.loads(stdout)
                if result.get('is_error'):
                    raise RuntimeError('Claude Code failed: ' + str(result.get('result', result.get('subtype'))))
                return
            raise TimeoutError('Claude Code did not request a model response or finish')
        finally:
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)

    def messages(self, aggressiveness=0):
        # CLI owns compaction. Never silently replace its request with ARL history.
        return self._messages

    async def step(self, text, *, truncated=False):
        from arl.envs.coding.swe import _import_sweenv
        ce = _import_sweenv()
        if self.done or self._pending is None:
            raise RuntimeError('step outside a pending Claude Code request')
        self._turns += 1
        self._feedback = []
        try:
            if truncated:
                raise ValueError('response truncated; emit a shorter complete response')
            blocks = translate_response(text, {t['function']['name'] for t in self.tools})
            if sum(b['type'] == 'tool_use' for b in blocks) > self.env.cfg.max_calls_per_turn:
                raise ValueError('too many workspace tool calls in one response')
        except (ValueError, KeyError, TypeError) as exc:
            self._messages = [*self._messages, {'role': 'assistant', 'content': text},
                              {'role': 'user', 'content': f'Invalid model response: {exc}'}]
            if self._turns >= self.env.cfg.max_turns:
                return await self.finish('max turns')
            return ce.StepOutput(0., False, [], {'format_ok': 0., 'n_invalid_calls': 1})
        self._pending.set_result(model_response(blocks, self._body['model']))
        self._pending = None
        try:
            await self._advance()
            if self._tool_error:
                raise RuntimeError(self._tool_error)
        except Exception as exc:
            self._infra = True
            self._error = str(exc)
            return await self.finish('infrastructure failure')
        if self._pending is None or self._turns >= self.env.cfg.max_turns:
            return await self.finish()
        return ce.StepOutput(0., False, list(self._feedback), {'format_ok': 1., 'n_tool_calls': len(self._feedback)})

    async def finish(self, reason=''):
        if self.done:
            raise RuntimeError('finish after Claude Code episode ended')
        self.done = True
        await self._stop()
        self.env._turns = self._turns
        result = await self.env._finish(infra=self._infra)
        result.tool_results = list(self._feedback)
        return result

    def summary(self):
        return {**self.env.summary(), 'harness': 'claude_code', 'claude_code_version': VERSION,
                'model_bridge': 'arl.local_tokens', 'workspace_tools': 'mcp',
                'turns': self._turns, 'harness_error': getattr(self, '_error', '')}

    async def _stop(self):
        if self._pending and not self._pending.done():
            self._pending.cancel()
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(asyncio.shield(self._completion), 10)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._completion
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._tmp:
            self._tmp.cleanup()
            self._tmp = None

    async def close(self):
        if not self._closed:
            await self._stop()
            await self.env.close()
            self._closed = True


async def run(*, tasks, source, sampling_client, renderer, params, operation='train',
              env_name='coding.swe', uid_prefix='', log_turns=False):
    if env_name not in ('swe', 'coding.swe'):
        raise ValueError('claude_code currently supports coding.swe only')
    check_dependencies()
    class Source:
        def task_key(self, task): return source.task_key(task)
        def make_env(self, task, uid): return ClaudeCodeEnv(source.make_env(task, uid))
    from arl.agent.rollout import run_agent_rollout
    return await run_agent_rollout(tasks, Source(), sampling_client, renderer, params,
        train=operation in ('collect', 'train'), uid_prefix=uid_prefix, log_turns=log_turns)
