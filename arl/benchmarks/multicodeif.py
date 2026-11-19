"""MultiCodeIF data/generation/constraint evaluation adapter.

Uses a pinned external checkout's checks and self-repair prompt. Generated code
is parsed, never executed here. Compilation/functional execution are not scored.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'envs/instruction_following/benchmarks/code/multicodeif'
CACHE = ROOT / '.cache/benchmarks'
SUITES = ('single_level', 'multi_level')


@dataclass(frozen=True)
class Task:
    task_id: str
    file: str
    row: dict
    checks: tuple[dict, ...]

    def public(self):
        # Expected answers/reference implementations are evaluator-only.
        return {'task_id': self.task_id, 'prompt': self.row['prompt']}


def setup(upstream_root=None, dependencies=None):
    import fcntl
    repo = Path(upstream_root or CACHE / 'MultiCodeIF').resolve()
    deps = Path(dependencies or CACHE / 'multicodeif-deps').resolve()
    pin = json.loads((ENTRY / 'upstream.json').read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / 'multicodeif-setup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', pin['repository'], str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', pin['commit']], check=True)
        revision(repo)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--target', str(deps),
                        '-r', str(ENTRY / 'requirements-rules.txt')], check=True)
    return {'upstream_root': str(repo), 'dependencies': str(deps), 'commit': pin['commit']}


def revision(repo):
    if not (repo / 'dataset/instructions').is_dir():
        raise ValueError(f'MultiCodeIF checkout missing: {repo}; run script/setup_multicodeif.sh first')
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
    commit = result.stdout.strip()
    expected = json.loads((ENTRY / 'upstream.json').read_text())['commit']
    if commit != expected:
        raise ValueError(f'MultiCodeIF revision {commit} differs from pinned {expected}')
    changed = subprocess.run(['git', '-C', str(repo), 'diff', '--quiet', 'HEAD', '--', 'src', 'dataset'])
    if changed.returncode:
        raise ValueError('MultiCodeIF evaluator/data have local changes; use the pinned clean checkout')
    return commit


def load_tasks(repo, *, suite='single_level', constraints=None, languages=None, limit=0,
               task_split='all', test_fraction=.1, split_seed=0):
    if suite not in SUITES:
        raise ValueError(f'suite must be one of {SUITES}')
    if limit < 0:
        raise ValueError('limit must be >= 0')
    files = sorted((Path(repo) / 'dataset/instructions' / suite).glob('*.json'))
    selected = set(constraints or [p.stem for p in files])
    unknown = selected - {p.stem for p in files}
    if unknown:
        raise ValueError(f'unknown constraint files: {sorted(unknown)}')
    tasks = []
    seen = set()
    for file in files:
        if file.stem not in selected:
            continue
        rows = json.loads(file.read_text())
        if not isinstance(rows, list):
            raise ValueError(f'{file} must contain a list')
        for i, row in enumerate(rows):
            if not isinstance(row.get('prompt'), str) or not isinstance(row.get('answer'), str):
                raise ValueError(f'{file}:{i} lacks prompt/answer')
            task_id = f'{suite}/{file.stem}:{row["id"]}'
            if task_id in seen:
                raise ValueError(f'duplicate task ID {task_id}')
            seen.add(task_id)
            if suite == 'single_level':
                checks = (dict(row),)
            else:
                level = row['level']
                if not isinstance(level, int) or level < 0 or i < level:
                    raise ValueError(f'invalid multilevel chain at {task_id}')
                checks = tuple(dict(r) for r in rows[i - level:i + 1])
                if [r['level'] for r in checks] != list(range(level + 1)):
                    raise ValueError(f'broken multilevel chain at {task_id}')
                # The released multilevel suite only covers data_structure.
                if row['category'] != 'data_structure':
                    raise ValueError(f'unsupported released multilevel category: {row["category"]}')
            if languages and row['answer'].split(';', 1)[0].lower() not in {x.lower() for x in languages}:
                continue
            tasks.append(Task(task_id, file.stem, dict(row), checks))
    if not tasks:
        raise ValueError('no MultiCodeIF tasks selected')
    if task_split not in ('all', 'train', 'test') or not 0 <= test_fraction < 1:
        raise ValueError('task_split must be all/train/test and test_fraction must be in [0, 1)')
    if task_split != 'all':
        import random
        def group(task):
            # A multilevel chain must never be split across train and test.
            return (task.file, task.checks[0]['id']) if suite == 'multi_level' else task.task_id
        groups = list(dict.fromkeys(group(task) for task in tasks))
        random.Random(split_seed).shuffle(groups)
        n_test = int(len(groups) * test_fraction)
        if test_fraction and len(groups) > 1:
            n_test = max(1, n_test)
        heldout = set(groups[:n_test])
        tasks = [task for task in tasks if (group(task) in heldout) == (task_split == 'test')]
        if not tasks:
            raise ValueError(f'no tasks in {task_split} split')
    return tasks[:limit] if limit else tasks


def inspect_dataset(repo):
    result = {}
    for suite in SUITES:
        tasks = load_tasks(repo, suite=suite)
        result[suite] = {'tasks': len(tasks), 'files': dict(sorted(Counter(t.file for t in tasks).items()))}
    return result


def response_text(value):
    if isinstance(value, str):
        return value
    # Official generations: response=[[prompt, completion]].
    if isinstance(value, list) and value and isinstance(value[0], list) and len(value[0]) == 2:
        text = value[0][1]
        if isinstance(text, str):
            return text
    raise ValueError('response must be a string or official [[prompt, completion]] format')


def load_predictions(path, tasks):
    path = Path(path)
    outputs = {}
    if path.is_dir():
        for stem in sorted({t.file for t in tasks}):
            files = [p for p in path.glob('*.json') if p.stem == stem or p.stem.startswith(stem + '_')]
            if len(files) != 1:
                raise ValueError(f'expected exactly one official generation file for {stem}, found {len(files)}')
            rows = json.loads(files[0].read_text())
            for row in rows:
                key = f'{tasks[0].task_id.split("/", 1)[0]}/{stem}:{row["id"]}'
                if key in outputs:
                    raise ValueError(f'duplicate prediction {key}')
                outputs[key] = [response_text(row['response'])]
    else:
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()] if path.suffix == '.jsonl' else json.load(f)
        if not isinstance(rows, list):
            raise ValueError('predictions must be JSON array or JSONL records')
        for row in rows:
            key = row.get('task_id')
            if not isinstance(key, str) or key in outputs:
                raise ValueError(f'missing or duplicate prediction task_id: {key!r}')
            values = row.get('responses')
            if values is None:
                values = [response_text(row['response'])]
            if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
                raise ValueError(f'{key}: responses must be a nonempty list of strings')
            outputs[key] = values
    missing = {task.task_id for task in tasks} - outputs.keys()
    if missing:
        raise ValueError(f'missing {len(missing)} predictions, e.g. {sorted(missing)[0]}')
    return outputs


def chat(config, messages, *, judge=False):
    prefix = 'judge_' if judge else ''
    base_url = config.get(prefix + 'base_url')
    model = config.get(prefix + 'model')
    if not base_url or not model:
        raise ValueError(f'{prefix}base_url and {prefix}model must be configured')
    key_env = config.get(prefix + 'api_key_env', 'MULTICODEIF_JUDGE_API_KEY' if judge else 'MULTICODEIF_API_KEY')
    key = os.environ.get(key_env, '')
    body = dict(model=model, messages=messages,
                temperature=config.get('judge_temperature' if judge else 'temperature', 0.0),
                max_tokens=config.get('judge_max_tokens' if judge else 'max_tokens', 512 if judge else 4096))
    if not judge and config.get('chat_template_kwargs'):
        body['chat_template_kwargs'] = config['chat_template_kwargs']
    request = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
        **({'Authorization': 'Bearer ' + key} if key else {})})
    try:
        with urllib.request.urlopen(request, timeout=config.get('request_timeout', 180)) as response:
            result = json.load(response)
    except urllib.error.HTTPError as e:
        # Do not echo endpoint response bodies: gateways may include credentials.
        raise RuntimeError(f'{"judge" if judge else "model"} API HTTP {e.code}') from None
    choice = result['choices'][0]
    text = choice['message'].get('content')
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError('model API returned no completion text')
    if choice.get('finish_reason') == 'length':
        raise RuntimeError('model completion truncated at the token budget')
    if judge and text.strip().upper() not in ('YES', 'NO'):
        raise RuntimeError('judge must return exactly YES or NO')
    return text


def evaluate_round(repo, deps, requests, config, output_dir):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(deps) + os.pathsep + str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    payload = {'repo': str(repo), 'requests': requests, 'config': config}
    process = subprocess.run([sys.executable, str(ENTRY / 'worker.py')],
        input=json.dumps(payload), capture_output=True, text=True, env=env, cwd=output_dir,
        timeout=config.get('evaluation_timeout', 600))
    if process.returncode:
        raise RuntimeError('MultiCodeIF evaluator failed: ' + process.stderr[-2000:])
    try:
        result = json.loads(process.stdout)
    except ValueError as e:
        raise RuntimeError('MultiCodeIF evaluator returned invalid JSON: ' + process.stderr[-1000:]) from e
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def summarize(results):
    checks = [check for result in results for check in result['checks']]
    scored = [c for c in checks if c['status'] == 'scored']
    passed = sum(c['passed'] for c in scored)
    complete_tasks = [r for r in results if r['status'] == 'scored']
    return {'tasks': len(results), 'tasks_scored': len(complete_tasks), 'checks_requested': len(checks),
            'checks_scored': len(scored), 'checks_passed': passed,
            'coverage': len(scored) / len(checks) if checks else 0.,
            'constraint_accuracy_scored': passed / len(scored) if scored else None,
            'all_constraints_passed_scored': sum(r['passed'] for r in complete_tasks) / len(complete_tasks) if complete_tasks else None,
            'complete': bool(checks) and len(scored) == len(checks),
            'functional_correctness': 'not_evaluated', 'compilation': 'not_evaluated',
            'unavailable_or_error': dict(Counter(c['status'] for c in checks if c['status'] != 'scored'))}


def validate_options(config):
    allowed = {'action', 'upstream_root', 'dependencies', 'suite', 'constraints', 'languages', 'limit',
               'predictions', 'output_dir', 'base_url', 'model', 'api_key_env', 'temperature', 'max_tokens',
               'judge_base_url', 'judge_model', 'judge_api_key_env', 'judge_temperature', 'judge_max_tokens',
               'request_timeout', 'evaluation_timeout', 'max_rounds', 'concurrency', 'export_tasks',
               'chat_template_kwargs', 'model_path', 'harness', 'task_split', 'test_fraction', 'split_seed'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'unknown MultiCodeIF options: {sorted(unknown)}')
    if config.get('action', 'inspect') not in ('inspect', 'evaluate', 'export'):
        raise ValueError('action must be inspect, evaluate or export')
    if config.get('task_split', 'all') not in ('all', 'train', 'test'):
        raise ValueError('task_split must be all, train or test')
    if not isinstance(config.get('test_fraction', .1), (float, int)) or not 0 <= config.get('test_fraction', .1) < 1:
        raise ValueError('test_fraction must be in [0, 1)')
    if not isinstance(config.get('split_seed', 0), int):
        raise ValueError('split_seed must be an integer')
    if config.get('suite', 'single_level') not in SUITES:
        raise ValueError(f'suite must be one of {SUITES}')
    for field in ('constraints', 'languages'):
        if field in config and (not isinstance(config[field], list) or not all(isinstance(v, str) for v in config[field])):
            raise ValueError(f'{field} must be a list of strings')
    for field, low, high in (('max_rounds', 1, 16), ('concurrency', 1, None), ('limit', 0, None)):
        if field in config and (not isinstance(config[field], int) or isinstance(config[field], bool)
                                or config[field] < low or (high is not None and config[field] > high)):
            raise ValueError(f'invalid {field}')


def benchmark(**config):
    validate_options(config)
    from arl.harnesses import get_harness, resolve_pair
    pair = resolve_pair('instruction_following.multicodeif', config.get('harness'), operation='evaluate')
    selected_harness = get_harness(pair.harness)
    repo = Path(config.get('upstream_root', CACHE / 'MultiCodeIF')).resolve()
    deps = Path(config.get('dependencies', CACHE / 'multicodeif-deps')).resolve()
    commit = revision(repo)
    action = config.get('action', 'evaluate' if config.get('predictions') or config.get('model') else 'inspect')
    if action == 'inspect':
        return {'benchmark': 'instruction_following.multicodeif', 'commit': commit,
                'upstream_root': str(repo), 'datasets': inspect_dataset(repo),
                'evaluation_scope': 'official_constraint_checks', 'harness': pair.harness,
                'training': 'local_on_policy_grpo', 'training_entry': 'experiments/multicodeif_grpo.yaml'}
    if action not in ('evaluate', 'export'):
        raise ValueError('action must be inspect, evaluate or export')
    tasks = load_tasks(repo, suite=config.get('suite', 'single_level'), constraints=config.get('constraints'),
                       languages=config.get('languages'), limit=config.get('limit', 0),
                       task_split=config.get('task_split', 'all'), test_fraction=config.get('test_fraction', .1),
                       split_seed=config.get('split_seed', 0))
    if action == 'export':
        if not config.get('export_tasks'):
            raise ValueError('export action requires export_tasks path')
        target = Path(config['export_tasks'])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w') as f:
            for task in tasks:
                f.write(json.dumps(task.public(), ensure_ascii=False) + '\n')
        return {'tasks': len(tasks), 'export_tasks': str(target)}
    if not config.get('predictions') and not (config.get('base_url') and config.get('model')):
        raise ValueError('evaluate requires predictions or base_url + model')
    rounds = config.get('max_rounds', 1)
    concurrency = config.get('concurrency', 4)
    if not isinstance(rounds, int) or not 1 <= rounds <= 16 or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError('max_rounds must be 1..16; concurrency must be >= 1')
    predictions = load_predictions(config['predictions'], tasks) if config.get('predictions') else None
    output = Path(config.get('output_dir', ROOT / 'runs' / f'multicodeif_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}')).resolve()
    # Never overwrite a previous run's predictions/results.
    output.mkdir(parents=True, exist_ok=False)
    all_results, current, texts = [], {}, {}
    provenance = {'benchmark': 'instruction_following.multicodeif', 'harness': pair.harness, 'upstream_commit': commit,
                  'suite': config.get('suite', 'single_level'), 'evaluation_scope': 'official_constraint_checks',
                  'functional_correctness': 'not_evaluated',
                  'compatibility': ['lazy_guesslang_import', 'tree_sitter_legacy_api_bridge'],
                  'sampling_model': config.get('model'),
                  'judge_model': config.get('judge_model'), 'requested_rounds': rounds,
                  'config': {k: v for k, v in config.items() if k not in ('base_url', 'judge_base_url')}}
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    for round_idx in range(rounds):
        active = [task for task in tasks if task.task_id not in current or
                  (current[task.task_id]['status'] == 'scored' and not current[task.task_id]['passed'])]
        if not active:
            break
        contexts = {}
        for task in active:
            previous = current.get(task.task_id)
            contexts[task.task_id] = selected_harness.make_messages(
                prompt=task.row['prompt'], previous_response=texts[task.task_id] if round_idx else None,
                failed_checks=[c for c in previous['checks'] if not c.get('passed')] if previous else None,
                repair_template=previous['repair_template'] if previous else None)
        def produce(task):
            if predictions is not None:
                responses = predictions[task.task_id]
                if round_idx >= len(responses):
                    return task.task_id, None, 'offline prediction has no response for this repair round'
                return task.task_id, responses[round_idx], None
            try:
                text = selected_harness.run(model=config, messages=contexts[task.task_id])
                return task.task_id, text, None
            except Exception as e:
                return task.task_id, None, str(e)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            generated = list(pool.map(produce, active))
        lookup = {task.task_id: task for task in tasks}
        requests = []
        for key, text, error in generated:
            task = lookup[key]
            texts[key] = text or ''
            requests.append({'task_id': key, 'response': text, 'generation_error': error,
                             'row': task.row, 'checks': task.checks, 'suite': config.get('suite', 'single_level')})
        evaluated = evaluate_round(repo, deps, requests, config, output)
        for result in evaluated['results']:
            current[result['task_id']] = result
        ordered = [current[t.task_id] for t in tasks]
        summary = summarize(ordered)
        all_results.append({'round': round_idx + 1, **summary})
        with (output / 'trajectories.jsonl').open('a') as f:
            for key, text, error in generated:
                f.write(json.dumps({'round': round_idx + 1, 'task_id': key, 'response': text,
                    'generation_error': error, 'messages': contexts[key],
                    'generation_source': 'offline_predictions' if predictions is not None else 'model_api',
                    'evaluation': current[key]}, ensure_ascii=False) + '\n')
        report = {'provenance': provenance, 'rounds': all_results, 'final': summary,
                  'results': ordered, 'output_dir': str(output)}
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return {'output_dir': str(output), 'report': str(output / 'report.json'),
            'rounds': all_results, 'final': all_results[-1]}
