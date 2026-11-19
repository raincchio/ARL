"""Trusted ProgramBench Docker-host RPC worker; agent commands run as user agent."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile


def run(argv, timeout=120, limit=16000):
    with tempfile.TemporaryFile() as stream:
        try:
            proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
        size = stream.tell(); stream.seek(0)
        text = stream.read(limit).decode(errors='replace')
    return {'exit_code': code, 'output': text+('\n[output truncated]' if size > limit else '')}


def instances(req):
    import programbench
    from programbench.utils.load_data import load_all_instances
    root = Path(programbench.__file__).parent
    manifest = req['manifest']
    if importlib.metadata.version('programbench') != manifest['version']:
        raise ValueError('ProgramBench package version differs from pinned version')
    for name, sha in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'ProgramBench source modified or differs from pinned version: {name}')
    data = [i for i in load_all_instances() if not i['instance_id'].startswith('testorg__')]
    digest = hashlib.sha256()
    for instance in sorted(data, key=lambda i: i['instance_id']):
        for name in ('task.yaml', 'tests.json'):
            key = instance['instance_id']+'/'+name
            digest.update(key.encode()+hashlib.sha256((root/'data/tasks'/key).read_bytes()).hexdigest().encode())
    if digest.hexdigest() != manifest['dataset_sha256']:
        raise ValueError('ProgramBench task/ignore metadata differs from pinned dataset')
    return {i['instance_id']: i for i in data}


def grade(req, data):
    iid = req['instance_id']
    if iid not in data: raise ValueError('unknown ProgramBench task')
    os.environ['PROGRAMBENCH_BLOB_DIR'] = req['blob_dir']
    os.environ['HF_HUB_OFFLINE'] = '1'
    from programbench import constants
    from programbench.eval.eval_batch import _evaluate_instance
    constants.DOCKER_RUN_ARGS.extend(['--label', 'arl.programbench=1', '--label', f'arl.programbench.run={req["name"]}'])
    root = Path(req['work_dir']).resolve()/req['name']
    root.mkdir(parents=True, exist_ok=True)
    archive = root/iid/'submission.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if req['action'] == 'grade':
        # Official copy_submission semantics: reference executable may be unreadable;
        # tar can return 1 while still writing a valid source archive.
        with archive.open('wb') as out:
            proc = subprocess.run(['docker', 'exec', '--user', 'agent', req['name'], 'tar', '-czf', '-', '-C', '/workspace', '.'],
                                  stdout=out, stderr=subprocess.PIPE, timeout=600)
        if not archive.stat().st_size:
            raise RuntimeError(f'workspace archive failed: {proc.stderr.decode(errors="replace")[-500:]}')
    else:
        import shutil
        candidate = Path(req['submissions_dir']).resolve()/iid/'submission.tar.gz'
        if not candidate.is_file(): raise ValueError(f'missing submission: {candidate}')
        shutil.copyfile(candidate, archive)
    result = _evaluate_instance(instance_id=iid, instance=data[iid], source_dir=root, target_dir=root,
        force=True, image_tag=req['manifest']['image_tag'], docker_cpus=int(req['eval_cpus']),
        branch_workers=int(req['branch_workers']), branch_retries=1)
    if result is None: raise RuntimeError('official evaluator produced no summary')
    summary = result.model_dump()
    return {'summary': summary, 'archive': str(archive), 'report': str(root/iid/f'{iid}.eval.json'),
            'archive_warning': proc.stderr.decode(errors='replace')[-1000:] if req['action']=='grade' and proc.returncode else None}


def cleanup(name):
    owned = run(['docker', 'ps', '-aq', '--filter', f'label=arl.programbench.run={name}'], timeout=30)
    if owned['exit_code']: raise RuntimeError(owned['output'])
    containers = owned['output'].split()
    # Only this trial's labeled containers, never other users' containers/images.
    if containers:
        result = run(['docker', 'rm', '-f', *containers], timeout=120)
        if result['exit_code']: raise RuntimeError(result['output'])


def dispatch(req):
    action = req['action']
    if action in ('inventory', 'health', 'grade', 'evaluate'):
        data = instances(req)
        if action == 'inventory':
            return {'tasks': [{'instance_id': iid} for iid in sorted(data)], 'commit': req['manifest']['commit']}
        if action == 'health':
            version = run(['docker', 'version', '--format', '{{.Server.Version}}'], timeout=15)
            if version['exit_code']: raise RuntimeError(version['output'])
            selected = req.get('instance_ids', [])
            for iid in selected:
                if iid not in data: raise ValueError(f'unknown task: {iid}')
                image = f'programbench/{iid.replace("__", "_1776_")}:{req["manifest"]["image_tag"]}'
                result = run(['docker', 'image', 'inspect', image], timeout=20, limit=200)
                if result['exit_code']: raise RuntimeError(f'ProgramBench image not prepared: {image}')
                if not (Path(req['blob_dir'])/iid).is_dir(): raise RuntimeError(f'ProgramBench test blobs not prepared: {iid}')
            revision = Path(req['blob_dir'])/'REVISION'
            return {'ready': True, 'tasks': len(data), 'selected': len(selected), 'version': req['manifest']['version'],
                    'commit': req['manifest']['commit'], 'dataset_sha256': req['manifest']['dataset_sha256'],
                    'blob_revision': revision.read_text().strip() if revision.exists() else None}
    name = req['name']
    if not re.fullmatch(r'arl-pb-[a-f0-9]{32}', name): raise ValueError('invalid owned ProgramBench container name')
    if action in ('grade', 'evaluate'):
        def expired(*_): raise TimeoutError('ProgramBench grading deadline exceeded')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(int(req['grade_timeout']))
        try: return grade(req, data)
        finally: cleanup(name)
    if action == 'start':
        iid = req['instance_id']
        # An allowed ID must be present in the pinned official inventory.
        if iid not in instances(req): raise ValueError('unknown task')
        image = f'programbench/{iid.replace("__", "_1776_")}:{req["manifest"]["image_tag"]}'
        result = run(['docker', 'run', '--pull=never', '--rm', '-d', '--name', name,
            '--label', 'arl.programbench=1', '--label', f'arl.programbench.run={name}',
            '--network', 'none', '--user', 'agent', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--pids-limit', '1024', '--memory', req['memory'], '--memory-swap', req['memory'],
            '--cpus', str(req['cpus']), '-w', '/workspace', '--entrypoint', 'bash', image, '-c', 'sleep 7h'])
        if result['exit_code']: raise RuntimeError(result['output'])
        return {'image': image}
    if action == 'exec':
        if req.get('interpreter') is not None:
            argv = ['docker', 'exec', '--user', 'agent', '-w', '/workspace']
            for key, value in req.get('environment', {}).items():
                argv.extend(['-e', f'{key}={value}'])
            argv.extend([name, *req['interpreter'], req['command']])
            try:
                result = subprocess.run(argv, text=True, encoding='utf-8', errors='replace',
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=int(req['command_timeout']))
                return {'exit_code': result.returncode, 'output': result.stdout, 'exception_info': ''}
            except subprocess.TimeoutExpired as exc:
                output = exc.output or b''
                return {'exit_code': -1, 'output': output.decode(errors='replace') if isinstance(output, bytes) else output,
                        'exception_info': f'An error occurred while executing the command: {exc}'}
        return run(['docker', 'exec', '--user', 'agent', '-w', '/workspace',
                    '-e', 'PAGER=cat', '-e', 'MANPAGER=cat', name, 'timeout', '--signal=KILL',
                    str(req['command_timeout']), 'bash', '-c', req['command']],
                    timeout=int(req['command_timeout'])+30, limit=int(req['output_limit']))
    if action == 'close':
        cleanup(name)
        return {'closed': True}
    raise ValueError(f'unknown ProgramBench action {action}')


if __name__ == '__main__':
    try:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr): result = dispatch(request)
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({'infra_error': True, 'error': f'{type(exc).__name__}: {exc}'}))
