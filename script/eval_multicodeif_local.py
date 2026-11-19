#!/usr/bin/env python3
"""Small local-model MultiCodeIF run on scheduler-assigned GPUs.

Submit through the existing cluster CMD queue; CUDA_VISIBLE_DEVICES is assigned
by that scheduler. Holds the shared ARM GPU lease and stops only its own server.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='/models/Qwen/Qwen3.6-35B-A3B')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_dir / 'status.json'
    def publish(state, **kwargs):
        record = dict(state=state, host=socket.gethostname(), model_path=args.model_path,
                      gpus=os.environ.get('CUDA_VISIBLE_DEVICES', ''), **kwargs)
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(status)
        print(json.dumps(record), flush=True)
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if not cards or any(not c.isdigit() for c in cards):
        raise RuntimeError('scheduler must assign physical CUDA_VISIBLE_DEVICES')
    if not (Path(args.model_path) / 'config.json').is_file():
        raise RuntimeError('model directory does not exist')
    sys.path.insert(0, '/root/workspace/ARM')
    from rlhf import gpu_lease
    lease = gpu_lease.try_acquire(cards, 'arl-multicodeif', str(run_dir))
    if lease is None:
        publish('blocked', error='scheduler-assigned GPUs have a foreign lease')
        return 1
    server = None
    def stop(signum, frame):
        # Stop inference first so outstanding HTTP calls unblock promptly.
        if server is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
        raise KeyboardInterrupt(f'stopped by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
        usage = {r.split(',')[0].strip(): [int(x.strip()) for x in r.split(',')[1:]] for r in rows.splitlines()}
        if any(usage[c][0] >= 1500 or usage[c][1] != 0 for c in cards):
            publish('blocked', error='scheduler-assigned GPUs are occupied')
            return 1
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        model = 'arl-multicodeif-qwen36-35b-a3b'
        command = [sys.executable, '/root/workspace/AgentBench/agentbench/vllm_main.py',
            'serve', args.model_path, '--served-model-name', model, '--host', '127.0.0.1', '--port', str(port),
            '--tensor-parallel-size', str(len(cards)), '--gpu-memory-utilization', '0.8',
            '--max-model-len', '16384', '--max-num-seqs', '4', '--language-model-only', '--enforce-eager']
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        publish('starting', port=port)
        with (run_dir / 'server.log').open('w') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, pass_fds=tuple(lease.fds))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'vLLM exited {server.returncode}; see server.log')
            try:
                with opener.open(f'http://127.0.0.1:{port}/health', timeout=2):
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError('vLLM did not become ready within 900s')
        publish('evaluating', port=port)
        config = {
            'suite': 'single_level', 'constraints': ['code_convention_naming_convention', 'data_structure_type'],
            'languages': ['python'], 'limit': 0, 'max_rounds': 3, 'concurrency': 2,
            'max_tokens': 4096, 'temperature': 0., 'request_timeout': 240,
            'chat_template_kwargs': {'enable_thinking': False},
            'base_url': f'http://127.0.0.1:{port}/v1', 'model': model, 'model_path': args.model_path,
            'output_dir': str(run_dir / 'benchmark')}
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
        from arl.benchmarks.multicodeif import benchmark
        result = benchmark(**config)
        (run_dir / 'result.json').write_text(json.dumps(result, indent=2))
        if not result['final']['complete']:
            raise RuntimeError('model smoke has incomplete checks; see benchmark/report.json')
        publish('complete', result=result)
        return 0
    except BaseException as e:
        publish('failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        if server is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        lease.release()


if __name__ == '__main__':
    raise SystemExit(main())
