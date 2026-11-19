#!/usr/bin/env python3
"""Read local experiment processes and saved/live trajectory progress (no GPU required)."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def outcome_counts(path):
    completed = dropped = 0
    try:
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # The writer may still be appending its last line.
                completed += 1
                dropped += bool(row.get('dropped') or row.get('summary', {}).get('infra_error'))
    except OSError:
        pass
    return completed, dropped


def local_processes(config):
    processes = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(arg == 'arl.experiments' for arg in argv) and any(
                    arg.endswith('.json') and Path(arg).resolve() == config for arg in argv):
                processes.append(int(proc.name))
        except OSError:
            continue
    return processes


def snapshot(run):
    phases = []
    for name in ('baseline', 'train', 'evaluate'):
        config = (run / f'{name}.json').resolve()
        document = read_json(config)
        plan = document.get('train' if name == 'train' else 'evaluate', {})
        output = Path(plan.get('output_dir', run / name))
        if not output.is_absolute():
            output = config.parent / output
        processes = local_processes(config)
        if not output.exists() and not processes:
            continue
        options = plan.get('options', {})
        completed, dropped = outcome_counts(output / 'outcomes.jsonl')
        phases.append(dict(phase=name, pids=processes,
            status=read_json(output / 'manifest.json').get('status', 'unknown'),
            outcomes_saved=completed, outcomes_dropped=dropped,
            total=options.get('task_limit', 0) * options.get('group_size', 1),
            live=[read_json(path) for path in sorted(output.glob('rollout_progress.rank*.json'))]))
    return dict(run=str(run), hostname=socket.gethostname(), phases=phases)


def render(state):
    lines = [f"机器: {state['hostname']}", f"Run: {state['run']}"]
    for phase in state['phases']:
        pids = ','.join(map(str, phase['pids'])) or '无本机进程'
        lines.append(f"\n{phase['phase']}: {phase['status']} | PID: {pids}")
        if phase['phase'] != 'train':
            lines.append(f"  累计已落盘: {phase['outcomes_saved']}/{phase['total'] or '?'} 条轨迹"
                         f"，其中异常/丢弃 {phase['outcomes_dropped']}")
        if not phase['live']:
            lines.append('  暂无批内快照（旧进程或尚未开始 rollout）；落盘数量不含当前未完成批次。')
        for live in phase['live']:
            if not live:
                continue
            age = max(0, time.time() - live['updated_at'])
            owner = f"{live['hostname']}:{live['pid']}"
            lines.append(f"  {live['batch']} rank={live['rank']} [{live['status']}] {owner}")
            allocation = (f"涉及 {live['batch_size']} 题，轨迹跨 rank 分配"
                          if live.get('shard_trajectories') else
                          f"{live['batch_size']} × {live['samples_per_task']}")
            lines.append(f"  本批完成: {live['completed']}/{live['total']}"
                         f"（{allocation}），丢弃 {live['dropped']}")
            lines.append(f"  已初始化 {live['initialized']} | 生成/排队 {live['generating']}"
                         f" | 环境执行/判分 {live['environment']}")
            lines.append(f"  已完成轮数 {live['turns_completed']} | 已返回 token {live['generated_tokens']}"
                         f" | 快照距今 {age:.0f}s（历史快照不代表进程仍在运行）")
    if not state['phases']:
        lines.append('尚无阶段输出或本机实验进程。')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', type=Path, help='run directory; defaults to newest runs directory')
    parser.add_argument('--watch', type=float, metavar='SECONDS', help='refresh interval, e.g. --watch 5')
    parser.add_argument('--json', action='store_true', help='print machine-readable snapshots')
    parser.add_argument('--gpu', action='store_true', help='also query GPU utilization/memory')
    args = parser.parse_args(argv)
    if args.watch is not None and args.watch <= 0:
        parser.error('--watch must be positive')
    run = args.run
    if run is None:
        runs = [path for path in (ROOT / 'runs').iterdir() if path.is_dir()]
        if not runs:
            parser.error('no runs found')
        run = max(runs, key=lambda path: path.stat().st_mtime)
    run = run.resolve()
    if not run.is_dir():
        parser.error(f'run directory does not exist: {run}')
    try:
        while True:
            state = snapshot(run)
            print(json.dumps(state, ensure_ascii=False) if args.json else render(state), flush=True)
            if args.gpu:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv'], capture_output=True, text=True, timeout=5)
                    print(result.stdout or result.stderr, flush=True)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    print(f'GPU 状态不可用: {exc}', flush=True)
            if args.watch is None:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
