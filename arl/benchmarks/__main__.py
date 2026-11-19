"""List or invoke benchmark entries without initializing the training stack."""
import argparse
from dataclasses import asdict
import json

from . import BenchmarkPendingError, list_benchmarks, run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark 入口；默认评测，各入口分别声明训练接入状态。")
    parser.add_argument("name", nargs="?", help="category-qualified name or short benchmark name")
    parser.add_argument("--mode", choices=("benchmark", "train"), default="benchmark")
    parser.add_argument("--list", action="store_true", help="list registered benchmarks")
    parser.add_argument("--category", choices=("coding", "instruction_following"))
    parser.add_argument("--group")
    parser.add_argument("--json", action="store_true", help="print listing as JSON")
    parser.add_argument("--config", help="JSON file with benchmark-specific options")
    parser.add_argument("--harness", help="explicit compatible harness for this benchmark")
    args = parser.parse_args(argv)
    if args.list or args.name is None:
        if args.config or args.harness:
            parser.error("--config requires a benchmark name and cannot be combined with --list")
        if args.list and args.name:
            parser.error("--list cannot be combined with a benchmark name")
        specs = list_benchmarks(category=args.category, group=args.group)
        if args.json:
            print(json.dumps([asdict(spec) for spec in specs], ensure_ascii=False, indent=2))
        else:
            labels = {"pending": "待完成", "ready": "已接入", "partial": "部分接入"}
            for spec in specs:
                benchmark = labels.get(spec.benchmark_status, spec.benchmark_status)
                train = labels.get(spec.training_status, spec.training_status)
                print(f"{spec.name}\t{spec.display_name}\t{spec.group_label}\tbenchmark={benchmark}\ttrain={train}")
        return 0
    if args.category or args.group or args.json:
        parser.error("--category, --group and --json are listing options")
    try:
        options = {}
        if args.config:
            from pathlib import Path
            options = json.loads(Path(args.config).read_text())
            if not isinstance(options, dict) or set(options) & {"name", "mode"}:
                raise ValueError("benchmark config must be a JSON object without name/mode overrides")
        if args.harness:
            if 'harness' in options and options['harness'] != args.harness:
                raise ValueError('conflicting harness in --harness and config')
            options['harness'] = args.harness
        result = run_benchmark(args.name, mode=args.mode, **options)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except BenchmarkPendingError as e:
        print(str(e))
        return 2
    except (ValueError, OSError, RuntimeError) as e:
        parser.error(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
