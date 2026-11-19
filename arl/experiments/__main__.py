"""Run one independently configured experiment phase, defaulting to evaluation."""
import argparse
from dataclasses import asdict
import json

from .config import PHASES, load_experiment


def compatibility_matrix():
    from arl.envs import get_env_spec, list_environments
    from arl.benchmarks import list_benchmarks
    result = []
    native = [get_env_spec(name) for name in list_environments()]
    names = {spec.name for spec in native}
    for spec in [*native, *(spec for spec in list_benchmarks() if spec.name not in names)]:
        result.append({'env': spec.name, 'default_harness': spec.default_harness,
                       'supported_harnesses': list(spec.supported_harnesses), 'interface': spec.interface})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', help='YAML/JSON experiment configuration')
    parser.add_argument('--phase', choices=PHASES, default='evaluate')
    parser.add_argument('--check', action='store_true', help='validate the selected phase without creating a backend')
    parser.add_argument('--matrix', action='store_true', help='list configured env/harness relationships')
    args = parser.parse_args(argv)
    if args.matrix:
        if args.config or args.check:
            parser.error('--matrix cannot be combined with a config or --check')
        print(json.dumps(compatibility_matrix(), ensure_ascii=False, indent=2))
        return 0
    if not args.config:
        parser.error('config is required')
    try:
        plan = load_experiment(args.config).plan(args.phase)
        from .runner import run_phase, validate_plan
        if args.check:
            validate_plan(plan)
            result = {'validated': True, **plan.to_dict()}
        else:
            result = run_phase(plan)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, NotImplementedError, RuntimeError, OSError) as e:
        print(str(e))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
