"""Resolve harness/environment pairs before loading models or creating backends."""
from dataclasses import dataclass
from typing import Literal

from .registry import HarnessPendingError, get_harness

Operation = Literal['collect', 'train', 'evaluate']
TRAIN_TRACE_FIELDS = frozenset({'token_ids', 'logprobs', 'loss_mask'})


class HarnessCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedPair:
    env: str
    harness: str
    operation: Operation
    interface: str
    trace_fields: tuple[str, ...]


def resolve_pair(env: str, harness: str | None = None, *, operation: Operation,
                 model_backend: str | None = None) -> ResolvedPair:
    if operation not in ('collect', 'train', 'evaluate'):
        raise ValueError(f'unknown operation {operation!r}')
    from arl.envs import find_env_spec
    target = find_env_spec(env)
    if target is None:
        from arl.benchmarks import BenchmarkPendingError, get_benchmark
        target = get_benchmark(env)
        if operation == 'train' and target.training_status != 'ready':
            raise BenchmarkPendingError(target.name, mode='train')
        if operation != 'train' and target.benchmark_status not in ('ready', 'partial'):
            raise BenchmarkPendingError(target.name, mode='benchmark')
    selected = harness or target.default_harness
    if not selected:
        raise HarnessCompatibilityError(f'{target.name} has no configured default harness')
    spec = get_harness(selected)
    interface = dict(getattr(target, 'backend_interfaces', ())).get(model_backend, target.interface)
    if spec.status != 'ready':
        raise HarnessPendingError(spec.name)
    if spec.name not in target.supported_harnesses or interface not in spec.interfaces:
        raise HarnessCompatibilityError(
            f'harness {spec.name!r} is incompatible with env {target.name!r}; '
            f'supported: {", ".join(target.supported_harnesses) or "none"}')
    if operation not in spec.operations:
        raise HarnessCompatibilityError(f'harness {spec.name!r} does not support {operation}')
    if operation == 'train':
        if interface != 'agent_messages' or model_backend not in (None, 'local'):
            raise HarnessCompatibilityError('GRPO requires a local on-policy token sampler; API/offline text is not training data')
        missing = TRAIN_TRACE_FIELDS - set(spec.trace_fields)
        if missing:
            raise HarnessCompatibilityError(f'harness {spec.name!r} cannot train: missing {sorted(missing)}')
    if model_backend is not None and model_backend not in spec.model_backends:
        raise HarnessCompatibilityError(
            f'harness {spec.name!r} does not support model backend {model_backend!r}; '
            f'supported: {", ".join(spec.model_backends)}')
    if spec.name == 'mini_swe_agent':
        from .mini_swe import check_dependencies
        check_dependencies()
    if spec.name == 'claude_code':
        from .claude_code import check_dependencies
        check_dependencies()
    return ResolvedPair(target.name, spec.name, operation, interface,
                        tuple(spec.trace_fields) if interface == 'agent_messages' else ())
