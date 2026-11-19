"""Environment registry organized by coding and instruction_following categories.

Canonical names are category-qualified (coding.cuda, coding.swe). Existing cuda
and swe config values remain aliases. Registrations do not import environment
implementations or their optional dependencies until a source is constructed.
"""
from __future__ import annotations

from .coding import ENVIRONMENTS as CODING_ENVIRONMENTS
from .instruction_following import ENVIRONMENTS as INSTRUCTION_FOLLOWING_ENVIRONMENTS
from .registry import EnvironmentCategory, EnvironmentSpec

_ENVIRONMENTS_BY_CATEGORY = {
    "coding": CODING_ENVIRONMENTS,
    "instruction_following": INSTRUCTION_FOLLOWING_ENVIRONMENTS,
}
_ENVIRONMENTS = {
    name: spec
    for specs in _ENVIRONMENTS_BY_CATEGORY.values()
    for spec in specs
    for name in (spec.name, *spec.aliases)
}


def list_environment_categories() -> tuple[str, ...]:
    return tuple(_ENVIRONMENTS_BY_CATEGORY)


def list_environments(category: EnvironmentCategory | None = None) -> tuple[str, ...]:
    if category is not None and category not in _ENVIRONMENTS_BY_CATEGORY:
        raise ValueError(f"unknown environment category {category!r}; available: {list_environment_categories()}")
    categories = _ENVIRONMENTS_BY_CATEGORY.values() if category is None else (_ENVIRONMENTS_BY_CATEGORY[category],)
    return tuple(spec.name for specs in categories for spec in specs)


def find_env_spec(name: str) -> EnvironmentSpec | None:
    return _ENVIRONMENTS.get(name)


def get_env_spec(name: str) -> EnvironmentSpec:
    try:
        return _ENVIRONMENTS[name]
    except KeyError:
        # A benchmark scaffold is not a training environment. Fail explicitly
        # before the trainer can allocate model weights or create a backend.
        from arl.benchmarks import BenchmarkPendingError, find_benchmark
        benchmark = find_benchmark(name)
        if benchmark is not None:
            raise BenchmarkPendingError(benchmark.name, mode="train") from None
        category = name.split(".", 1)[0]
        if category in _ENVIRONMENTS_BY_CATEGORY and not _ENVIRONMENTS_BY_CATEGORY[category]:
            raise ValueError(f"environment category {category!r} has no registered environments") from None
        raise ValueError(f"unknown env {name!r}; available: {', '.join(list_environments())}") from None


def make_task_source(name: str, **kwargs):
    return get_env_spec(name).make_task_source(**kwargs)


def make_task_source_from_config(cfg):
    return get_env_spec(cfg.env).from_training_config(cfg)
