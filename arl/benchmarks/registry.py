"""Benchmark inventory and explicit pending status, separate from training environments."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Literal

BenchmarkMode = Literal["benchmark", "train"]


class BenchmarkPendingError(NotImplementedError):
    def __init__(self, name: str, *, mode: BenchmarkMode):
        if mode not in ("benchmark", "train"):
            raise ValueError(f"unknown benchmark mode {mode!r}")
        self.name, self.mode = name, mode
        purpose = "benchmark 评测接入" if mode == "benchmark" else "训练接入"
        super().__init__(f"待完成：{name} 的{purpose}。")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    category: str
    group: str
    group_label: str
    directory: str
    benchmark_status: str
    training_status: str
    related_environment: str | None = None
    interface: str | None = None
    default_harness: str | None = None
    supported_harnesses: tuple[str, ...] = ()

    @property
    def path(self) -> Path:
        return Path(__file__).resolve().parents[2] / self.directory

    def run(self, mode: BenchmarkMode = "benchmark", **kwargs):
        """Dispatch to this benchmark's entry.py; never default to training."""
        if mode not in ("benchmark", "train"):
            raise ValueError(f"unknown benchmark mode {mode!r}; available: benchmark, train")
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location(f"arl_benchmark_{self.name.replace('.', '_')}", self.path / "entry.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load benchmark entry {self.name}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, mode)(**kwargs)


@lru_cache(maxsize=1)
def _catalog() -> tuple[BenchmarkSpec, ...]:
    rows = json.loads(Path(__file__).with_name("catalog.json").read_text())
    specs = tuple(BenchmarkSpec(**{**row, "supported_harnesses": tuple(row.get("supported_harnesses", ()))}) for row in rows)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate benchmark names in catalog")
    return specs


def list_benchmarks(category: str | None = None, group: str | None = None) -> tuple[BenchmarkSpec, ...]:
    if category is not None and category not in ("coding", "instruction_following"):
        raise ValueError(f"unknown benchmark category {category!r}")
    return tuple(spec for spec in _catalog()
                 if (category is None or spec.category == category) and (group is None or spec.group == group))


def find_benchmark(name: str) -> BenchmarkSpec | None:
    matches = [spec for spec in _catalog() if name in (spec.name, spec.name.split('.', 1)[1])]
    if len(matches) > 1:
        raise ValueError(f"ambiguous benchmark name {name!r}; use its category-qualified name")
    return matches[0] if matches else None


def get_benchmark(name: str) -> BenchmarkSpec:
    spec = find_benchmark(name)
    if spec is None:
        raise ValueError(f"unknown benchmark {name!r}; use --list to see available names")
    return spec


def run_benchmark(name: str, mode: BenchmarkMode = "benchmark", **kwargs):
    return get_benchmark(name).run(mode=mode, **kwargs)
