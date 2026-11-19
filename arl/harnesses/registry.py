"""Agent harness registrations, independent of benchmark data and grading."""
from dataclasses import dataclass
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


class HarnessPendingError(NotImplementedError):
    def __init__(self, name):
        self.name = name
        super().__init__(f'待完成：{name} harness 接入。')


@dataclass(frozen=True)
class HarnessSpec:
    name: str
    display_name: str
    directory: str
    status: str = 'pending'
    aliases: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    trace_fields: tuple[str, ...] = ()
    model_backends: tuple[str, ...] = ()

    def ensure_ready(self):
        if self.status != 'ready':
            raise HarnessPendingError(self.name)

    def load(self):
        self.ensure_ready()
        spec = spec_from_file_location(f'arl_harness_{self.name}', self.path / 'entry.py')
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load harness {self.name}')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def make_messages(self, **kwargs):
        return self.load().make_messages(**kwargs)

    def make_renderer(self, tokenizer, **kwargs):
        return self.load().make_renderer(tokenizer, **kwargs)

    @property
    def path(self):
        return Path(__file__).resolve().parents[2] / self.directory

    def run(self, **kwargs):
        return self.load().run(**kwargs)


@lru_cache(maxsize=1)
def list_harnesses() -> tuple[HarnessSpec, ...]:
    rows = json.loads(Path(__file__).with_name('catalog.json').read_text())
    tuple_fields = ('aliases', 'operations', 'interfaces', 'trace_fields', 'model_backends')
    result = tuple(HarnessSpec(**{**row, **{k: tuple(row.get(k, ())) for k in tuple_fields}}) for row in rows)
    names = [name for spec in result for name in (spec.name, *spec.aliases)]
    if len(set(names)) != len(names):
        raise ValueError('duplicate harness name/alias')
    return result


def get_harness(name: str) -> HarnessSpec:
    for spec in list_harnesses():
        if name in (spec.name, *spec.aliases):
            return spec
    raise ValueError(f'unknown harness {name!r}; available: {", ".join(s.name for s in list_harnesses())}')


def run_harness(name: str, **kwargs):
    return get_harness(name).run(**kwargs)
