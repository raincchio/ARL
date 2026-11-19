"""Strict, lightweight experiment configuration; no model/backend imports."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from arl.harnesses import resolve_pair

PHASES = ('collect', 'train', 'evaluate')


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str = 'local'
    base_url: str | None = None
    api_key_env: str | None = None
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('model.name must be a nonempty string')
        if self.backend not in ('local', 'openai_compatible', 'offline'):
            raise ValueError(f'unknown model backend {self.backend!r}')
        if self.backend == 'openai_compatible' and not self.base_url:
            raise ValueError('openai_compatible model requires base_url')
        if self.base_url:
            from urllib.parse import urlsplit
            url = urlsplit(self.base_url)
            if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError('base_url must be an HTTP(S) service URL without embedded credentials/query; use api_key_env')
        if self.backend != 'openai_compatible' and self.base_url is not None:
            raise ValueError('base_url only applies to openai_compatible models')
        if not isinstance(self.options, dict):
            raise ValueError('model.options must be an object')
        if set(self.options) & {'name', 'model', 'model_name', 'backend', 'base_url', 'api_key', 'headers', 'output_dir', 'harness', 'env', 'environments', 'api_key_env', 'log_path', 'eval_harness', 'dataset', 'input_data', 'task_split'}:
            raise ValueError('model.options cannot override model identity, routing or credentials; use api_key_env')


@dataclass(frozen=True)
class PhasePlan:
    phase: str
    env: str
    harness: str
    model: ModelConfig
    options: dict
    output_dir: str | None = None
    split: str = 'train'
    environments: tuple[dict, ...] = ()

    @property
    def identity(self):
        identity = {'env': self.env, 'harness': self.harness, 'model': asdict(self.model)}
        if self.environments:
            identity['environments'] = list(self.environments)
        return identity

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    models: dict[str, ModelConfig]
    stages: dict
    defaults: dict
    name: str = 'experiment'
    base_dir: Path = field(default_factory=Path.cwd)

    def plan(self, phase='evaluate'):
        if phase not in PHASES:
            raise ValueError(f'unknown phase {phase!r}')
        if phase not in self.stages:
            raise ValueError(f'experiment has no {phase!r} phase')
        raw = {**self.defaults, **self.stages[phase]}
        unknown = set(raw) - {'env', 'harness', 'model', 'options', 'output_dir', 'split', 'environments'}
        if unknown:
            raise ValueError(f'unknown {phase} fields: {sorted(unknown)}')
        env, model_name = raw.get('env'), raw.get('model')
        if not isinstance(model_name, str) or model_name not in self.models:
            raise ValueError(f'{phase}.model must reference a model in models')
        model = self.models[model_name]
        if not isinstance(raw.get('options', {}), dict):
            raise ValueError(f'{phase}.options must be an object')
        options = {**self.defaults.get('options', {}), **raw.get('options', {})}
        reserved = {'environments', 'env', 'harness', 'model', 'model_name', 'base_url', 'api_key_env', 'api_key',
                    'headers', 'output_dir', 'log_path', 'eval_harness', 'dataset', 'input_data', 'task_split'}
        if set(options) & reserved:
            raise ValueError(f'{phase}.options cannot override routing/model/output or silently import offline data: {sorted(set(options) & reserved)}')
        if 'environments' in raw:
            if phase != 'train':
                raise ValueError('environments mixtures currently apply to train; collect/evaluate select their own env')
            if 'env' in self.stages[phase]:
                raise ValueError('choose train.env or train.environments, not both')
            if raw.get('split', 'train') != 'train':
                raise ValueError('training cannot use the test split')
            from arl.envs.mixture import normalize_members
            members = normalize_members(raw['environments'], harness=raw.get('harness'), model_backend=model.backend)
            members = tuple({**member, 'options': self._resolve_options(member['options'], member['env'])} for member in members)
            output = str(self._path(raw['output_dir'])) if raw.get('output_dir') else None
            return PhasePlan(phase, 'mixture', 'per_environment', model,
                self._resolve_options(options, ''), output, 'train', members)
        if not isinstance(env, str):
            raise ValueError(f'{phase}.env is required')
        pair = resolve_pair(env, raw.get('harness'), operation=phase, model_backend=model.backend)
        split = raw.get('split', 'train' if phase != 'evaluate' else 'test')
        if split not in ('train', 'test', 'all') or (pair.interface == 'agent_messages' and split == 'all'):
            raise ValueError('split must be train/test; text benchmarks also allow explicit all')
        if phase == 'train' and split != 'train':
            raise ValueError('training cannot use the test split')
        if model.backend == 'offline' and not options.get('predictions') and options.get('action') not in ('inspect', 'export'):
            raise ValueError('offline models require options.predictions')
        if model.backend != 'offline' and options.get('predictions'):
            raise ValueError('options.predictions requires an offline model; choose API or replay explicitly')
        options = self._resolve_options(options, pair.env)
        output = str(self._path(raw['output_dir'])) if raw.get('output_dir') else None
        return PhasePlan(phase, pair.env, pair.harness, model, options, output, split)

    def _resolve_options(self, options, env):
        options = dict(options)
        for key in ('predictions', 'upstream_root', 'dependencies', 'export_tasks', 'load_checkpoint_path', 'ifeval_python', 'ifeval_upstream', 'multicodeif_upstream', 'multicodeif_dependencies'):
            if options.get(key):
                options[key] = str(self._path(options[key]))
        if env in ('instruction_following.ifeval', 'instruction_following.multicodeif', 'coding.programbench'):
            for key in ('train_set', 'test_set'):
                if options.get(key) and options[key] not in ('ifeval', 'programbench', 'multicodeif'):
                    options[key] = str(self._path(options[key]))
        return options

    def _path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.base_dir / path).resolve()


def from_dict(raw, *, base_dir=None):
    if not isinstance(raw, dict):
        raise ValueError('experiment must be an object')
    unknown = set(raw) - {'version', 'name', 'models', 'defaults', *PHASES}
    if unknown:
        raise ValueError(f'unknown experiment fields: {sorted(unknown)}')
    if raw.get('version', 1) != 1:
        raise ValueError('unsupported experiment version')
    if not isinstance(raw.get('models'), dict) or not raw['models']:
        raise ValueError('models must be a nonempty mapping')
    models = {}
    for name, config in raw['models'].items():
        if not isinstance(config, dict):
            raise ValueError(f'model {name!r} must be an object')
        try:
            models[name] = ModelConfig(**config)
        except TypeError as e:
            raise ValueError(f'invalid model {name!r}: {e}') from e
    defaults = raw.get('defaults', {})
    if not isinstance(defaults, dict) or not isinstance(defaults.get('options', {}), dict):
        raise ValueError('defaults and defaults.options must be objects')
    stages = {key: raw[key] for key in PHASES if key in raw}
    if not stages or any(not isinstance(stage, dict) for stage in stages.values()):
        raise ValueError('provide at least one phase as an object')
    return Experiment(models, stages, defaults, raw.get('name', 'experiment'), Path(base_dir or Path.cwd()))


def load_experiment(path):
    path = Path(path).resolve()
    if path.suffix.lower() == '.json':
        raw = json.loads(path.read_text())
    else:
        import yaml
        raw = yaml.safe_load(path.read_text())
    return from_dict(raw, base_dir=path.parent)
