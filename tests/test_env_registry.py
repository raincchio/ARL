"""Category routing, legacy aliases, optional imports and relocated package roots."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from arl.envs import (get_env_spec, list_environment_categories, list_environments,
                      make_task_source, make_task_source_from_config)

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentRegistryTests(unittest.TestCase):
    def test_coding_and_instruction_following_are_separate_categories(self):
        self.assertEqual(list_environment_categories(), ('coding', 'instruction_following'))
        self.assertEqual(list_environments('coding'), ('coding.cuda', 'coding.swe', 'coding.programbench'))
        self.assertEqual(list_environments('instruction_following'), ('instruction_following.multicodeif', 'instruction_following.ifeval'))
        self.assertEqual(get_env_spec('ifeval').category, 'instruction_following')

    def test_legacy_names_resolve_to_the_same_registration(self):
        for short in ('cuda', 'swe'):
            self.assertIs(get_env_spec(short), get_env_spec('coding.' + short))

    def test_unknown_environment_or_category_fails_without_fallback(self):
        for name in ('coding.missing', 'instruction_following.missing', 'instruction_following', 'missing'):
            with self.assertRaisesRegex(ValueError, 'unknown env|no registered environments'):
                get_env_spec(name)
        with self.assertRaisesRegex(ValueError, 'unknown environment category'):
            list_environments('missing')

    def test_legacy_adapter_imports_keep_class_identity(self):
        from arl.envs.cuda import CudaSourceConfig, CudaTaskSource, _import_cudaenv
        from arl.envs.swe import SweSourceConfig, SweTaskSource, _import_sweenv
        from arl.envs.coding import cuda, swe
        self.assertIs(CudaSourceConfig, cuda.CudaSourceConfig)
        self.assertIs(CudaTaskSource, cuda.CudaTaskSource)
        self.assertIs(SweSourceConfig, swe.SweSourceConfig)
        self.assertIs(SweTaskSource, swe.SweTaskSource)
        self.assertIs(_import_cudaenv, cuda._import_cudaenv)
        self.assertIs(_import_sweenv, swe._import_sweenv)

    def test_factory_uses_correct_config_for_canonical_names_and_aliases(self):
        for short, class_name in (('cuda', 'CudaTaskSource'), ('swe', 'SweTaskSource')):
            for name in (short, 'coding.' + short):
                with patch(f'arl.envs.coding.{short}.{class_name}') as factory:
                    source = make_task_source(name, train_set='local_tasks.jsonl', max_turns=7)
                self.assertIs(source, factory.return_value)
                cfg = factory.call_args.args[0]
                self.assertEqual(cfg.train_set, 'local_tasks.jsonl')
                self.assertEqual(cfg.max_turns, 7)

    def test_training_config_dispatch_preserves_backend_specific_options(self):
        from arl.grpo import GRPOConfig
        for short, class_name, keys in (
            ('cuda', 'CudaTaskSource', dict(endpoint='http://example', enable_analyze=True)),
            ('swe', 'SweTaskSource', dict(ssh_host='example', command_timeout=17)),
        ):
            for name in (short, 'coding.' + short):
                cfg = GRPOConfig(model_name='model', log_path='/tmp/test', env=name,
                    eval_endpoint='http://example', env_enable_analyze=True,
                    swe_ssh_host='example', swe_command_timeout=17)
                with patch(f'arl.envs.coding.{short}.{class_name}') as factory:
                    make_task_source_from_config(cfg)
                source_cfg = factory.call_args.args[0]
                for key, value in keys.items():
                    self.assertEqual(getattr(source_cfg, key), value)
                self.assertEqual(source_cfg.train_set, 'ops6k' if short == 'cuda' else 'verified')

    def test_relocated_environment_packages_are_importable(self):
        from arl.envs.coding.cuda import _DEFAULT_ROOT, _import_cudaenv
        from arl.envs.coding.swe import _import_sweenv
        self.assertEqual(Path(_DEFAULT_ROOT), ROOT / 'envs' / 'coding' / 'CudaEnv')
        self.assertTrue(Path(_import_cudaenv().__file__).is_relative_to(ROOT / 'envs' / 'coding' / 'CudaEnv'))
        self.assertTrue(Path(_import_sweenv().__file__).is_relative_to(ROOT / 'envs' / 'coding' / 'SweEnv'))

    def test_registry_import_keeps_environment_dependencies_lazy(self):
        code = '''
import sys
import arl.envs
assert arl.envs.list_environments() == ('coding.cuda', 'coding.swe', 'coding.programbench', 'instruction_following.multicodeif', 'instruction_following.ifeval')
for name in ('arl.envs.coding.cuda', 'arl.envs.coding.swe', 'cudaenv', 'sweenv', 'docker', 'swebench', 'vllm', 'ifevalenv', 'nltk', 'arl.envs.instruction_following.ifeval', 'arl.envs.instruction_following.multicodeif'):
    assert name not in sys.modules, name
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
