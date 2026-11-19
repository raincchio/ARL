"""CPU validation of the physical attention-slot bounds in the vLLM overlay."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

from arl.models.vllm_overlay import prepare_overlay, patch_source, FILES


class OverlayTests(unittest.TestCase):
    def test_overlay_covers_physical_slots_and_never_changes_installed_sources(self):
        spec=importlib.util.find_spec('vllm')
        if not spec:self.skipTest('vLLM not installed')
        installed=Path(spec.origin).parent
        before={f:(installed/f).read_bytes() for f in FILES}
        with tempfile.TemporaryDirectory() as cache:
            overlay=prepare_overlay(cache)
            effective=installed if overlay is None else overlay/'vllm'
            for filename in FILES:
                text=(effective/filename).read_text()
                tree=ast.parse(text)
                method='init_routed_experts_capturer' if 'worker' in filename else '__init__'
                owner='GPUModelRunner' if 'worker' in filename else 'Scheduler'
                klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
                function=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name==method)
                selected=[]
                for node in function.body:
                    for child in ast.walk(node):
                        if isinstance(child,ast.Assign) and any(
                            isinstance(t,ast.Name) and t.id=='attn_block_size' or
                            isinstance(t,ast.Attribute) and t.attr=='max_num_kv_tokens' for t in child.targets):
                            selected.append(child)
                config=NS(num_blocks=101,kv_cache_groups=[NS(kv_cache_spec=NS(block_size=16)),
                                                         NS(kv_cache_spec=NS(block_size=64))])
                self_obj=NS(routed_experts_attn_gid=1,kv_cache_config=config)
                module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
                exec(compile(module,'<slot-bound-test>','exec'),{'self':self_obj,'kv_cache_config':config})
                self.assertEqual(self_obj.max_num_kv_tokens,101*64)
                self.assertLess(100*64+63,self_obj.max_num_kv_tokens)
            self.assertEqual(prepare_overlay(cache),overlay)
            if overlay is not None:
                env={**os.environ,'PYTHONPATH':str(overlay)}
                origin=subprocess.check_output([sys.executable,'-c',
                    'import importlib.util; print(importlib.util.find_spec("vllm").origin)'],env=env,text=True)
                self.assertEqual(Path(origin.strip()),overlay/'vllm/__init__.py')
            else:
                self.assertFalse(list(Path(cache).iterdir()))
        self.assertEqual({f:(installed/f).read_bytes() for f in FILES},before)

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(RuntimeError):patch_source('self.max_num_kv_tokens = 1')


if __name__=='__main__':unittest.main()
