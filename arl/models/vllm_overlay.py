"""Project-local vLLM 0.19 routed-experts buffer fix; never edit site-packages."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

FILES = ('v1/core/sched/scheduler.py', 'v1/worker/gpu_model_runner.py')


def patch_source(source, worker=False):
    base = 'self.kv_cache_config' if worker else 'kv_cache_config'
    old = (f'            self.max_num_kv_tokens = (\n                {base}.num_blocks // num_groups\n'
           '            ) * min_block_size') if not worker else (
           f'        self.max_num_kv_tokens = (\n            {base}.num_blocks // num_groups\n'
           '        ) * min_block_size')
    indent = '        ' if worker else '            '
    new = (indent + '# [ARM patch] Physical attention block IDs span the whole block pool.\n' +
           indent + f'attn_block_size = {base}.kv_cache_groups[self.routed_experts_attn_gid].kv_cache_spec.block_size\n' +
           indent + f'self.max_num_kv_tokens = {base}.num_blocks * attn_block_size')
    if source.count(old) != 1:
        raise RuntimeError('Unsupported vLLM routed-experts allocation source; refusing to patch a different implementation')
    result = source.replace(old, new, 1)
    compile(result, '<vllm-router-overlay>', 'exec')
    return result


def prepare_overlay(cache=None):
    """Link unchanged modules; copy only the scheduler and model-runner fixes."""
    spec = importlib.util.find_spec('vllm')
    if spec is None or not spec.origin:
        raise RuntimeError('router replay requires vLLM')
    source_root = Path(spec.origin).parent.resolve()
    version = importlib.metadata.version('vllm')
    sources = {name: (source_root/name).read_text() for name in FILES}
    if all('[ARM patch]' in text for text in sources.values()):
        return None
    if version.split('+')[0] != '0.19.0':
        raise RuntimeError(f'Local router overlay supports vLLM 0.19.0, found {version}')
    patched = {name: patch_source(sources[name], worker='worker/' in name) for name in FILES}
    digest = hashlib.sha256((str(source_root)+version+''.join(patched.values())).encode()).hexdigest()
    cache = Path(cache) if cache is not None else Path(__file__).resolve().parents[2]/'.cache/benchmarks/vllm-router-replay'
    root = cache/digest[:16]
    metadata = dict(version=version, source_root=str(source_root), digest=digest,
                    files={name: hashlib.sha256(text.encode()).hexdigest() for name,text in patched.items()})
    if (root/'overlay.json').exists():
        if json.loads((root/'overlay.json').read_text()) != metadata:
            raise RuntimeError('Existing vLLM overlay metadata differs')
        for name,text in patched.items():
            if (root/'vllm'/name).read_text() != text:
                raise RuntimeError('Existing vLLM overlay source differs')
        return root
    def link_tree(source, dest, prefix=''):
        dest.mkdir(parents=True,exist_ok=True)
        for child in source.iterdir():
            if child.name=='__pycache__':
                continue
            relative=prefix+child.name
            target=dest/child.name
            if relative in patched:
                target.write_text(patched[relative])
            elif any(name.startswith(relative+'/') for name in patched):
                link_tree(child,target,relative+'/')
            elif not target.exists() and not target.is_symlink():
                target.symlink_to(child,target_is_directory=child.is_dir())
    link_tree(source_root,root/'vllm')
    (root/'overlay.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return root
