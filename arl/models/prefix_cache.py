"""Explicit vLLM prefix-cache options, independent of scheduler budgets."""


def engine_args(enabled=None, mode='none', model_config=None):
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError('enable_prefix_caching must be True, False, or unset')
    if mode not in ('none', 'align', 'all'):
        raise ValueError('mamba_cache_mode must be none, align, or all')
    if mode != 'none' and enabled is not True:
        raise ValueError('mamba_cache_mode=align/all requires enable_prefix_caching=True')
    cfg = (model_config or {}).get('text_config') or model_config or {}
    if mode == 'all' and str(cfg.get('model_type', '')).startswith('qwen3_5'):
        raise ValueError('This Qwen model does not support mamba_cache_mode=all; use align')
    if enabled is None:
        return {}
    result = dict(enable_prefix_caching=enabled, mamba_cache_mode=mode)
    if mode == 'align':
        result['enable_chunked_prefill'] = True
    return result
