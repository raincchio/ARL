"""ARM training switches, independent of environment and harness semantics."""
import math
import os


def validate_training_options(cfg):
    format_weight = getattr(cfg, 'format_reward_weight', 0.)
    if not math.isfinite(format_weight) or format_weight < 0:
        raise ValueError('format_reward_weight must be finite and >= 0')
    from arl.models.prefill import engine_args as prefill_engine_args
    prefill_engine_args(getattr(cfg,'sampling_prefill_tokens',0),getattr(cfg,'sampling_prefill_batch_tokens',0))
    from arl.models.prefix_cache import engine_args as prefix_cache_engine_args
    prefix_cache_engine_args(getattr(cfg,'enable_prefix_caching',None),getattr(cfg,'mamba_cache_mode','none'))
    for key in ('sampling_replay', 'router_replay'):
        if not isinstance(getattr(cfg, key), bool):
            raise ValueError(f'{key} must be boolean')
    if not isinstance(cfg.think_budget, int) or isinstance(cfg.think_budget, bool) or cfg.think_budget < 0:
        raise ValueError("think_budget must be an integer >= 0")
    if cfg.think_budget and not cfg.enable_thinking:
        raise ValueError("think_budget requires enable_thinking=True")
    if cfg.sampling_replay:
        if not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
            raise ValueError('sampling_replay requires finite temperature > 0')
        if not 0 < cfg.top_p <= 1:
            raise ValueError('sampling_replay requires 0 < top_p <= 1')
        if not isinstance(cfg.top_k, int) or isinstance(cfg.top_k, bool):
            raise ValueError('sampling_replay requires integer top_k')
        if (cfg.loss_fn_config or {}).get('kl_beta', 0):
            raise ValueError('sampling_replay requires kl_beta=0')


def configure_training_runtime(cfg):
    """Set before constructing vLLM/trainer; do not inherit stale replay modes."""
    validate_training_options(cfg)
    os.environ['KERNEL_RL_SAMPLER_LOGPROBS_MODE'] = 'processed_logprobs' if cfg.sampling_replay else 'raw_logprobs'
    os.environ['KERNEL_RL_ROUTER_REPLAY'] = '1' if cfg.router_replay else '0'


def training_loss_config(cfg, gen_cap):
    config = dict(cfg.loss_fn_config or {})
    config.update(clip_low_threshold=float(cfg.clip_low), clip_high_threshold=float(cfg.clip_high),
                  dr_grpo_const_len=float(gen_cap))
    if cfg.sampling_replay:
        config.update(replay_temperature=float(cfg.temperature), replay_top_k=int(cfg.top_k),
                      replay_top_p=float(cfg.top_p))
    return config


async def training_phase(client, phase):
    from arl import colocate
    if colocate.enabled():
        await client.colocate_phase(phase)


def thinking_sampling_options(cfg, tokenizer):
    """Same two-stage sampling budget as ARM, using this harness's tokenizer."""
    if not cfg.think_budget:
        return {}
    force = list(tokenizer.encode(cfg.think_force_text, add_special_tokens=False))
    end = list(tokenizer.encode('</think>', add_special_tokens=False))
    if not end or not any(force[i:i+len(end)] == end for i in range(len(force)-len(end)+1)):
        raise ValueError('think_force_text must contain the tokenizer sequence for </think>')
    return dict(think_budget=cfg.think_budget, think_force_ids=force, think_end_ids=end)
