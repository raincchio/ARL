"""API/offline constraint evaluation and explicit on-policy GRPO training."""


def benchmark(**kwargs):
    from arl.benchmarks.multicodeif import benchmark as run
    return run(**kwargs)


def train(**kwargs):
    import asyncio
    if not kwargs.get('model_name') or not kwargs.get('log_path'):
        raise ValueError('MultiCodeIF 训练已接通；请用 experiments/multicodeif_grpo.yaml --phase train，或提供 model_name/log_path')
    from arl.grpo import GRPOConfig, run_grpo
    kwargs.setdefault('train_set', 'multicodeif')
    return asyncio.run(run_grpo(GRPOConfig(env='instruction_following.multicodeif', **kwargs)))
