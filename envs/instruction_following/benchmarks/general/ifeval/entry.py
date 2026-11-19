"""IFEval inventory/offline official evaluation and explicit native GRPO entry."""


def benchmark(**kwargs):
    from arl.benchmarks.ifeval import benchmark
    return benchmark(**kwargs)


def train(**kwargs):
    import asyncio
    if not kwargs.get('model_name') or not kwargs.get('log_path'):
        raise ValueError('IFEval 训练已接通；请用 python3 -m arl.experiments experiments/ifeval.yaml --phase train，或提供 model_name/log_path')
    from arl.grpo import GRPOConfig, run_grpo
    kwargs.setdefault('train_set', 'ifeval')
    kwargs.setdefault('ifeval_reward_mode', 'strict')
    return asyncio.run(run_grpo(GRPOConfig(env='instruction_following.ifeval', **kwargs)))
