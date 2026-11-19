"""Single-turn instruction following over the shared token sampling runtime."""


def make_renderer(tokenizer, **kwargs):
    from arl.agent.render import AgentRenderer
    return AgentRenderer(tokenizer, **kwargs)


async def run(*, tasks, source, sampling_client, renderer, params, operation,
              env_name=None, uid_prefix='', log_turns=False):
    if env_name != 'instruction_following.ifeval':
        raise ValueError('direct harness currently supports instruction_following.ifeval only')
    if operation not in ('collect', 'train', 'evaluate'):
        raise ValueError(f'unsupported direct operation {operation}')
    from arl.agent.rollout import run_agent_rollout
    # This is the shared token/logprob bookkeeping engine; the single-turn env
    # owns termination and has no tools. No arl_agent entry is invoked.
    return await run_agent_rollout(tasks, source, sampling_client, renderer, params,
        train=operation in ('collect', 'train'), uid_prefix=uid_prefix, log_turns=log_turns)
