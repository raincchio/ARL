"""The existing ARL agent loop, exposed as a selectable harness."""


def make_renderer(tokenizer, **kwargs):
    from arl.agent.render import AgentRenderer
    return AgentRenderer(tokenizer, **kwargs)


async def run(*, tasks, source, sampling_client, renderer, params, operation,
              env_name=None, uid_prefix='', log_turns=False):
    if env_name != 'coding.cuda':
        raise ValueError('arl_agent is restricted to CUDA operator / KernelBench environments')
    if operation not in ('collect', 'train', 'evaluate'):
        raise ValueError(f'unsupported ARL agent operation {operation!r}')
    from arl.agent.rollout import run_agent_rollout
    # Collection needs the same sampled-token bookkeeping as training. Evaluation
    # intentionally avoids building datums, while still retaining turn transcripts.
    return await run_agent_rollout(tasks, source, sampling_client, renderer, params,
        train=operation in ('collect', 'train'), uid_prefix=uid_prefix, log_turns=log_turns)
