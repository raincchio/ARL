"""Model prompting and official feedback iteration, independent of scoring."""


def make_messages(*, prompt, previous_response=None, failed_checks=None, repair_template=None):
    if previous_response is not None:
        if repair_template is None:
            raise ValueError('self-repair requires the benchmark feedback template')
        failures = [check['constraint'] + ': ' + check['answer'] for check in failed_checks or ()]
        prompt = repair_template.format(answer='; '.join(failures), prev_code=previous_response, prompt=prompt)
    return [{'role': 'user', 'content': prompt}]


def make_renderer(tokenizer, **kwargs):
    from arl.agent.render import AgentRenderer
    return AgentRenderer(tokenizer, **kwargs)


async def run_local(*, tasks, source, sampling_client, renderer, params, operation,
                    env_name, uid_prefix='', log_turns=False):
    if env_name != 'instruction_following.multicodeif':
        raise ValueError('multicodeif_self_repair only supports instruction_following.multicodeif')
    if operation not in ('collect', 'train', 'evaluate'):
        raise ValueError(f'unsupported operation {operation}')
    from arl.agent.rollout import run_agent_rollout
    return await run_agent_rollout(tasks, source, sampling_client, renderer, params,
        train=operation in ('collect', 'train'), uid_prefix=uid_prefix, log_turns=log_turns)


def run(*, model=None, messages=None, **kwargs):
    if model is None:
        return run_local(**kwargs)
    from arl.benchmarks.multicodeif import chat
    return chat(model, messages if messages is not None else make_messages(**kwargs))
