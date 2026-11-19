"""Serializable prefill controls shared by training and standalone inference."""

CONFIG_KEY = 'arl_prefill_budget'


def engine_args(tokens=0, batch_tokens=0, additional_config=None):
    for name, value in (('prefill_tokens', tokens), ('prefill_batch_tokens', batch_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if batch_tokens and tokens > batch_tokens:
        raise ValueError('prefill_tokens cannot exceed prefill_batch_tokens')
    batch = batch_tokens or (max(2048, tokens) if tokens else 0)
    result = {}
    if batch:
        result.update(max_num_batched_tokens=batch, enable_chunked_prefill=True)
    if tokens:
        result.update(
            scheduler_cls='arl.models.prefill_scheduler.BudgetAsyncScheduler',
            async_scheduling=True,
            additional_config={**(additional_config or {}), CONFIG_KEY: {
                'version': 1, 'tokens': tokens,
            }},
        )
    return result
