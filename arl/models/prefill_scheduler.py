"""vLLM scheduler budget without changing its compilation/profiling range."""
import logging

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from arl.models.prefill import CONFIG_KEY

logger = logging.getLogger(__name__)


class BudgetAsyncScheduler(AsyncScheduler):
    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        config = vllm_config.additional_config.get(CONFIG_KEY, {})
        tokens = config.get('tokens')
        if config.get('version') != 1 or isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError('Invalid serialized ARL prefill budget')
        limit = self.scheduler_config.max_num_batched_tokens
        if tokens > limit:
            raise ValueError(f'prefill_tokens={tokens} exceeds compilation budget {limit}')
        if self.cache_config.mamba_cache_mode == 'align' and tokens % self.block_size:
            raise ValueError(f'prefill_tokens={tokens} must be a multiple of the actual cache block_size={self.block_size}')
        previous = self.max_num_scheduled_tokens
        if previous < tokens:
            raise ValueError(f'prefill_tokens={tokens} exceeds engine scheduling limit {previous}')
        self.max_num_scheduled_tokens = tokens
        self.arl_prefill_status = dict(tokens=tokens, batch_tokens=limit,
                                      block_size=self.block_size,
                                      mamba_cache_mode=self.cache_config.mamba_cache_mode)
        logger.info('[prefill] fixed per-engine scheduling budget: %s', self.arl_prefill_status)
