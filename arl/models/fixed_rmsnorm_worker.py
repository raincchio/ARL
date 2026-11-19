"""Spawn/TP-safe installation before vLLM loads, compiles or captures the model."""
from vllm.v1.worker.gpu_worker import Worker
from vllm.logger import init_logger

from arl.models.fixed_rmsnorm import CONFIG, CONFIG_KEY, install

logger = init_logger(__name__)


class FixedRMSNormWorker(Worker):
    def __init__(self, vllm_config, *args, **kwargs):
        requested = vllm_config.additional_config.get(CONFIG_KEY)
        if requested != CONFIG:
            raise ValueError(f"Unexpected fixed RMSNorm config: {requested!r}; expected {CONFIG!r}")
        text = vllm_config.model_config.hf_text_config
        if getattr(text, "model_type", None) not in ("qwen3_5", "qwen3_5_text") or text.hidden_size != 5120:
            raise ValueError("Fixed RMSNorm worker requires Qwen3.5/3.8 hidden_size=5120")
        install()
        super().__init__(vllm_config, *args, **kwargs)

    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        logger.info("[sampler] fixed RMSNorm active: %s", self.arm_fixed_rmsnorm_status())

    def arm_fixed_rmsnorm_status(self):
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        model = self.model_runner.model
        count = sum(isinstance(m, GemmaRMSNorm) and m.weight.numel() == 5120 for m in model.modules())
        if not count:
            raise RuntimeError("Fixed RMSNorm enabled but no matching layers found")
        return {"rank": self.rank, "config": dict(CONFIG), "matching_layers": count,
                "installed_version": GemmaRMSNorm._arm_fixed_rmsnorm_version}
