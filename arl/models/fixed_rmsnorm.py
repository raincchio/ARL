"""Fixed Gemma-style RMSNorm for the 5120-wide Qwen3.5/3.8 vLLM sampler.

Enabled by default for supported models; KERNEL_RL_FIXED_RMSNORM=0 disables it.
Configuration is sent through vLLM
additional_config (including its compile-cache hash), not worker environment
inheritance. This module does not patch Transformers or the training backward.
"""
from __future__ import annotations

import os

CONFIG_KEY = "arm_fixed_rmsnorm"
CONFIG = {"version": 1, "hidden_size": 5120, "rblock": 8192,
          "xblock": 1, "num_warps": 8, "num_stages": 1}


def engine_args(model_config: dict) -> dict:
    raw = os.environ.get("KERNEL_RL_FIXED_RMSNORM", "").strip().lower()
    if raw in ("0", "false", "no"):
        return {}
    if raw not in ("", "1", "true", "yes"):
        raise ValueError("KERNEL_RL_FIXED_RMSNORM must be 0 or 1")
    cfg = model_config.get("text_config") or model_config
    supported = (cfg.get("model_type") in ("qwen3_5", "qwen3_5_text")
                 and cfg.get("hidden_size") == 5120)
    if not supported:
        if raw:
            raise ValueError("Fixed RMSNorm currently requires Qwen3.5/3.8 with hidden_size=5120")
        return {}
    return {"worker_cls": "arl.models.fixed_rmsnorm_worker.FixedRMSNormWorker",
            "additional_config": {CONFIG_KEY: dict(CONFIG)}}


def install() -> None:
    """Install once, inside each vLLM GPU worker, BEFORE model construction."""
    import torch
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from arl.models.fixed_rmsnorm_kernel import fixed_gemma_rmsnorm

    if getattr(GemmaRMSNorm, "_arm_fixed_rmsnorm_version", None) == CONFIG["version"]:
        return
    original_native = GemmaRMSNorm.forward_native
    original_cuda = GemmaRMSNorm.forward_cuda

    def forward(self, x, residual=None):
        if x.shape[-1] != CONFIG["hidden_size"]:
            return original_native(self, x, residual)
        if residual is not None:
            # Match vLLM GemmaRMSNorm: FP16 adds in FP32; BF16 keeps BF16 residual.
            x = x.float() + residual.float() if x.dtype == torch.float16 else x + residual
            residual = x
        out = fixed_gemma_rmsnorm(x, self.weight, self.variance_epsilon)
        return out if residual is None else (out, residual)

    def forward_cuda(self, x, residual=None):
        if x.shape[-1] != CONFIG["hidden_size"]:
            return original_cuda(self, x, residual)
        return forward(self, x, residual)

    GemmaRMSNorm.forward_native = forward
    GemmaRMSNorm.forward_cuda = forward_cuda
    GemmaRMSNorm._arm_fixed_rmsnorm_version = CONFIG["version"]
    from vllm.logger import init_logger
    init_logger(__name__).info("[sampler] fixed GemmaRMSNorm installed: %s", CONFIG)
