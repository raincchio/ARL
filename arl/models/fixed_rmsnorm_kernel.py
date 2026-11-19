"""Inference-only opaque custom op: fixed reduction order, CUDA-graph compatible."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


@triton.jit
def _kernel(X, W, Y, M, EPS: tl.constexpr):
    # Same reduction and arithmetic as the experimentally verified 8192/8 kernel.
    row = tl.program_id(0)
    col = tl.arange(0, 8192)
    mask = (row < M) & (col < 5120)
    x = tl.load(X + row * 5120 + col, mask=mask, other=0.0).to(tl.float32)
    square = x * x
    total = tl.sum(square, 0)
    weight = tl.load(W + col, mask=col < 5120, other=0.0).to(tl.float32)
    inv = libdevice.rsqrt(total / 5120.0 + EPS)
    y = (x * inv) * (1.0 + weight)
    tl.store(Y + row * 5120 + col, y, mask=mask)


@torch.library.custom_op("arm::fixed_gemma_rmsnorm_v1", mutates_args=())
def fixed_gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda or x.shape[-1] != 5120 or weight.shape != (5120,):
        raise ValueError("fixed GemmaRMSNorm requires CUDA input [...,5120] and weight [5120]")
    if weight.device != x.device:
        raise ValueError("RMSNorm input and weight must be on the same CUDA device")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Unsupported RMSNorm dtype: {x.dtype}")
    contiguous = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(contiguous)
    rows = contiguous.numel() // 5120
    if rows:
        _kernel[(rows,)](contiguous, weight, out, rows, eps, num_warps=8, num_stages=1)
    return out


@fixed_gemma_rmsnorm.register_fake
def _fake(x, weight, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
