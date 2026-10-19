"""Value network on top of a (detached) policy hidden state: optional causal tower + MLP head.

Shared by the online critic (``training_client``) and the offline experiments
(``rlhf/ppo/value_lora.py``, ``value_head_sweep.py``), so both run the same module.

Why a tower: a head that reads ONE position's hidden vector cannot pick what to look
at, and most of the predictable return lives in the prompt (previous turns'
feedback). A few causal self-attention blocks over the whole sequence can fetch it.
Offline (Qwen3.8-27B, improve γ=1 return): frozen single-position head ev 0.22 ->
2-block tower 0.41-0.43, better than a LoRA-on-trunk critic at a fraction of the
cost (no backward through the trunk).

Causality is required: V_t is a baseline for the action at t, so it may only see
tokens <= t. With right padding, padded positions sit after every real token, so
they never influence a real position either.
"""

from __future__ import annotations

import torch


class Head(torch.nn.Module):
    """[..., d_in] -> [...]: linear / 1-hidden-layer MLP / 2-hidden-layer MLP, optional input LayerNorm."""

    def __init__(self, d_in: int, arch: str = "linear", hidden: int = 1024, in_norm: str = "none"):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_in) if in_norm == "ln" else torch.nn.Identity()
        if arch == "linear":
            self.body, d = torch.nn.Identity(), d_in
        elif arch == "mlp1":
            self.body, d = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU()), hidden
        elif arch == "mlp2":
            self.body = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
                                            torch.nn.Linear(hidden, hidden), torch.nn.GELU())
            d = hidden
        else:
            raise ValueError(f"unknown head arch {arch!r} (linear / mlp1 / mlp2)")
        self.out = torch.nn.Linear(d, 1)

    def forward(self, x):
        return self.out(self.body(self.norm(x))).squeeze(-1)


class _Block(torch.nn.Module):
    """Pre-LN causal self-attention + FFN. Residual output layers are zero-initialised,
    so a fresh block is the identity."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = torch.nn.LayerNorm(dim), torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.o = torch.nn.Linear(dim, dim)
        self.ff = torch.nn.Sequential(torch.nn.Linear(dim, 4 * dim), torch.nn.GELU(), torch.nn.Linear(4 * dim, dim))
        for m in (self.o, self.ff[2]):
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):  # [B, T, D]
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.ln2(x))


class Tower(torch.nn.Module):
    """Hidden states [T, H] or [B, T, H] -> LayerNorm -> Linear(H, dim) -> n causal blocks -> same rank, last dim = dim.

    No extra positional encoding: the causal mask gives order, and the trunk's hidden
    states already carry RoPE-derived position information. Runs under bf16 autocast
    on CUDA (SDPA flash kernel -> attention memory linear in T); params stay fp32."""

    def __init__(self, d_in: int, dim: int = 1024, n_layers: int = 2, heads: int = 8):
        super().__init__()
        self.ln_in = torch.nn.LayerNorm(d_in)
        self.proj = torch.nn.Linear(d_in, dim)
        self.blocks = torch.nn.ModuleList(_Block(dim, heads) for _ in range(n_layers))

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        squeeze = hs.dim() == 2
        dev = hs.device.type
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            x = self.proj(self.ln_in(hs.float()))
            if squeeze:
                x = x[None]
            for b in self.blocks:
                x = b(x)
        x = x.float()
        return x[0] if squeeze else x


class ValueNet(torch.nn.Module):
    """Optional tower + head: hidden [B, T, H] -> values [B, T]."""

    def __init__(self, d_in: int, arch: str = "linear", tower_layers: int = 2, tower_dim: int = 1024,
                 tower_heads: int = 8, head_arch: str = "linear", head_hidden: int = 1024):
        super().__init__()
        if arch == "tower":
            self.tower = Tower(d_in, tower_dim, tower_layers, tower_heads)
            d = tower_dim
        elif arch == "linear":
            self.tower, d = None, d_in
        else:
            raise ValueError(f"value arch {arch!r} -- only linear / tower")
        self.head = Head(d, head_arch, head_hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = hidden.float() if self.tower is None else self.tower(hidden)
        return self.head(h)
