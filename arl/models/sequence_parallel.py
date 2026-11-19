"""Ulysses sequence parallelism for packed Qwen3.5-MoE text training.

One ZeRO data-parallel world cooperates on one packed row at a time. The
original data owner computes its PPO loss. The autograd gather sums the
owner's gradient back into the position shards before ZeRO averages parameters.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_nn

from arl.models import router_replay
from arl.utils import distributed as du


def enabled():
    return os.environ.get("KERNEL_RL_SP16", "0") == "1"


class _Exchange(torch.autograd.Function):
    """[local positions, all heads, ...] <-> [all positions, local heads, ...]."""

    @staticmethod
    def forward(ctx, x, inverse):
        ctx.inverse = inverse
        world = dist.get_world_size()
        seq, heads, *dims = x.shape
        if inverse:
            send = x.reshape(world, seq // world, heads, *dims)
        else:
            send = x.reshape(seq, world, heads // world, *dims).transpose(0, 1)
        send = send.contiguous()
        received = torch.empty_like(send)
        dist.all_to_all_single(received, send)
        if inverse:
            return received.transpose(0, 1).reshape(seq // world, world * heads, *dims)
        return received.flatten(0, 1)

    @staticmethod
    def backward(ctx, grad):
        return _Exchange.apply(grad, not ctx.inverse), None


def _gather_positions(x):
    return _Exchange.apply(x[0], False).unsqueeze(0)


def _scatter_positions(x):
    return _Exchange.apply(x[0], True).unsqueeze(0)


def _full_attention(module, query, key, value, attention_mask, *, cu_seq_lens_q,
                    max_length_q, scaling, dropout=0.0, **kwargs):
    from flash_attn import flash_attn_varlen_func

    repeats = max(1, dist.get_world_size() // key.shape[1])
    q = _gather_positions(query.transpose(1, 2))[0]
    k = _gather_positions(key.repeat_interleave(repeats, dim=1).transpose(1, 2))[0]
    v = _gather_positions(value.repeat_interleave(repeats, dim=1).transpose(1, 2))[0]
    out = flash_attn_varlen_func(q, k, v, cu_seq_lens_q, cu_seq_lens_q,
                                 max_length_q, max_length_q, dropout,
                                 softmax_scale=scaling, causal=True)
    return _scatter_positions(out.unsqueeze(0)), None


def _depthwise_conv(kernel, x, weight, bias, **kwargs):
    # Convolution channels are independent. Give each rank full position
    # history for its own channels, including the packed sequence boundaries.
    full = _gather_positions(x.transpose(1, 2)).transpose(1, 2)
    rank, world = dist.get_rank(), dist.get_world_size()
    w = weight.chunk(world)[rank]
    b = bias.chunk(world)[rank] if bias is not None else None
    out = kernel(x=full, weight=w, bias=b, **kwargs)
    return _scatter_positions(out.transpose(1, 2)).transpose(1, 2)


def _delta_rule(kernel, q, k, v, *, g, beta, **kwargs):
    q, k, v, g, beta = [_gather_positions(x) for x in (q, k, v, g, beta)]
    out, _ = kernel(q, k, v, g=g, beta=beta, **kwargs)
    return _scatter_positions(out), None


def validate_model(config, world):
    if world != 16:
        raise ValueError("SP16 requires exactly 16 training ranks")
    cfg = config.get("text_config", config)
    if cfg.get("model_type") not in ("qwen3_5_moe", "qwen3_5_moe_text"):
        raise ValueError("SP16 requires a Qwen3.5-MoE text model (including Qwen3.6-35B-A3B)")
    kv = cfg["num_key_value_heads"]
    if (cfg["num_attention_heads"] % world or (kv % world and world % kv)
            or cfg["linear_num_value_heads"] % world):
        raise ValueError("Qwen query, KV or linear attention head counts are incompatible with SP16")


def install(model):
    from transformers import AttentionInterface

    validate_model(model.config.to_dict(), dist.get_world_size())
    AttentionInterface.register("arl_ulysses", _full_attention)
    for block in model.model.layers:
        if block.block_type == "full_attention":
            attn = block.self_attn
            # Keep the text model's FlashAttention mask builder and sampling config.
            attn.config = copy.copy(attn.config)
            attn.config._attn_implementation = "arl_ulysses"
        else:
            attn = block.linear_attn
            if attn.causal_conv1d_fn is None:
                raise RuntimeError("SP16 needs causal-conv1d for packed linear attention")
            attn.causal_conv1d_fn = partial(_depthwise_conv, attn.causal_conv1d_fn)
            attn.chunk_gated_delta_rule = partial(_delta_rule, attn.chunk_gated_delta_rule)


@dataclass
class Pack:
    owner: int
    indices: list[int]
    lengths: list[int]
    route_shape: tuple[int, ...]


def schedule(chunks, per):
    """Share small descriptors only. Trajectories and full routing stay on their owner."""
    local = []
    for indices in chunks:
        route = next((per[i][8] for i in indices if per[i][8] is not None), None)
        local.append(Pack(dist.get_rank(), indices, [per[i][4] for i in indices],
                          tuple(route.shape[1:]) if route is not None else ()))
    return [pack for packs in du.all_gather_object(local, group=du.gloo_group()) for pack in packs]


def _scatter_cpu(tensor, owner, local_len, tail, dtype, fill):
    # Routed experts can be much larger than the token IDs. Scatter on CPU to
    # avoid staging the full routing tensor on a single accelerator.
    received = torch.empty((local_len, *tail), dtype=dtype)
    parts = None
    if dist.get_rank() == owner:
        missing = local_len * dist.get_world_size() - tensor.size(0)
        if missing:
            tensor = torch.cat((tensor, tensor.new_full((missing, *tail), fill)))
        parts = [part.view(torch.uint8) for part in tensor.contiguous().split(local_len)]
    dist.scatter(received.view(torch.uint8), parts, src=owner, group=du.gloo_group())
    return received


def prepare(pack, per, device, pad_id, head_chunks, replay):
    world = dist.get_world_size()
    total = sum(pack.lengths)
    local_len = max(1, head_chunks, (total + world - 1) // world)
    padding = local_len * world - total
    lengths = pack.lengths + ([padding] if padding else [])
    offsets = torch.tensor([0, *lengths]).cumsum(0)
    seq_idx = torch.arange(len(lengths), dtype=torch.int32).repeat_interleave(torch.tensor(lengths))
    start = dist.get_rank() * local_len
    positions = torch.arange(start, start + local_len) - offsets[seq_idx[start:start + local_len].long()]
    owned = dist.get_rank() == pack.owner
    tokens = None
    if owned:
        ids = torch.cat([per[i][0] for i in pack.indices])
        targets = torch.cat([per[i][1][:per[i][4]] for i in pack.indices])
        tokens = torch.stack((ids, targets), dim=1)
    tokens = _scatter_cpu(tokens, pack.owner, local_len, (2,), torch.long, pad_id).to(device)
    if replay:
        routes = None
        if pack.route_shape:
            if owned:
                routes = router_replay.build_index([per[i][8] for i in pack.indices], pack.lengths, "cpu")
            routes = _scatter_cpu(routes, pack.owner, local_len, pack.route_shape, torch.int16, -1).to(device)
        router_replay.begin(routes)
    cu = offsets.to(device=device, dtype=torch.int32)
    args = dict(input_ids=tokens[:, 0].contiguous()[None], position_ids=positions.to(device)[None],
                cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=max(lengths),
                max_length_k=max(lengths), seq_idx=seq_idx.to(device)[None])
    slots = [(0, off) for off in offsets[:len(pack.lengths)].tolist()]
    return args, tokens[:, 1].contiguous(), slots


def gather_output(local, total):
    # Backward reduce-scatters the single owner's loss gradient to all ranks.
    return torch.cat(dist_nn.all_gather(local.contiguous()))[:total]
