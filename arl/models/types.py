"""Self-hosted replacements for the Tinker SDK data types.

These are plain, dependency-free containers that mirror the small subset of the
``tinker`` public data API that arl actually uses (see the seam inventory).
They carry NO network behaviour -- the hosted clients are replaced separately in
``arl.models.{sampling,training_client,service}``.

Only text chunks are supported (arl is a text-only task); image chunks from
the original Tinker API are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Scalar aliases
# --------------------------------------------------------------------------- #
StopReason: TypeAlias = Literal["stop", "length"]
# The loss function selector. We only implement "ppo" (importance-sampling GRPO).
LossFnType: TypeAlias = Literal["ppo", "importance_sampling", "cross_entropy"]


# --------------------------------------------------------------------------- #
# Model input: a sequence of token chunks
# --------------------------------------------------------------------------- #
class ModelInputChunk:
    """Base class for a chunk of model input."""

    length: int


@dataclass
class EncodedTextChunk(ModelInputChunk):
    """A run of already-tokenized text."""

    tokens: list[int]

    @property
    def length(self) -> int:  # type: ignore[override]
        return len(self.tokens)


@dataclass
class ModelInput:
    """A tokenized prompt, stored as a list of chunks.

    Mirrors ``tinker.ModelInput``: supports ``.chunks``, ``.length``,
    ``ModelInput.from_ints(...)`` and ``.to_ints()``.
    """

    chunks: list[ModelInputChunk] = field(default_factory=list)

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        if not tokens:
            return cls(chunks=[])
        return cls(chunks=[EncodedTextChunk(tokens=list(tokens))])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    @property
    def length(self) -> int:
        return sum(c.length for c in self.chunks)

    def to_ints(self) -> list[int]:
        """Flatten all text chunks into a single token-id list.

        Non-text chunks are unsupported and raise (arl is text-only).
        """
        out: list[int] = []
        for c in self.chunks:
            if isinstance(c, EncodedTextChunk):
                out.extend(c.tokens)
            else:
                raise TypeError(f"Non-text chunk {type(c).__name__} is not supported")
        return out


# --------------------------------------------------------------------------- #
# TensorData: opaque wrapper around a tensor/array, matching Tinker's API
# --------------------------------------------------------------------------- #
@dataclass
class TensorData:
    """Wrapper for a dense numeric array, matching ``tinker.TensorData``.

    Stored internally as a torch tensor. Only the methods arl relies on
    are implemented: ``from_torch``/``from_numpy``/``to_torch``/``to_numpy``/
    ``shape``/``tolist``.
    """

    _tensor: torch.Tensor

    @classmethod
    def from_torch(cls, t: torch.Tensor) -> "TensorData":
        return cls(_tensor=t.detach().cpu())

    @classmethod
    def from_numpy(cls, a: np.ndarray) -> "TensorData":
        return cls(_tensor=torch.as_tensor(a))

    def to_torch(self) -> torch.Tensor:
        return self._tensor

    def to_numpy(self) -> np.ndarray:
        return self._tensor.numpy()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._tensor.shape)

    def tolist(self) -> list:
        return self._tensor.tolist()


# --------------------------------------------------------------------------- #
# Datum: one training example
# --------------------------------------------------------------------------- #
@dataclass
class Datum:
    """A single training datum: a model input plus per-token loss inputs.

    ``loss_fn_inputs`` maps names (e.g. ``target_tokens``, ``logprobs``,
    ``advantages``, ``mask``) to :class:`TensorData`.
    """

    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]


# --------------------------------------------------------------------------- #
# Sampling / optimizer parameter bags
# --------------------------------------------------------------------------- #
@dataclass
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 1.0
    stop: list[str] | list[int] | None = None
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    # 思考预算 (见 SamplingClient.sample_async): >0 且小于 max_tokens 时, 思考段写满这么多
    # token 还没出 ``think_end_ids`` 就把 ``think_force_ids`` (过渡语 + </think>) 接上去,
    # 再接着采答案; 两段合计仍 <= max_tokens。0 = 关。
    think_budget: int = 0
    think_force_ids: list[int] | None = None
    think_end_ids: list[int] | None = None
    affinity_key: str | None = None  # one trajectory stays on one sampler replica


@dataclass
class AdamParams:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None


# --------------------------------------------------------------------------- #
# Sampling response types
# --------------------------------------------------------------------------- #
@dataclass
class SampledSequence:
    tokens: list[int]
    logprobs: list[float] | None = None
    stop_reason: StopReason = "stop"
    # 逐 token 的 loss 掩码 (None = 全 1)。思考预算强塞进去的 token 不是策略采的, 记 0。
    loss_mask: list[float] | None = None
    think_forced: bool = False
    # 路由回放 (arl/models/router_replay.py): vLLM 回传的 [prompt+output-1, layers, top_k]
    # 专家 id (np.ndarray, uint8)。没开回放时 None。
    routed_experts: Any = None


@dataclass
class SampleResponse:
    sequences: list[SampledSequence]


@dataclass
class ForwardBackwardOutput:
    """Result of one forward_backward call, mirroring the field loop.py reads."""

    loss_fn_outputs: list[dict[str, TensorData]]
    metrics: dict[str, float] = field(default_factory=dict)


# A submodule-style alias so `from ... import types as tinker; tinker.types.X` works.
class _TypesNamespace:
    EncodedTextChunk = EncodedTextChunk
    ModelInput = ModelInput
    ModelInputChunk = ModelInputChunk
    TensorData = TensorData
    Datum = Datum
    LossFnType = LossFnType


types: Any = _TypesNamespace()
