"""Token/message completers.

Interfaces (``TokenCompleter``, ``TokensWithLogprobs``, ``StopCondition``,
``MessageCompleter``) are vendored from tinker_cookbook.completers. The hosted
``TinkerTokenCompleter`` is replaced by :class:`LocalTokenCompleter`, which
samples from a local vLLM-backed :class:`~arl.models.sampling.SamplingClient`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from arl.models import types as tinker

if TYPE_CHECKING:
    from arl.models.renderers import Message, Renderer
    from arl.models.sampling import SamplingClient

# Stop sequences (strings) or stop token IDs.
StopCondition: TypeAlias = "list[str] | list[int]"


@dataclass
class TokensWithLogprobs:
    """A sequence of token IDs with optional log-probabilities and a stop reason."""

    tokens: list[int]
    maybe_logprobs: list[float] | None
    stop_reason: tinker.StopReason = "stop"
    # 逐 token 的 loss 掩码, None = 全 1 (见 SampledSequence.loss_mask)。
    maybe_mask: list[float] | None = None
    # 采样时的 MoE 路由, 覆盖 prompt + 本次输出 (见 SampledSequence.routed_experts)。
    maybe_routing: Any = None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


class TokenCompleter:
    """Abstract interface for generating tokens from a prompt."""

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        raise NotImplementedError


class MessageCompleter:
    """Abstract interface for generating message responses."""

    async def __call__(self, messages: "list[Message]") -> "Message":
        raise NotImplementedError


@dataclass
class LocalTokenCompleter(TokenCompleter):
    """Token completer backed by a local (vLLM) sampling client.

    Drop-in replacement for ``tinker_cookbook.completers.TinkerTokenCompleter``.
    """

    sampling_client: "SamplingClient"
    max_tokens: int
    temperature: float = 1.0
    # Nucleus / top-k truncation. Defaults are the no-op values (full distribution);
    # reasoning models (QwQ et al.) want their generation_config's 0.95/40 instead --
    # pure temperature sampling lets the tail through and invites repetition loops.
    top_p: float = 1.0
    top_k: int = -1
    context_window: int | None = None

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        max_tokens = self.max_tokens
        if self.context_window is not None:
            max_tokens = min(max_tokens, self.context_window - model_input.length)
            if max_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({model_input.length}) exceeds context window "
                    f"({self.context_window}). No room for generation."
                )

        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                stop=stop,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            ),
        )

        sampled_seq = sample_result.sequences[0]
        assert sampled_seq.logprobs is not None
        return TokensWithLogprobs(
            tokens=sampled_seq.tokens,
            maybe_logprobs=sampled_seq.logprobs,
            stop_reason=sampled_seq.stop_reason,
            maybe_routing=sampled_seq.routed_experts,
        )


# Backwards-compatible alias so existing imports keep working during migration.
TinkerTokenCompleter = LocalTokenCompleter
