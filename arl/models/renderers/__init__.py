"""Slim, self-contained message renderers.

Replaces the subset of tinker_cookbook.renderers that r3lm uses:
``get_renderer(name, tokenizer)`` returning a :class:`Renderer` exposing
``build_generation_prompt``, ``parse_response`` and ``get_stop_sequences``,
plus the ``Message`` / ``Renderer`` types.

Prompts are built with the tokenizer's official chat template, so tokens are
in-distribution for the model. Sampler/trainer token consistency is guaranteed
by construction: the env renders the prompt once and the *same* token ids are
fed to both the vLLM sampler and the DeepSpeed trainer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeAlias

from typing_extensions import TypedDict

from arl.models import types as tinker
from arl.models.tokenizer import Tokenizer


class Message(TypedDict, total=False):
    role: str
    content: str
    tool_calls: list[Any]


class ParseTermination(StrEnum):
    STOP = "stop"
    LENGTH = "length"


class Renderer:
    """Renders conversations to token sequences and parses responses back."""

    def __init__(self, tokenizer: Tokenizer, enable_thinking: bool | None = None,
                 reasoning_effort: str | None = None):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        # Qwen3.8's chat template takes a reasoning_effort of xhigh/medium/low and
        # turns it into a sentence at the top of the system prompt; ``None`` = do not
        # pass it at all (the template's own default then applies -- xhigh on
        # Qwen3.8, i.e. the longest chains). Templates that do not know the variable
        # ignore it, so this is inert on QwQ / Qwen3.6. See _RENDERER_CONFIG.
        self.reasoning_effort = reasoning_effort

    # -- prompt construction ------------------------------------------------ #
    def build_generation_prompt(self, messages: list[Message]) -> tinker.ModelInput:
        kwargs: dict[str, Any] = dict(add_generation_prompt=True, tokenize=True)
        extra: dict[str, Any] = {}
        if self.enable_thinking is not None:
            # Qwen3 / QwQ templates accept enable_thinking; ignored otherwise.
            extra["enable_thinking"] = self.enable_thinking
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        if extra:
            try:
                token_ids = self._encode(messages, **extra, **kwargs)
            except TypeError:
                token_ids = self._encode(messages, **kwargs)
        else:
            token_ids = self._encode(messages, **kwargs)
        return tinker.ModelInput.from_ints(list(token_ids))

    def _encode(self, messages: list[Message], **kwargs) -> list[int]:
        out = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        # apply_chat_template may return a list[int] or a BatchEncoding.
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return list(out)

    # -- response parsing --------------------------------------------------- #
    def parse_response(self, tokens: list[int]) -> tuple[Message, ParseTermination]:
        stop_ids = set(self._stop_token_ids())
        ended = ParseTermination.STOP if (tokens and tokens[-1] in stop_ids) else ParseTermination.LENGTH
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return {"role": "assistant", "content": text}, ended

    # -- stop conditions ---------------------------------------------------- #
    def _stop_token_ids(self) -> list[int]:
        ids: list[int] = []
        for tok in ("<|im_end|>",):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                ids.append(tid)
        if self.tokenizer.eos_token_id is not None:
            ids.append(int(self.tokenizer.eos_token_id))
        return list(dict.fromkeys(ids))  # dedup, preserve order

    def get_stop_sequences(self) -> list[int]:
        return self._stop_token_ids()


class RoleColonRenderer(Renderer):
    """Fallback renderer for base models without a chat template."""

    def build_generation_prompt(self, messages: list[Message]) -> tinker.ModelInput:
        parts = [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
        text = "\n".join(parts) + "\nassistant:"
        return tinker.ModelInput.from_ints(self.tokenizer.encode(text))


# reasoning_effort (2026-09-19): only Qwen3.8's template knows this variable. It maps
# xhigh/medium/low to one sentence prepended to the system prompt, and **defaults to
# xhigh** -- "think carefully through the task, validate key assumptions, consider
# plausible alternatives", i.e. the longest chains it has. ``medium`` is the neutral
# setting (the template adds nothing at all); ``low`` adds "keep your thinking brief
# and focused". ``None`` = never pass the variable, which leaves each model on its own
# default -- that is what the pre-existing names keep doing, so QwQ / Qwen3.6 render
# byte-for-byte as before.
# Which one a run uses is a **prompt-口径** choice: runs rendered with different
# entries are not comparable on length metrics. Pick it in the conf (renderer_name),
# never silently.
_RENDERER_CONFIG = {
    "qwen3": dict(enable_thinking=True, reasoning_effort=None),
    "qwen3_medium": dict(enable_thinking=True, reasoning_effort="medium"),
    "qwen3_low": dict(enable_thinking=True, reasoning_effort="low"),
    "qwen3_instruct": dict(enable_thinking=False, reasoning_effort=None),
    "qwen3_disable_thinking": dict(enable_thinking=False, reasoning_effort=None),
    "llama3": dict(enable_thinking=None, reasoning_effort=None),
    "deepseekv3": dict(enable_thinking=None, reasoning_effort=None),
}


def get_renderer(name: str, tokenizer: Tokenizer | None = None) -> Renderer:
    """Return a renderer by name. ``qwen3*``/``llama3``/``deepseekv3`` use the
    tokenizer's chat template; ``role_colon`` is a plain-text fallback.
    The ``qwen3_medium`` / ``qwen3_low`` variants only differ by the
    reasoning_effort handed to the template (Qwen3.8; see _RENDERER_CONFIG)."""
    if tokenizer is None:
        raise ValueError("get_renderer requires a tokenizer for the self-hosted backend")
    if name == "role_colon":
        return RoleColonRenderer(tokenizer)
    cfg = _RENDERER_CONFIG.get(name)
    if cfg is None:
        raise ValueError(
            f"Unknown renderer: {name}. Supported: {sorted(_RENDERER_CONFIG) + ['role_colon']}"
        )
    return Renderer(tokenizer, enable_thinking=cfg["enable_thinking"],
                    reasoning_effort=cfg.get("reasoning_effort"))


__all__ = ["Message", "Renderer", "RoleColonRenderer", "ParseTermination", "get_renderer"]
