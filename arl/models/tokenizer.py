"""Tokenizer helpers, vendored from tinker_cookbook.tokenizer_utils.

Backend-agnostic: wraps HuggingFace ``AutoTokenizer`` with a small custom
registry. No Tinker/cloud dependency.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    Tokenizer: TypeAlias = Any

_CUSTOM_TOKENIZER_REGISTRY: dict[str, Callable[[], Tokenizer]] = {}


def register_tokenizer(name: str, factory: Callable[[], Tokenizer]) -> None:
    _CUSTOM_TOKENIZER_REGISTRY[name] = factory


def get_registered_tokenizer_names() -> list[str]:
    return list(_CUSTOM_TOKENIZER_REGISTRY.keys())


def is_tokenizer_registered(name: str) -> bool:
    return name in _CUSTOM_TOKENIZER_REGISTRY


def unregister_tokenizer(name: str) -> bool:
    if name in _CUSTOM_TOKENIZER_REGISTRY:
        del _CUSTOM_TOKENIZER_REGISTRY[name]
        return True
    return False


def get_tokenizer(model_name: str) -> Tokenizer:
    """Return a tokenizer, checking the custom registry before HuggingFace."""
    if (tokenizer := _CUSTOM_TOKENIZER_REGISTRY.get(model_name)) is not None:
        return tokenizer()
    return _get_hf_tokenizer(model_name)


@cache
def _get_hf_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    model_name = model_name.split(":")[0]
    if os.path.isdir(model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    kwargs: dict[str, Any] = {}
    if os.environ.get("HF_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"):
        kwargs["trust_remote_code"] = True
    return AutoTokenizer.from_pretrained(model_name, **kwargs)
