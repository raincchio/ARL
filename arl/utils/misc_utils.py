"""Small dependency-free helpers, vendored from tinker_cookbook.utils.misc_utils."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    """Measure wall-clock time of a block and store it in ``metrics['time/{key}']``."""
    logger.info(f"Starting {key}")
    tstart = time.time()
    yield
    logger.info(f"{key} took {time.time() - tstart:.2f} seconds")
    metrics[f"time/{key}"] = time.time() - tstart


safezip = cast(type[zip], lambda *args, **kwargs: zip(*args, **kwargs, strict=True))
"""A strict ``zip`` that raises ``ValueError`` when iterables differ in length."""


def all_same(xs: list[Any]) -> bool:
    """True if every element equals the first (or the list is empty)."""
    return all(x == xs[0] for x in xs)


def dict_mean(list_of_dicts: list[dict[str, float | int]]) -> dict[str, float]:
    key2values: dict[str, list] = {}
    for d in list_of_dicts:
        for k, v in d.items():
            key2values.setdefault(k, []).append(v)
    return {k: float(np.mean(values)) for k, values in key2values.items()}
