"""A drop-in ``tinker`` namespace backed by the self-hosted backend.

Existing r3lm code does ``import tinker`` and references ``tinker.Datum``,
``tinker.ModelInput``, ``tinker.AdamParams`` (pure data types) as well as
``tinker.ServiceClient`` / ``tinker.TrainingClient`` / ``tinker.SamplingClient``
(the clients). Swapping ``import tinker`` for ``from arl.models import
tinker_compat as tinker`` keeps all of that working.

Data types are imported eagerly (cheap). The three client classes pull in
vLLM/DeepSpeed, so they are resolved lazily via ``__getattr__`` to keep importing
this module fast.
"""

from __future__ import annotations

from arl.models.types import (  # noqa: F401
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LossFnType,
    ModelInput,
    ModelInputChunk,
    SampledSequence,
    SampleResponse,
    SamplingParams,
    StopReason,
    TensorData,
    types,
)


def __getattr__(name: str):
    if name == "ServiceClient":
        from arl.models.service import LocalServiceClient

        return LocalServiceClient
    if name == "TrainingClient":
        from arl.models.training_client import LocalTrainingClient

        return LocalTrainingClient
    if name == "SamplingClient":
        from arl.models.sampling import SamplingClient

        return SamplingClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
