"""Minimal SPMD (data-parallel) helpers for the ``deepspeed --num_gpus=N`` launcher.

Under the launcher, N processes each run the whole training loop. These helpers let
the loop:
  - discover its rank/world_size,
  - gate file I/O and logging to rank 0,
  - round-robin shard each batch's problem-groups across ranks,
  - reduce scalar metrics to a correct global view,
  - barrier between the rank-0 weight write and every rank's vLLM reload.

torch.distributed is initialized by DeepSpeed (``deepspeed.initialize``); these
helpers no-op cleanly when running single-process (WORLD_SIZE==1).
"""

from __future__ import annotations

import os


def dist_info() -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` from env (defaults 0/1/0)."""
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _pg_ready() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _reduce_device():
    import torch

    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    return torch.device("cpu")


def barrier() -> None:
    if _pg_ready():
        import torch.distributed as dist

        dist.barrier()


def destroy() -> None:
    """Tear down the default process group so ranks exit cleanly.

    Without this, SPMD ranks can deadlock at interpreter shutdown: NCCL/GC
    finalizers run in an unsynchronized order and a rank blocks in an
    uninterruptible collective wait (state ``D``) that even SIGKILL can't clear
    until the kernel wait returns.
    """
    if _pg_ready():
        import torch.distributed as dist

        dist.destroy_process_group()


def shard_list(items: list, rank: int, world_size: int) -> list:
    """Round-robin shard: rank ``r`` gets ``items[r::world_size]``."""
    if world_size <= 1:
        return items
    return items[rank::world_size]


def _all_reduce_scalar(value: float, op: str) -> float:
    if not _pg_ready():
        return value
    import torch
    import torch.distributed as dist

    reduce_op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op]
    t = torch.tensor([float(value)], dtype=torch.float64, device=_reduce_device())
    dist.all_reduce(t, op=reduce_op)
    return float(t.item())


def all_reduce_sum(value: float) -> float:
    return _all_reduce_scalar(value, "sum")


def all_reduce_max(value: float) -> float:
    return _all_reduce_scalar(value, "max")


_STATS_GROUP = None
_STATS_TIMEOUT_MIN = 10


def stats_group():
    """Lazily create (once) a gloo subgroup for progress/statistics reduces.

    Separate from :func:`gloo_group` purely for its timeout. That one carries 120
    minutes because it ships real work (sampling requests) that legitimately takes
    that long; borrowing it for bookkeeping would turn a wedged rank into a two-hour
    silent hang. Ten minutes is far more than a 5-float CPU reduce among ranks that
    have all just finished the same batch can need, and bounds the damage.

    COLLECTIVE on first call: every rank must reach it.
    """
    global _STATS_GROUP
    if not _pg_ready():
        return None
    if _STATS_GROUP is None:
        from datetime import timedelta

        import torch.distributed as dist

        _STATS_GROUP = dist.new_group(
            backend="gloo", timeout=timedelta(minutes=_STATS_TIMEOUT_MIN)
        )
    return _STATS_GROUP


def all_reduce_sum_list_gloo(values: list[float]) -> list[float]:
    """Sum a small vector across ranks in ONE collective, over GLOO (CPU).

    For progress/statistics only, and deliberately not on the NCCL default group:

      * it must not contend with (or wedge behind) the NCCL traffic doing real work
        -- weight sync and the sampler's gather/scatter;
      * an unhealthy rank makes an NCCL collective hang until the watchdog aborts the
        whole job, and losing a finished rollout to a bookkeeping collective is
        exactly the failure mode ``run()``'s final reduce already guards against.
        (With remote eval the generated kernels no longer run on the training GPUs,
        so the specific "poisoned by an illegal access" path is off the table -- but
        a wedged rank is not, and this reduce is worth nothing either way.)

    Gloo runs on CPU, on a group with a bounded timeout (see ``stats_group``).

    COLLECTIVE: every rank must call this the same number of times, so only call it
    from points every rank reaches deterministically (e.g. a batch boundary), never
    from a timer.
    """
    if not _pg_ready():
        return list(values)
    import torch
    import torch.distributed as dist

    t = torch.tensor([float(v) for v in values], dtype=torch.float64)  # CPU for gloo
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=stats_group())
    return [float(x) for x in t.tolist()]


_GLOO_GROUP = None


def gloo_group(timeout_minutes: int = 120):
    """Lazily create (once) and return a gloo subgroup over all ranks.

    Used to ship python objects (sampling requests/results) between ranks
    without touching the NCCL default group: gloo runs on CPU, doesn't hold
    GPU buffers, and can take a long timeout without an NCCL watchdog killing
    the job mid-rollout. Collective: EVERY rank must call this the first time.
    Returns None when single-process.
    """
    global _GLOO_GROUP
    if not _pg_ready():
        return None
    if _GLOO_GROUP is None:
        from datetime import timedelta

        import torch.distributed as dist

        _GLOO_GROUP = dist.new_group(
            backend="gloo", timeout=timedelta(minutes=timeout_minutes)
        )
    return _GLOO_GROUP


def gather_object_to_rank0(obj, group=None) -> list | None:
    """Collective gather of an arbitrary picklable object; rank 0 gets the list.

    Returns the per-rank list on rank 0, None on other ranks (and a 1-element
    list when single-process).
    """
    if not _pg_ready():
        return [obj]
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    out = [None] * world_size if rank == 0 else None
    dist.gather_object(obj, out, dst=0, group=group)
    return out


def all_gather_object(obj, group=None) -> list:
    """Collective gather of a picklable object; EVERY rank gets the full list.

    Same call as :func:`gather_object_to_rank0` minus the dst, for the case where
    the result has to be identical on every rank rather than only on rank 0 (e.g.
    a metric each rank then logs or asserts on). Single-process: ``[obj]``.
    """
    if not _pg_ready():
        return [obj]
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    out: list = [None] * world_size
    dist.all_gather_object(out, obj, group=group)
    return out


def scatter_object_from_rank0(objs: list | None, group=None):
    """Collective scatter from rank 0; every rank gets its element.

    ``objs`` must be a world_size-长 list on rank 0, ignored elsewhere.
    Single-process: returns objs[0].
    """
    if not _pg_ready():
        return objs[0]
    import torch.distributed as dist

    out = [None]
    dist.scatter_object_list(out, objs, src=0, group=group)
    return out[0]


def reduce_weighted_metrics(
    metrics: dict,
    mean_keys: list[str],
    local_weight: float,
    sum_keys: list[str] = (),
) -> dict:
    """Return ``metrics`` with a correct GLOBAL view of the given keys.

    ``mean_keys`` are weight-averaged by ``local_weight`` (e.g. per-rank trajectory
    count) across ranks; ``sum_keys`` are summed. All ranks get the same result.
    No-op when single-process. Batches every value into one all_reduce.
    """
    if not _pg_ready():
        return metrics
    import torch
    import torch.distributed as dist

    mean_keys = list(mean_keys)
    sum_keys = list(sum_keys)
    packed = [float(local_weight)]
    packed += [float(metrics.get(k, 0.0)) * float(local_weight) for k in mean_keys]
    packed += [float(metrics.get(k, 0.0)) for k in sum_keys]
    t = torch.tensor(packed, dtype=torch.float64, device=_reduce_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vals = t.tolist()
    global_weight = vals[0]
    out = dict(metrics)
    for i, k in enumerate(mean_keys, start=1):
        out[k] = (vals[i] / global_weight) if global_weight > 0 else 0.0
    for j, k in enumerate(sum_keys, start=1 + len(mean_keys)):
        out[k] = vals[j]
    return out
