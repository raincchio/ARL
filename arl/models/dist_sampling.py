"""Distributed sampling client: all ranks funnel prompts to rank 0's shared engine.

Under the shared-sampler architecture (GPU_IMPROVEMENTS.md 改进②) only rank 0
holds the vLLM engine (TP over SAMPLER_GPUS). Sampling becomes an explicit
collective phase:

    every rank:  results = await client.sample_batch(requests)   # COLLECTIVE
                 - gloo gather(requests) -> rank 0
                 - rank 0: fan out k concurrent n=1 generates per request
                 - gloo scatter(results) -> every rank gets its own back

``sample_batch`` MUST be called by every rank each batch -- with an empty list
if the rank has no prompts -- exactly like ``compute_values``: the gather and
scatter are collectives, and a rank that skips them deadlocks the job.

Failed samples come back as ``None`` entries (callers already tolerate missing
group members, see ``ppo_loop``'s former ``_sample_k`` semantics).

``submit_and_poll`` is the non-blocking variant (ARMv2's async rollout)::

    every rank:  done = await client.submit_and_poll(new_requests, timeout)
                 - gloo gather(new requests) -> rank 0
                 - rank 0: ensure_future each one into a PERSISTENT in-flight table
                 - rank 0: wait up to `timeout`, then harvest whatever finished
                 - gloo scatter(finished) -> every rank gets its own back

The difference that matters is ``_run_requests``'s ``asyncio.gather``: it waits
for the SLOWEST sequence in the batch, so a batch costs max(), not mean(), and
every sequence in it is released at the same instant. vLLM underneath is already
continuous-batching -- that gather is a pure software barrier on top of it, and
it is what makes downstream (eval / RM) load arrive in bursts. ``submit_and_poll``
drops the barrier: a sequence is handed back on the first poll after it finishes,
and requests submitted in different waves coexist in the engine.

Both entry points share one engine and one gloo group, so each is its own
collective phase: every rank must agree on which one it is calling. They compose
across phases -- ARMv2 drives rollout with ``submit_and_poll`` and still runs its
held-out eval through ``sample_batch``; generations already in flight keep
decoding in the background while ``sample_batch`` awaits its own batch, and get
harvested by the next poll. ARM's synchronous PPO uses ``sample_batch`` only.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from arl.models import types as tinker
from arl.utils import distributed as dist_utils

logger = logging.getLogger(__name__)


@dataclass
class SampleRequest:
    """One prompt's sampling job; picklable so it survives gather_object."""

    prompt_ids: list[int]
    num_samples: int  # = group_size; expanded rank-0-side into k concurrent n=1 calls
    max_tokens: int
    temperature: float
    stop: list | None = None
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    extras: dict = field(default_factory=dict)
    # 思考预算 (见 tinker.SamplingParams / SamplingClient._sample_with_think_budget)。0 = 关。
    think_budget: int = 0
    think_force_ids: list[int] | None = None
    think_end_ids: list[int] | None = None


class DistributedSamplingClient:
    """rank 0 owns the real engine; every rank shares the collective interface.

    Also keeps ``sample_async`` (single-prompt, rank-0-only) so non-SPMD callers
    (single-process debugging) keep working unchanged.
    """

    def __init__(self, engine):
        self.engine = engine  # SamplingEngine on rank 0, None elsewhere
        rank, world_size, _ = dist_utils.dist_info()
        self._rank, self._world_size = rank, world_size
        if rank == 0 and engine is None and world_size > 1:
            raise ValueError("rank 0 must hold the sampling engine")
        # rank-0-side persistent in-flight table for submit_and_poll:
        #   (origin_rank, req_id) -> Task. Survives across calls -- that IS the
        #   feature: requests submitted in wave N may come back in wave N+3.
        self._inflight: dict[tuple[int, str], asyncio.Task] = {}

    # ---- collective batch interface --------------------------------------- #
    async def sample_batch(
        self, requests: list[SampleRequest]
    ) -> list[list[tinker.SampledSequence | None]]:
        """COLLECTIVE. Returns, per request, ``num_samples`` sequences (None = failed).

        Every rank must call this in the same phase, empty list included.
        """
        if self._world_size <= 1:
            return await self._run_requests(requests)

        group = dist_utils.gloo_group()
        gathered = dist_utils.gather_object_to_rank0(requests, group=group)
        if self._rank == 0:
            flat: list[SampleRequest] = [r for per_rank in gathered for r in per_rank]
            flat_results = await self._run_requests(flat)
            # regroup by originating rank
            out: list[list] = []
            i = 0
            for per_rank in gathered:
                out.append(flat_results[i : i + len(per_rank)])
                i += len(per_rank)
        else:
            out = None
        return dist_utils.scatter_object_from_rank0(out, group=group)

    # ---- non-blocking batch interface (ARMv2 async rollout) ---------------- #
    async def submit_and_poll(
        self, requests: list[SampleRequest], timeout: float = 5.0,
        abort: list[str] | None = None,
    ) -> list[tuple[str, tinker.SampledSequence | None]]:
        """COLLECTIVE. Submit ``requests``, return whatever has FINISHED by now.

        ``abort``: req_ids (this rank's) whose in-flight generation should be
        cancelled -- the rank-0 task is cancelled, which vLLM's ``generate``
        turns into an engine-side abort (V1 catches CancelledError). Aborted ids
        never come back from a poll. Used by ppo_loop's over-provisioned groups
        (extra_samples) to stop the trajectories it decided not to keep.

        Returned pairs are ``(req_id, sequence)`` for this rank only; ``sequence``
        is None when that generation failed. The ids come back in completion
        order and generally belong to EARLIER calls -- callers must key on the id
        rather than on submission order.

        Every rank must call this in the same phase, empty list included: the
        gather and scatter are collectives.

        ``req_id`` must be set in ``SampleRequest.extras['req_id']`` and be unique
        per origin rank while in flight. ``num_samples`` must be 1 -- the k>1 fan
        out of ``sample_batch`` has no meaning once results are polled
        individually; callers that want k samples submit k requests.
        """
        for req in requests:
            if req.num_samples != 1:
                raise ValueError(
                    "submit_and_poll requires num_samples=1 (submit k requests "
                    f"for k samples); got {req.num_samples}"
                )
            if not req.extras.get("req_id"):
                raise ValueError("submit_and_poll requires extras['req_id']")

        aborts = [str(r) for r in (abort or [])]
        if self._world_size <= 1:
            self._abort(aborts, origin=0)
            self._start(requests, origin=0)
            return [(rid, seq) for _origin, rid, seq in await self._poll(timeout)]

        group = dist_utils.gloo_group()
        gathered = dist_utils.gather_object_to_rank0((requests, aborts), group=group)
        if self._rank == 0:
            for origin, (per_rank, per_abort) in enumerate(gathered):
                self._abort(per_abort, origin=origin)
                self._start(per_rank, origin=origin)
            out: list[list] = [[] for _ in range(self._world_size)]
            for origin, rid, seq in await self._poll(timeout):
                out[origin].append((rid, seq))
        else:
            out = None
        return dist_utils.scatter_object_from_rank0(out, group=group)

    @property
    def n_inflight(self) -> int:
        """rank-0-side count of generations still running (0 on other ranks)."""
        return len(self._inflight)

    def _abort(self, rids: list[str], origin: int) -> None:
        """rank-0-side: cancel in-flight generations; popped first so ``_poll``
        never hands their (cancelled) tasks back."""
        for rid in rids:
            task = self._inflight.pop((origin, rid), None)
            if task is not None and not task.done():
                task.cancel()

    def _start(self, requests: list[SampleRequest], origin: int) -> None:
        """rank-0-side: fire each request into the engine WITHOUT awaiting it."""
        for req in requests:
            key = (origin, str(req.extras["req_id"]))
            if key in self._inflight:
                logger.warning(f"duplicate req_id in flight, dropping: {key}")
                continue
            self._inflight[key] = asyncio.ensure_future(self._one(req))

    async def _poll(
        self, timeout: float
    ) -> list[tuple[int, str, tinker.SampledSequence | None]]:
        """rank-0-side: wait up to ``timeout``, then harvest everything done.

        Waiting for ALL_COMPLETED-or-timeout (rather than FIRST_COMPLETED) makes
        the caller's wave cadence a predictable ``timeout``: FIRST_COMPLETED would
        return after a few milliseconds at steady state and spin the outer loop --
        two gloo collectives, a metrics row and a wandb point per spin.
        """
        if self._inflight:
            await asyncio.wait(
                list(self._inflight.values()),
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
        done = []
        for key, task in list(self._inflight.items()):
            if not task.done():
                continue
            del self._inflight[key]
            origin, rid = key
            try:
                seq = task.result()
            except Exception as e:  # noqa: BLE001 -- lost samples are tolerated
                logger.warning(f"sample failed: {type(e).__name__}: {e}")
                seq = None
            done.append((origin, rid, seq))
        return done

    async def _one(self, req: SampleRequest) -> tinker.SampledSequence | None:
        try:
            sr = await self._sample_one(req)
            return sr.sequences[0]
        except Exception as e:  # noqa: BLE001 -- lost samples are tolerated
            logger.warning(f"sample failed: {type(e).__name__}: {e}")
            return None

    async def _run_requests(
        self, requests: list[SampleRequest]
    ) -> list[list[tinker.SampledSequence | None]]:
        """rank-0-side: k concurrent n=1 generates per request (engine batches them)."""
        tasks = [self._one(req) for req in requests for _ in range(req.num_samples)]
        flat = await asyncio.gather(*tasks)
        out, i = [], 0
        for req in requests:
            out.append(list(flat[i : i + req.num_samples]))
            i += req.num_samples
        return out

    async def _sample_one(self, req: SampleRequest) -> tinker.SampleResponse:
        from arl.models.sampling import SamplingClient

        return await SamplingClient(self.engine).sample_async(
            prompt=tinker.ModelInput.from_ints(req.prompt_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=req.stop,
                top_p=req.top_p,
                top_k=req.top_k,
                seed=req.seed,
                think_budget=req.think_budget,
                think_force_ids=req.think_force_ids,
                think_end_ids=req.think_end_ids,
            ),
        )

    # ---- single-prompt passthrough (rank 0 / single-process only) ---------- #
    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
    ) -> tinker.SampleResponse:
        if self.engine is None:
            raise RuntimeError(
                "sample_async needs the engine (rank 0); non-rank-0 callers must "
                "use the collective sample_batch()"
            )
        from arl.models.sampling import SamplingClient

        return await SamplingClient(self.engine).sample_async(
            prompt=prompt, num_samples=num_samples, sampling_params=sampling_params
        )
