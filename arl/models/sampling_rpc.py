"""Per-rollout async connections to the training job's shared sampler."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import secrets
import struct
from contextlib import asynccontextmanager

import numpy as np

from arl.models.dist_sampling import SampleRequest
from arl.models.types import SampledSequence
from arl.utils import distributed as dist_utils

logger = logging.getLogger(__name__)
_SIZE = struct.Struct('!I')
_MAX_METADATA = 16 << 20
_MAX_ROUTING = 1 << 30
_KEY_SIZE = 32


async def _read_metadata(reader):
    size, = _SIZE.unpack(await reader.readexactly(_SIZE.size))
    if size > _MAX_METADATA:
        raise ValueError('sampler metadata is too large')
    return json.loads(await reader.readexactly(size))


def _write_metadata(writer, value):
    payload = json.dumps(value, separators=(',', ':')).encode()
    if len(payload) > _MAX_METADATA:
        raise ValueError('sampler metadata is too large')
    writer.write(_SIZE.pack(len(payload)))
    writer.write(payload)


async def _write_sequence(writer, sequence):
    if sequence is None:
        _write_metadata(writer, None)
    else:
        metadata = vars(sequence).copy()
        routing = metadata.pop('routed_experts')
        if routing is not None:
            routing = np.ascontiguousarray(routing)
            if routing.nbytes > _MAX_ROUTING:
                raise ValueError('sampler router trace is too large')
            metadata['routing'] = [routing.dtype.str, routing.shape, routing.nbytes]
        _write_metadata(writer, metadata)
        if routing is not None and routing.nbytes:
            writer.write(memoryview(routing).cast('B'))
    await writer.drain()


async def _read_sequence(reader):
    metadata = await _read_metadata(reader)
    if metadata is None:
        return None
    routing = metadata.pop('routing', None)
    sequence = SampledSequence(**metadata)
    if routing is not None:
        dtype, shape, size = routing
        dtype = np.dtype(dtype)
        if dtype.type not in (np.uint8, np.int16) or not 0 <= size <= _MAX_ROUTING:
            raise ValueError('invalid sampler router trace')
        buffer = bytearray(await reader.readexactly(size))
        sequence.routed_experts = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    return sequence


class _GroupSelection:
    """Rank 0 picks the first eligible siblings as their completions arrive."""

    def __init__(self, group_size, total, sharded):
        self.group_size = group_size
        self.total = total
        self.sharded = sharded
        self.groups = {}
        self.failed = False

    async def dispatch(self, operation, fields, origin):
        if operation == 'abort':
            self.failed = True
            for decision, _, _ in self.groups.values():
                if not decision.done():
                    decision.set_result({'failed': True})
            return None

        group, *outcome = fields
        key = group if self.sharded else (origin, group)
        if key not in self.groups:
            self.groups[key] = asyncio.get_running_loop().create_future(), set(), []
        decision, finished, winners = self.groups[key]
        if self.failed and not decision.done():
            decision.set_result({'failed': True})
        if operation == 'wait':
            # Disconnecting a subscriber must not cancel the shared decision.
            return await asyncio.shield(decision)
        if operation != 'complete':
            raise ValueError(f'unknown rollout selection operation: {operation}')
        tid, eligible = outcome
        if not decision.done() and tid not in finished:
            finished.add(tid)
            if eligible:
                winners.append(tid)
            if len(winners) == self.group_size:
                decision.set_result({'keep': winners})
            elif len(finished) == self.total:
                decision.set_result({'keep': None})
        return None


async def _serve(reader, writer, key, sample, selection, world_size):
    async def read_request():
        if not hmac.compare_digest(await reader.readexactly(_KEY_SIZE), key):
            raise ValueError('invalid sampler access key')
        origin, operation, fields = await _read_metadata(reader)
        if not isinstance(origin, int) or not 0 < origin < world_size:
            raise ValueError('invalid sampler origin rank')
        return origin, operation, fields

    tasks = []
    try:
        origin, operation, fields = await asyncio.wait_for(read_request(), timeout=30)
        if operation == 'sample':
            request = SampleRequest(**fields)
            if request.num_samples != 1:
                raise ValueError('async rollout requests require one sample')
            work = sample(request, origin)
        elif selection is not None:
            work = selection.dispatch(operation, fields, origin)
        else:
            raise ValueError('group selection is disabled for this rollout')
        response = asyncio.create_task(work)
        disconnected = asyncio.create_task(reader.read(1))
        tasks = [response, disconnected]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if disconnected not in done:
            if operation == 'sample':
                await _write_sequence(writer, response.result())
            else:
                _write_metadata(writer, response.result())
                await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    except Exception:
        logger.exception('sampler connection failed')
    finally:
        writer.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _request(address, key, rank, operation, fields):
    reader, writer = await asyncio.wait_for(asyncio.open_connection(*address), timeout=30)
    try:
        writer.write(key)
        _write_metadata(writer, [rank, operation, fields])
        await writer.drain()
        if operation == 'sample':
            return await _read_sequence(reader)
        return await _read_metadata(reader)
    finally:
        writer.close()


@asynccontextmanager
async def rollout_sampler(client, *, selection_options=None):
    """Exchange the sampler address once, then keep serving until every rank drains."""
    rank, world_size, _ = dist_utils.dist_info()

    async def sample(request, origin=rank):
        return await client._one(client._with_affinity(request, origin))

    selection = (_GroupSelection(*selection_options)
                 if rank == 0 and selection_options is not None else None)

    async def select(operation, *fields):
        return await selection.dispatch(operation, fields, rank)

    if world_size == 1:
        yield sample, select
        return

    group = dist_utils.gloo_group()
    server = None
    handlers = {}
    endpoint = None
    startup_error = None
    if rank == 0:
        key = secrets.token_bytes(_KEY_SIZE)

        def connected(reader, writer):
            task = asyncio.create_task(_serve(reader, writer, key, sample, selection, world_size))
            handlers[task] = writer
            task.add_done_callback(handlers.pop)

        host = os.environ.get('MASTER_ADDR', '127.0.0.1')
        try:
            server = await asyncio.start_server(connected, host, 0)
            address = server.sockets[0].getsockname()[:2]
            endpoint = address, key
        except OSError as exc:
            startup_error = str(exc)

    try:
        endpoint, startup_error = (await asyncio.to_thread(
            dist_utils.all_gather_object, (endpoint, startup_error), group=group))[0]
        if startup_error is not None:
            raise RuntimeError(f'could not start async rollout sampler: {startup_error}')

        async def remote_sample(request):
            address, key = endpoint
            return await _request(address, key, rank, 'sample', vars(request))

        async def remote_select(operation, *fields):
            address, key = endpoint
            return await _request(address, key, rank, operation, fields)

        error = None
        try:
            yield (sample, select) if rank == 0 else (remote_sample, remote_select)
        except Exception as exc:
            error = exc
        # A locally finished rank must keep its event loop running here. Rank 0
        # may still be generating replies for ranks whose trajectories are longer.
        failures = await asyncio.to_thread(dist_utils.all_gather_object, error is not None, group=group)
        if error is not None:
            raise error
        if any(failures):
            raise RuntimeError('async rollout failed on another rank')
    finally:
        if server is not None:
            server.close()
            active = handlers.copy()
            for task, writer in active.items():
                writer.close()
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            await server.wait_closed()
