"""Exact BF16 transfer from a resident trainer into planned rollout parameters."""

import ctypes
import math
from time import perf_counter

METHODS = (
    "send-direct",
    "receive-direct",
    "both-direct",
    "rank-shards",
    "rank-batch",
    "rank-lanes",
    "rank-columns",
)


def prepare_receiver(worker, manifest, method):
    import torch
    from streaming import mapped_destinations
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.distributed.weight_transfer.base import ParamMeta

    named = [
        ParamMeta(n, getattr(torch, dt), tuple(shape)) for n, dt, shape in manifest
    ]
    import torch_npu
    from vllm.model_executor.model_loader.reload.plan import plan_loads

    model = worker.get_model()
    plan = plan_loads(model, named)
    targets, _ = mapped_destinations(model, named, plan=plan)
    worker.transfer_targets = targets
    worker.column_targets = {}
    ranges = {name: (start, value.numel()) for name, (start, value) in targets.items()}
    if method == "rank-columns":
        for meta in named:
            writes = plan.get_dest(meta.name)
            if (
                meta.name in targets
                or len(meta.shape) != 2
                or not writes
                or len(writes) != 1
            ):
                continue
            box, dest = writes[0]
            rows, width = meta.shape
            if (
                len(box) != 2
                or box[0] != (0, rows)
                or not dest.is_contiguous()
                or dest.dtype != meta.dtype
            ):
                continue
            start, end = box[1]
            if (
                not 0 <= start < end <= width
                or end - start == width
                or dest.numel() != rows * (end - start)
                or torch_npu.get_npu_format(dest) not in (0, 2)
            ):
                continue
            worker.column_targets[meta.name] = dest.reshape(-1).view(torch.uint8)
            ranges[meta.name] = (start * 2, (end - start) * 2, width * 2, rows)
    return get_tensor_model_parallel_rank(), ranges


def routes_for(manifest, rank_ranges, method, minimum):
    routes = {}
    ranks = [ranges for _, ranges in sorted(rank_ranges)]
    for name, dtype, shape in manifest:
        if dtype != "bfloat16":
            raise ValueError(f"Expected BF16 checkpoint, got {dtype}")
        size = math.prod(shape) * 2
        if size < minimum or not all(name in r for r in ranks):
            continue
        spans = [tuple(r[name]) for r in ranks]
        if method == "rank-columns" and all(len(span) == 4 for span in spans):
            stride, rows = spans[0][2:]
            if (
                len(spans) == 2
                and stride * rows == size
                and all(s[2:] == (stride, rows) for s in spans)
                and sorted((s[0], s[1]) for s in spans)
                == [(0, stride // 2), (stride // 2, stride // 2)]
                and stride % 2 == 0
            ):
                routes[name] = spans
            continue
        if any(len(s) != 2 for s in spans):
            continue
        if method in ("rank-shards", "rank-batch", "rank-lanes", "rank-columns"):
            end = 0
            for start, length in sorted(spans):
                if start != end or length <= 0:
                    break
                end = start + length
            else:
                if (
                    len(spans) > 1
                    and end == size
                    and len({length for _, length in spans}) == 1
                ):
                    routes[name] = spans
        elif all(span == (0, size) for span in spans):
            routes[name] = spans
    return routes


def point_to_point(group, buf, peer, stream, sending):
    from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
        aclrtStream_t,
        buffer_type,
        hcclComm_t,
        hcclDataTypeEnum,
    )

    op = group.hccl.lib.HcclSend if sending else group.hccl.lib.HcclRecv
    op.argtypes = [
        buffer_type,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_uint32,
        hcclComm_t,
        aclrtStream_t,
    ]
    op.restype = ctypes.c_int
    group.hccl.HCCL_CHECK(
        op(
            buf.data_ptr(),
            buf.numel(),
            hcclDataTypeEnum.from_torch(buf.dtype),
            peer,
            group.comm,
            stream.npu_stream,
        )
    )


class HcclPeerBuffer(ctypes.Structure):
    _fields_ = [
        ("operation", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("dtype", ctypes.c_int),
        ("peer", ctypes.c_uint32),
    ]


def batch_shards(group, routes, views, cap, sending, stream):
    from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
        aclrtStream_t,
        hcclComm_t,
        hcclDataTypeEnum,
    )

    fn = group.hccl.lib.HcclBatchSendRecv
    fn.argtypes = [
        ctypes.POINTER(HcclPeerBuffer),
        ctypes.c_uint32,
        hcclComm_t,
        aclrtStream_t,
    ]
    fn.restype = ctypes.c_int
    count = 0
    for name, spans in routes.items():
        value = views[name]
        dtype = hcclDataTypeEnum.from_torch(value.dtype)
        length = spans[0][1]
        for i in range(0, length, cap):
            n = min(cap, length - i)
            if sending:
                items = [
                    HcclPeerBuffer(0, value.data_ptr() + origin + i, n, dtype, rank)
                    for rank, (origin, _) in enumerate(spans, 1)
                ]
            else:
                items = [HcclPeerBuffer(1, value.data_ptr() + i, n, dtype, 0)]
            buffers = (HcclPeerBuffer * len(items))(*items)
            group.hccl.HCCL_CHECK(
                fn(buffers, len(items), group.comm, stream.npu_stream)
            )
            count += n * len(items)
    stream.synchronize()
    return count


def make_lane(group, port, rank):
    import torch
    from vllm_ascend.distributed.weight_transfer.hccl_common import (
        stateless_init_process_group,
    )

    with torch.npu.device(group.device):
        lane = stateless_init_process_group(
            "127.0.0.1", port, rank, 2, group.device.index
        )
        return lane, torch.npu.Stream(device=group.device)


def receiver_lane(worker, ports):
    import torch

    group = worker.weight_transfer_engine.model_update_group
    free = torch.npu.mem_get_info(group.device)[0]
    group.peer_lanes = {group.rank: make_lane(group, ports[group.rank - 1], 1)}
    return free - torch.npu.mem_get_info(group.device)[0]


def close_lanes(group):
    for lane, _ in getattr(group, "peer_lanes", {}).values():
        lane.close()
    group.peer_lanes = {}


def close_receiver_lanes(worker):
    close_lanes(worker.weight_transfer_engine.model_update_group)


def lane_shards(group, routes, views, cap, sending):
    import torch

    moved = 0
    slots = {}
    for name, spans in routes.items():
        for rank, (lane, stream) in group.peer_lanes.items():
            span = spans[rank - 1]
            if len(span) == 4:
                offset, width, stride, rows = span
                step = cap // width
                for r in range(0, rows, step):
                    height = min(step, rows - r)
                    n = width * height
                    if sending:
                        if rank not in slots:
                            slots[rank] = torch.empty(
                                cap, dtype=torch.uint8, device=group.device
                            )
                        wire = slots[rank][:n]
                        stream.synchronize()
                        with torch.npu.stream(stream):
                            wire.view(height, width).copy_(
                                views[name].view(rows, stride)[
                                    r : r + height, offset : offset + width
                                ]
                            )
                        stream.synchronize()
                    else:
                        wire = views[name][r * width : r * width + n]
                    lane.broadcast(wire, src=0, stream=stream)
                    moved += n
                continue
            offset, length = span
            start = offset if sending else 0
            for i in range(0, length, cap):
                n = min(cap, length - i)
                lane.broadcast(
                    views[name][start + i : start + i + n], src=0, stream=stream
                )
                moved += n
    for _, stream in group.peer_lanes.values():
        stream.synchronize()
    return moved


def large_weights(group, routes, views, mode, cap, sending):
    import torch

    stream = torch.npu.current_stream()
    stream.synchronize()
    if mode in ("rank-lanes", "rank-columns"):
        return lane_shards(group, routes, views, cap, sending)
    if mode == "rank-batch":
        return batch_shards(group, routes, views, cap, sending, stream)
    stage = (mode == "receive-direct" and sending) or (
        mode == "send-direct" and not sending
    )
    scratch = (
        torch.empty(cap, dtype=torch.uint8, device=group.device) if stage else None
    )
    count = 0
    for name, spans in routes.items():
        value = views[name]
        shard = mode == "rank-shards"
        for rank, (origin, length) in enumerate(spans if shard else spans[:1], 1):
            if shard and not sending and group.rank != rank:
                continue
            start = origin if sending else 0
            for i in range(0, length, cap):
                n = min(cap, length - i)
                slab = value[start + i : start + i + n]
                wire = scratch[:n] if stage else slab
                if stage and sending:
                    wire.copy_(slab)
                    stream.synchronize()
                if shard:
                    point_to_point(group, wire, rank if sending else 0, stream, sending)
                else:
                    group.broadcast(wire, src=0, stream=stream)
                if stage and not sending:
                    stream.synchronize()
                    slab.copy_(wire)
                    stream.synchronize()
                count += n
    stream.synchronize()
    return count


def install_receiver(worker, routes, mode):
    from vllm_ascend.distributed.weight_transfer import hccl_engine
    from vllm_ascend.distributed.weight_transfer.bounded_tensor import (
        bounded_broadcast_consumer,
    )

    targets = worker.transfer_targets

    def receive(metadata, **kwargs):
        remaining = [row for row in metadata if row[0] not in routes]
        ts = perf_counter()
        moved = large_weights(
            kwargs["group"],
            routes,
            {
                n: (targets[n][1] if n in targets else worker.column_targets[n])
                for n in routes
            },
            mode,
            kwargs["buffer_size_bytes"],
            False,
        )
        split = perf_counter()
        if remaining:
            bounded_broadcast_consumer(
                iter(remaining), direct_targets=targets, **kwargs
            )
        worker.transport_stats = {
            "direct_bytes": moved,
            "remainder_bytes": sum(
                math.prod(s) * d.itemsize for _, (s, d) in remaining
            ),
            "direct_ms": (split - ts) * 1000,
            "remainder_ms": (perf_counter() - split) * 1000,
        }

    hccl_engine.bounded_broadcast_consumer = receive


def receiver_stats(worker):
    return worker.transport_stats


def install_trainer(trainer, engine, loop, method, minimum):
    import asyncio

    import torch
    import torch_npu
    from vllm_ascend.distributed.weight_transfer import hccl_engine
    from vllm_ascend.distributed.weight_transfer.bounded_tensor import (
        bounded_broadcast_producer,
    )

    manifest = [
        (m.name, str(m.dtype).split(".")[-1], list(m.shape))
        for m in trainer.source.metadata
    ]
    original = trainer.client.start_weight_update

    def first_start():
        original()
        ranges = asyncio.run_coroutine_threadsafe(
            engine.collective_rpc(prepare_receiver, args=(manifest, method)), loop
        ).result(timeout=180)
        if method in ("rank-lanes", "rank-columns"):
            from vllm.utils.network_utils import get_open_ports_list

            group = trainer.model_update_group
            ports = list(get_open_ports_list(len(ranges)))
            ready = asyncio.run_coroutine_threadsafe(
                engine.collective_rpc(receiver_lane, args=(ports,)), loop
            )
            free = torch.npu.mem_get_info(group.device)[0]
            group.peer_lanes = {
                i: make_lane(group, port, 0) for i, port in enumerate(ports, 1)
            }
            trainer.comm_setup_bytes = {
                "receivers": ready.result(timeout=180),
                "sender": free - torch.npu.mem_get_info(group.device)[0],
            }
        routes = routes_for(manifest, ranges, method, minimum)
        if not routes:
            raise ValueError(f"No eligible direct weights for {method}")
        asyncio.run_coroutine_threadsafe(
            engine.collective_rpc(install_receiver, args=(routes, method)), loop
        ).result(timeout=180)
        trainer.transport_routes = routes
        trainer.client.start_weight_update = original

    def send(iterator, metadata, **kwargs):
        routes = trainer.transport_routes
        rows = list(metadata)
        pairs = dict(iterator)
        views = {}
        for name in routes:
            tensor = pairs[name].detach()
            if (
                tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
                or tensor.device != kwargs["group"].device
                or torch_npu.get_npu_format(tensor) not in (0, 2)
            ):
                raise ValueError(f"Unsupported trainer storage for {name}")
            views[name] = tensor.view(-1).view(torch.uint8)
        start = perf_counter()
        moved = large_weights(
            kwargs["group"], routes, views, method, kwargs["buffer_size_bytes"], True
        )
        split = perf_counter()
        rest = [row for row in rows if row[0] not in routes]
        if rest:
            bounded_broadcast_producer(
                ((n, pairs[n]) for n, _ in rest), iter(rest), **kwargs
            )
        trainer.transport_stats = {
            "direct_bytes": moved,
            "direct_ms": (split - start) * 1000,
            "remainder_ms": (perf_counter() - split) * 1000,
        }

    trainer.client.start_weight_update = first_start
    hccl_engine.bounded_broadcast_producer = send
