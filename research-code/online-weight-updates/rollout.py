"""Version-synchronous batched rollouts and a pure HCCL broadcast reference."""

import asyncio
from time import perf_counter

PROMPTS = tuple(
    tuple(1000 + (53 * j + i * 97) % 30000 for j in range(512)) for i in range(16)
)


async def batch(engine, tag):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=0,
        max_tokens=128,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )

    async def one(i, prompt):
        tokens = []
        first = None
        async for response in engine.generate(
            {"prompt_token_ids": list(prompt)}, params, f"{tag}-{i}"
        ):
            new = response.outputs[0].token_ids
            if new and first is None:
                first = perf_counter()
            tokens.extend(new)
        if len(tokens) != 128:
            raise ValueError(f"Incomplete rollout for {tag}-{i}")
        return first, tokens

    answers = await asyncio.gather(*(one(i, p) for i, p in enumerate(PROMPTS)))
    return min(t for t, _ in answers), [tokens for _, tokens in answers]


async def run_suite(engine, pool, trainer, sources, cycles, record, path, reference):
    import json

    from experiment import generate, save

    loop = asyncio.get_running_loop()
    rows = record["rollouts"] = []
    versions = record["rollout_outputs"] = {}
    await batch(engine, "batch-warmup")
    for cycle in range(cycles):
        version = "instruct" if cycle % 2 == 0 else "base"
        for every in (1, 4, 0) if cycle % 2 == 0 else (4, 1, 0):
            trainer.source = sources[version]
            start = perf_counter()
            if every:
                await engine.pause_generation(mode="wait", clear_cache=True)
                drained = perf_counter()
                await loop.run_in_executor(pool, trainer.send_weights)
                loaded = perf_counter()
                await engine.resume_generation()
            else:
                drained = loaded = start
            resumed = perf_counter()
            observations = []
            for i in range(every or 2):
                first, actual = await batch(engine, f"r{cycle}-{every}-{i}")
                finished = perf_counter()
                observations.append((first, finished, actual))
            end = observations[-1][1]
            sentinel, _ = await generate(engine, f"sentinel-{cycle}-{every}")
            if sentinel != json.loads(
                (reference / version / "inference.json").read_text()
            ):
                raise ValueError(
                    f"Cold reference mismatch for rollout policy {version}"
                )
            reference_tokens = versions.setdefault(version, observations[0][2])
            divergent = sum(
                one != ref
                for _, _, actual in observations
                for one, ref in zip(actual, reference_tokens)
            )
            rows.append(
                {
                    "cycle": cycle,
                    "version": version,
                    "batches_per_update": every,
                    "tokens": sum(
                        len(seq) for _, _, actual in observations for seq in actual
                    ),
                    "cold_sentinel_match": True,
                    "divergent_sequences": divergent,
                    "cycle_ms": 1000 * (end - start),
                    "first_token_ms": 1000 * (observations[0][0] - start),
                    "drain_ms": 1000 * (drained - start),
                    "transfer_ms": 1000 * (loaded - drained),
                    "resume_ms": 1000 * (resumed - loaded),
                    "pause_ms": 1000 * (resumed - start) if every else 0,
                }
            )
            save(path, record)


def wire_receive(worker, sizes, rounds):
    import torch

    group = worker.weight_transfer_engine.model_update_group
    stream = torch.npu.current_stream()
    slot = torch.empty(max(sizes), dtype=torch.uint8, device=group.device)
    for _ in range(rounds):
        for size in sizes:
            group.broadcast(slot[:size], src=0, stream=stream)
        stream.synchronize()


def wire_send(trainer, source, cap, rounds):
    import torch

    group = trainer.model_update_group
    with torch.npu.device(group.device):
        packed = torch.cat(
            [v.detach().view(-1).view(torch.uint8) for _, v in source.iter_full()]
        )
        chunks = packed.split(cap)
        stream = torch.npu.current_stream()
        stream.synchronize()
        times = []
        for _ in range(rounds):
            start = perf_counter()
            for x in chunks:
                group.broadcast(x, src=0, stream=stream)
            stream.synchronize()
            times.append(1000 * (perf_counter() - start))
    return times


async def wire_reference(engine, pool, trainer, source, cap):
    import math

    n = sum(math.prod(m.shape) * m.dtype.itemsize for m in source.metadata)
    sizes = [min(cap, n - i) for i in range(0, n, cap)]
    recv = asyncio.create_task(engine.collective_rpc(wire_receive, args=(sizes, 5)))
    sent, _ = await asyncio.gather(
        asyncio.get_running_loop().run_in_executor(
            pool, wire_send, trainer, source, cap, 5
        ),
        recv,
    )
    return {
        "checkpoint_bytes": n,
        "bucket_bytes": cap,
        "messages": len(sizes),
        "warmup_ms": sent[0],
        "ms": sent[1:],
    }


def shard_wire_worker(worker, message_sizes, remaining, cap, rounds):
    import torch

    group = worker.weight_transfer_engine.model_update_group
    slot = torch.empty(cap, dtype=torch.uint8, device=group.device)
    lane, stream = group.peer_lanes[group.rank]
    common = torch.npu.current_stream()
    for _ in range(rounds):
        for n in message_sizes:
            lane.broadcast(slot[:n], src=0, stream=stream)
        stream.synchronize()
        for n in remaining:
            group.broadcast(slot[:n], src=0, stream=common)
        common.synchronize()


def shard_wire_sender(trainer, source, cap, prepared, rounds):
    import torch

    group = trainer.model_update_group
    with torch.npu.device(group.device):
        weights = dict(source.iter_full())
        peers = {i: [] for i in group.peer_lanes}
        for name, spans in trainer.transport_routes.items():
            value = weights[name].detach().view(-1).view(torch.uint8)
            for rank, span in enumerate(spans, 1):
                offset, width = span[:2]
                if len(span) == 2:
                    chunks = value[offset : offset + width].split(cap)
                else:
                    stride, rows = span[2:]
                    slab = (
                        value.view(rows, stride)[:, offset : offset + width]
                        .contiguous()
                        .view(-1)
                    )
                    chunks = slab.split((cap // width) * width)
                peers[rank].extend(chunks)
        common = torch.npu.current_stream()
        rest = torch.cat(
            [
                v.detach().view(-1).view(torch.uint8)
                for name, v in weights.items()
                if name not in trainer.transport_routes
            ]
        ).split(cap)
        torch.npu.synchronize()
        sizes = [[x.numel() for x in chunks] for chunks in peers.values()]
        if sizes[0] != sizes[1]:
            raise ValueError("Wire reference requires equal rank partitions")
        prepared.set_result((sizes[0], [x.numel() for x in rest]))
        times = []
        for _ in range(rounds):
            t = perf_counter()
            for i in range(len(sizes[0])):
                for rank, (lane, stream) in group.peer_lanes.items():
                    lane.broadcast(peers[rank][i], src=0, stream=stream)
            for _, stream in group.peer_lanes.values():
                stream.synchronize()
            for x in rest:
                group.broadcast(x, src=0, stream=common)
            common.synchronize()
            times.append((perf_counter() - t) * 1000)
        return times


async def shard_wire_reference(engine, pool, trainer, source, cap):
    from concurrent.futures import Future

    loop = asyncio.get_running_loop()
    ready = Future()

    def sender():
        try:
            return shard_wire_sender(trainer, source, cap, ready, 5)
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            raise

    send = loop.run_in_executor(pool, sender)
    sizes, rest = await asyncio.wrap_future(ready)
    times, _ = await asyncio.gather(
        send, engine.collective_rpc(shard_wire_worker, args=(sizes, rest, cap, 5))
    )
    return {
        "sharded_messages_per_rank": len(sizes),
        "common_messages": len(rest),
        "rank_bytes": sum(sizes) + sum(rest),
        "warmup_ms": times[0],
        "ms": times[1:],
    }
