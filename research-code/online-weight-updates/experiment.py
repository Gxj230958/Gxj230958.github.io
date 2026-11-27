"""Compare online HCCL weight updates at a fixed vLLM KV cache capacity."""

import argparse
import asyncio
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

PROMPTS = ([9707, 11, 1268, 525, 498, 30], [785, 4096, 374, 279, 2038, 315])


def save(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def memory(worker, reset=False):
    import torch

    torch.npu.synchronize()
    if reset:
        torch.npu.reset_peak_memory_stats()
    return {
        "allocated": torch.npu.memory_allocated(),
        "peak": torch.npu.max_memory_allocated(),
        "reserved": torch.npu.memory_reserved(),
        "device_free": torch.npu.mem_get_info()[0],
        "alloc_retries": torch.npu.memory_stats().get("num_alloc_retries"),
    }


def parameter_state(worker, directory, action):
    import torch
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    rank = get_tensor_model_parallel_rank()
    parameters = dict(worker.get_model().named_parameters())
    addresses = {name: value.data_ptr() for name, value in parameters.items()}
    if action == "addresses":
        return addresses
    path = Path(directory) / f"rank-{rank}"
    if action == "save":
        path.mkdir(parents=True, exist_ok=True)
        save(path / "names.json", sorted(parameters))
    elif json.loads((path / "names.json").read_text()) != sorted(parameters):
        raise ValueError("Cold and updated parameter inventories differ")
    for name, tensor in parameters.items():
        value = tensor.detach().cpu().contiguous()
        file = path / (name + ".pt")
        if action == "save":
            torch.save(value, file)
        else:
            expected = torch.load(
                file, map_location="cpu", weights_only=True, mmap=True
            )
            if (
                expected.shape != value.shape
                or expected.dtype != value.dtype
                or not torch.equal(
                    value.reshape(-1).view(torch.uint8),
                    expected.reshape(-1).view(torch.uint8),
                )
            ):
                raise ValueError(f"Cold reference mismatch on rank {rank}: {name}")
    return addresses


class Client:
    def __init__(self, engine, loop):
        self.engine = engine
        self.loop = loop

    def wait(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(
            timeout=180
        )

    def init_weight_transfer_engine(self, info):
        from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

        self.wait(
            self.engine.init_weight_transfer_engine(
                WeightTransferInitRequest(init_info=info)
            )
        )

    def start_weight_update(self):
        self.wait(self.engine.start_weight_update())

    def update_weights(self, info):
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

        self.wait(
            self.engine.update_weights(WeightTransferUpdateRequest(update_info=info))
        )

    def finish_weight_update(self, weight_version=None):
        self.wait(self.engine.finish_weight_update(weight_version))


def trainer_init(args, engine, loop):
    import torch
    from transformers import AutoModelForCausalLM
    from vllm.distributed.weight_transfer.base import ModuleSource
    from vllm.utils.network_utils import get_open_port
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerInitInfo,
        HCCLTrainerWeightTransferEngine,
    )

    torch.npu.set_device(args.tp)
    sources = {
        key: ModuleSource(
            AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(
                f"npu:{args.tp}"
            )
        )
        for key, path in (("base", args.base), ("instruct", args.instruct))
    }
    torch.npu.synchronize()
    init = HCCLTrainerInitInfo(
        rank=0,
        master_address="127.0.0.1",
        master_port=get_open_port(),
        world_size=args.tp + 1,
        packed=args.mode != "unpacked",
        packed_transfer_mode="bounded"
        if args.mode in ("bounded", "stream", "stream-all")
        else "legacy",
        packed_buffer_size_bytes=args.bucket_mib * 2**20,
        packed_num_buffers=args.slots,
    )
    trainer = HCCLTrainerWeightTransferEngine.trainer_init(
        init, client=Client(engine, loop), source=sources["base"]
    )
    return trainer, sources


async def generate(engine, request_prefix):
    from vllm import SamplingParams

    async def one(i, prompt):
        first = None
        params = SamplingParams(
            temperature=0, max_tokens=16, ignore_eos=True, detokenize=False, logprobs=1
        )
        async for output in engine.generate(
            {"prompt_token_ids": list(prompt)}, params, f"{request_prefix}-{i}"
        ):
            item = output.outputs[0]
            if first is None and item.token_ids:
                first = perf_counter()
        return {
            "token_ids": list(item.token_ids),
            "logprobs": [
                dist[t].logprob for dist, t in zip(item.logprobs, item.token_ids)
            ],
        }, first

    results = await asyncio.gather(*(one(i, p) for i, p in enumerate(PROMPTS)))
    return [r[0] for r in results], min(r[1] for r in results)


async def study(args, record):
    from vllm import AsyncEngineArgs
    from vllm import __file__ as vllm_file
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm_ascend.distributed.weight_transfer import hccl_engine

    record.update(vllm_source=vllm_file, ascend_source=hccl_engine.__file__)
    loop = asyncio.get_running_loop()
    target = args.instruct if args.reference == "instruct" else args.base
    kv_bytes = args.kv_mib * 2**20
    if args.kv_blocks:
        from inventory import inventory

        _, per_token = inventory(args.base)
        kv_bytes = args.kv_blocks * 128 * per_token // args.tp
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=str(target),
            hf_config_path=str(args.base),
            tensor_parallel_size=args.tp,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=not args.graph,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4],
            }
            if args.graph
            else {},
            kv_cache_memory_bytes=kv_bytes,
            max_model_len=args.context,
            max_num_seqs=32,
            max_num_batched_tokens=2048,
            enable_prefix_caching=False,
            distributed_executor_backend="mp",
            additional_config={"weight_nz_mode": 0},
            weight_transfer_config={"backend": "hccl"},
            seed=0,
        )
    )
    reference = args.oracle / f"tp{args.tp}-{'graph' if args.graph else 'eager'}"
    pool = ThreadPoolExecutor(max_workers=1)
    trainer = None
    try:
        cfg = engine.vllm_config.cache_config
        blocks = cfg.num_gpu_blocks
        record.update(
            blocks=blocks,
            block_size=cfg.block_size,
            kv_tokens=cfg.kv_cache_size_tokens or blocks * cfg.block_size,
        )
        if args.reference:
            output, _ = await generate(engine, "reference")
            folder = reference / args.reference
            folder.mkdir(parents=True, exist_ok=True)
            if args.compare_reference:
                await engine.collective_rpc(
                    parameter_state, args=(str(folder), "compare")
                )
                if output != json.loads((folder / "inference.json").read_text()):
                    raise ValueError("Cold inference differs between reference builds")
            else:
                await engine.collective_rpc(parameter_state, args=(str(folder), "save"))
                save(folder / "inference.json", output)
            record["status"] = "reference ready"
            return
        await generate(engine, "warmup")
        record["phase"] = "trainer_init"
        save(args.output, record)
        if args.mode == "auto":
            from selector import select

            baseline = await engine.collective_rpc(memory)
            physical_budget = min(r["device_free"] for r in baseline)
            budget = (
                min(physical_budget, args.budget_mib * 2**20)
                if args.budget_mib
                else physical_budget
            )
            plan_start = perf_counter()
            choice, candidates = select(args.base, budget)
            plan_seconds = perf_counter() - plan_start
            args.mode, args.slots, args.bucket_mib = (
                choice["mode"],
                choice["slots"],
                choice["bucket_mib"],
            )
            args.threshold_mib = choice["threshold_bytes"] / 2**20
            record["selection"] = {
                "budget": budget,
                "before_hccl": baseline,
                "selector_cpu_seconds": plan_seconds,
                "choice": choice,
                "candidates": candidates,
            }
            save(args.output, record)
        trainer, sources = await loop.run_in_executor(
            pool, trainer_init, args, engine, loop
        )
        if args.mode in ("stream", "stream-all"):
            from streaming import install

            await engine.collective_rpc(
                install,
                args=(
                    args.threshold_mib * 2**20,
                    "all" if args.mode == "stream-all" else "vocab",
                    args.wrong_row,
                ),
            )
        if args.layers:
            from probes import start_materialization

            await engine.collective_rpc(start_materialization)
        initial = await engine.collective_rpc(parameter_state, args=("", "addresses"))
        for i in range(args.warmups + args.samples):
            version = "instruct" if i % 2 == 0 else "base"
            trainer.source = sources[version]
            before = await engine.collective_rpc(memory, args=(True,))
            record["phase"] = "update"
            save(args.output, record)
            if args.profile and i >= args.warmups:
                from probes import start_trace

                await engine.collective_rpc(
                    start_trace, args=(str(args.profile / f"round-{i}"),)
                )
            start = perf_counter()
            await engine.pause_generation(mode="wait", clear_cache=True)
            await loop.run_in_executor(pool, trainer.send_weights)
            await engine.resume_generation()
            resumed = perf_counter()
            if args.profile and i >= args.warmups:
                from probes import stop_trace

                await engine.collective_rpc(stop_trace)
            update_memory = (
                await engine.collective_rpc(memory) if args.attribute else None
            )
            record["phase"] = "generation"
            if args.attribute:
                save(args.output, record)
            output, first = await generate(engine, str(i))
            end = perf_counter()
            after = await engine.collective_rpc(memory)
            folder = reference / version
            if args.verify:
                record["phase"] = "verification"
                current = await engine.collective_rpc(
                    parameter_state, args=(str(folder), "compare")
                )
                if current != initial or output != json.loads(
                    (folder / "inference.json").read_text()
                ):
                    raise ValueError(f"{version} cold inference/address mismatch")
            record["rounds"].append(
                {
                    "version": version,
                    "warmup": i < args.warmups,
                    "pause_ms": (resumed - start) * 1000,
                    "first_token_ms": (first - start) * 1000,
                    "cycle_ms": (end - start) * 1000,
                    "tokens": sum(len(r["token_ids"]) for r in output),
                    "memory_before": before,
                    "memory_after": after,
                    "update_peak": [
                        a["peak"] - b["allocated"]
                        for a, b in zip(update_memory, before)
                    ]
                    if update_memory
                    else None,
                    "update_alloc_retries": [
                        a["alloc_retries"] - b["alloc_retries"]
                        for a, b in zip(update_memory, before)
                    ]
                    if update_memory
                    and all(a["alloc_retries"] is not None for a in update_memory)
                    else None,
                    "extra_peak": [
                        a["peak"] - b["allocated"] for a, b in zip(after, before)
                    ],
                }
            )
        if args.mode in ("stream", "stream-all"):
            from streaming import plan_details

            record["plan"] = await engine.collective_rpc(plan_details)
        if args.layers:
            from probes import materializations

            record["materializations"] = await engine.collective_rpc(materializations)
        record["status"] = "passed"
    finally:
        if trainer is not None:
            await loop.run_in_executor(pool, trainer.shutdown)
        pool.shutdown(wait=False)
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--instruct", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--reference", choices=["base", "instruct"])
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Compare an independent cold start with saved tensors and inference",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[
            "unpacked",
            "cached-packed",
            "bounded",
            "stream",
            "stream-all",
            "auto",
        ],
        default="unpacked",
    )
    parser.add_argument(
        "--reload", choices=["direct", "layerwise"], default="layerwise"
    )
    parser.add_argument("--slots", type=int, choices=[1, 2], default=2)
    parser.add_argument("--tp", type=int, choices=[1, 2], default=1)
    parser.add_argument("--bucket-mib", type=int, default=128)
    parser.add_argument("--kv-mib", type=int, default=2048)
    parser.add_argument(
        "--kv-blocks", type=int, help="Exact number of 128-token KV blocks per rank"
    )
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--layers", action="store_true")
    parser.add_argument(
        "--profile", type=Path, help="Separate receiver traces per measured update"
    )
    parser.add_argument(
        "--wrong-row",
        action="store_true",
        help="Displace the rank-1 embedding source by one row (negative control)",
    )
    parser.add_argument(
        "--budget-mib",
        type=int,
        help="Application cap on additional receiver memory for session selection",
    )
    parser.add_argument("--attribute", action="store_true")
    parser.add_argument("--threshold-mib", type=int, default=0)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode in ("auto", "stream", "stream-all") and args.reload != "direct":
        parser.error("Stream-to-destination requires direct reload")
    if args.mode == "auto" and not args.budget_mib:
        parser.error("auto requires the available update budget in MiB")
    if args.wrong_row and not (args.tp == 2 and args.verify):
        parser.error("The row displacement control needs TP2 cold verification")
    os.environ.update(
        VLLM_RELOAD_MODE=args.reload, VLLM_ALLOW_INSECURE_SERIALIZATION="1"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cards": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "phase": "startup",
        "status": "running",
        "rounds": [],
    }
    save(args.output, record)
    try:
        asyncio.run(study(args, record))
        gc.collect()
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        save(args.output, record)


if __name__ == "__main__":
    main()
