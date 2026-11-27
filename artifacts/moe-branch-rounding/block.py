"""Full expert forward/backward with a single changed input-gradient function."""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu
from block_inputs import routed_inputs
from paired_statistics import balanced_orders, statistics
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeSparseMoeBlock,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralSparseMoeBlock,
)
from verl.models.transformers.npu_patch import (
    _qwen3_sparse_moe_routed_forward_npu as routed,
)
from verl.utils.kernel.npu import qwen3_moe, qwen3_moe_triton
from variants import separate_dx, wide_dx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["block", "profile", "hf-default", "static"])
    p.add_argument("model", choices=["qwen", "boundary", "mixtral"])
    p.add_argument("--tokens", nargs="+", type=int, required=True)
    p.add_argument("--routes", nargs="+", default=["uniform", "hot"])
    p.add_argument(
        "--original-tile",
        action="store_true",
        help="Use the four-row Qwen packing tile on Mixtral",
    )
    p.add_argument("--shard", type=Path)
    p.add_argument("--states", type=Path)
    args = p.parse_args()
    run = Path(os.environ["ROUTING_RUN_DIR"])
    torch.npu.set_device(0)
    if not qwen3_moe._supported_runtime(0):
        raise RuntimeError("Unmatched Ascend runtime")
    torch.manual_seed(47)
    e, k, h, i = {
        "qwen": (128, 8, 2048, 768),
        "boundary": (8, 1, 2048, 768),
        "mixtral": (8, 2, 4096, 14336),
    }[args.model]
    with torch.device("meta"):
        if args.model == "mixtral":
            model = MixtralSparseMoeBlock(
                MixtralConfig(
                    hidden_size=h,
                    intermediate_size=i,
                    num_local_experts=e,
                    num_experts_per_tok=k,
                )
            )
        else:
            model = Qwen3MoeSparseMoeBlock(
                Qwen3MoeConfig(
                    hidden_size=h,
                    moe_intermediate_size=i,
                    num_experts=e,
                    num_experts_per_tok=k,
                )
            )
    if args.model == "mixtral":
        model.experts.config._experts_implementation = (
            "grouped_mm" if args.stage == "hf-default" else "eager"
        )
    model = model.to(dtype=torch.bfloat16).to_empty(device="npu")
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0, 0.02)
        learned_gate = None
        if args.shard is not None:
            from safetensors import safe_open

            with safe_open(args.shard, framework="pt") as tensors:
                prefix = "model.layers.0.block_sparse_moe."
                learned_gate = tensors.get_tensor(prefix + "gate.weight").bfloat16()
                for expert in range(e):
                    src = prefix + f"experts.{expert}."
                    model.experts.gate_up_proj[expert, :i].copy_(
                        tensors.get_tensor(src + "w1.weight")
                    )
                    model.experts.gate_up_proj[expert, i:].copy_(
                        tensors.get_tensor(src + "w3.weight")
                    )
                    model.experts.down_proj[expert].copy_(
                        tensors.get_tensor(src + "w2.weight")
                    )
        model.gate.weight.zero_()
        model.gate.weight[:, :e].copy_(torch.eye(e))
    original = qwen3_moe_triton.branch_group_dx
    candidate = (
        wide_dx if args.model == "mixtral" and not args.original_tile else original
    )
    implementations = {"base": separate_dx, "candidate": candidate}
    override = (
        nullcontext()
        if args.model == "qwen"
        else patch.object(qwen3_moe, "_use_packed_qwen3", lambda *a: True)
    )
    with override:
        for tokens in args.tokens:
            for distribution in args.routes:
                with torch.no_grad():
                    if distribution == "text":
                        model.gate.weight.copy_(learned_gate)
                        hidden = torch.load(
                            args.states, map_location="cpu", weights_only=True
                        )["states"][:, :tokens]
                        expected = model.gate(hidden.reshape(tokens, h).npu())[2].cpu()
                    else:
                        model.gate.weight.zero_()
                        model.gate.weight[:, :e].copy_(torch.eye(e))
                        hidden, expected = routed_inputs(
                            tokens, distribution, experts=e, topk=k
                        )
                if h > hidden.shape[-1]:
                    hidden = torch.cat(
                        (
                            hidden,
                            torch.randn(
                                1, tokens, h - hidden.shape[-1], dtype=hidden.dtype
                            )
                            * 0.02,
                        ),
                        -1,
                    )
                x = hidden.npu().requires_grad_()
                folder = run / f"{tokens}-{distribution}"
                folder.mkdir()

                def step(native=False):
                    model.zero_grad(set_to_none=True)
                    x.grad = None
                    if native and args.stage == "static":
                        _, y, logits = routed(model, x, use_packed=False)
                    elif native:
                        flat = x.reshape(tokens, h)
                        logits, probabilities, ids = model.gate(flat)
                        y = model.experts(flat, ids, probabilities)
                    else:
                        _, y, logits = routed(model, x, use_packed=True)
                    (
                        y.float().square().mean()
                        + 0.01 * logits.float().square().mean()
                    ).backward()
                    return y.detach().reshape(tokens, h)

                def snapshot(native=False):
                    y = step(native)
                    return {
                        name: v.detach().cpu()
                        for name, v in [
                            ("output", y),
                            ("input", x.grad),
                            *((n, p.grad) for n, p in model.named_parameters()),
                        ]
                    }

                if args.stage in ("hf-default", "static"):
                    base = snapshot(True)
                    with patch.object(qwen3_moe_triton, "branch_group_dx", candidate):
                        other = snapshot()
                        errors = {
                            name: max(
                                float((a.float() - b.float()).abs().max())
                                for a, b in zip(
                                    v.reshape(-1).split(1 << 20),
                                    base[name].reshape(-1).split(1 << 20),
                                    strict=True,
                                )
                            )
                            for name, v in other.items()
                        }
                        del base, other

                        def hf_timing(arm):
                            torch.npu.synchronize()
                            start = time.perf_counter_ns()
                            step(arm == "base")
                            torch.npu.synchronize()
                            return (time.perf_counter_ns() - start) / 1e6

                        for _ in range(30):
                            hf_timing("base")
                            hf_timing("candidate")
                        rows = []
                        for idx, order in enumerate(balanced_orders(100, 47)):
                            row = dict(pair=idx, order=order)
                            for arm in order:
                                row[arm] = hf_timing(arm)
                            rows.append(row)
                        (folder / f"{args.stage}-vs-C.json").write_text(
                            json.dumps(
                                dict(
                                    samples=rows,
                                    statistics=statistics(rows),
                                    max_abs=errors,
                                )
                            )
                        )
                    continue
                observations = {}
                permute = torch_npu.npu_moe_token_permute

                def observe_permute(value, ids, *a, **kw):
                    result = permute(value, ids, *a, **kw)
                    observations.update(ids=ids.cpu(), mapping=result[1].cpu())
                    return result

                reference = None
                differences = {}
                for arm, fn in implementations.items():

                    def observed(grad, weight, counts):
                        if not grad.is_contiguous():
                            raise AssertionError("unexpected grad layout")
                        observations["counts"] = counts.cpu()
                        return fn(grad, weight, counts)

                    with (
                        patch.object(qwen3_moe_triton, "branch_group_dx", observed),
                        patch.object(
                            torch_npu, "npu_moe_token_permute", observe_permute
                        ),
                    ):
                        values = snapshot()
                    if not torch.equal(
                        observations["ids"], expected
                    ) or not torch.equal(
                        observations["counts"], expected.flatten().bincount(minlength=e)
                    ):
                        raise AssertionError("route mismatch or backward path missed")
                    if reference is None:
                        reference, mapping = values, observations["mapping"]
                    else:
                        if not torch.equal(mapping, observations["mapping"]):
                            raise AssertionError("B/C permutation differs")
                        for name, value in values.items():
                            if torch.equal(value, reference[name]):
                                continue
                            unequal, max_abs = 0, 0.0
                            for a, b in zip(
                                value.reshape(-1).split(1 << 20),
                                reference[name].reshape(-1).split(1 << 20),
                                strict=True,
                            ):
                                unequal += int((a != b).sum())
                                max_abs = max(
                                    max_abs, float((a.float() - b.float()).abs().max())
                                )
                            differences[name] = dict(unequal=unequal, max_abs=max_abs)
                    observations.clear()
                contract = dict(
                    exact=[key for key in reference if key not in differences],
                    differences=differences,
                    counts=expected.flatten().bincount(minlength=e).tolist(),
                    route_order_and_mapping_equal=True,
                    confirmed_contiguous_dx_hits=["B", "C"],
                )
                if args.model == "mixtral":
                    native = snapshot(True)
                    contract["HF_eager_max_abs_difference"] = {
                        n: max(
                            float((a.float() - b.float()).abs().max())
                            for a, b in zip(
                                v.reshape(-1).split(1 << 20),
                                reference[n].reshape(-1).split(1 << 20),
                                strict=True,
                            )
                        )
                        for n, v in native.items()
                    }
                    contract["weights"] = (
                        str(args.shard) if args.shard else "seeded initialization"
                    )
                    del native
                del reference, values
                (folder / "contract.json").write_text(json.dumps(contract, indent=2))
                if differences:
                    print(
                        "Exact branch comparison failed",
                        tokens,
                        distribution,
                        differences,
                        flush=True,
                    )
                    continue
                if args.stage == "profile":
                    for arm, fn in implementations.items():
                        with patch.object(qwen3_moe_triton, "branch_group_dx", fn):
                            for _ in range(3):
                                step()
                            torch.npu.synchronize()
                            with torch_npu.profiler.profile(
                                activities=[
                                    torch_npu.profiler.ProfilerActivity.CPU,
                                    torch_npu.profiler.ProfilerActivity.NPU,
                                ],
                                record_shapes=True,
                                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                                    str(folder / arm)
                                ),
                            ):
                                step()
                                torch.npu.synchronize()
                    continue

                def measured(arm):
                    with patch.object(
                        qwen3_moe_triton, "branch_group_dx", implementations[arm]
                    ):
                        torch.npu.synchronize()
                        start = time.perf_counter_ns()
                        step()
                        torch.npu.synchronize()
                        return (time.perf_counter_ns() - start) / 1e6

                for _ in range(30):
                    for arm in implementations:
                        measured(arm)
                rows = []
                with (folder / "samples.jsonl").open("w") as out:
                    for idx, order in enumerate(balanced_orders(100, 47)):
                        row = dict(pair=idx, order=order)
                        for arm in order:
                            row[arm] = measured(arm)
                        rows.append(row)
                        print(json.dumps(row), file=out, flush=True)
                memory = {}
                for arm in implementations:
                    model.zero_grad(set_to_none=True)
                    x.grad = None
                    torch.npu.empty_cache()
                    torch.npu.reset_peak_memory_stats()
                    measured(arm)
                    memory[arm] = [
                        torch.npu.max_memory_allocated(),
                        torch.npu.max_memory_reserved(),
                    ]
                if args.model == "mixtral":

                    def eager_time():
                        torch.npu.synchronize()
                        start = time.perf_counter_ns()
                        step(True)
                        torch.npu.synchronize()
                        return (time.perf_counter_ns() - start) / 1e6

                    for _ in range(30):
                        eager_time()
                    native_rows = []
                    for idx, order in enumerate(balanced_orders(100, 47)):
                        row = dict(pair=idx, order=order)
                        for arm in order:
                            row[arm] = (
                                eager_time() if arm == "base" else measured("base")
                            )
                        native_rows.append(row)
                    (folder / "hf-eager-vs-B.json").write_text(
                        json.dumps(
                            {
                                "samples": native_rows,
                                "statistics": statistics(native_rows),
                            }
                        )
                    )
                result = dict(
                    model=args.model,
                    shape=[e, k, h, i],
                    pack_rows=1 if candidate is wide_dx else 4,
                    tokens=tokens,
                    distribution=distribution,
                    stats=statistics(rows),
                    memory_bytes_allocated_reserved=memory,
                )
                (folder / "result.json").write_text(json.dumps(result, indent=2))
                print(tokens, distribution, result["stats"]["improvement"], flush=True)


if __name__ == "__main__":
    main()
