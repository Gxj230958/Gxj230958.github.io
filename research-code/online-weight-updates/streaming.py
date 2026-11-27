"""Copy compatible checkpoint slabs into destinations from the live reload plan."""

import math
import time

VOCAB = {"model.embed_tokens.weight", "lm_head.weight"}


def install(worker, threshold=0, scope="vocab", wrong_row=False):
    import torch
    import torch_npu
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
    from vllm.distributed.weight_transfer.base import ParamMeta
    from vllm.model_executor.model_loader.reload.plan import plan_loads
    from vllm_ascend.distributed.weight_transfer import hccl_engine

    model = worker.get_model()
    native = hccl_engine.bounded_broadcast_consumer
    targets = None

    def receive(metadata, **kwargs):
        nonlocal targets
        items = tuple(metadata)
        if targets is None:
            started = time.perf_counter()
            named = [
                ParamMeta(name, dtype, tuple(shape)) for name, (shape, dtype) in items
            ]
            plan = plan_loads(model, named)
            targets = {}
            fallback = {}
            for meta in named:
                size = math.prod(meta.shape) * meta.dtype.itemsize
                fallback[meta.name] = size
                if (
                    (scope == "vocab" and meta.name not in VOCAB)
                    or not meta.shape
                    or size < threshold
                ):
                    continue
                writes = plan.get_dest(meta.name)
                if not writes or len(writes) != 1:
                    continue
                box, dest = writes[0]
                width = math.prod(meta.shape[1:])
                if (
                    len(box) != len(meta.shape)
                    or any(
                        span != (0, dim) for span, dim in zip(box[1:], meta.shape[1:])
                    )
                    or dest.dtype != meta.dtype
                    or not dest.is_contiguous()
                    or torch_npu.get_npu_format(dest) not in (0, 2)
                    or dest.numel() != (box[0][1] - box[0][0]) * width
                ):
                    continue
                start_row = box[0][0]
                if (
                    wrong_row
                    and get_tensor_model_parallel_rank() == 1
                    and meta.name == "model.embed_tokens.weight"
                ):
                    start_row -= 1
                targets[meta.name] = (
                    start_row * width * meta.dtype.itemsize,
                    dest.reshape(-1).view(torch.uint8),
                )
                del fallback[meta.name]
            worker.study_plan = {
                "direct_names": list(targets),
                "fallback_max_bytes": max(fallback.values(), default=0),
                "planning_seconds": time.perf_counter() - started,
            }
        return native(iter(items), **kwargs, direct_targets=targets)

    hccl_engine.bounded_broadcast_consumer = receive


def plan_details(worker):
    return worker.study_plan
