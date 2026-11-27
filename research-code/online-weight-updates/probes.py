"""Receiver-side instrumentation used in separate attribution and profiling sessions."""

from pathlib import Path


def start_materialization(worker):
    import torch
    from vllm.model_executor.model_loader.reload import layerwise

    original = layerwise.materialize_layer
    observations = []

    def observe(layer, info):
        before = torch.npu.memory_allocated()
        result = original(layer, info)
        inputs = {}
        for _, bound in info.loaded_weights:
            for tensor in bound.arguments.values():
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu":
                    storage = tensor.untyped_storage()
                    inputs[storage.data_ptr()] = storage.nbytes()
        observations.append(
            {
                "layer": type(layer).__name__,
                "allocation": torch.npu.memory_allocated() - before,
                "buffered_inputs": sum(inputs.values()),
            }
        )
        return result

    layerwise.materialize_layer = observe
    worker.study_materializations = observations


def materializations(worker):
    return worker.study_materializations


def start_trace(worker, directory):
    from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler

    worker.study_profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        on_trace_ready=tensorboard_trace_handler(str(Path(directory))),
    )
    worker.study_profiler.start()


def stop_trace(worker):
    worker.study_profiler.stop()
    del worker.study_profiler
