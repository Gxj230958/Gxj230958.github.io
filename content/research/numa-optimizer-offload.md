---
title: "Concurrent CPU Adam Updates for DeepSpeed Offload"
summary: "DeepSpeed’s native CPU Adam update held Python’s interpreter lock, stalling other workers. Releasing the lock after copying each call’s optimizer state sped up complete Qwen3-0.6B training steps by 1.151× in a four-worker offload pipeline on one Ascend 910B. Merged in DeepSpeed #9035."
weight: -1
cover: "/covers/minjia_zhang.svg"
coverAlt: "Two CPU workers update buckets in sequence with the GIL held and can overlap after release. The measured median speedup on Qwen3-0.6B is 1.151×."
code: "https://github.com/deepspeedai/DeepSpeed/blob/71ee91cf28a58432322eacaadf9c46613237baf8/csrc/adam/cpu_adam_impl.cpp"
report: "/reports/deepspeed-cpu-adam.pdf"
pr: "https://github.com/deepspeedai/DeepSpeed/pull/9035"
---

### What changes during an update

A training thread can now schedule gradient transfers and weight returns while CPU workers update independent parameter buckets. DeepSpeed still holds the global interpreter lock (GIL) while updating the shared step state. Each native call takes a snapshot with its own gradient scale and optional BF16 copy destination, then releases the GIL for the update. Two calls can share an optimizer ID while operating on separate tensors.

{{< figure class="cover" src="/projects/deepspeed-cpu-adam/figures/per-call-state.svg" link="/projects/deepspeed-cpu-adam/figures/per-call-state.svg" alt="Two calls share an optimizer ID. Before releasing the GIL, each snapshots its own gradient scale and optional BF16 copy destination. CPU workers update independent buckets, and the training thread can submit transfers." >}}

*Figure 1. A plain update uses scale 1. A scaled update can write a BF16 copy of its own bucket. Each call finishes setting up its private state before its native kernel begins.* [Vector PDF](/projects/deepspeed-cpu-adam/figures/per-call-state.pdf)

### End-to-end training results

{{< figure class="cover" src="/projects/deepspeed-cpu-adam/figures/paired-training.svg" link="/projects/deepspeed-cpu-adam/figures/paired-training.svg" alt="Three paired seeds yield complete-step speedups of 1.145×, 1.151× and 1.156×. The optimizer phase saves 90.6 to 99.1 ms per run median. Forward and backward phase savings range from −2.3 to +1.8 ms." >}}

*Figure 2. Each point in A is one timed training step. The vertical ticks mark run medians. In B, positive values indicate time saved after releasing the GIL. Each phase uses its own run median.* [Vector PDF](/projects/deepspeed-cpu-adam/figures/paired-training.pdf) · [Raw runs](https://github.com/Gxj230958/numa-optimizer-offload/tree/main/results/cpu_adam_comparison)

Across three paired seeds, median step times fell from 729 to 637 ms, 721 to 626 ms, and 723 to 625 ms. Both builds used the same four-worker pipeline, compiler flags and inputs. Each run used two warmup steps and six timed steps. Peak allocated NPU memory was 3.764 GiB in all six runs. Gradient norms matched for every paired step, and the largest absolute loss difference was 4.77 × 10⁻⁷. [Paired measurements](https://github.com/Gxj230958/numa-optimizer-offload/blob/main/results/cpu_adam_comparison/summary.json)

### NUMA-aware offload study

The wider study measured the effect of overlapping gradient staging, CPU AdamW and weight returns. Against independently tuned synchronous offload, the pipeline sped up complete training steps by 1.43×, 1.52× and 1.59× on Qwen3-0.6B, 1.7B and 4B, respectively. Against DeepSpeed ZeRO-Offload, the pipeline was faster on 0.6B, had overlapping timings on 1.7B and took 8.52% longer per step on 4B. [Experiment code](https://github.com/Gxj230958/numa-optimizer-offload) · [Full offload study](https://github.com/Gxj230958/numa-optimizer-offload/blob/main/reports/technical_report.md)
