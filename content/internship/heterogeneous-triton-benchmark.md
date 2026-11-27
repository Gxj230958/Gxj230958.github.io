---
title: "Heterogeneous Triton Benchmark"
weight: 1
summary: "I built a benchmark for porting 101 production Triton kernels from vLLM and SGLang to Ascend 910B. I organized the kernels into three tiers of migration complexity and designed numerical checks for hardware differences, including FP8 quantization on a processor without native FP8 arithmetic."
cover: "/projects/heterogeneous-triton-benchmark/cover.svg"
coverAlt: "A GPU Triton kernel maps to the 910B. Its Cube unit handles matrix arithmetic, while the highlighted Vector path converts quantized values to FP8 for storage."
code: "https://github.com/Gxj230958/HeterogeneousTritonBenchmark"
codeLabel: "GitHub"
---

*Figure 1. Porting a GPU kernel starts with the operations it needs. A matrix multiply can use the 910B Cube unit for supported input types. A fused activation that writes FP8 takes the Vector path and converts its output in software.*

## How I chose the 101 tasks

I used kernels from two inference engines, **61 from vLLM and 40 from SGLang**. For each kernel I kept the upstream GPU Triton source, paired it with an independent PyTorch reference, and specified public and held-out input shapes. I organized the suite around the dependencies a port must preserve. Three examples show how that changes the engineering problem. [Kernel selection](https://github.com/Gxj230958/HeterogeneousTritonBenchmark/blob/700c942f3fbb11e4ab39f9cf272578354b8a7769/docs/kernel-select.md)

| Tier | Tasks | What the port must preserve |
|---|---:|---|
| L1 · Single operation | 36 | **KV-cache indexing.** Gather each request’s active page IDs into one vector, preserving request order and the final partial page. |
| L2 · Fused operation | 35 | **Gated activation and quantization.** The activation, group reduction and output conversion share intermediate values. The kernel must return both quantized values and the scale that gives them meaning. |
| L3 · Multi-stage computation | 30 | **Chunked recurrent update.** A chunk consumes the previous state and writes the state for the next chunk. A rounding difference can propagate across chunk boundaries. |

## When a GPU kernel writes FP8

The 910B can store FP8 values. Its Cube and Vector units lack native FP8 arithmetic. That distinction determined what I included in the benchmark. I excluded matrix products that require native FP8 multiply–accumulate, and kept a vLLM mixture-of-experts kernel whose **output** must be FP8. Its activation and quantization arithmetic can run at higher precision. [Selection criteria](https://github.com/Gxj230958/HeterogeneousTritonBenchmark/blob/700c942f3fbb11e4ab39f9cf272578354b8a7769/docs/kernel-select.md) · [Reference implementation](https://github.com/Gxj230958/HeterogeneousTritonBenchmark/blob/700c942f3fbb11e4ab39f9cf272578354b8a7769/tasks/level2/L2_13_silu_mul_fp8_quant_deep_gemm/oracle_pytorch.py)

{{< figure class="cover" src="/projects/heterogeneous-triton-benchmark/fp8.svg" link="/projects/heterogeneous-triton-benchmark/fp8.svg" alt="SiLU and multiplication produce FP32 values. Each group of 128 shares one scale. Software conversion produces E4M3 values, and the kernel returns the FP32 scale separately. At 1.0 the next E4M3 value is 1.125." >}}

*Figure 2. Each valid token contributes groups of 128 activation values. The reference floors each group’s largest absolute value at 10⁻¹⁰, then divides by 448 to obtain its scale. The kernel writes one E4M3 value per activation and one FP32 scale per group.*

I checked the two outputs separately. For E4M3, I compared positions in the ordered set of representable values. Near a rounding boundary, **1.000 and 1.125 are one step apart**. The grader permits a one-step discrepancy for at most 1% of finite entries and rejects larger deviations. It checks the FP32 scales against a floating-point tolerance. In the recorded 910B evaluation, this port completed correctness checks and timing on **44 of 44 required shapes**. [Numerical grading](https://github.com/Gxj230958/HeterogeneousTritonBenchmark/blob/700c942f3fbb11e4ab39f9cf272578354b8a7769/docs/reference/correctness_model.md) · [Device records](https://github.com/Gxj230958/HeterogeneousTritonBenchmark/blob/700c942f3fbb11e4ab39f9cf272578354b8a7769/scripts/internal/evidence/streaming-evaluation-2026-10-10/task-results.md)

## What passed on the 910B

In the device evaluation, **73 of 101 tasks** completed numerical validation and timing on every required shape. Together those tasks covered **3,078 input shapes**. The tier breakdown shows where migration became harder for the evaluated candidates.

{{< figure class="cover" src="/projects/heterogeneous-triton-benchmark/device-results.svg" link="/projects/heterogeneous-triton-benchmark/device-results.svg" alt="Full-task device coverage reached 32 of 36 single-operation tasks, 24 of 35 fused tasks, and 17 of 30 multi-stage tasks." >}}

*Figure 3. Each bar gives the fraction of tasks in its tier with complete device records for both correctness and timing. The denominator includes every task in the tier. Results include the campaign’s recorded retries.* [Per-task results](https://github.com/Gxj230958/HeterogeneousTritonBenchmark/blob/700c942f3fbb11e4ab39f9cf272578354b8a7769/scripts/internal/evidence/streaming-evaluation-2026-10-10/task-results.md)

Of the 28 remaining tasks, 15 had numerical failures and six had compilation failures. Five were blocked by an artifact or contract issue, one had a device execution failure, and one candidate violated the task contract. Keeping those outcomes separate makes it possible to decide whether to debug the kernel’s arithmetic, its compilation, or the evaluation inputs. [Experiment report](https://github.com/Gxj230958/HeterogeneousTritonBenchmark/blob/700c942f3fbb11e4ab39f9cf272578354b8a7769/docs/experiments/streaming-evaluation-2026-10-10.md)
