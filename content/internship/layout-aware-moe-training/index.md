---
title: "Layout-aware MoE Training: When Does Grouping Expert Gradients Pay Off?"
layout: "combined"
weight: 4
summary: "Grouped gate and up gradients preserve BF16 rounding and cut complete Qwen3 block time by 7.95–37.20% against a matched baseline on Ascend 910B. The gain reaches 2.05–2.26% in eight-NPU actor updates and reverses on a long Mixtral workload."
cover: "/projects/layout-aware-moe-training/figures/grouped-backward.svg"
coverAlt: "Gate and up gradients and weight views enter one grouped matrix multiplication. Two separate BF16 products are added to form the input gradient."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/artifacts/moe-branch-rounding"
codeLabel: "GitHub"
report: "/papers/moe-branch-rounding/"
---

A gated expert sends each token through two projections, gate and up. In the backward pass, preparing their inputs can take longer than either matrix multiplication. On Ascend 910B, computing both products with one native call reduced **complete Qwen3 expert-block time by 7.95–37.20%** with routing, the forward pass and weight-gradient computation held fixed. The same change saved **2.05–2.26% per fixed-batch training update** across three eight-NPU sessions.

## One native call preserves both BF16 rounding points

The original schedule computes two products, writes each in BF16, then adds them to get the input gradient (dX). A single wider dot product could round at a different point. I instead give each expert two internal compute groups. A packing kernel places its gate rows immediately before its up rows, and a weight view exposes the same order without a copy. The native call writes both BF16 products. A small combine kernel widens them for addition and writes dX in BF16.

{{< figure src="/papers/moe-branch-rounding/figures/branch-schedule.svg" alt="Separate and grouped schedules both produce two BF16 branch products before addition. Two experts with three rows and one row become four internal groups of sizes three, three, one and one." caption="**Figure 1.** Two experts receiving three rows and one row become four internal matrix-multiplication groups of sizes 3, 3, 1 and 1. The router still sees two experts." >}}

```python
branch_weight = weight.transpose(1, 2).view(2 * e, i, h)
branches = torch_npu.npu_grouped_matmul(
    [packed], [branch_weight], bias=None, group_list=groups,
    split_item=2, group_type=0, group_list_type=1,
)[0]
```

## What the matched baseline isolates

The full layout patch also packs the forward projection and writes weight gradients in the parameter’s physical layout. Against the split reference in verl, that patch reduced block time by **67.89%** (geometric mean across 16 Qwen3 cases). To isolate dX, I put the same forward and weight-gradient paths in both arms and timed the complete forward and backward pass again.

**Table 1. Qwen3 block time with only the dX schedule changed.** Times are medians in milliseconds from 100 balanced timing pairs after 30 warmups per arm. “Hot” sends tokens to a fixed set of experts.

| Rows · routing | Separate | Grouped | Saved |
|:---|---:|---:|---:|
| 128 · uniform | 4.556 | 3.466 | 23.94% |
| 128 · hot | 3.098 | 1.946 | 37.20% |
| 8,192 · uniform | 15.457 | 14.229 | 7.95% |
| 8,192 · hot | 13.565 | 12.357 | 8.91% |

{{< figure src="/projects/moe-branch-rounding/figures/qwen-dx-profile.svg" alt="Device trace on 128 hot Qwen3 rows. The two-call input-gradient segment takes 1.436 ms and the grouped segment takes 0.224 ms. The magnified grouped lane separates pack, native matmul and combine." caption="**Figure 2.** On the 128-row hotspot, four baseline slicing tasks took 1.351 ms. Packing and combining took 0.162 ms together. Native matrix multiplication took 0.066 ms and 0.060 ms, respectively. Each profiled lane has its own time origin. Table 1 reports separately measured, unprofiled block times." >}}

## When grouping costs more than it saves

On a pretrained Mixtral layer with its learned router, grouping saved **28.58%** at 128 text rows and **3.96%** at 8,192, then ran **1.20% slower** at 32,768. These workloads contain 1, 64 and 256 independent 128-token windows. Two device measurements explain why routing and input size matter.

{{< moe-cost-panels >}}**Figure 3.** In the narrow eight-expert control (a), concentrating work in one expert makes packing and combining more expensive. On the wide Mixtral layer (b), grouping still saves staging time at 32,768 rows, but the native matrix products take an additional 10.62 ms. Bars sum device-task durations within each family.{{< /moe-cost-panels >}}

## How much reaches a training update?

The 48-layer Qwen3 experiment tested the entire layout patch on 16 NPUs. The 24-layer experiment changed only dX on eight NPUs. Each fixed-batch PPO session restored a common training state and measured the slowest rank across 40 balanced pairs. In the matched experiment, every rank produced identical checked gradients, parameter updates and Adam moments.

**Table 2. Median actor update time.** Update times are in seconds, from baseline to grouped. Brackets give paired-bootstrap 95% intervals for the percentage saved.

| Training run | Median (s) | Saved [95% CI] |
|:---|---:|---:|
| Full patch · 48 layers | 32.730 → 31.259 | 4.49% [4.08–4.80] |
| dX only · session 1 | 3.110 → 3.040 | 2.26% [2.06–2.45] |
| dX only · session 2 | 3.106 → 3.042 | 2.07% [1.82–2.34] |
| dX only · session 3 | 3.106 → 3.042 | 2.05% [1.89–2.29] |

Session 2 also shows the distinction between median and tail latency. Its p95 rose from 3.151 to 3.374 s. The timing experiments used synthetic advantages and cover the actor update. Rollout throughput lies outside the measurement.

## When rank arrival changes collective completion

In a separate 16-NPU experiment, only rank 0 fused the cast and divide that prepare a gradient bucket for reduction. A controlled 50 ms release offset made rank 0 arrive first or last. Both regimes saved about 3.8 ms of local work. The collective finished earlier only when rank 0 arrived last.

**Table 3. Median paired time saved, in milliseconds.** Each arrival regime used 100 pairs. Brackets give 95% bootstrap intervals for completion savings.

| Rank 0 arrives | Preparation saved | Completion saved [95% CI] |
|:---|---:|---:|
| First | 3.809 | −0.004 [−0.027, 0.030] |
| Last | 3.801 | 3.783 [3.745, 3.828] |

**Training quality.** Longer runs enabled both the expert change and a communication hook. After 200 full-model updates across three seeds, the largest one-sided 95% upper bound on the final increase in negative log-likelihood across three evaluation splits was 0.001080 nats/token, below the prespecified 0.005 limit. In separate strict full-model checks under the native runtime, 91 normalization gradients and two parameter updates exceeded their tolerances.
