---
title: "When Does Grouping Expert Gradients Pay Off?"
weight: 9
summary: "One grouped matrix multiplication preserves separate BF16 branch rounding in a gated expert. Against a matched packed baseline, Qwen3 block time fell 7.95–37.20% and eight-NPU PPO update time fell 2.05–2.26%. On real Mixtral text, increasing the row count reversed the benefit."
cover: "/covers/moe-branch-rounding.svg"
coverAlt: "Each expert contributes two internal GMM groups. Packed gradients and a weight view feed one native call, whose two BF16 branch products are added to form the input gradient."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/artifacts/moe-branch-rounding"
report: "/papers/moe-branch-rounding/"
---

*Figure 1. For each expert e, groups 2e and 2e+1 compute the gate and up contributions to its input gradient. Packing arranges gradient rows in this order. A weight view exposes the same order in existing storage. Each product is written in BF16 before the combine kernel adds the pair.* [Vector PDF](/projects/moe-branch-rounding/figures/expert-branch-cover.pdf)

## What the matched baseline isolates

Both arms use the same router, token permutation, packed forward projection and physical weight-gradient layout. Only the gate/up **input-gradient** schedule changes. Separate native products form baseline B. The grouped schedule is C.

{{< figure class="cover" src="/projects/moe-branch-rounding/figures/matched-qwen.svg" link="/projects/moe-branch-rounding/figures/matched-qwen.svg" alt="Four Qwen3 cases comparing separate and grouped dX. Time reductions are 23.94%, 37.20%, 7.95% and 8.91%." >}}

*Figure 2. Each line connects the complete forward–backward median times for one workload. Each row uses 100 balanced pairs after 30 warmups per arm. The last column gives the paired-bootstrap 95% interval for the reduction in percent.* [Vector PDF](/projects/moe-branch-rounding/figures/matched-qwen.pdf) · [Case data](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/analysis/block.csv)

## Inside the input-gradient kernel

For an expert with n assigned rows, the pack kernel writes n into both entries of the group-count vector. The packed gradient stores all n gate rows followed by all n up rows for that expert.


```python
if part == 0:
    tl.store(groups + 2 * expert, count)
    tl.store(groups + 2 * expert + 1, count)
```

For each expert, the prefix sum `start` locates its original rows. In the packed tensor the gate rows begin at `2 * start` and the up rows at `2 * start + count`.

```python
tl.store(
    packed + (2 * start + rows[:, None]) * INTERMEDIATE + columns[None, :],
    gate,
    mask,
)
tl.store(
    packed + (2 * start + count + rows[:, None]) * INTERMEDIATE + columns[None, :],
    up,
    mask,
)
```

The native call consumes the packed gradient and the weight view. Its BF16 output tensor stores two distinct products per expert.

```python
branch_weight = weight.transpose(1, 2).view(2 * e, i, h)
branches = torch_npu.npu_grouped_matmul(
    [packed],
    [branch_weight],
    bias=None,
    group_list=groups,
    split_item=2,
    group_type=0,
    group_list_type=1,
)[0]
```

The combine kernel aligns the two BF16 rows for the same token, widens them for addition and writes the result in BF16.

```python
result = (gate.to(tl.float32) + up.to(tl.float32)).to(tl.bfloat16)
tl.store(output + (start + rows[:, None]) * H + columns[None, :], result, mask)
```

[Full TritonAscend packing and combine kernels](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/qwen3_moe_triton.py) · [Two-call baseline](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/variants.py)

## Which device work becomes faster?

In the 128-row Qwen3 hotspot, four slicing tasks in the baseline consume **1.351 ms**. Two account for 0.712 and 0.629 ms. Grouping removes all four. The replacement pack and combine tasks take **0.162 ms** together. The two dX matrix products themselves take 66.1 µs in B and the single grouped product takes 59.6 µs in C.

{{< figure class="cover" src="/projects/moe-branch-rounding/figures/qwen-dx-profile.svg" link="/projects/moe-branch-rounding/figures/qwen-dx-profile.svg" alt="Actual Ascend device timelines for dX on 128 hot Qwen3 rows. Four slices dominate the separate arm. The grouped arm contains pack, GMM and combine." >}}

*Figure 3. Device tasks in the contiguous gate/up dX segment of one independently profiled block pass per arm. Each lane starts at its own first task and retains gaps between device events. Lane-end labels give elapsed milliseconds. The lower strip magnifies the grouped arm. Independent, unprofiled measurements of the full block give 3.098 ms for B and 1.946 ms for C.* [Vector PDF](/projects/moe-branch-rounding/figures/qwen-dx-profile.pdf) · [Device events with timestamps](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/analysis/qwen-hot-dx-tasks.csv)

On a 24-layer Qwen3 actor, three eight-NPU sessions found 2.26%, 2.07% and 2.05% lower median fixed-batch PPO update time. The [English paper](/papers/moe-branch-rounding/) follows the same change onto a pretrained Mixtral layer, including a length where separate products run faster. The [earlier integration](/projects/moe-training-ascend/) evaluates the complete layout patch.

[Download English paper (PDF)](/papers/moe-branch-rounding/paper.pdf)
