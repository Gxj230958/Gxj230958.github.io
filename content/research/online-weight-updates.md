---
title: "Preserving KV Cache Capacity During Online Weight Updates on Ascend"
summary: "I built a bounded HCCL transport for online weight updates on Ascend. Each broadcast fit in a 128 MiB buffer, yet the receiver still reconstructed a 1,039.5 MiB vocabulary matrix. Streaming those fragments into the live Qwen2.5-7B model preserved 15,360 additional KV tokens over per-parameter direct reload."
weight: -2
cover: "/covers/online-weight-updates.svg"
coverAlt: "Single-slot vocabulary streaming raised the largest verified KV allocation from 834,944 to 850,304 tokens over per-parameter direct reload, a gain of 15,360 tokens."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/research-code/online-weight-updates"
codeLabel: "Study code on GitHub"
pr: "https://github.com/vllm-project/vllm-ascend/pull/21288"
report: "/projects/online-weight-updates/report.pdf"
artifact: "/projects/online-weight-updates/study.zip"
layout: "online-weight-updates"
---

**If each broadcast fits in 128 MiB, why does a weight update still need more than a gigabyte of scratch memory?** I ran into this question after building a [bounded HCCL transport for vLLM-Ascend](https://github.com/vllm-project/vllm-ascend/pull/21288). That PR let a large checkpoint tensor span several broadcasts. The native loader still expected an entire input tensor, so the receiver assembled the fragments into a temporary copy. During an online policy update, that copy competes for memory with the rollout worker’s already allocated KV cache.

## From a bounded message to an in-place write

{{< figure src="/projects/online-weight-updates/figures/receiver.svg" link="/projects/online-weight-updates/figures/receiver.pdf" alt="The PR reassembles fragments in a full temporary vocabulary matrix before loading the live weight. The study sends the same bounded fragments and copies the relevant byte range straight from the receive slot into the existing parameter." >}}

**Figure 1.** Follow the blue fragment of a Qwen2.5-7B vocabulary matrix. Both paths broadcast the same bytes in the same order. In the study receiver, vLLM’s load plan identifies the contiguous rows owned by each rank. Weights with other layouts can use the original bounded reassembly path.

The 7B checkpoint has two independent vocabulary matrices, each 1,039.5 MiB. Streaming just these two leaves an MLP projection of 129.5 MiB as the largest input still requiring reassembly. With two 128 MiB receive slots, the predicted transport footprint is **2 × 128 + 129.5 = 385.5 MiB**. This gives a prediction to check before raising the cache quota.

## How much memory and KV capacity did this recover?

{{< figure class="kv-panel" src="/projects/online-weight-updates/figures/receiver-peak.svg" link="/projects/online-weight-updates/figures/measured-results.pdf" alt="A. At 6,144 KV blocks on the same device pair, the whole-cycle receiver peak above baseline was 1,295.5 MiB for bounded reassembly and 385.5 MiB for vocabulary streaming. Both used two 128 MiB slots." >}}
{{< figure class="kv-panel" src="/projects/online-weight-updates/figures/capacity-boundary.svg" link="/projects/online-weight-updates/figures/measured-results.pdf" alt="B. Per-parameter direct reload passed at 6,523 blocks and encountered update OOM at 6,524. Single-slot vocabulary streaming passed at 6,643 and encountered update OOM at 6,644." >}}

**Figure 2.** (A) Receiver peak allocated bytes above the pre-update baseline, measured across the whole update-and-generation cycle at 786,432 KV tokens. Both policies used direct reload on the same Ascend 910B2C pair. Each bar is the median of three session maxima, with 50 measured cycles per session. (B) Largest successful allocation and the next tested block for each policy, repeated on its original device pair. [SVG](/projects/online-weight-updates/figures/measured-results.svg) · [PDF](/projects/online-weight-updates/figures/measured-results.pdf)

Panel A isolates the receiver change at a fixed quota. The **910 MiB** reduction matches the difference between the vocabulary matrix and the largest remaining input. In a separate single-slot probe at 6,550 blocks, bounded reassembly ran out of memory during the update. Vocabulary streaming completed the update and generation on another device pair.

Panel B asks how far the worker can raise the cache quota and still update and generate. One slot leaves more room for KV. Relative to per-parameter direct reload, the single-slot receiver gained **120 blocks × 128 tokens = 15,360 tokens (+1.84%)**.

## Correctness and the operating point

After switching from Base to Instruct and back, every named rank-local parameter matched an independent cold load byte for byte. Two fixed prompts also reproduced the cold model’s greedy tokens and selected-token log probabilities. Shifting the embedding by one row on TP rank 1 made the check identify that rank and parameter.

The largest feasible quota had a latency cost. On the same device pair, single-slot median pause rose from **1.02 s at 6,550 blocks to 5.13 s at 6,643**, alongside 36–47 allocator retries per update at the upper quota. With room for overlap at a common quota of 6,144 blocks, **two-slot** vocabulary streaming instead shortened median pause by **6.3%** relative to per-parameter direct reload. The [English report](/projects/online-weight-updates/report.pdf) covers the budget selector, two held-out layouts and the full comparison across twelve strategies.
