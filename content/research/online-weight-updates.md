---
title: "From Bounded Broadcasts to Rank-Local Policy Updates on Ascend"
summary: "I built bounded HCCL broadcasts for vLLM-Ascend, then used the model’s load plan to send each rollout rank only its weight shards. On the same three Ascend 910B2C NPUs, Qwen2.5-7B update pauses fell 22.8% and concurrent rollout throughput rose 4.6% with an update every batch."
weight: -2
cover: "/covers/online-weight-updates.svg"
coverAlt: "TP2 update pause on identical hardware: 668 ms with all-eligible broadcast and 516 ms with parallel row and column shards."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/research-code/online-weight-updates"
codeLabel: "Study code on GitHub"
pr: "https://github.com/vllm-project/vllm-ascend/pull/21288"
report: "/projects/online-weight-updates/report.pdf"
artifact: "/projects/online-weight-updates/study.zip"
layout: "online-weight-updates"
---

**A rollout worker must install every new policy before it can generate the next batch.** With one trainer and two inference ranks, the original transport sends the entire 14.19 GiB checkpoint to each receiver. My rank-aware transport sends 7.19 GiB to each. Over five same-device sessions, the median update pause fell from **668 to 516 ms**. Updating every batch of sixteen requests increased completed output throughput by **4.6%**.

## Why bounding a message still left a gigabyte of scratch

My [upstream vLLM-Ascend PR](https://github.com/vllm-project/vllm-ascend/pull/21288) lets a large weight span multiple bounded HCCL broadcasts. A 128 MiB message fits the buffer, yet the loader still wants a complete input. On Qwen2.5-7B the receiver reassembled a 1,039.5 MiB vocabulary matrix before installing it. I used vLLM’s load plan to copy each fragment directly into live model storage.

{{< figure src="/projects/online-weight-updates/figures/receiver.svg" link="/projects/online-weight-updates/figures/receiver.pdf" alt="The original PR reassembles a vocabulary matrix. The streaming receiver copies the intersecting fragment directly into existing rank-local weights." >}}

**Figure 1.** Both rows use the original broadcast order. In a matched two-slot experiment, eliminating vocabulary reassembly reduced the whole-cycle receiver peak from **1,295.5 to 385.5 MiB** at a fixed cache allocation. [Memory and capacity measurements](/projects/online-weight-updates/figures/measured-results.pdf)

The original capacity sweep observed 6,643 successful 128-token blocks with single-slot vocabulary streaming versus 6,523 for per-parameter reload on their respective device pairs. A later repeat of the vocabulary policy encountered update OOM at 6,643 on its original pair and passed at 6,550. In the same campaign, the all-eligible single-slot and direct-broadcast single-slot receivers both completed two updates at 6,643.

## Which rank actually needs each byte?

The copy trace suggested a limit to further tuning on one receiver. Direct broadcast cut receiver copy activity from about 31 to 4 ms, but most of the old copies already overlapped communication. TP1 update pause stayed near 870 ms. With two inference ranks, the load plan exposed another opportunity. Each rank owns different rows or columns of the same weight.

I gave each trainer–receiver pair its own bounded HCCL stream and communicator. A row shard comes straight from the trainer’s weight. For a column shard, the trainer gathers that rank’s columns into a reusable slot and transfers the bytes into the rank’s final parameter. Remaining inputs use the original broadcast. Both ranks finish before rollout resumes. Sending the shards serially took about 782 ms in the warmed correctness probe. Parallel row transfers reached **568 ms** in formal sessions. Handling columns as well brought the pause to **516 ms**.

{{< figure class="kv-singleplot" src="/projects/online-weight-updates/figures/updatephases_tp2.svg" link="/projects/online-weight-updates/figures/updatephases.pdf" alt="TP2 pause falls from 668 ms to 516 ms. The dashed line marks the full-payload broadcast reference." >}}

**Figure 2.** Median pause on the same three 910B2C devices at 14,342 KV blocks per rank. The baseline already streams all eligible weights into their final destinations. The dashed line is a prepacked broadcast of the full checkpoint. Five independently initialized sessions per configuration. [Full TP1 and TP2 comparison](/projects/online-weight-updates/figures/updatephases.pdf)

## What shorter pauses buy in a rollout

{{< figure class="kv-singleplot" src="/projects/online-weight-updates/figures/rolloutfrequency_tp2.svg" link="/projects/online-weight-updates/figures/rolloutfrequency.pdf" alt="Completed output tokens per second across five sessions at one update per batch, one per four batches, and a fixed policy." >}}

**Figure 3.** Each batch contains sixteen 512-token prompts with 128 greedy outputs each. Rates include the pause, update, resume and generation. Points are session medians and whiskers span the five sessions. [Full throughput figure](/projects/online-weight-updates/figures/rolloutfrequency.pdf)

With one update per batch, the TP2 rate went from **543.8 to 568.7 output tokens/s**. At one update per four batches, the paired gain was **2.0%**. Updating occupied 17.7% of the original every-batch cycle. The extra HCCL communicators used about 408 MiB of free HBM per inference rank at initialization.

For Base → Instruct and the reverse switch, every named parameter matched an independent cold load on both ranks. Fixed prompts reproduced the reference tokens and selected-token log probabilities. The report includes memory accounting, held-out layouts, first-token latency and the single-communicator controls.
