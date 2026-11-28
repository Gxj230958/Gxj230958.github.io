---
title: "From Bounded Broadcasts to Rank-Local Policy Updates on Ascend"
summary: "I built bounded HCCL broadcasts for vLLM-Ascend, then used the model’s load plan to send each rollout rank only its weight shards. On the same three Ascend 910B2C NPUs, Qwen2.5-7B update pauses fell 22.8% and concurrent rollout throughput rose 4.6% with an update every batch."
weight: -2
cover: "/covers/online-weight-updates.svg?v=2"
coverAlt: "Two trainer matrices split by rows and columns. Parallel HCCL links deliver blue shards to rank 0 and amber shards to rank 1, cutting the median Qwen2.5-7B update pause from 668 to 516 ms."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/research-code/online-weight-updates"
codeLabel: "Study code on GitHub"
pr: "https://github.com/vllm-project/vllm-ascend/pull/21288"
report: "/projects/online-weight-updates/report.pdf"
artifact: "/projects/online-weight-updates/study.zip"
layout: "online-weight-updates"
---

**A rollout worker must install every new policy before it can generate the next batch.** With one trainer and two inference ranks, full broadcast delivers the entire 14.19 GiB checkpoint to each receiver. Routing the weights by rank delivers 7.19 GiB to each. Over five same-device sessions, the median update pause fell from **668 to 516 ms**. Updating every batch of sixteen requests increased completed output throughput by **4.6%**.

## Where the extra allocation comes from

My [vLLM-Ascend contribution](https://github.com/vllm-project/vllm-ascend/pull/21288) lets a large weight span several bounded HCCL broadcasts. A 128 MiB receive slot can carry a 1,039.5 MiB vocabulary matrix piece by piece. The loader can still demand a second copy of the whole matrix. The highlighted fragment below shows how vLLM’s load plan lets the receiver write directly to the right bytes in the live model.

{{< figure src="/projects/online-weight-updates/figures/receiver.svg" link="/projects/online-weight-updates/figures/receiver.pdf" alt="The hatched fragment travels from the trainer to one receive slot. Reassembly allocates a complete temporary vocabulary matrix. The direct path writes the same fragment into its final byte range in the live weight." caption="**Figure 1.** Both paths receive the same fragments in the same order. At a fixed KV allocation and two 128 MiB slots, writing fragments in place reduced the receiver’s peak extra allocation from **1,295.5 to 385.5 MiB**." >}}

In the initial capacity sweep the single-slot vocabulary policy completed an update at 6,643 KV blocks. On its original pair of devices a later run reached 6,550 blocks and ran out of memory at 6,643. In the same campaign, all-eligible single-slot streaming and single-slot direct broadcast each completed two updates at 6,643 blocks. [Memory and capacity measurements](/projects/online-weight-updates/figures/measured-results.pdf)

## Which rank actually needs each byte?

The load plan gives both the destination address and the rank that owns it. On one inference rank, direct broadcast cut receiver copy activity from about 31 to 4 ms, but the copies already overlapped communication and the update pause stayed near 870 ms. With two inference ranks, each owns different rows or columns of the same weight. Sending both ranks the full tensor moves bytes each will discard.

Each trainer–receiver pair gets its own HCCL stream and communicator. Row shards transfer directly from the trainer’s weights. For column shards, the trainer packs each rank’s columns into a reusable slot. Both routes land in the final parameter. Shared inputs still use full broadcast, and generation resumes after both ranks finish.

{{< figure src="/projects/online-weight-updates/figures/parallel-handoff.svg" link="/projects/online-weight-updates/figures/parallel-handoff.pdf" alt="Five sessions per strategy give median full-update pauses of 667.7 ms for broadcast, 567.6 ms for parallel rows and 515.8 ms for parallel rows and columns. Two separate sender traces show 412.0 and 411.1 ms with overlapping HCCL calls." caption="**Figure 2.** **A.** Five paired sessions on the same three 910B2C NPUs, with 14,342 KV blocks per inference rank. The complete pause fell by **22.6–23.0%** across pairs as payload per receiver fell from **14.19 to 7.19 GiB**. **B.** Two separately profiled updates show communication calls active concurrently on the trainer. Each clock starts at its first HCCL call. White gaps indicate idle communication intervals." >}}

## What shorter pauses buy in a rollout

**Table 1. Completed output tokens per second at three update frequencies.** Each batch contains sixteen 512-token prompts with 128 generated tokens per request. The two rate columns compare full broadcast with parallel row and column shards and include both update and generation time. The gain column gives the median change within five session pairs, followed by the observed range. The fixed-policy runs measure generation with weights already resident.

| Update cadence | Broadcast | Shards | Paired gain |
|:---|---:|---:|---:|
| Every batch | 543.8 | 568.7 | **+4.6%** (3.2–7.3%) |
| Every four batches | 625.6 | 636.8 | **+2.0%** (0.6–5.2%) |
| Fixed policy | 655.6 | 663.4 | Control |

Updates occupied **17.7%** of the baseline cycle at one update per batch. Holding generation time fixed, shortening that pause by **22.8%** predicts **4.2%** more output tokens per second, close to the measured **4.6%**. The additional communicators used approximately 408 MiB of device memory per inference rank.

For Base → Instruct and the reverse switch, every named parameter matched an independent cold load on both ranks. Fixed prompts reproduced the reference tokens and selected-token log probabilities. The report includes the memory accounting, first-token latency and single-communicator controls.
