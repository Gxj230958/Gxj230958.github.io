---
title: "Preserving KV Cache Capacity During Online Weight Updates on Ascend"
summary: "A rollout worker can serve from its KV cache and still run out of memory when a new policy arrives. Streaming Qwen2.5-7B vocabulary weights directly into the live model cut update scratch from 1,064 to 257.5 MiB and preserved 15,360 additional KV tokens (+1.84%). At a fixed cache size, a two-slot receiver shortened median update pauses by 6.3%."
weight: -2
cover: "/covers/online-weight-updates.svg"
coverAlt: "Largest verified KV allocation during Qwen2.5-7B weight updates on Ascend 910B2C: 834,944 tokens with per-parameter reload and 850,304 with in-place vocabulary streaming."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/research-code/online-weight-updates"
codeLabel: "Study code on GitHub"
pr: "https://github.com/vllm-project/vllm-ascend/pull/21288"
report: "/projects/online-weight-updates/report.pdf"
artifact: "/projects/online-weight-updates/study.zip"
---

### Where the extra capacity comes from

Each Qwen2.5-7B vocabulary matrix occupies 1,039.5 MiB. Even with 128 MiB broadcast buffers, reconstructing an entire matrix on the receiver competes with the KV cache. The receiver uses vLLM's load plan to copy each fragment's rank-local bytes into existing parameters. With one slot, the largest remaining input is 129.5 MiB, giving 257.5 MiB of update scratch. All ranks keep the same HCCL broadcast order.

### Capacity and pause time

The per-parameter receiver completed an update at 834,944 KV tokens. Single-slot vocabulary streaming reached 850,304. Both encountered an update OOM one block above their respective endpoints on the original card pairs. At a common quota of 786,432 tokens, dual-slot vocabulary streaming reduced median pause by 6.3% and increased replay throughput by 5.2%. At the single-slot capacity limit, allocator retries accompanied a rise in median pause from 1.02 to 5.13 seconds.

### Implementation and evidence

The upstream vLLM-Ascend PR implements bounded HCCL transport. The study code adds the in-place receiver and budget selector. The experiment harness compares updated weights with independent cold loads. The published tables summarize capacity, holdout predictions and 1,800 timed update cycles. The two held-out layouts are Qwen2.5-7B with tensor parallelism across two receiver ranks and Qwen2.5-1.5B with tied vocabulary weights.
