---
title: "When Does Grouping Expert Gradients Pay Off?"
weight: 9
summary: "One grouped matrix multiplication preserves separate BF16 branch rounding in a gated expert. Against a matched packed baseline, Qwen3 block time fell 7.95–37.20% and eight-NPU PPO update time fell 2.05–2.26%. On real Mixtral text, increasing the row count reversed the benefit."
cover: "/covers/moe-branch-rounding.svg"
coverAlt: "Mixtral expert-block time fell 28.6% at 128 text rows and 4.0% at 8,192, then rose 1.2% at 32,768."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/artifacts/moe-branch-rounding"
report: "/papers/moe-branch-rounding/"
---

A packed gated expert has two independently rounded input-gradient products. Mapping each expert–branch pair to its own native GMM group preserves those writebacks. The matched baseline uses the same packed forward and weight gradients.

- Fourteen block comparisons reproduce the reference outputs and gradients exactly. Independent device traces locate two distinct causes of a slowdown, concentrated staging work on narrow experts and longer native GMM execution on wide experts.
- On a 24-layer Qwen3 actor, three independently launched sessions give 2.26%, 2.07% and 2.05% lower median fixed-batch PPO update latency. Each session contains 40 paired updates on eight Ascend 910B2C devices.
- The earlier [layout-aware training integration](/projects/moe-training-ascend/) measured the full patch. This follow-up isolates the contribution of backward branch grouping and tests transfer on a pretrained Mixtral layer.

[Download English paper (PDF)](/papers/moe-branch-rounding/paper.pdf)
