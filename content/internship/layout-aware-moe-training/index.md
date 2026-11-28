---
title: "Layout-aware MoE Training: When Does Grouping Expert Gradients Pay Off?"
layout: "combined"
weight: 4
summary: "Grouped gate and up gradients preserve BF16 rounding and cut complete Qwen3 block time by 7.95–37.20% against a matched baseline on Ascend 910B. The gain reaches 2.05–2.26% in eight-NPU actor updates and reverses on a long Mixtral workload."
cover: "/projects/layout-aware-moe-training/figures/grouped-backward.svg"
coverAlt: "Gate and up gradients and weight views enter one grouped matrix multiplication. Two separate BF16 products are added to form the input gradient."
code: "https://github.com/Gxj230958/Gxj230958.github.io/tree/main/artifacts/moe-branch-rounding"
report: "/papers/moe-branch-rounding/"
---
