---
title: "Tree-Shared Attention Scheduling for Ascend Cube"
summary: "Uses subset dynamic programming to group shared-prefix requests, accounting for Cube padding, KV continuity and merge cost. Achieves 1.315× geometric-mean speedup over per-request tree execution on 81 held-out shapes, with 1,224 layer comparisons on pretrained Qwen3-4B."
weight: 8
tagline: "Subset-DP grouping with hardware-aware costs for tree-shared decode attention on one Ascend 910B NPU, integrated and validated end-to-end on pretrained Qwen3-4B."
cover: "/covers/yufei_ding.svg"
code: "https://github.com/Gxj230958/tree-attention-scheduling"
report: "https://github.com/Gxj230958/tree-attention-scheduling/blob/main/reports/technical_report.md"
pr: ""
---
- Subset dynamic program prices each tree grouping with full descendant cost plus Cube query-row padding, KV-layout continuity splits, and log-sum-exp merge terms; extends the FastTree (MLSys'25) cost-model line.
- 128 requests x 8192 tokens sharing 8128: 2.523 ms resident attention vs 2.839 ms for the faster vendor fused-operator layout (1.126x); 1.315x geometric-mean speedup across 81 held-out tree shapes.
- Pretrained Qwen3-4B on shared KV pools: 1,224 layer comparisons pass against FP32 and vendor attention (max rel L2 0.218%); cached plans cut per-step planning-plus-attention from 3.896 ms to 0.606 ms.
