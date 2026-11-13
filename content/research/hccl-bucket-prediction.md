---
title: "Execution-Graph Prediction and DDP Bucket Selection for HCCL Training"
summary: "Predicts distributed step time from measured gradient-readiness graphs and HCCL costs, then selects DDP bucket sizes. Held-out 2/4/8-NPU prediction errors are 7.77%/5.74%/4.23%; selected configurations finish within 3.29% of the measured best."
weight: 7
tagline: "Critical-path prediction from measured gradient-readiness dependency graphs and automatic DDP bucket sizing, validated on 2/4/8 Ascend 910B NPUs over HCCL."
cover: "/covers/minlan_yu.svg"
code: "https://github.com/Gxj230958/hccl-bucket-prediction"
report: "https://github.com/Gxj230958/hccl-bucket-prediction/blob/main/REPORT.md"
pr: ""
---
- Dependency graphs from NPU-event gradient-readiness times predict held-out step time with 7.77%/5.74%/4.23% MAPE on 2/4/8 NPUs, vs 33.59%/19.97%/22.65% for additive models.
- Evaluated 16 held-out DDP bucket configurations across 48 independent runs of Qwen3-0.6B; predicted and actual bucket layouts matched in all 1,296 rank-steps.
- HCCL cost tables from 4 KiB–512 MiB microbenchmarks with a Cube-contended variant; selected bucket within 3.29% of measured best, no gain over the 25 MiB default.
