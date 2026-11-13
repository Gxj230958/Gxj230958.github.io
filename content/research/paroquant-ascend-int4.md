---
title: "ParoQuant on Ascend: Fused Rotation and Packed INT4 Execution"
summary: "Fuses channel scaling and eight rotation stages, then executes published ParoQuant checkpoints with native packed INT4 weights. Reduces peak memory by 38.9–42.1% versus cached FP16 execution of the same weights, with end-to-end quality validation on Qwen3-4B and 8B."
weight: 4
tagline: "Fused eight-stage rotation kernels and native packed-INT4 execution for ParoQuant Qwen3-4B/8B on Ascend 910B, cutting peak memory 38.9-42.1% with validated quality."
cover: "/covers/zhijian_liu.svg"
code: "https://github.com/Gxj230958/paroquant-ascend-int4"
report: "https://github.com/Gxj230958/paroquant-ascend-int4/blob/main/RESEARCH.md"
pr: "https://github.com/z-lab/paroquant/pull/57"
---
- Runs the published Qwen3-4B/8B-PARO checkpoints on Ascend 910B unchanged: one Triton Ascend kernel fuses channel scaling and all eight group-128 pairwise rotation stages.
- One-time AWQ-to-native INT4 repacking feeds the W4A16 matrix operator; decode uses one row per program, prefill eight, with a tiled Triton unpack-and-GEMM path as ablation.
- Peak allocated memory drops 38.9-42.1% vs cached FP16 execution of the same quantized weights; decode is 45.6-83.6x the naive expand path and 5.0-18.1% below the dense fused baseline.
- Full-set quality: WikiText-2 perplexity 13.8587 (4B) / 9.8820 (8B, 299,077 targets); GSM8K 89.54% / 87.11% (all 1319); all 252 linears per model validated, max rel. L2 0.00266.
