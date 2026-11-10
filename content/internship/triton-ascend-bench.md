---
title: "TritonAscendBench Benchmark Suite (Open Source)"
weight: 3
summary: "Released a tiered benchmark with 101 core AI Triton operators for evaluating cross-hardware kernel migration."
code: "https://github.com/Gxj230958/TritonAscendBench"
---

TritonAscendBench is a public benchmark for Triton kernel migration across GPU and Ascend NPU. It is organized into three difficulty levels: single operator, fused operator, and multi-stage pipeline. It contains 101 representative AI operators, each paired with reference GPU Triton implementations and PyTorch golden outputs. Each operator covers dozens of input shapes split into seen and held-out test sets to measure generalization.

GitHub: <https://github.com/Gxj230958/TritonAscendBench>
