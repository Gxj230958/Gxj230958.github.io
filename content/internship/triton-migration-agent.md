---
title: "LLM Agent for Triton Operator Migration (GPU → Ascend NPU)"
weight: 2
summary: "Developed an LLM-agent workflow to automate migrating Triton kernels from GPU to Ascend NPU, lowering the expertise barrier for heterogeneous operator porting and optimization."
---

This work targets the Ascend heterogeneous computing ecosystem. I built an agent framework to automate the migration and optimization of GPU Triton operators onto NPU Triton. The goal is to reduce heavy manual engineering work and the requirement for deep heterogeneous computing expertise. I constructed the TritonAscendBench benchmark suite to evaluate migration quality, then collected CoT traces and performed supervised fine-tuning on Qwen variants using the benchmark. The tuned model reaches >50% accuracy on the benchmark, close to the stronger Qwen3.8-max-xhigh baseline. Ongoing work includes hard-example mining from failure cases and on-policy distillation to inject domain knowledge into model weights, targeting a closed loop of compilation validation, profiling and iterative optimization to cut hardware adaptation cost.
