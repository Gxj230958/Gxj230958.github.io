---
title: "NUMA-Aware Optimizer Offload for LLM Training"
weight: 6
tagline: "Bounded gradient-transfer, CPU AdamW and weight write-back pipeline for full-parameter Qwen3 training on Ascend 910B, with a DeepSpeed GIL patch."
cover: "/covers/minjia_zhang.svg"
code: "https://github.com/Gxj230958/numa-optimizer-offload"
report: "https://github.com/Gxj230958/numa-optimizer-offload/blob/main/reports/technical_report.md"
pr: "https://github.com/deepspeedai/DeepSpeed/pull/9035"
---
- Pipelined CPU-NPU offload gives 1.430x/1.516x/1.590x complete-step speedups over tuned synchronous offload on Qwen3-0.6B/1.7B/4B (33 processes, 198 timed updates).
- Joint calibration over 8/32/128 MiB buckets, depths 1/2/4, 4-32 threads and local/remote NUMA affinity selects 8 MiB + depth 4 + 8 local threads on complete-step time.
- Offloading trains 4B at 2,048 tokens where the tested resident optimizer OOMs at 256; GIL release in DeepSpeed CPUAdam adds a median 1.151x paired training speedup.
