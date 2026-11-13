---
title: "Cache-Aware Fair Scheduling for LLM Serving on Ascend NPU"
summary: "Extends vLLM-Ascend with executed-work accounting for cached prefixes, prefill chunks, dynamic batches and preemption. On Qwen3-4B, reduces service gaps by 16.16% on shared prefixes and 22.73% on bursts versus profiled VTC, at 4.32% and 0.79% median throughput cost."
weight: 3
tagline: "Executed-work service accounting for fair multi-user LLM inference under prefix caching, chunked prefill and dynamic batching, built on vLLM-Ascend."
cover: "/covers/danyang_zhuo.svg"
code: "https://github.com/Gxj230958/cache-aware-llm-scheduling"
report: "https://github.com/Gxj230958/cache-aware-llm-scheduling/blob/main/REPORT.md"
pr: ""
---
- Extends vLLM-Ascend with weighted virtual-service scheduling, executed-chunk accounting, per-batch reservations and preemption recomputation, retaining the VTC (OSDI'24) framework.
- Charges actual query/context work: cached prefixes skip projection/MLP but stay attention keys; executed prefill chunks, decode contexts, shared batch overhead and recomputation after preemption are all metered.
- Qwen3-4B on one Ascend 910B: 48 policy runs, 1,536 requests, 46,080 tokens; 16.16% smaller service gap on shared prefixes and 22.73% on bursts vs profiled VTC, with 4.32%/0.79% median throughput costs.
- Mixed-length workload shows a 122.60% larger gap and 7.65% lower throughput: joint accounting is a workload-dependent tradeoff, not a universal fairness win.
