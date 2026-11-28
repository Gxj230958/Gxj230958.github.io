---
title: "Cache-Aware Fair Scheduling for LLM Serving on Ascend NPU"
summary: "I changed how vLLM Ascend assigns service to tenants when requests share a batch or rebuild evicted KV state. On a mixed-length Qwen3-4B stream, the maximum gap between backlogged tenants fell from 2.55 to 1.15 s. Across eight paired seeds, the policy had a 13.4% smaller service gap and 1.55% lower throughput than profiled virtual token counting."
weight: 3
cover: "/covers/danyang_zhuo.svg?v=2"
coverAlt: "Paired maximum service-gap differences versus profiled VTC across four prefix and arrival conditions, with 95% intervals."
code: "https://github.com/Gxj230958/cache-aware-llm-scheduling"
codeLabel: "Code and data"
report: "/projects/cache-aware-llm-scheduling/report.pdf"
---

**How should a server divide the cost of a batch shared by several tenants?** On the original Qwen3-4B mixed-length stream, the fitted fixed dispatch term covered 94.9% of predicted service. Splitting it equally produced a 2.55 s maximum gap in independently measured service. Weighting it by executed prefill and decode work brought the gap to 1.22 s. Accounting for replay separately brought it to 1.15 s.

![Fixed-dispatch share and the four original mixed-length ablation arms](/projects/cache-aware-llm-scheduling/mixed-mechanism.png)

The evaluation crosses warm document-prefix reuse and burst arrivals while holding every request’s length fixed. It compares four accounting rules with FCFS, token VTC, profiled VTC, the original executed-work scheduler and single-worker DLPM. The report includes a separately calibrated Qwen3-1.7B transfer experiment and an external BurstGPT arrival-and-length trace.

On the shared-prefix burst condition the paired gap was **37.0% smaller** than profiled VTC. Across all four conditions the change was **-13.4%**, with a 95% seed-bootstrap interval from **-16.7% to -9.7%**. Each of the eight seeds ran all policies on the same physical device. [Per-run measurements](https://github.com/Gxj230958/cache-aware-llm-scheduling/blob/main/results/attribution-summary/4b/runs.csv)
