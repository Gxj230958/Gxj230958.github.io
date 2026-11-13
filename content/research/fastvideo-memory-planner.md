---
title: "Budget-Driven Activation Memory Planning for FastVideo Training"
summary: "Combines tensor-lifetime modeling with a mixed-integer planner to select saved, recomputed or CPU-offloaded activations. Wan2.1-1.3B fits a 24 GiB budget at 23.498 GiB peak and takes 14.97% less step time than full recomputation; supports multi-role DMD2 and FSDP2 training."
weight: 5
tagline: "Event-liveness model plus MILP planner that selects per-block activation save, recompute, or CPU offload under a strict memory budget for video-diffusion training on Ascend 910B."
cover: "/covers/hao_zhang.svg"
code: "https://github.com/Gxj230958/fastvideo-memory-planner"
report: "https://github.com/Gxj230958/fastvideo-memory-planner/blob/main/research/report.md"
pr: "https://github.com/hao-ai-lab/FastVideo/pull/2055"
---
- MILP planner over an execution-event liveness model picks save/recompute/offload per block under a budget; Wan2.1-1.3B at 24 GiB: 25 saved + 5 recomputed blocks, 23.498 GiB measured peak allocated.
- Complete training step 471.82 ms, 14.97% faster than full recomputation, validated in three independent held-out processes; plans reach 49-frame inputs and two-rank FSDP2 where full-save is infeasible at 17 frames.
- Cross-role constraints cover DMD2 student/teacher/critic lifetimes, satisfying a 54 GiB budget in every run; upstream as draft PRs #2055 (planner, CI green) and #2053 (HCCL NPU stream fix).
