---
title: "Cost-Aware Mixed-Precision Scheduling for Diffusion Transformers"
summary: "Selects a fixed INT8/BF16 table from paired error and complete layer costs across 144 modules and five sampling segments. At batch 8, all 720 decisions retain BF16, avoiding the 11.2% latency increase of uniform W8A8; uniform W8A8 saves 20.4% peak memory."
weight: 2
tagline: "Measured-cost per-timestep-segment, per-module INT8/BF16 scheduling for DiT-XL/2 on Ascend 910B, with paired error-and-cost calibration and full-generation FID evaluation."
cover: "/covers/tony_geng.svg"
code: "https://github.com/Gxj230958/cost-aware-dit-precision"
report: "https://github.com/Gxj230958/cost-aware-dit-precision/blob/main/RESEARCH.md"
pr: ""
---
- 144 linear modules x 5 sampling-step segments form 720 precision decisions, fixed before weight preparation from measured W8A8-vs-BF16 error and measured complete layer cost; no online profiling.
- At batch 8 the cost-aware table keeps BF16 for all 720 decisions, avoiding the 11.2% latency increase of uniform W8A8 (2.81677 vs 2.53193 s/batch, 50 DDIM steps + VAE); W8A8 instead cuts peak memory 20.4%.
- Cost decomposition across ND/FRACTAL_NZ layouts locates the INT8 payoff boundary: batch-64 NZ fc2 reaches 0.880x BF16 layer time, yet uniform W8A8 stays 2.39% slower over full generation at batch 64.
- 15,000 matched evaluation images; official ADM FID on 5000 images per distinct strategy: BF16 17.8916, uniform W8A8 16.4195, error-only 17.8618 (single matched sets, finite-sample variation noted).
