---
title: "When does grouping expert gradients pay off? A BF16 training study on Ascend 910B"
layout: paper
---

[Download paper (PDF)](/papers/moe-branch-rounding/paper.pdf) · [Code and paired measurements](https://github.com/Gxj230958/Gxj230958.github.io/tree/main/artifacts/moe-branch-rounding)

## Abstract

A gated mixture-of-experts layer projects each input through two branches. The backward pass must add their contributions to the input gradient. Combining the branches into a wider matrix product can move the BF16 rounding point and change the gradient. We instead map each expert–branch pair to a distinct group in one native grouped matrix multiplication. A matched baseline uses the same packed forward projection and physical weight-gradient layout, isolating the value of grouping the backward branches. Across four Qwen3 workloads on Ascend 910B2C, grouping reduces complete expert-block forward and backward time by 7.95–37.20%. On a pretrained Mixtral layer with its learned router, the reduction falls from 28.58% at 128 text rows to 3.96% at 8,192, then reverses to a 1.20% regression at 32,768. Device traces distinguish two costs that can reverse the benefit: uneven staging work and increased execution time within native matrix multiplications. On eight NPUs, the change reduces median fixed-batch PPO update time by 2.05–2.26% across three independently launched sessions. All fourteen block comparisons match the reference outputs and gradients exactly. The results identify both a reproducible training benefit and workloads where separate products remain faster.

## 1. Introduction

Conditional computation makes expert training sensitive to two kinds of shape. The model fixes each expert’s weight dimensions. The router determines how many token rows reach that expert on each step. A backend may therefore execute different amounts of data movement for two batches that require the same number of multiply-adds. The gate and up projections of a gated expert expose a further choice. Their weights are already adjacent in memory, but the backward reference rounds the products separately. A systems optimization must account for both the memory layout and the arithmetic at that boundary.

An earlier verl integration on this platform changed the forward projection, weight-gradient layout and input-gradient computation together. That package reduced full-block time by 67.89% as a geometric-mean reduction over 16 cases, and median update time by 4.49% in a deterministic 48-layer, 16-device replay. These measurements leave a practical decision unresolved. Once the forward and weight gradients are already efficient, does a separate packing kernel and a combine kernel earn their place in the backward pass?

This study makes three contributions.

1. **A branch-preserving schedule.** We express two independently rounded products per expert as groups of a single native call. A matched two-call baseline isolates the change at the input-gradient boundary.
2. **An experimental account of two crossovers.** Changing only routing concentration reverses the benefit on a narrow expert geometry. Increasing input length reverses it on a pretrained wide layer. Independent traces show different changes in device work.
3. **A distributed training measurement.** We connect the block result to 120 paired, fixed-state PPO updates on a 24-layer, 15.577B-parameter Qwen3 actor, comparing slowest-rank latency and the resulting optimizer state.

## 2. Background and numerical constraint

Let *E* be the expert count, *k* the experts selected per token, *T* the number of token rows and *M = Tk* the total assignments. Expert *e* receives *nₑ* rows with ∑ₑ nₑ = M. For hidden width *H* and intermediate width *I*, an expert projects its input *X* into *G = XWgᵀ* and *U = XWuᵀ*, applies SiLU(G) elementwise to gate *U*, and projects the result back to width *H*. The router that selects experts is distinct from this internal gate.

Write *Q* for a BF16 writeback and *dG*, *dU* for upstream branch gradients. The reference computes

**dX = Q( Q(dG Wg) + Q(dU Wu) ).**　　　　　　　　　　　　　　　　　(1)

The placement of *Q* affects finite inputs. Suppose the two accumulated products are 1 + 2⁻⁹ and −1. The first rounds to 1 in BF16, so (1) returns zero. Adding the high-precision products before rounding gives the representable BF16 value 2⁻⁹. Preserving the product boundaries lets a new schedule be checked directly against existing training semantics.

## 3. Method

### 3.1 One native call with two outputs per expert

The packed forward stores each expert’s gate and up weights together. Baseline B takes the two slices of the incoming gradient and their corresponding weight views, issues two native grouped matrix multiplications (GMMs), and adds the BF16 outputs. Candidate C packs gradient rows in expert-major, branch-minor order. It reinterprets the physical weights as 2E matrices of shape I × H and supplies row counts [n₀, n₀, n₁, n₁, …]. The native GMM writes separate BF16 products. A combine kernel reads corresponding rows, adds in FP32 and writes BF16. Router assignments, the forward pass and direct computation of the weight gradient in [output, input] order are identical in B and C.

![Two schedules with identical BF16 product boundaries](/papers/moe-branch-rounding/figures/branch-schedule.svg)

*Figure 1. Gate/up input gradients for a packed expert. Each colored product box denotes a BF16 writeback. Grouping doubles the number of internal compute groups while preserving the router’s expert count and each branch’s rows.*

### 3.2 The price of grouping

The staged gradient has shape 2M × I. The native call writes 2M × H BF16 branch elements before combination. These shapes imply a packing buffer of 4MI bytes and a branch-output buffer of 4MH bytes. They describe logical tensors. Peak live memory also depends on buffer lifetimes and the reference implementation’s own temporaries. Two costs may offset the saved slices and launch: a hotspot concentrates row loops in the packing and combine kernels, and the native GMM can execute a different schedule with 2E groups.

Tile feasibility depends on width. At Mixtral’s I = 14,336, compiling the four-row Qwen tiles requests 512 KiB for packing and 256 KiB for combining, against a 192 KiB per-core unified-buffer limit. One packing row and two combining rows fit. We fixed these tile sizes before timing Mixtral. Expert weights are packed once at load time.

### 3.3 A testable dispatch hypothesis

We used the dimensionless proxy **R = EHI / [Tk(H + I)]** to compare weight dimensions with the number of activation elements subject to staging. From the Qwen3 and narrow control cases we fixed this rule: select C for R ≥ 1. Below one, predict B faster if a single expert receives over half of all assignments. Otherwise predict a practical tie (±1%) and choose B. This hypothesis predicts the preferred schedule given feasible tiles. Its value is measured by the time spent on the selected implementation relative to always choosing B or C.

## 4. Experimental setup

**Platform and workloads.** The host has Ascend 910B2C devices with 64 GiB HBM per device. The experiments used CANN 8.5.1, PyTorch 2.9.0, torch_npu 2.9.0.post1, TritonAscend 3.2.0.dev20260322 and Transformers 5.5.3. Qwen3 uses (E, k, H, I) = (128, 8, 2048, 768) [1]. A routing control holds the widths fixed and changes (E, k) to (8, 1). Mixtral uses (8, 2, 4096, 14336) [2]. All Mixtral cases load layer-zero expert weights. Controlled inputs prescribe uniform or fixed-hotspot routing. Text inputs pass WikiText-2 tokens through the pretrained embedding, first causal attention, residual and normalization before reaching the learned router. The three sizes comprise 1, 64 and 256 independent 128-token windows.

**Baselines and unit of measurement.** A is the pinned verl split-forward implementation. B and C share the packed forward, direct weight gradients and contiguous upstream gradient. Both execute the router, token permutation, experts, unpermutation, scalar objective and backward pass on the same device within a case. We also compare the installed Transformers grouped_mm default against C as a deployment comparison. Same-device profiling and emptied-cache memory replays run separately from latency sampling.

**Sampling and correctness.** Each block case has 30 warmups per arm and 100 timed pairs with 50 in each order. Time reduction is 100 × [1 − median(C)/median(B)]. Intervals are 95% within-case percentile bootstrap intervals using 10,000 resamples of complete pairs. Separate observations verify ordered routes, counts, permutation, output, input gradient and all parameter gradients. All fourteen matched block cases pass.

**Distributed replay.** The actor uses the first 24 pretrained Qwen3-30B-A3B-Instruct-2507 layers, embeddings, final norm and output head through verl’s registered FSDP training entry. Eight ranks use BF16 compute, FP32 AdamW, activation checkpointing, backward-post prefetch and two 128-token microbatches per rank. Nine prefix updates establish nonzero optimizer state. Each replay restores common weights, optimizer state, input and RNG. Each timed sample is the maximum synchronized update time across ranks. Three independent launches at seed 47 each supply 30 warmups per arm and 40 balanced pairs.

## 5. Evaluation

### 5.1 How much remains after the baseline is already packed?

Grouping saves roughly 1.1–1.2 ms per Qwen3 forward–backward pass. This translates to 23.94% and 37.20% for 128 rows under uniform and hot routing, and 7.95% and 8.91% for 8,192 rows. In the short hot trace, B issues four GMM Slice tasks totaling 1.351 ms. C replaces them with 0.162 ms of packing and combining. The full block goes from seven native GMM launches to six. Peak allocated memory is the same for B and C in each Qwen3 case. For 128 rows, peak reserved memory falls from 2.892 to 2.489 GB.

![Fourteen matched block comparisons across model shapes and input distributions](/papers/moe-branch-rounding/figures/routing-and-length.svg)

*Figure 2. Reduction in complete expert-block time relative to B. Positive values favor grouping. Whiskers show paired 95% intervals and can be narrower than markers. In panel a the two 32,768-row cases use the eight-expert top-1 control. In panel b “text” denotes learned routing. Each point summarizes 100 pairs.*

### 5.2 Why can the same transformation change sign?

With eight narrow experts, 32,768 rows and top-1 routing, spreading work evenly gives C a 0.59% advantage. Sending every row to one expert makes C 5.09% slower. B’s four slices total about 0.26 ms under either distribution. C’s pack and combine time rises from 0.529 to 1.273 ms. In the hot case the two arms spend 3.604 and 3.607 ms in native GMMs. This supports a staging explanation: the candidate partitions row loops by expert, and the concentrated input leaves fewer partitions with useful work.

Mixtral exhibits a different crossover. With the learned router on real text, the measured B → C times are 10.399 → 7.427 ms at 128 rows, 70.797 → 67.993 ms at 8,192 and 290.847 → 294.327 ms at 32,768. In the last case C’s staging still costs less than B’s slices, 7.417 versus 8.849 ms. Total native GMM time instead rises by 10.624 ms. The trace locates the increase in a family of device tasks. Establishing why the library schedules the wide groups differently would require measurements inside the native GMM.

![Staging and matrix multiplication costs in independent device traces](/papers/moe-branch-rounding/figures/device-costs.svg)

*Figure 3. Device-task durations from separate profiled block passes. Panel a compares slice materialization in B with packing plus combination in C for the narrow eight-expert control. Panel b subtracts B from C within each task family for the pretrained Mixtral text workloads. These task sums and the unprofiled wall times measure different quantities.*

### 5.3 Does the rule transfer to another model?

Before timing Mixtral, the rule selected C for all six uniform, hot and text cases at 128 and 8,192 rows. All six reductions exceeded 1%. Always-C matched the rule. We then fixed a separate length test at 32,768 rows, where R ≈ 0.389. With distinct top-2 assignments a single expert can receive at most half the assignments, so the rule predicted a tie and selected B for both hot and text. B won both. The measured C penalties were 1.16% and 1.20%. Both intervals exclude zero and cross the 1% practical threshold. The choice was correct even though the point estimates fell outside the predicted band.

Counting each of the eight cases once, the rule matches the sum of the faster per-case medians, 872.173 ms, compared with 879.771 ms for always-C and 888.766 ms for always-B. This descriptive aggregation combines two sequentially specified cohorts. Row count alone suffices for these three lengths on this layer.

**Sensitivity to the comparator.** The installed Transformers default takes 10.484 and 82.750 ms on the 128- and 8,192-row text cases, against 7.490 and 67.633 ms for C in those independent comparisons. Maximum absolute output differences from that default are 0.015625 and 0.125. The exactness result above concerns B and C. On the narrow hotspot C also beats A’s static fallback by 1.24%, despite losing to the stronger B by 5.09%.

### 5.4 Does the gain survive a sharded optimizer step?

Across the three sessions, median update time falls by 2.26%, 2.07% and 2.05%. The within-pair median savings are 71.6, 62.3 and 66.2 ms. Every rank records 48 invocations per implementation during the same-state check. Ordered routes, selected weights, permutations, counts, outputs and loss match. Clipped gradients, parameter updates and both Adam moments are elementwise equal. Across ranks and sessions peak allocated memory is 39.015 GB for B and 39.013 GB for C. Peak reserved memory is 56.566 and 56.751 GB, respectively.

![All measured update pairs and per-session p95 on eight NPUs](/papers/moe-branch-rounding/figures/actor-updates.svg)

*Figure 4. Fixed-batch PPO updates on eight Ascend 910B2Cs. Each gray point is one paired difference in maximum-rank wall time. Diamonds and horizontal intervals show the median paired difference and its 95% interval. Panel b shows per-arm p95. In session 2 the median improves while p95 rises.*

| Session | Median B → C (s) | Reduction [paired 95% interval] | p95 B → C (s) |
|---|---|---|---|
| 1 | 3.110 → 3.040 | 2.26% [2.06%, 2.45%] | 3.169 → 3.097 |
| 2 | 3.106 → 3.042 | 2.07% [1.82%, 2.34%] | 3.151 → 3.374 |
| 3 | 3.106 → 3.042 | 2.05% [1.89%, 2.29%] | 3.416 → 3.338 |

## 6. Discussion and limitations

A scheduler faces two independently variable costs. Routing concentration changes the work distribution in the staging kernels. Increasing width and assignment count can change how much device time the grouped products themselves consume. A useful admission test must consider both. On the measured Mixtral layer, input length gives a cheap decision without reading route counts back to the host. The threshold’s generality across further layers and intermediate lengths remains open.

The study covers one accelerator and software stack, one pretrained Mixtral layer, synthetic routing controls and a deterministic fixed-batch Qwen3 actor with synthetic advantages. Session replication measures runtime variation at one training state. The p95 regression in session 2 limits claims about tail latency. Rollout throughput and compute-only long-run quality require separate experiments. In earlier native-runtime full-model experiments, some RMSNorm gradient and update thresholds were exceeded, and the earlier long-trajectory quality study also changed communication. The matched replay establishes numerical agreement in the configuration reported here.

## 7. Related work

**Sparse expert execution.** MegaBlocks formulates dynamic expert routing as block-sparse computations on GPUs [3]. Mixtral demonstrates a model in which each token selects two experts per layer [2]. Here the assignments remain fixed while the two backward products *within* each selected expert are scheduled differently.

**Data movement and tiling.** SonicMoE studies activation traffic, tile efficiency and grouped GPU kernels [4]. The Ascend verl optimization practice also uses GMM and merged expert weights [5]. The contribution examined here is the incremental value of a schedule that preserves separate BF16 product writebacks after forward packing and weight-gradient layout have already been addressed.

**Distributed training.** FSDP shards training state and coordinates parameter materialization and gradient reduction [6]. That execution context motivates measuring the synchronized slowest rank. A block speedup alone gives no estimate of how much latency remains exposed in the optimizer step.

## 8. Conclusion

One native call can retain both BF16 product boundaries and deliver a measurable training benefit. Across the workloads studied here, the decisive quantity is the full cost of staging and executing the regrouped products. On short Qwen3 blocks the saved slice work dominates. On a narrow routing hotspot or a sufficiently long pretrained Mixtral input, separate products win. The next test is whether native GMM timing at intermediate widths and lengths can predict that boundary on another layer before running a complete training update.

## References

1. Qwen Team. *Qwen3-30B-A3B-Instruct-2507*, model card, 2025. [Checkpoint and configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
2. A. Q. Jiang et al. *Mixtral of Experts*, 2024. [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
3. T. Gale, D. Narayanan, C. Young and M. Zaharia. *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*, 2022. [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)
4. W. Guo et al. *SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations*, 2025, revised 2026. [arXiv:2512.14080](https://arxiv.org/abs/2512.14080)
5. verl project. *Add Qwen3-235B NPU Long Sequence Optimizing Practice*. [Implementation discussion, PR 5835](https://github.com/verl-project/verl/pull/5835)
6. Y. Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*, 2023. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)

## Research artifacts

[Reproduction commands](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/README.md) · [Reference and wide-layer implementation](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/variants.py) · [verl patch](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/expert-backward.patch) · [Core block estimates](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/analysis/block.csv) · [Length extrapolation](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/analysis/length-extrapolation.csv) · [Device traces, summarized](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/analysis/profile.csv) · [Actor estimates](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/analysis/actor.csv) · [Frozen transfer predictions](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/prediction.json) · [Length-test predictions](https://github.com/Gxj230958/Gxj230958.github.io/blob/main/artifacts/moe-branch-rounding/length-prediction.json)

**Vector figures.** [Figure 1 (PDF)](/papers/moe-branch-rounding/figures/branch-schedule.pdf) · [Figure 2 (PDF)](/papers/moe-branch-rounding/figures/routing-and-length.pdf) · [Figure 3 (PDF)](/papers/moe-branch-rounding/figures/device-costs.pdf) · [Figure 4 (PDF)](/papers/moe-branch-rounding/figures/actor-updates.pdf)
