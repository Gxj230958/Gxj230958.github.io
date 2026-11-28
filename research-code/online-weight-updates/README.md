# Preserving KV Cache Capacity During Online Weight Updates on Ascend

This research artifact accompanies the Qwen2.5 experiments on Ascend 910B2C. It contains the receiver, session budget selector, experiment harness, and the numerical summaries used in the English report. [Project page](https://gxj230958.github.io/research/online-weight-updates/) · [Report (PDF)](https://gxj230958.github.io/projects/online-weight-updates/report.pdf)

## Implementation

- `streaming.py` obtains contiguous byte ranges from the live vLLM load plan. `streaming.patch` integrates direct destinations into the bounded HCCL receive loop and adds its CPU regression cases.
- `selector.py` enumerates transport choices for a post-KV memory budget. `inventory.py` reads BF16 checkpoint headers.
- `experiment.py` runs cold references, live updates, deterministic generation and allocator measurements. `probes.py` provides the optional layer and device tracing hooks.

## Reproducing a receiver run

Use vLLM at `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15` and vLLM-Ascend at `dcc78d0773a0ebd45275148c18d02a3145356aa7`. The latter contains the underlying bounded transport. [Upstream HCCL implementation](https://github.com/vllm-project/vllm-ascend/pull/21288/files)

Apply the following patches to the Ascend checkout in order. Build the pinned platform plugin and native operators for the environment in section 4 of the report, activate CANN/NNAL, and make both source trees and this directory importable.

```bash
study_dir="$PWD"
git -C "$STUDY_ASCEND_SRC" apply "$study_dir/patches/lifecycle-baseline.patch"
git -C "$STUDY_ASCEND_SRC" apply "$study_dir/patches/baseline-stream-cache-control.patch"
git -C "$STUDY_ASCEND_SRC" apply "$study_dir/streaming.patch"
export PYTHONPATH="$STUDY_VLLM_SRC:$STUDY_ASCEND_SRC:$study_dir${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

Set `BASE` and `INSTRUCT` to the two local BF16 Qwen2.5-7B checkpoint directories. The following TP1 example assumes two idle, healthy 910B2C devices with physical IDs 0 and 1. The receiver occupies the first device and the trainer the second. Use the IDs available on your host.

```bash
for version in base instruct; do
  ASCEND_RT_VISIBLE_DEVICES=0 python experiment.py \
    --base "$BASE" --instruct "$INSTRUCT" --reference "$version" \
    --oracle ./oracle --output "./results/reference-$version.json"
done
ASCEND_RT_VISIBLE_DEVICES=0,1 python experiment.py \
  --base "$BASE" --instruct "$INSTRUCT" --oracle ./oracle \
  --mode stream --reload direct --slots 1 --kv-blocks 6550 \
  --warmups 0 --samples 2 --verify --attribute --output ./results/update.json
```

## Reading the result tables

- `results/capacity.csv` gives observed successes and failures for four 7B TP1 policies. One block holds 128 tokens. The bounded two-slot row gives a bracket; the other three rows have adjacent observed endpoints on their original device pairs.
- `results/holdouts.csv` compares predicted and observed peaks and block counts for two unseen layouts. The TP2 maximum pass exceeds the minimum OOM because outcomes overlapped across sessions.
- `results/fixed-capacity.csv` gives the median of three independent session statistics for each strategy at 786,432 KV tokens. Each session contains 50 measured updates after warm-up. Replay throughput includes updates and generation.

## Reproducing the website figures

From this directory, run `python plot_project.py --output figures` with Matplotlib installed. Each figure is exported as SVG and PDF. The receiver illustration traces one hatched fragment from the trainer into live model storage, with a separate temporary matrix on the reassembly path.

`results/receiver-layout.csv` gives the two streamed tensor sizes and the largest remaining input from the 7B BF16 checkpoint inventory. `results/matched-peaks.csv` gives the largest whole-cycle receiver allocation increment across 50 measured cycles in each session, measured from the pre-update baseline. Both two-slot policies used the same physical NPU pair, direct reload, 128 MiB slots and 6,144 KV blocks. The figure reports the median of three independent session maxima per policy. Capacity endpoints in the adjacent panel come from the separate observations in `results/capacity.csv`.

## Concurrent rollout performance

The trainer holds both BF16 checkpoints in device memory. Use one visible rollout NPU for TP1, or two for TP2, followed by one trainer NPU. Create cold references for both versions with the corresponding `--tp` setting before running the version sentinels. At TP2 the all-eligible broadcast baseline is `--mode stream-all`. The row-shard transport is `--mode rank-lanes`. The row-and-column transport is `--mode rank-columns --direct-min-mib 16`. The peers receive into live weights; shared inputs stay on the common broadcast communicator.

```bash
for version in base instruct; do
  ASCEND_RT_VISIBLE_DEVICES=0,1 python experiment.py --tp 2 --base "$BASE" --instruct "$INSTRUCT" \
    --reference "$version" --oracle ./oracle --output "./reference-tp2-$version.json"
done
ASCEND_RT_VISIBLE_DEVICES=0,1,2 python experiment.py --tp 2 --base "$BASE" --instruct "$INSTRUCT" \
  --oracle ./oracle --reload direct --mode rank-columns --direct-min-mib 16 --kv-mib 50200 \
  --warmups 3 --samples 12 --rollout-cycles 2 --output ./tp2-session.json
python analyze_acceleration.py --from-tables results --output figures
```

Experiment manifests under `configs/` give the device group, output stem and command-line flags for each session. Map the devices to the equivalent local topology and supply local checkpoint and oracle paths. Five independent sessions per configuration measure alternating Base and Instruct with a complete update every batch or every four batches. Each batch has sixteen 512-token prompts and 128 generated tokens per request. The fixed-policy control runs two batches without an update. `results/acceleration-sessions.csv` records pause, replay rate and peak allocation. `results/rollout-sessions.csv` gives output rate, first-token latency and same-version sequence variation. The separate device intervals and prepacked broadcast references characterize communication and copying.

## Reproducing the project page

```bash
python plot_project.py --output figures
python plot_handoff.py --output figures
```

`receiver.svg` and `parallel-handoff.svg` also have PDF versions. The upper panel of `parallel-handoff` plots the session medians in `results/acceleration-sessions.csv`. The lower panel uses `results/sender-hccl-relative.csv`, which records the start and duration of each HCCL operation relative to the first operation in its profiling round. `trace_acceleration.py` exports these intervals alongside the profile summary. `rollout-table.md` is generated from `results/rollout-sessions.csv` using matching session IDs for each pair.
