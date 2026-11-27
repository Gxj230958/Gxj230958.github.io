# BF16 branch grouping for MoE training on Ascend 910B

The verl patch includes the packed training path, native grouped products, TritonAscend pack/combine kernels and numerical tests. The matched baseline and Mixtral width adaptation are in [variants.py](variants.py). [English paper](https://gxj230958.github.io/papers/moe-branch-rounding/) · [Download PDF](https://gxj230958.github.io/papers/moe-branch-rounding/paper.pdf).

## Run a matched block comparison

Use a clean checkout of verl at `8e48d72de4766228c36623c90f9f5c49ee5b45f3`. The measured environment uses CANN 8.5.1, PyTorch 2.9.0, torch_npu 2.9.0.post1, TritonAscend 3.2.0.dev20260322 and Transformers 5.5.3. With the artifact directory as the working directory and the target checkout in the shell variable `VERL_CHECKOUT`, run on an idle Ascend 910B2C:

```bash
git -C "$VERL_CHECKOUT" apply "$PWD/expert-backward.patch"
export PYTHONPATH="$VERL_CHECKOUT:$PWD"
export ROUTING_RUN_DIR="$(mktemp -d)"
export ASCEND_RT_VISIBLE_DEVICES=0
source /usr/local/Ascend/cann-8.5.1/set_env.sh
npu-smi info
python block.py block qwen --tokens 128 8192 --routes uniform hot
```

The harness times the full forward and backward pass, checks the input gradient and all parameter gradients, and saves the paired observations. Qwen and the eight-expert control build their input directly. The paper's Mixtral measurements use pretrained first-layer weights and text hidden states.

## Inspect the measurements

[Block samples](results/) contain the two elapsed times and the execution order for each pair. [Summary tables](analysis/) include the ratio-of-medians reductions, within-case bootstrap intervals, trace task sums, policy comparisons and eight-device actor observations. The original six-case transfer predictions and the separately specified two-case length test are in the two prediction JSON files.

Run `python paper_figures.py` with Matplotlib and NumPy to regenerate all four figures and the cover from these files, without an NPU.

The [project overview](https://gxj230958.github.io/research/moe-branch-rounding/) includes the [kernel source](qwen3_moe_triton.py). Run `python project_figures.py` to draw its mechanism diagram, matched comparison and device timeline from the archived measurements.
