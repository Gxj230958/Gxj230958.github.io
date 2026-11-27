"""Two native BF16 branch products, on the same contiguous upstream gradient as 2E."""

from verl.utils.kernel.npu.qwen3_moe import _grouped_mm


def separate_dx(grad, weight, counts):
    gate, up = grad.chunk(2, dim=-1)
    wg, wu = weight.chunk(2, dim=-1)
    return _grouped_mm(gate, wg.transpose(1, 2), counts) + _grouped_mm(
        up, wu.transpose(1, 2), counts
    )


def wide_dx(grad, weight, counts):
    """One row of the padded 16,384-column gate/up gradient per packing tile."""
    import torch
    import triton
    from verl.utils.kernel.npu.qwen3_moe_triton import _pack, _combine

    e, h, two_i = weight.shape
    i = two_i // 2
    m = grad.shape[0]
    packed = grad.new_empty(2 * m, i)
    groups = counts.new_empty(2 * e)
    _pack[(e, 8)](
        grad,
        counts,
        packed,
        groups,
        e,
        i,
        triton.next_power_of_2(e),
        triton.next_power_of_2(i),
        1,
    )
    branches = _grouped_mm(packed, weight.transpose(1, 2).view(2 * e, i, h), groups)
    out = torch.empty((m, h), device=grad.device, dtype=grad.dtype)
    _combine[(e, 8)](
        branches,
        counts,
        out,
        e,
        h,
        triton.next_power_of_2(e),
        triton.next_power_of_2(h),
        2,
    )
    return out
