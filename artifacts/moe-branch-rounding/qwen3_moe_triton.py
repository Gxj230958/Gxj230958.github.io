# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lift gate/up branches into separate GMM groups, retaining BF16 roundings."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(
    grad,
    counts,
    packed,
    groups,
    E: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_I: tl.constexpr,
    ROWS: tl.constexpr,
):
    expert, part = tl.program_id(0), tl.program_id(1)
    experts = tl.arange(0, BLOCK_E)
    values = tl.load(counts + experts, experts < E, 0).to(tl.int32)
    start = tl.sum(tl.where(experts < expert, values, 0), 0)
    count = tl.load(counts + expert).to(tl.int32)
    if part == 0:
        tl.store(groups + 2 * expert, count)
        tl.store(groups + 2 * expert + 1, count)
    columns = tl.arange(0, BLOCK_I)
    for offset in range(part * ROWS, count, 8 * ROWS):
        rows = offset + tl.arange(0, ROWS)
        mask = (rows[:, None] < count) & (columns[None, :] < INTERMEDIATE)
        gate = tl.load(
            grad + (start + rows[:, None]) * 2 * INTERMEDIATE + columns[None, :],
            mask,
            0,
        )
        up = tl.load(
            grad + (start + rows[:, None]) * 2 * INTERMEDIATE + INTERMEDIATE + columns[None, :],
            mask,
            0,
        )
        tl.store(
            packed + (2 * start + rows[:, None]) * INTERMEDIATE + columns[None, :],
            gate,
            mask,
        )
        tl.store(
            packed + (2 * start + count + rows[:, None]) * INTERMEDIATE + columns[None, :],
            up,
            mask,
        )


@triton.jit
def _combine(
    branches,
    counts,
    output,
    E: tl.constexpr,
    H: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS: tl.constexpr,
):
    expert, part = tl.program_id(0), tl.program_id(1)
    experts = tl.arange(0, BLOCK_E)
    values = tl.load(counts + experts, experts < E, 0).to(tl.int32)
    start = tl.sum(tl.where(experts < expert, values, 0), 0)
    count = tl.load(counts + expert).to(tl.int32)
    columns = tl.arange(0, BLOCK_H)
    for offset in range(part * ROWS, count, 8 * ROWS):
        rows = offset + tl.arange(0, ROWS)
        mask = (rows[:, None] < count) & (columns[None, :] < H)
        gate = tl.load(branches + (2 * start + rows[:, None]) * H + columns[None, :], mask, 0)
        up = tl.load(
            branches + (2 * start + count + rows[:, None]) * H + columns[None, :],
            mask,
            0,
        )
        result = (gate.to(tl.float32) + up.to(tl.float32)).to(tl.bfloat16)
        tl.store(output + (start + rows[:, None]) * H + columns[None, :], result, mask)


def branch_group_dx(grad, weight, counts):
    """Compute paired input gradients with two independent BF16 products."""
    import torch_npu

    e, h, two_i = weight.shape
    if weight.stride() != (h * two_i, 1, h) or not grad.is_contiguous():
        raise ValueError("Branch grouping requires the physical HF weight layout and contiguous gradient")
    if grad.dtype != torch.bfloat16 or weight.dtype != grad.dtype or counts.dtype != torch.int64:
        raise ValueError("Expected BF16 operands and direct int64 counts")
    i = two_i // 2
    packed = torch.empty((2 * grad.shape[0], i), dtype=grad.dtype, device=grad.device)
    groups = torch.empty((2 * e,), dtype=counts.dtype, device=counts.device)
    output = torch.empty((grad.shape[0], h), dtype=grad.dtype, device=grad.device)
    if not grad.shape[0]:
        return output
    _pack[(e, 8)](
        grad,
        counts,
        packed,
        groups,
        e,
        i,
        triton.next_power_of_2(e),
        triton.next_power_of_2(i),
        4,
    )
    branch_weight = weight.transpose(1, 2).view(2 * e, i, h)
    branches = torch_npu.npu_grouped_matmul(
        [packed],
        [branch_weight],
        bias=None,
        group_list=groups,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]
    _combine[(e, 8)](
        branches,
        counts,
        output,
        e,
        h,
        triton.next_power_of_2(e),
        triton.next_power_of_2(h),
        4,
    )
    return output
