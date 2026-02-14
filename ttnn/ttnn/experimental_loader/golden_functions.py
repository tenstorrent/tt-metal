# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

# set golden functions


def _golden_function(input_tensor, *args, **kwargs):
    import torch

    return torch.exp(input_tensor)


ttnn.attach_golden_function(ttnn.exp, _golden_function)


def _golden_function(
    input_tensor,
    kv_input_tensor,
    *,
    num_heads,
    num_kv_heads,
    transpose_k_heads=True,
    **_,
):
    import torch

    if num_kv_heads is None:
        num_kv_heads = num_heads

    batch_size, Z, sequence_size, hidden_size = input_tensor.shape
    head_size = hidden_size // num_heads

    query = torch.reshape(input_tensor, (batch_size, sequence_size, num_heads, head_size))
    query = torch.permute(query, (0, 2, 1, 3)).contiguous().clone()

    batch_size, Z, sequence_size, hidden_size = kv_input_tensor.shape
    head_size = hidden_size // num_kv_heads // 2
    split_tensors = kv_input_tensor.split(kv_input_tensor.shape[-1] // (2 * num_kv_heads), dim=-1)
    key = torch.concat(split_tensors[::2], dim=-1)
    value = torch.concat(split_tensors[1::2], dim=-1)

    key = torch.reshape(key, (batch_size, sequence_size, num_kv_heads, head_size))
    value = torch.reshape(value, (batch_size, sequence_size, num_kv_heads, head_size))

    key = torch.permute(key, (0, 2, 1, 3)).contiguous().clone()
    value = torch.permute(value, (0, 2, 1, 3)).contiguous().clone()
    if transpose_k_heads:
        key = torch.permute(key, (0, 1, 3, 2)).contiguous().clone()

    return query, key, value


ttnn.attach_golden_function(ttnn.experimental.create_qkv_heads_from_separate_tensors, _golden_function)


def _golden_function(tensor, grid_size, shard_spec, num_slices, slice, *args, **kwargs):
    tensor = tensor.reshape(1, 1, -1, tensor.shape[-1])
    slice_size = tensor.shape[-2] // num_slices
    start = slice * slice_size
    stop = start + slice_size
    tensor = tensor[:, :, start:stop, :]
    return tensor


ttnn.attach_golden_function(ttnn.interleaved_to_sharded_partial, _golden_function)


def _golden_function(slice, tensor, num_slices, slice_id, *args, **kwargs):
    original_shape = tensor.shape
    tensor = tensor.reshape(1, 1, -1, tensor.shape[-1])
    slice_size = tensor.shape[-2] // num_slices
    start = slice_id * slice_size
    stop = start + slice_size
    tensor[:, :, start:stop, :] = slice
    return tensor.reshape(original_shape)


ttnn.attach_golden_function(ttnn.sharded_to_interleaved_partial, _golden_function)


def _golden_function(in0, in1, math_op, dim, *args, **kwargs):
    import torch

    if dim in {ttnn.BcastOpDim.W, ttnn.BcastOpDim.H, ttnn.BcastOpDim.HW}:
        # Perform the operation
        if math_op == ttnn.BcastOpMath.ADD:
            res = in0 + in1
        elif math_op == ttnn.BcastOpMath.SUB:
            res = in0 - in1
        elif math_op == ttnn.BcastOpMath.MUL:
            res = in0 * in1
        else:
            raise AssertionError("Invalid math operation")

        # Handle ALL dimension mismatches
        if res.shape != in0.shape:
            slices = []
            for i, (res_dim, in0_dim) in enumerate(zip(res.shape, in0.shape)):
                if res_dim > in0_dim and in0_dim == 1:
                    # Truncate any dimension that is size 1 in in0
                    slices.append(slice(0, 1))
                elif res_dim >= in0_dim:
                    # Take first in0_dim elements
                    slices.append(slice(0, in0_dim))
                else:
                    slices.append(slice(None))

            res = res[tuple(slices)]

        return res
    else:
        raise AssertionError("Invalid bcast dimension")


ttnn.attach_golden_function(ttnn.bcast, _golden_function)


def _nop_golden_function(input_tensor, *args, **kwargs):
    return input_tensor


ttnn.attach_golden_function(ttnn.interleaved_to_sharded, _nop_golden_function)
ttnn.attach_golden_function(ttnn.sharded_to_interleaved, _nop_golden_function)
ttnn.attach_golden_function(ttnn.reshard, _nop_golden_function)
ttnn.attach_golden_function(ttnn.tilize, _nop_golden_function)
