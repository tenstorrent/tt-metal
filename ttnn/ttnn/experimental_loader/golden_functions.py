# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn.operations.golden_common import golden_to_output_dtype

# set golden functions


def _golden_function(input_tensor, *args, fast_and_approximate_mode=False, **kwargs):
    import torch

    result = torch.exp(input_tensor)
    if fast_and_approximate_mode:
        # Fast exp is specified against exact exp within a 5% relative error, so degenerate outputs
        # use that bound instead of the default near-exact allclose tolerance.
        ttnn.decorators.set_golden_comparison_config(
            result, method="allclose", scope="degenerate", rtol=0.05, atol=1e-6
        )
    return result


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


def _golden_function(tensor, grid_size, shard_spec, num_slices, slice, *args, output_dtype=None, **kwargs):
    tensor = tensor.reshape(1, 1, -1, tensor.shape[-1])
    slice_size = tensor.shape[-2] // num_slices
    start = slice * slice_size
    stop = start + slice_size
    tensor = tensor[:, :, start:stop, :]
    return golden_to_output_dtype(tensor, output_dtype)


ttnn.attach_golden_function(ttnn.interleaved_to_sharded_partial, _golden_function)


def _golden_function(slice, tensor, num_slices, slice_id, *args, output_dtype=None, **kwargs):
    original_shape = tensor.shape
    tensor = tensor.reshape(1, 1, -1, tensor.shape[-1])
    slice_size = tensor.shape[-2] // num_slices
    start = slice_id * slice_size
    stop = start + slice_size
    tensor[:, :, start:stop, :] = golden_to_output_dtype(slice, output_dtype)
    return tensor.reshape(original_shape)


ttnn.attach_golden_function(ttnn.sharded_to_interleaved_partial, _golden_function)


def _golden_function(in0, in1, math_op, dim, *args, **kwargs):
    import torch

    if dim in {ttnn.BcastOpDim.W, ttnn.BcastOpDim.H, ttnn.BcastOpDim.HW}:
        # The device broadcasts only the first column (W), row (H) or element (HW) of in1's tile,
        # even when in1 spans a full tile along the broadcast dimension.
        if dim in {ttnn.BcastOpDim.W, ttnn.BcastOpDim.HW}:
            in1 = in1[..., :1]
        if dim in {ttnn.BcastOpDim.H, ttnn.BcastOpDim.HW}:
            in1 = in1[..., :1, :]

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


def _sharding_conversion_golden_function(input_tensor, *args, output_dtype=None, **kwargs):
    # Every sharding-conversion overload ends its positional arguments with an optional output_dtype.
    if output_dtype is None and args and isinstance(args[-1], ttnn.DataType):
        output_dtype = args[-1]
    return golden_to_output_dtype(input_tensor, output_dtype)


def _tilize_golden_function(input_tensor, *args, dtype=None, **kwargs):
    return golden_to_output_dtype(input_tensor, dtype)


ttnn.attach_golden_function(ttnn.interleaved_to_sharded, _sharding_conversion_golden_function)
ttnn.attach_golden_function(ttnn.sharded_to_interleaved, _sharding_conversion_golden_function)
ttnn.attach_golden_function(ttnn.reshard, _nop_golden_function)
ttnn.attach_golden_function(ttnn.tilize, _tilize_golden_function)


def _slice_write_golden_function(input_tensor, output_tensor, start, end, step, *args, **kwargs):
    slices = tuple(slice(int(begin), int(stop), int(stride)) for begin, stop, stride in zip(start, end, step))
    # Mutate the destination golden so callers retaining output_tensor observe each write.
    output_tensor[slices] = input_tensor
    return output_tensor


ttnn.attach_golden_function(ttnn.experimental.slice_write, _slice_write_golden_function)


def _broadcast_to_golden_function(input, output_shape, *args, output=None, **kwargs):
    result = input.broadcast_to(tuple(output_shape))
    if output is not None:
        # Mirror output= writes so retained local and global destination goldens stay current.
        output.copy_(result)
        return output
    return result


ttnn.attach_golden_function(ttnn.experimental.broadcast_to, _broadcast_to_golden_function)


def _indexed_fused_update_cache_golden_function(
    cache_tensor1,
    input_tensor1,
    cache_tensor2,
    input_tensor2,
    physical_update_idxs_tensor,
):
    output_tensor1 = cache_tensor1.clone()
    output_tensor2 = cache_tensor2.clone()
    rows_per_page = cache_tensor1.shape[2]
    total_cache_rows = cache_tensor1.shape[0] * rows_per_page

    for source_row, physical_row in enumerate(physical_update_idxs_tensor.reshape(-1)[: input_tensor1.shape[2]]):
        physical_row = int(physical_row)
        if physical_row < 0 or physical_row >= total_cache_rows:
            continue
        physical_page, row_in_page = divmod(physical_row, rows_per_page)
        output_tensor1[physical_page, :, row_in_page, :] = input_tensor1[0, :, source_row, :]
        output_tensor2[physical_page, :, row_in_page, :] = input_tensor2[0, :, source_row, :]

    return output_tensor1, output_tensor2


ttnn.attach_golden_function(
    ttnn.experimental.indexed_fused_update_cache,
    _indexed_fused_update_cache_golden_function,
)


def _composite_example_golden_function(input_tensor, *args, **kwargs):
    # composite_example applies the example copy op twice; the value is the input unchanged.
    return input_tensor


ttnn.attach_golden_function(ttnn.composite_example, _composite_example_golden_function)


def _composite_example_multiple_return_golden_function(
    input_tensor, return_output1=True, return_output2=True, *args, **kwargs
):
    # Each requested output is a copy of the input; unrequested outputs are None.
    return [
        input_tensor if return_output1 else None,
        input_tensor if return_output2 else None,
    ]


ttnn.attach_golden_function(ttnn.composite_example_multiple_return, _composite_example_multiple_return_golden_function)


def _dram_prefetcher_golden_function(tensors, *args, **kwargs):
    import torch

    # dram_prefetcher returns an otherwise unspecified 32x32 synchronization tensor.
    output = torch.empty((32, 32), dtype=tensors[0].dtype)
    ttnn.decorators.set_golden_comparison_config(output, method="skip", scope="all")
    return output


ttnn.attach_golden_function(ttnn.dram_prefetcher, _dram_prefetcher_golden_function)

# generic_op executes a user-supplied program descriptor; it has no fixed semantics to reference.
ttnn.attach_golden_function(ttnn.generic_op, golden_function=None)

# test_hang_device_operation intentionally hangs the device for testing; it has no value golden.
ttnn.attach_golden_function(ttnn.test_hang_device_operation, golden_function=None)
