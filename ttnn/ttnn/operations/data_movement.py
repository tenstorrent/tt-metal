# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import ttnn
import ttnn.decorators
from ttnn.operations import integer_golden


def _preprocess_golden_function_inputs(args, kwargs):
    input_tensor, args, kwargs = ttnn.reflection.pop_argument("input_tensor", args, kwargs)
    pad_arg, args, kwargs = ttnn.reflection.pop_argument("padding", args, kwargs)

    rank = len(input_tensor.shape)
    input_shape = list(input_tensor.shape)

    # ttnn.pad has two overloads:
    #   A) pad(input, padding, value, ...)          where padding is a list of (start, end) pairs
    #   B) pad(input, output_padded_shape, input_tensor_start, value, ...)  (legacy shape form)
    # Distinguish them by whether the elements of the second argument are (start, end) pairs or ints.
    if len(pad_arg) == 0 or isinstance(pad_arg[0], (list, tuple)):
        padding = list(pad_arg)
        value, args, kwargs = ttnn.reflection.pop_argument("value", args, kwargs)
    else:
        padded_shape = list(pad_arg)
        input_tensor_start, args, kwargs = ttnn.reflection.pop_argument("input_tensor_start", args, kwargs)
        value, args, kwargs = ttnn.reflection.pop_argument("value", args, kwargs)
        padding = [
            (input_tensor_start[i], padded_shape[i] - input_shape[i] - input_tensor_start[i]) for i in range(rank)
        ]

    # A shorter padding list applies to the trailing dimensions; leading dims are left unpadded.
    if len(padding) > rank:
        raise RuntimeError("ttnn.pad: padding len can't be larger than input tensor rank")
    if len(padding) < rank:
        padding = [(0, 0)] * (rank - len(padding)) + list(padding)

    for start, end in padding:
        if start < 0 or end < 0:
            raise RuntimeError("ttnn.pad: padding must be non-negative")

    input_tensor = ttnn.to_torch(input_tensor)

    # Device-only kwargs (use_multicore, sub_core_grids, memory_config, queue_id) are irrelevant to the golden.
    return (input_tensor, padding, value), {}


def _golden_function(input_tensor, padding, *args, value=0, **_):
    import torch

    # Global comparison path passes raw ttnn.pad args; support both overloads:
    # (padding_pairs, value) and (output_padded_shape, input_tensor_start, value).
    if len(padding) == 0 or isinstance(padding[0], (list, tuple)):
        pad_pairs = list(padding)
        if args:
            value = args[0]
    else:
        input_tensor_start = args[0]
        if len(args) > 1:
            value = args[1]
        input_shape = list(input_tensor.shape)
        pad_pairs = [
            (input_tensor_start[i], padding[i] - input_shape[i] - input_tensor_start[i])
            for i in range(len(input_shape))
        ]

    rank = len(input_tensor.shape)
    if len(pad_pairs) < rank:
        pad_pairs = [(0, 0)] * (rank - len(pad_pairs)) + list(pad_pairs)

    torch_padding = []
    for dimension in reversed(pad_pairs):
        torch_padding.append(dimension[0])
        torch_padding.append(dimension[1])
    return torch.nn.functional.pad(input_tensor, pad=torch_padding, mode="constant", value=value)


def _postprocess_golden_function_outputs(output_tensor, args, kwargs):
    output_tensor = ttnn.decorators.default_postprocess_golden_function_outputs(output_tensor, args, kwargs)
    # Padding always turns the intended shape to the shape with tile padding. For simplicity of the operation
    output_tensor = ttnn.reshape(output_tensor, shape=output_tensor.padded_shape)
    return output_tensor


ttnn.attach_golden_function(
    ttnn.pad,
    golden_function=_golden_function,
    preprocess_golden_function_inputs=_preprocess_golden_function_inputs,
    postprocess_golden_function_outputs=_postprocess_golden_function_outputs,
)


def _golden_function(input_tensor: ttnn.Tensor, order: Tuple[int, ...], **_):
    if len(input_tensor.shape) != len(order):
        raise RuntimeError(
            "The number of dimensions in the tensor input does not match the length of the desired ordering"
        )

    return input_tensor.permute(order).contiguous().clone()


def _golden_function(input_tensor, dims, **_):
    import torch

    return torch.permute(input_tensor, dims)


ttnn.attach_golden_function(ttnn.permute, golden_function=_golden_function)


def _golden_function(tensors, dim=0, groups=1, **_):
    import torch

    def grouped_concat(activations, residuals, groups):
        """
        Concatenate activations and residuals with flexible interleaving based on groups.

        Args:
            activations (torch.Tensor): Activation tensor with shape [N, H, W, C].
            residuals (torch.Tensor): Residual tensor with shape [N, H, W, C].
            groups (int): Number of groups to split channels into.

        Returns:
            torch.Tensor: Concatenated tensor with interleaved groups.
        """

        assert (
            activations.shape[:-1] == residuals.shape[:-1]
        ), "Activations and residuals must have the same shape in all dims but -1"

        N, H, W, activation_channels = activations.shape
        assert activation_channels % groups == 0, "Channel count must be divisible by the number of groups"

        N, H, W, residual_channels = residuals.shape
        assert residual_channels % groups == 0, "Channel count must be divisible by the number of groups"

        act_groups = activations.view(N, H, W, groups, activation_channels // groups)
        res_groups = residuals.view(N, H, W, groups, residual_channels // groups)

        # Interleave activations and residuals along the channel axis
        interleaved = torch.cat([act_groups, res_groups], dim=-1)  # Shape: [N, H, W, groups, 2 * group_size]

        # Reshape to combine groups and channels correctly
        interleaved = interleaved.permute(0, 1, 2, 3, 4).reshape(N, H, W, residual_channels + activation_channels)

        return interleaved

    return grouped_concat(tensors[0], tensors[1], groups=groups) if groups > 1 else torch.concat(tensors, dim)


ttnn.attach_golden_function(
    ttnn.concat,
    golden_function=_golden_function,
)


def _golden_function(input, dim, index, *, sparse_grad=False, out=None, **_):
    import torch

    return torch.gather(input, dim, index.to(torch.int64), sparse_grad=sparse_grad, out=out)


ttnn.attach_golden_function(ttnn.gather, golden_function=_golden_function)


def _golden_function(tensor, repeats, dim=0, **_):
    import torch

    return torch.repeat_interleave(tensor, repeats, dim=dim)


ttnn.attach_golden_function(ttnn.repeat_interleave, golden_function=_golden_function)


def _golden_function(tensor, shape, **_):
    repeat_dims = [int(shape[i]) for i in range(len(shape))]
    # ttnn.repeat allows fewer repeat dims than tensor rank (pads leading 1s);
    # torch.repeat requires len(dims) >= tensor.dim(), so pad to match.
    if len(repeat_dims) < tensor.dim():
        repeat_dims = [1] * (tensor.dim() - len(repeat_dims)) + repeat_dims
    return tensor.repeat(*repeat_dims)


ttnn.attach_golden_function(ttnn.repeat, golden_function=_golden_function)


def _golden_function(input_tensor: ttnn.Tensor, scale_factor: Tuple[float, float], **_):
    import torch

    input_tensor = input_tensor.permute(0, 3, 1, 2)
    ret = torch.nn.functional.upsample(input_tensor, scale_factor=scale_factor)
    ret = ret.permute(0, 2, 3, 1)
    return ret


ttnn.attach_golden_function(
    ttnn.upsample,
    golden_function=_golden_function,
)


def _golden_function(input_tensor, slice_start=None, slice_end=None, slice_step=None, *args, **kwargs):
    if slice_start is None:
        slice_start = kwargs.get("starts")
    if slice_end is None:
        slice_end = kwargs.get("ends")
    if slice_step is None:
        slice_step = kwargs.get("steps", kwargs.get("slice_step"))
    if slice_step is None:
        slice_step = [1] * len(slice_start)
    slices = tuple(slice(int(s), int(e), int(st)) for s, e, st in zip(slice_start, slice_end, slice_step))
    return input_tensor[slices]


ttnn.attach_golden_function(ttnn.slice, golden_function=_golden_function)


def _golden_function(input_tensor, dim1, dim2, *args, **kwargs):
    import torch

    return torch.transpose(input_tensor, dim1, dim2)


ttnn.attach_golden_function(ttnn.transpose, golden_function=_golden_function)


def _golden_function(input_tensors, dim, *args, **kwargs):
    import torch

    return torch.stack(list(input_tensors), dim)


ttnn.attach_golden_function(ttnn.stack, golden_function=_golden_function)


def _golden_function(input_tensor, split_size, dim=0, *args, **kwargs):
    import torch

    return list(torch.split(input_tensor, split_size, dim=dim))


ttnn.attach_golden_function(ttnn.split, golden_function=_golden_function)


def _golden_function(input_tensor, chunks, dim, *args, **kwargs):
    import torch

    return list(torch.chunk(input_tensor, chunks, dim=dim))


ttnn.attach_golden_function(ttnn.chunk, golden_function=_golden_function)


def _golden_function(input_tensor, dim=None, *args, **kwargs):
    import torch

    if dim is None:
        return torch.squeeze(input_tensor)
    if isinstance(dim, (list, tuple)):
        return torch.squeeze(input_tensor, tuple(dim))
    return torch.squeeze(input_tensor, dim)


ttnn.attach_golden_function(ttnn.squeeze, golden_function=_golden_function)


def _golden_function(input_tensor, dim, *args, **kwargs):
    import torch

    return torch.unsqueeze(input_tensor, dim)


ttnn.attach_golden_function(ttnn.unsqueeze, golden_function=_golden_function)


def _golden_function(input_tensor, dim, start, length, *args, **kwargs):
    import torch

    return torch.narrow(input_tensor, dim, start, length)


ttnn.attach_golden_function(ttnn.narrow, golden_function=_golden_function)


def _golden_function(input_tensor, shape, *args, **kwargs):
    return input_tensor.reshape(list(shape))


ttnn.attach_golden_function(ttnn.view, golden_function=_golden_function)


def _golden_function(input_tensor, output_shape, *args, **kwargs):
    return input_tensor.expand(list(output_shape))


ttnn.attach_golden_function(ttnn.expand, golden_function=_golden_function)


def _golden_function(input_tensor, W, Z, Y, X, *args, **kwargs):
    return input_tensor.reshape(int(W), int(Z), int(Y), int(X))


ttnn.attach_golden_function(ttnn.reshape_on_device, golden_function=_golden_function)


def _golden_function(input_tensor, shifts, dim=None, *args, **kwargs):
    import torch

    shifts_arg = shifts if isinstance(shifts, int) else tuple(shifts)
    if dim is None:
        return torch.roll(input_tensor, shifts_arg)
    dims_arg = dim if isinstance(dim, int) else tuple(dim)
    return torch.roll(input_tensor, shifts_arg, dims=dims_arg)


ttnn.attach_golden_function(ttnn.roll, golden_function=_golden_function)


def _golden_function(input_tensor, *args, **kwargs):
    return input_tensor


ttnn.attach_golden_function(ttnn.move, golden_function=_golden_function)


def _golden_function(input, dim, index, src, *args, **kwargs):
    import torch

    return torch.scatter(input, dim, index.to(torch.int64), src)


ttnn.attach_golden_function(ttnn.scatter, golden_function=_golden_function)


def _golden_function(input, dim, index, src, *args, **kwargs):
    import torch

    return torch.scatter_add(input, dim, index.to(torch.int64), src)


ttnn.attach_golden_function(ttnn.scatter_add, golden_function=_golden_function)


def _golden_function(input, dim, index, value, *args, **kwargs):
    import torch

    return torch.index_fill(input, dim, index.to(torch.int64), value)


ttnn.attach_golden_function(ttnn.index_fill, golden_function=_golden_function)


def _golden_function(batch_id, input_tensor_a, input_tensor_b, *args, dim=0, **kwargs):
    import torch

    output_tensor = input_tensor_a.clone()
    for source_index, target_index in enumerate(batch_id.flatten().tolist()):
        output_tensor.index_copy_(
            dim, torch.tensor([int(target_index)]), input_tensor_b.index_select(dim, torch.tensor([source_index]))
        )
    return output_tensor


ttnn.attach_golden_function(ttnn.indexed_fill, golden_function=_golden_function)


def _golden_function(input, index, *args, **kwargs):
    import torch

    _, _, channels = input.shape
    gather_index = index.to(torch.int64).unsqueeze(-1).expand(-1, -1, channels)
    return torch.gather(input, 1, gather_index)


ttnn.attach_golden_function(ttnn.tosa_gather, golden_function=_golden_function)


def _golden_function(input, index, src, *args, **kwargs):
    import torch

    _, _, channels = input.shape
    scatter_index = index.to(torch.int64).unsqueeze(-1).expand(-1, -1, channels)
    return torch.scatter(input, 1, scatter_index, src)


ttnn.attach_golden_function(ttnn.tosa_scatter, golden_function=_golden_function)


def _golden_function(input_tensor, *args, skip_negative_entries=False, **kwargs):
    import torch

    if integer_golden.is_unsigned_dtype(input_tensor.dtype):
        # PyTorch cannot add UInt32 directly; widen and restore to preserve TTNN wraparound.
        input_tensor.copy_(integer_golden.binary(input_tensor, 1, torch.add))
    elif skip_negative_entries:
        keep = (input_tensor >= 0) & (input_tensor < (2**31 - 1))
        input_tensor.copy_(torch.where(keep, input_tensor + 1, input_tensor))
    else:
        input_tensor.add_(1)
    return input_tensor


ttnn.attach_golden_function(ttnn.plus_one, golden_function=_golden_function)


def _golden_function(buffer, shape, dtype, *args, **kwargs):
    import torch

    return torch.as_tensor(buffer, dtype=ttnn.ttnn_dtype_to_torch_dtype(dtype)).reshape(list(shape))


ttnn.attach_golden_function(ttnn.from_buffer, golden_function=_golden_function)


def _fold_nhwc(input, stride_h, stride_w, collapse_output):
    N, H, W, C = input.shape
    reshaped = input.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5)
    output_tensor = transposed.reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)
    if collapse_output:
        output_tensor = output_tensor.reshape(1, 1, N * (H // stride_h) * (W // stride_w), C * stride_h * stride_w)
    return output_tensor


def _parse_fold_padding(padding):
    if padding is None:
        return 0, 0, 0, 0, 0, 0
    if len(padding) == 2:
        return padding[0], padding[0], padding[1], padding[1], 0, 0
    if len(padding) == 4:
        return tuple(padding) + (0, 0)
    return tuple(padding)


def _golden_function_fold_transposed(input_tensor, stride_h, stride_w, padding, collapse_output):
    import torch

    # The transpose-based device path consumes NCHW and emits NHWC; six-element padding also aligns channels.
    pad_top, pad_bottom, pad_left, pad_right, pad_c_front, pad_c_back = _parse_fold_padding(padding)
    if pad_top or pad_bottom or pad_left or pad_right or pad_c_front or pad_c_back:
        input_tensor = torch.nn.functional.pad(
            input_tensor, (pad_left, pad_right, pad_top, pad_bottom, pad_c_front, pad_c_back), value=0.0
        )
    return _fold_nhwc(input_tensor.permute(0, 2, 3, 1), stride_h, stride_w, collapse_output)


def _golden_function(
    input,
    stride_h,
    stride_w,
    *args,
    padding=(0, 0),
    collapse_output=False,
    use_transpose_as_fold=False,
    **kwargs,
):
    import torch

    if use_transpose_as_fold:
        return _golden_function_fold_transposed(input, stride_h, stride_w, padding, collapse_output)

    pad_top, pad_bottom, pad_left, pad_right, pad_c_front, pad_c_back = _parse_fold_padding(padding)
    if pad_top or pad_bottom or pad_left or pad_right or pad_c_front or pad_c_back:
        input = torch.nn.functional.pad(
            input, (pad_c_front, pad_c_back, pad_left, pad_right, pad_top, pad_bottom), value=0.0
        )
    return _fold_nhwc(input, stride_h, stride_w, collapse_output)


ttnn.attach_golden_function(ttnn.fold, golden_function=_golden_function)


def _golden_function(input_tensor, *args, **kwargs):
    return input_tensor


ttnn.attach_golden_function(ttnn.untilize, golden_function=_golden_function)


def _golden_function(input_tensor, output_tensor_end, *args, **kwargs):
    slices = tuple(slice(0, int(end) + 1) for end in output_tensor_end)
    return input_tensor[slices]


ttnn.attach_golden_function(ttnn.untilize_with_unpadding, golden_function=_golden_function)


def _golden_function(input_tensor, output_tensor_shape, pad_value, *args, **kwargs):
    # output_tensor_shape describes physical tile padding; the logical output keeps the input shape.
    return input_tensor


ttnn.attach_golden_function(ttnn.tilize_with_val_padding, golden_function=_golden_function)


def _golden_function(input_tensor, *args, **kwargs):
    # Tile alignment is physical padding; the logical output keeps the input shape.
    return input_tensor


ttnn.attach_golden_function(ttnn.tilize_with_zero_padding, golden_function=_golden_function)


def _golden_function(input_tensor, fill_value, *args, **kwargs):
    return input_tensor


ttnn.attach_golden_function(ttnn.fill_implicit_tile_padding, golden_function=_golden_function)


def _golden_function(N, C, H, W, hOnes, wOnes, any, val_hi, val_lo, *args, **kwargs):
    import torch

    output_tensor = torch.full((N, C, H, W), float(val_lo), dtype=torch.float32)
    output_tensor[:, :, 0:hOnes, 0:wOnes] = float(val_hi)
    return output_tensor


ttnn.attach_golden_function(ttnn.fill_rm, golden_function=_golden_function)


def _golden_function(N, C, H, W, hOnes, wOnes, any, *args, **kwargs):
    import torch

    output_tensor = torch.zeros((N, C, H, W), dtype=torch.float32)
    output_tensor[:, :, 0:hOnes, 0:wOnes] = 1.0
    return output_tensor


ttnn.attach_golden_function(ttnn.fill_ones_rm, golden_function=_golden_function)


def _golden_function(cache_tensor, input_tensor, batch_idx, *args, update_idx=0, **kwargs):
    seq_len = input_tensor.shape[-2]
    cache_tensor[batch_idx : batch_idx + 1, :, update_idx : update_idx + seq_len, :] = input_tensor
    return cache_tensor


ttnn.attach_golden_function(ttnn.fill_cache, golden_function=_golden_function)


# Decode update: the input carries a single token per user with the user/batch axis in dim -2
# (shape [1, num_heads, batch, head_dim]); it is written into the cache at sequence position
# update_idx, reading from the padded batch starting at batch_offset.
def _golden_function(cache, input, update_idx, *args, batch_offset=0, **kwargs):
    num_users = cache.shape[0]
    for user in range(num_users):
        cache[user, :, update_idx, :] = input[0, :, batch_offset + user, :]
    return cache


ttnn.attach_golden_function(ttnn.update_cache, golden_function=_golden_function)


def _golden_function(cache, input, batch_index, *args, update_idx=0, **kwargs):
    seq_len = input.shape[-2]
    cache[batch_index : batch_index + 1, :, update_idx : update_idx + seq_len, :] = input
    return cache


ttnn.attach_golden_function(ttnn.kv_cache.fill_cache_for_user_, golden_function=_golden_function)


def _golden_function(cache, input, update_index, batch_offset=0, *args, **kwargs):
    num_users = cache.shape[0]
    for user in range(num_users):
        cache[user, :, update_index, :] = input[0, :, batch_offset + user, :]
    return cache


ttnn.attach_golden_function(ttnn.kv_cache.update_cache_for_token_, golden_function=_golden_function)


def _golden_function_copy(input_a, input_b, *_, _ttnn_global_golden=False, **__):
    # copy writes input_a into input_b in place and returns input_b; the value is input_a cast to input_b's dtype.
    result = input_a.to(input_b.dtype)
    if _ttnn_global_golden:
        input_b.copy_(result)
        return input_b
    return result


_golden_function_copy._ttnn_mutates_global_inputs = True
ttnn.attach_golden_function(ttnn.copy, golden_function=_golden_function_copy)


def _golden_function_sort(input_tensor, dim=-1, descending=False, stable=False, *_, **__):
    import torch

    values, indices = torch.sort(input_tensor, dim=dim, descending=descending, stable=stable)
    if not stable and values.shape[dim] > 1:
        adjacent_values = values.narrow(dim, 1, values.shape[dim] - 1)
        previous_values = values.narrow(dim, 0, values.shape[dim] - 1)
        adjacent_ties = adjacent_values == previous_values
        if bool(torch.any(adjacent_ties)):
            # Unstable sort may legally permute equal values differently; only its tied indices are non-unique.
            tie_mask = torch.zeros_like(indices, dtype=torch.bool)
            tie_mask.narrow(dim, 0, values.shape[dim] - 1).logical_or_(adjacent_ties)
            tie_mask.narrow(dim, 1, values.shape[dim] - 1).logical_or_(adjacent_ties)
            ttnn.decorators.set_golden_comparison_config(
                indices, method="allclose", scope="all", rtol=0.0, atol=0.0, mask=~tie_mask
            )
    return values, indices


ttnn.attach_golden_function(ttnn.sort, golden_function=_golden_function_sort)


def _golden_function_nonzero(input_tensor, *_, **__):
    import torch

    # The op returns (count, indices): count is [1, 1, 1, 8] with the non-zero count at [0, 0, 0, 0],
    # indices is [1, 1, 1, volume * 4] holding one flat (b, n, h, c) 4-tuple per non-zero element.
    # Both outputs are padded to a data-independent upper bound, so only the leading valid region is compared.
    if input_tensor.ndim != 4:
        raise ValueError(f"ttnn.nonzero golden requires rank-4 input, got rank {input_tensor.ndim}")

    coordinates = torch.nonzero(input_tensor, as_tuple=False)
    num_nonzero = coordinates.shape[0]

    count = torch.zeros((1, 1, 1, 8), dtype=torch.int64)
    count[0, 0, 0, 0] = num_nonzero
    count_mask = torch.zeros((1, 1, 1, 8), dtype=torch.bool)
    count_mask[0, 0, 0, 0] = True
    ttnn.decorators.set_golden_comparison_config(count, method="allclose", scope="all", mask=count_mask)

    flat_n = input_tensor.numel()
    indices = torch.zeros((1, 1, 1, flat_n * 4), dtype=torch.int64)
    if num_nonzero > 0:
        indices[0, 0, 0, : num_nonzero * 4] = coordinates.reshape(-1)
    indices_mask = torch.zeros((1, 1, 1, flat_n * 4), dtype=torch.bool)
    indices_mask[0, 0, 0, : num_nonzero * 4] = True
    ttnn.decorators.set_golden_comparison_config(indices, method="allclose", scope="all", mask=indices_mask)

    return [count, indices]


ttnn.attach_golden_function(ttnn.nonzero, golden_function=_golden_function_nonzero)


def _broadcast_quantization_arg(arg, input_tensor, axis):
    """Reshape a 1D per-channel scale/zero_point so it broadcasts against the input along `axis`."""
    import torch

    if not isinstance(arg, torch.Tensor) or axis is None:
        return arg
    rank = input_tensor.ndim
    axis_normalized = axis % rank
    broadcast_shape = [1] * rank
    broadcast_shape[axis_normalized] = arg.numel()
    return arg.reshape(broadcast_shape)


def _golden_function_quantize(input_tensor, scale, zero_point, *_, axis=None, dtype=None, **__):
    import torch

    # q = round(x / scale + zero_point); per-channel args broadcast along `axis`.
    scale = _broadcast_quantization_arg(scale, input_tensor, axis)
    zero_point = _broadcast_quantization_arg(zero_point, input_tensor, axis)
    output = torch.round(torch.div(input_tensor, scale) + zero_point)
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype) if dtype is not None else torch.int32
    return output.to(torch_dtype)


ttnn.attach_golden_function(ttnn.quantize, golden_function=_golden_function_quantize)


def _golden_function_dequantize(input_tensor, scale, zero_point, *_, axis=None, dtype=None, **__):
    import torch

    # x = (q - zero_point) * scale; per-channel args broadcast along `axis`.
    scale = _broadcast_quantization_arg(scale, input_tensor, axis)
    zero_point = _broadcast_quantization_arg(zero_point, input_tensor, axis)
    output = (input_tensor - zero_point) * scale
    if dtype is not None:
        output = output.to(ttnn.ttnn_dtype_to_torch_dtype(dtype))
    return output


ttnn.attach_golden_function(ttnn.dequantize, golden_function=_golden_function_dequantize)


def _golden_function_requantize(
    input_tensor, in_scale, in_zero_point, out_scale, out_zero_point, *_, axis=None, dtype=None, **__
):
    import torch

    # q' = round((x - in_zero_point) * in_scale / out_scale + out_zero_point).
    in_scale = _broadcast_quantization_arg(in_scale, input_tensor, axis)
    in_zero_point = _broadcast_quantization_arg(in_zero_point, input_tensor, axis)
    out_scale = _broadcast_quantization_arg(out_scale, input_tensor, axis)
    out_zero_point = _broadcast_quantization_arg(out_zero_point, input_tensor, axis)
    output = torch.round((input_tensor - in_zero_point) * (in_scale / out_scale) + out_zero_point)
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype) if dtype is not None else torch.int32
    return output.to(torch_dtype)


ttnn.attach_golden_function(ttnn.requantize, golden_function=_golden_function_requantize)


SliceParams = ttnn._ttnn.operations.data_movement.SliceParams
SliceInputs = ttnn._ttnn.operations.data_movement.SliceInputs
SliceDeviceOperation = ttnn._ttnn.operations.data_movement.SliceDeviceOperation
SliceTileProgramFactory = ttnn._ttnn.operations.data_movement.SliceTileProgramFactory

__all__ = []
