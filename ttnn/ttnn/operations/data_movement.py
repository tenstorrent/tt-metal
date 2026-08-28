# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import ttnn
import ttnn.decorators


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


# upsample not available in this build — golden function removed.


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

    if skip_negative_entries:
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


SliceParams = ttnn._ttnn.operations.data_movement.SliceParams
SliceInputs = ttnn._ttnn.operations.data_movement.SliceInputs
SliceDeviceOperation = ttnn._ttnn.operations.data_movement.SliceDeviceOperation
SliceTileProgramFactory = ttnn._ttnn.operations.data_movement.SliceTileProgramFactory

__all__ = []
