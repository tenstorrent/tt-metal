# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import ttnn
import ttnn.decorators


def _preprocess_golden_function_inputs(args, kwargs):
    input_tensor, args, kwargs = ttnn.reflection.pop_argument("input_tensor", args, kwargs)
    padding, args, kwargs = ttnn.reflection.pop_argument("padding", args, kwargs)

    if len(padding) != len(input_tensor.shape):
        raise RuntimeError("ttnn.pad: padding must be the same length as the input tensor rank")

    for start, end in padding:
        if start < 0 or end < 0:
            raise RuntimeError("ttnn.pad: padding must be non-negative")

    pad_start = tuple(start for start, _ in padding)
    *_, pad_start_height, pad_start_width = pad_start
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if pad_start_height % ttnn.TILE_SIZE != 0 or pad_start_width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.pad: padding end must be a multiple of the tile size on height and width for a tensor in tile layout"
            )

    pad_end = tuple(end for _, end in padding)
    *_, pad_end_height, pad_end_width = pad_end
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        if pad_end_height % ttnn.TILE_SIZE != 0 or pad_end_width % ttnn.TILE_SIZE != 0:
            raise RuntimeError(
                "ttnn.pad: padding end must be a multiple of the tile size on height and width for a tensor in tile layout"
            )

    input_tensor = ttnn.to_torch(input_tensor)

    return (input_tensor, padding, *args), kwargs


def _golden_function(input_tensor: ttnn.Tensor, padding, value):
    import torch

    torch_padding = []
    for dimension in reversed(padding):
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
    return tensor.repeat(shape[0], shape[1], shape[2], shape[3])


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


def _return_or_copy_to_output(result, output_tensor):
    if output_tensor is not None:
        output_tensor.copy_(result)
        return output_tensor
    return result


def _as_int_list(values):
    if hasattr(values, "detach"):
        values = values.detach().reshape(-1).tolist()
    return [int(value) for value in values]


def _to_torch_dtype(dtype):
    if dtype is None:
        return None

    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.bfloat8_b: torch.bfloat16,
        ttnn.bfloat4_b: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
        ttnn.uint32: torch.uint32,
    }.get(dtype)


def _golden_broadcast(input_tensor, sender_coord, **_):
    return input_tensor.clone()


ttnn.attach_golden_function(ttnn.broadcast, golden_function=_golden_broadcast)


def _golden_chunk(input_tensor, chunks, dim, **_):
    import torch

    return list(torch.chunk(input_tensor, chunks, dim=dim))


ttnn.attach_golden_function(ttnn.chunk, golden_function=_golden_chunk)


def _golden_expand(input_tensor, output_shape, **_):
    return input_tensor.expand(tuple(output_shape)).clone()


ttnn.attach_golden_function(ttnn.expand, golden_function=_golden_expand)


def _golden_index_fill(input, dim, index, value, **_):
    import torch

    return input.index_fill(dim, index.to(torch.int64), value)


ttnn.attach_golden_function(ttnn.index_fill, golden_function=_golden_index_fill)


def _golden_indexed_fill(batch_id, input_tensor_a, input_tensor_b, dim=0, **_):
    import torch

    output = input_tensor_a.clone()
    indices = batch_id.reshape(-1).to(torch.int64)
    for source_index, destination_index in enumerate(indices.tolist()):
        source = input_tensor_b.select(dim, source_index).unsqueeze(dim)
        output.index_copy_(dim, indices.new_tensor([destination_index]), source)
    return output


ttnn.attach_golden_function(ttnn.indexed_fill, golden_function=_golden_indexed_fill)


def _golden_narrow(input_tensor, dim, start, length, **_):
    import torch

    return torch.narrow(input_tensor, dim, start, length)


ttnn.attach_golden_function(ttnn.narrow, golden_function=_golden_narrow)


def _golden_roll(input_tensor, shifts, dim=None, **_):
    import torch

    return torch.roll(input_tensor, shifts, dims=dim)


ttnn.attach_golden_function(ttnn.roll, golden_function=_golden_roll)


def _golden_scatter(input, dim, index, src, *, reduce=None, **_):
    import torch

    index = index.to(torch.int64)
    if reduce is None:
        return torch.scatter(input, dim, index, src)
    reduction = {
        "add": "sum",
        "multiply": "prod",
        "max": "amax",
        "amax": "amax",
        "min": "amin",
        "amin": "amin",
    }.get(reduce)
    if reduction is None:
        return torch.scatter(input, dim, index, src)
    return torch.scatter_reduce(input, dim, index, src, reduce=reduction, include_self=True)


ttnn.attach_golden_function(ttnn.scatter, golden_function=_golden_scatter)


def _golden_scatter_add(input, dim, index, src, **_):
    import torch

    return torch.scatter_add(input, dim, index.to(torch.int64), src)


ttnn.attach_golden_function(ttnn.scatter_add, golden_function=_golden_scatter_add)


def _golden_slice(
    input_tensor,
    starts=None,
    ends=None,
    slice_step=None,
    *,
    slice_start=None,
    slice_end=None,
    steps=None,
    output_tensor=None,
    **_,
):
    starts = starts if starts is not None else slice_start
    ends = ends if ends is not None else slice_end
    steps = steps if steps is not None else slice_step
    starts = _as_int_list(starts)
    ends = _as_int_list(ends)
    steps = [1] * len(starts) if steps is None else _as_int_list(steps)
    result = input_tensor[tuple(slice(start, end, step) for start, end, step in zip(starts, ends, steps))]
    return _return_or_copy_to_output(result, output_tensor)


ttnn.attach_golden_function(ttnn.slice, golden_function=_golden_slice)


def _golden_split(input_tensor, split_size, dim=0, **_):
    import torch

    return list(torch.split(input_tensor, split_size, dim=dim))


ttnn.attach_golden_function(ttnn.split, golden_function=_golden_split)


def _golden_squeeze(input_tensor, dim=None, **_):
    import torch

    if dim is None:
        return torch.squeeze(input_tensor)
    if isinstance(dim, list):
        dim = tuple(dim)
    return torch.squeeze(input_tensor, dim)


ttnn.attach_golden_function(ttnn.squeeze, golden_function=_golden_squeeze)


def _golden_stack(input_tensors, dim, **_):
    import torch

    return torch.stack(input_tensors, dim=dim)


ttnn.attach_golden_function(ttnn.stack, golden_function=_golden_stack)


def _golden_tilize_with_val_padding(input_tensor, output_tensor_shape, pad_value, dtype=None, **_):
    import torch

    output_shape = tuple(int(value) for value in output_tensor_shape)
    output_dtype = _to_torch_dtype(dtype) or input_tensor.dtype
    output = torch.full(output_shape, pad_value, dtype=output_dtype, device=input_tensor.device)
    output[tuple(slice(0, size) for size in input_tensor.shape)] = input_tensor.to(output_dtype)
    return output


ttnn.attach_golden_function(ttnn.tilize_with_val_padding, golden_function=_golden_tilize_with_val_padding)


def _golden_tilize_with_zero_padding(input_tensor, output_dtype=None, **_):
    # Tile padding is physical padding; the operation's logical tensor is unchanged.
    torch_dtype = _to_torch_dtype(output_dtype)
    return input_tensor.clone() if torch_dtype is None else input_tensor.to(torch_dtype)


ttnn.attach_golden_function(ttnn.tilize_with_zero_padding, golden_function=_golden_tilize_with_zero_padding)


def _golden_tosa_gather(input, index, **_):
    import torch

    expanded_index = index.to(torch.int64).unsqueeze(-1).expand(*index.shape, input.shape[-1])
    return torch.gather(input, 1, expanded_index)


ttnn.attach_golden_function(ttnn.tosa_gather, golden_function=_golden_tosa_gather)


def _golden_tosa_scatter(input, index, src, **_):
    import torch

    expanded_index = index.to(torch.int64).unsqueeze(-1).expand(*index.shape, input.shape[-1])
    return torch.scatter(input, 1, expanded_index, src)


ttnn.attach_golden_function(ttnn.tosa_scatter, golden_function=_golden_tosa_scatter)


def _golden_transpose(input_tensor, dim1, dim2, **_):
    import torch

    return torch.transpose(input_tensor, dim1, dim2)


ttnn.attach_golden_function(ttnn.transpose, golden_function=_golden_transpose)


def _golden_unsqueeze(input_tensor, dim, **_):
    import torch

    return torch.unsqueeze(input_tensor, dim)


ttnn.attach_golden_function(ttnn.unsqueeze, golden_function=_golden_unsqueeze)


def _golden_untilize(input_tensor, **_):
    return input_tensor.clone()


ttnn.attach_golden_function(ttnn.untilize, golden_function=_golden_untilize)


def _golden_untilize_with_unpadding(input_tensor, output_tensor_end, **_):
    ends = _as_int_list(output_tensor_end)
    return input_tensor[tuple(slice(0, end + 1) for end in ends)]


ttnn.attach_golden_function(ttnn.untilize_with_unpadding, golden_function=_golden_untilize_with_unpadding)

SliceParams = ttnn._ttnn.operations.data_movement.SliceParams
SliceInputs = ttnn._ttnn.operations.data_movement.SliceInputs
SliceDeviceOperation = ttnn._ttnn.operations.data_movement.SliceDeviceOperation
SliceTileProgramFactory = ttnn._ttnn.operations.data_movement.SliceTileProgramFactory

__all__ = []
