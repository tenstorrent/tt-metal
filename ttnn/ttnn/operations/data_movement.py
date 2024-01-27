# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import tt_lib as ttl

import ttnn.core as ttnn
from ttnn.decorators import register_operation


def _torch_pad(input_tensor: ttnn.Tensor, padding, value):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    torch_padding = []
    for dimension in reversed(padding):
        torch_padding.append(dimension[0])
        torch_padding.append(dimension[1])

    return torch.nn.functional.pad(input_tensor, pad=torch_padding, mode="constant", value=value)


@register_operation(torch_function=_torch_pad, name="ttnn.pad")
def pad(input_tensor: ttnn.Tensor, padding: Tuple[Tuple[int, int], ...], value: Union[int, float]) -> ttnn.Tensor:
    r"""

    pad(input_tensor: ttnn.Tensor, padding: Tuple[Tuple[int, int], ...], value: Union[int, float]) -> ttnn.Tensor

    Pad tensor with constant value.

    Padded shape is accumulated if ttnn.pad is called on a tensor with padding.

    Args:
        * :attr:`input_tensor`: input tensor
        * :attr:`padding`: padding to apply. Each element of padding should be a tuple of 2 integers, with the first integer specifying the number of values to add before the tensor and the second integer specifying the number of values to add after the tensor.
        * :attr:`value`: value to pad with

    """

    if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
        raise RuntimeError("pad expects input tensor to be on device!")

    output_tensor = _torch_pad(input_tensor, padding, value)
    output_tensor = ttnn.from_torch(
        output_tensor, dtype=input_tensor.dtype, device=input_tensor.device, layout=input_tensor.layout
    )

    # output_tensor = ttnn.reshape(output_tensor, input_tensor.shape + padding)
    return output_tensor


def _torch_permute(input_tensor: ttnn.Tensor, order: Tuple[int, ...], **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.permute(input_tensor, order).contiguous().clone()


@register_operation(torch_function=_torch_permute, name="ttnn.permute")
def permute(input_tensor: ttnn.Tensor, order: Tuple[int, ...]) -> ttnn.Tensor:
    r"""
    permute(input_tensor: ttnn.Tensor, order: Tuple[int, ...]) -> ttnn.Tensor

    Permutes :attr:`input_tensor` using :attr:`order`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`order`: the desired ordering of dimensions.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.permute(tensor, (0, 1, 3, 2))
        >>> print(output.shape)
        [1, 1, 32, 64]

    """

    if not isinstance(order, tuple):
        raise RuntimeError("order must be a tuple")

    if not ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
        RuntimeError("input_tensor must be on device!")

    if len(input_tensor.shape) != len(order):
        raise RuntimeError(
            "The number of dimensions in the tensor input does not match the length of the desired ordering"
        )

    on_device = ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE)
    device = input_tensor.device
    layout = input_tensor.layout
    dtype = input_tensor.dtype
    rank = len(input_tensor.shape)

    if len(input_tensor.shape) < 4:
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value
        adjusted_order_for_4D_tensor = order
        while len(adjusted_order_for_4D_tensor) < 4:
            adjusted_order_for_4D_tensor = (0,) + tuple(x + 1 for x in adjusted_order_for_4D_tensor)
        order = adjusted_order_for_4D_tensor

    if ttnn.has_padding(input_tensor):
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

    ttl_input_tensor = input_tensor.value

    if ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE) and len(input_tensor.shape) == 4:
        output_tensor = ttnn.Tensor(ttl.tensor.permute(ttl_input_tensor, order))
        output_tensor = ttnn.to_layout(output_tensor, layout)
        rank_should_be_updated = len(output_tensor.shape) > rank
        while rank_should_be_updated:
            prior_rank = len(output_tensor.shape)
            output_tensor = ttnn.squeeze(output_tensor)
            rank_should_be_updated = prior_rank != len(output_tensor.shape) and len(output_tensor.shape) > rank

        if on_device and not ttnn.has_storage_type_of(output_tensor, ttnn.DEVICE_STORAGE_TYPE):
            output_tensor = ttnn.to_device(output_tensor, device)
        return output_tensor
    else:

        def torch_permute(tensor, order):
            return tensor.permute(order).contiguous().clone()

        tensor = ttnn.to_torch(tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_permute, function_name="torch.permute")(tensor, order)
        tensor = ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)
        return tensor


def _torch_concat(tensors, dim=0, **_):
    import torch

    torch_tensors = [ttnn.to_torch(tensor) for tensor in tensors]
    return torch.concat(torch_tensors, dim)


@register_operation(torch_function=_torch_concat, name="ttnn.concat")
def concat(
    tensors: Union[ttnn.Tensor, List[ttnn.Tensor]],
    dim: int = 0,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    concat(tensors: Union[ttnn.Tensor, List[ttnn.Tensor]], dim: int = 0) -> ttnn.Tensor

    Concats :attr:`tensors` in the given :attr:`dim`.

    Args:
        * :attr:`tensors`: the tensors to be concatenated.
        * :attr:`dim`: the concatenating dimension.

    Example::

        >>> tensor = ttnn.concat(ttnn.from_torch(torch.zeros((1, 1, 64, 32), ttnn.from_torch(torch.zeros((1, 1, 64, 32), dim=3)), device)

        >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.zeros((1, 1, 64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.concat(tensor1, tensor2, dim=4)
        >>> print(output.shape)
        [1, 1, 32, 64]

    """

    if len(tensors) < 2:
        raise RuntimeError("You must have at least two tensors to concat!")

    rank = len(tensors[0].shape)
    original_dim = dim
    if dim < 0:
        dim = rank + dim
    if dim < 0 or dim >= rank:
        raise RuntimeError(
            f"ttnn: Dimension out of range: dim {original_dim} cannot be used for tensors of rank {rank}"
        )

    for input_tensor in tensors:
        if not ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            raise RuntimeError("ttnn: All tensors must be on device!")

    dtype = tensors[0].dtype
    device = tensors[0].device
    layout = tensors[0].layout
    rank = len(tensors[0].shape)

    first_tensor = tensors[0]
    first_tensor_shape = first_tensor.shape
    for tensor in tensors:
        shape = tensor.shape
        if len(shape) != len(first_tensor_shape) or any(
            shape[i] != first_tensor_shape[i] for i in range(len(shape)) if i != dim
        ):
            raise ValueError(
                "All dimensions must be the same size except for the dimension along which the contenation is taking place."
            )

    all_tensors_are_tile_layout_without_padding = not any(
        tensor.layout != ttnn.TILE_LAYOUT or ttnn.has_padding(tensor) for tensor in tensors
    )

    if rank < 4 and all_tensors_are_tile_layout_without_padding:
        any_tensor_has_padding = any(ttnn.has_padding(tensor) for tensor in tensors)

        def convert_to_ttl_tensor(tensor):
            if any_tensor_has_padding:
                tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
            return ttnn.unsqueeze_to_4D(tensor).value

        ttl_tensors = [convert_to_ttl_tensor(tensor) for tensor in tensors]
        dim = dim + 4 - rank
        output_tensor = ttnn.Tensor(ttl.tensor.concat(ttl_tensors, dim=dim, output_mem_config=memory_config))
        output_tensor = ttnn.to_layout(output_tensor, layout)
        rank_should_be_updated = len(output_tensor.shape) > rank
        while rank_should_be_updated:
            prior_rank = len(output_tensor.shape)
            output_tensor = ttnn.squeeze(output_tensor)
            rank_should_be_updated = prior_rank != len(output_tensor.shape) and len(output_tensor.shape) > rank
        return output_tensor
    else:
        output_tensor = _torch_concat(tensors, dim=dim)

        return ttnn.from_torch(output_tensor, dtype=dtype, device=device, layout=layout)


def _torch_split(input_tensor: ttnn.Tensor, split_size, dim):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.split(input_tensor, split_size, dim=dim)


@register_operation(torch_function=_torch_split, name="ttnn.split")
def split(input_tensor: ttnn.Tensor, split_size: int, dim: int) -> ttnn.Tensor:
    r"""
    split(input_tensor: ttnn.Tensor, split_size: int, dim: int) -> Tuple[ttnn.Tensor, ...]

    Split tensor into chunks of :attr:`split_size` along :attr:`dim`.

    Args:
        * :attr:`input_tensor`: input tensor.
        * :attr:`split_size`: size of a single chunk.
        * :attr:`dim`:  dimension along which to split the tensor.
    """

    if not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
        raise RuntimeError("pad expects input tensor to be on device!")

    output_tensors = _torch_split(input_tensor, split_size, dim)
    output_tensors = tuple(
        ttnn.from_torch(output_tensor, device=input_tensor.device, dtype=input_tensor.dtype, layout=input_tensor.layout)
        for output_tensor in output_tensors
    )
    return output_tensors


def _torch_repeat_interleave(tensor, repeats, dim=0, **_):
    import torch

    if isinstance(repeats, ttnn.Tensor):
        repeats = ttnn.to_torch(repeats)

    return torch.repeat_interleave(ttnn.to_torch(tensor), repeats, dim=dim)


@register_operation(torch_function=_torch_repeat_interleave, name="ttnn.repeat_interleave")
def repeat_interleave(tensor: ttnn.Tensor, repeats: Union[ttnn.Tensor, int], dim: int = 0) -> ttnn.Tensor:
    r"""
    repeat_interleave(tensors: ttnn.Tensor, repeats : Union[ttnn.Tensor,int], dim: int = 0) -> ttnn.Tensor

    Repeats elements of a :attr:`tensor` in the given :attr:`dim`.

    Args:
        * :attr:`tensors`: the tensors to be concatenated.
        * :attr:`repeats`: The number of repetitions for each element. repeats is broadcasted to fit the shape of the given axis.
        * :attr:`dim`: the concatenating dimension.

    Example::

        >>> tensor = ttnn.repeats(ttnn.from_torch(torch.tensor([[1, 2], [3, 4]]), 2, dim=3)), device)
        >>> print(tensor)
        tensor([[1, 2],
        [1, 2],
        [3, 4],
        [3, 4]])

    """

    if not isinstance(tensor, ttnn.Tensor):
        raise RuntimeError("ttnn: Expected tensor argument to be a ttnn.Tensor")

    if not ttnn.has_storage_type_of(tensor, ttl.tensor.StorageType.DEVICE):
        raise RuntimeError("ttnn: Tensor must be on device!")

    if not isinstance(repeats, int) and not isinstance(repeats, ttnn.Tensor):
        raise RuntimeError("ttnn: Expected repeat to either be an int or a ttnn.Tensor")

    # For now, don't require the repeat tensor to be on device.
    # if type(repeats) == type(tensor) and not ttnn.has_storage_type_of(repeats, ttl.tensor.StorageType.DEVICE):
    #     raise RuntimeError("Repeats tensor must be on device!")

    rank_of_tensor = len(tensor.shape)
    if dim >= rank_of_tensor:
        dimension_range = f"[{-rank_of_tensor}, {rank_of_tensor - 1}]"
        raise RuntimeError(
            f"ttnn: Dimension out of range (expected to be in range of {dimension_range}, but got {dim})"
        )

    def custom_numel(tensor):
        total_elements = 1
        for dimension in tensor.shape:
            total_elements *= dimension
        return total_elements

    if isinstance(repeats, ttnn.Tensor):
        if tensor.shape[dim] != custom_numel(repeats):
            raise RuntimeError("ttnn: repeats must have the same size as input along dim")
        elif len(repeats.shape) != 1:
            raise RuntimeError("ttnn: repeats must be 0-dim or 1-dim tensor")

    dtype = tensor.dtype
    device = tensor.device
    layout = tensor.layout
    rank = len(tensor.shape)
    if dtype == ttnn.bfloat16 and rank == 4 and dim != 2 and dim != 3:
        ttl_input_tensor = tensor.value
        output_tensor = ttnn.Tensor(ttl.tensor.repeat_interleave(ttl_input_tensor, repeats, dim=dim))
        *batch, _, _ = output_tensor.shape
        *_, h, w = tensor.shape
        *_, padded_h, padded_w = tensor.shape.padded()
        if dim == 2:
            *_, h, _ = output_tensor.shape
            *_, padded_h, _ = output_tensor.shape.padded()
        elif dim == 3:
            *_, _, w = output_tensor.shape
            *_, _, padded_w = output_tensor.shape.padded()
        output_tensor = ttnn.reshape(output_tensor, shape=ttnn.Shape(batch + [h, w], batch + [padded_h, padded_w]))
        return output_tensor
    else:

        def torch_repeat_interleave(tensor, repeats, dim=dim):
            return _torch_repeat_interleave(tensor, repeats, dim)

        output_tensor = ttl.tensor.decorate_external_operation(
            torch_repeat_interleave, function_name="torch_repeat_interleave"
        )(tensor, repeats, dim=dim)
        return ttnn.from_torch(output_tensor, device=device, dtype=dtype, layout=layout)


__all__ = []
