# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import tt_lib as ttl

import ttnn.tensor as ttnn
from ttnn.decorators import decorate_operation


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


@decorate_operation(torch_function=_torch_pad)
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


@decorate_operation(torch_function=_torch_permute)
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

    ttl_input_tensor = input_tensor.value

    if len(input_tensor.shape) != len(order):
        raise RuntimeError(
            "The number of dimensions in the tensor input does not match the length of the desired ordering"
        )

    original_shape = tuple(input_tensor.shape)
    original_shape_padded = tuple(input_tensor.shape.padded())
    desired_shape = ttnn.Shape(
        list([original_shape[i] for i in order]), list([original_shape_padded[i] for i in order])
    )
    if ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE) and len(input_tensor.shape) == 4:
        output_tensor = ttnn.Tensor(ttl.tensor.permute(ttl_input_tensor, order))
        # permute is not currently keeping the original padding
        return ttnn.reshape(output_tensor, desired_shape)
    elif len(input_tensor.shape) < 4:
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value
        adjusted_order_for_4D_tensor = order
        while len(adjusted_order_for_4D_tensor) < 4:
            adjusted_order_for_4D_tensor = (0,) + tuple(x + 1 for x in adjusted_order_for_4D_tensor)
        output_tensor = ttnn.Tensor(ttl.tensor.permute(ttl_input_tensor, adjusted_order_for_4D_tensor))
        return ttnn.reshape(output_tensor, desired_shape)
    else:

        def torch_permute(tensor, order):
            return tensor.permute(order).contiguous().clone()

        tensor = ttnn.to_torch(tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_permute, function_name="torch.permute")(tensor, order)
        tensor = ttnn.from_torch(
            tensor, dtype=input_tensor.dtype, layout=input_tensor.layout, device=input_tensor.device
        )
        return tensor


def _torch_concat(tensors, dim=0, **_):
    import torch

    torch_tensors = [ttnn.to_torch(tensor) for tensor in tensors]
    return torch.concat(torch_tensors, dim)


# @decorate_operation(torch_function=_torch_concat)
def concat(tensors: Union[ttnn.Tensor, List[ttnn.Tensor]], dim: int = 0) -> ttnn.Tensor:
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

    for input_tensor in tensors:
        if not ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            raise RuntimeError("All tensors must be on device!")

    dtype = tensors[0].dtype
    device = tensors[0].device
    layout = tensors[0].layout

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

    output_tensor = _torch_concat(tensors, dim=0)

    return ttnn.from_torch(output_tensor, dtype=dtype, device=device, layout=layout)


def _torch_split(input_tensor: ttnn.Tensor, split_size, dim):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.split(input_tensor, split_size, dim=dim)


@decorate_operation(torch_function=_torch_split)
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


# @decorate_operation(torch_function=_torch_concat)
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
        raise RuntimeError("Expected tensor argument to be a ttnn.Tensor")

    if not ttnn.has_storage_type_of(tensor, ttl.tensor.StorageType.DEVICE):
        raise RuntimeError("Tensor must be on device!")

    if not isinstance(repeats, int) and not isinstance(repeats, ttnn.Tensor):
        raise RuntimeError("Expected repeat to either be an int or a ttnn.Tensor")

    if type(repeats) == tensor and not ttnn.has_storage_type_of(repeats, ttl.tensor.StorageType.DEVICE):
        raise RuntimeError("Repeats tensor must be on device!")

    dtype = tensor.dtype
    device = tensor.device
    layout = tensor.layout

    output_tensor = _torch_repeat_interleave(tensor, repeats, dim=0)

    return ttnn.from_torch(output_tensor, dtype=dtype, device=device, layout=layout)


__all__ = ["pad", "reshape", "permute", "concat", "split", "repeat_interleave"]
