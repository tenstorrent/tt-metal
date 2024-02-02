# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union


import tt_lib as ttl

import ttnn


def _zeros_like_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def _torch_zeros_like(input_tensor: ttnn.Tensor, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.zeros_like(input_tensor)


@ttnn.register_operation(
    name="ttnn.zeros_like",
    validate_input_tensors=_zeros_like_validate_input_tensors,
    torch_function=_torch_zeros_like,
)
def zeros_like(
    input_tensor: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    Returns a new tensor filled with zero by taking input tensor shape as reference.

    Args:
        * :attr:`input_tensor`: the input tensor for reference shape
    """

    ttl_input_tensor = input_tensor.value
    output_tensor = ttl.tensor.zeros_like(ttl_input_tensor, output_mem_config=memory_config)
    output_tensor = ttnn.Tensor(output_tensor)

    return output_tensor


def _ones_like_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def _torch_ones_like(input_tensor: ttnn.Tensor, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.ones_like(input_tensor)


@ttnn.register_operation(
    name="ttnn.ones_like",
    validate_input_tensors=_ones_like_validate_input_tensors,
    torch_function=_torch_ones_like,
)
def ones_like(
    input_tensor: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    Returns a new tensor filled with one by taking input tensor shape as reference.

    Args:
        * :attr:`input_tensor`: the input tensor for reference shape
    """

    ttl_input_tensor = input_tensor.value
    output_tensor = ttl.tensor.ones_like(ttl_input_tensor, output_mem_config=memory_config)
    output_tensor = ttnn.Tensor(output_tensor)

    return output_tensor


def _full_like_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
        layouts=(ttnn.ROW_MAJOR_LAYOUT,),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


def _torch_full_like(input_tensor: ttnn.Tensor, fill_value: float, **_):
    import torch

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)

    return torch.full_like(input_tensor, fill_value)


@ttnn.register_operation(
    name="ttnn.full_like",
    validate_input_tensors=_full_like_validate_input_tensors,
    torch_function=_torch_full_like,
)
def full_like(
    input_tensor: ttnn.Tensor,
    fill_value: float,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    Returns a new tensor filled with scalar value by taking input tensor shape as reference.

    Args:
        * :attr:`input_tensor`: the input tensor for reference shape
        * :attr:`fill_value`: the value to be filled
    """

    ttl_input_tensor = input_tensor.value
    output_tensor = ttl.tensor.full_like(ttl_input_tensor, fill_value, output_mem_config=memory_config)
    output_tensor = ttnn.Tensor(output_tensor)

    return output_tensor


def _torch_zeros(input_shape: ttnn.Shape, **_):
    import torch

    input_shape = ttnn.from_device(input_shape)
    input_shape = ttnn.to_layout(input_shape, ttnn.ROW_MAJOR_LAYOUT)
    input_shape = ttnn.to_torch(input_shape)

    return torch.zeros(input_shape)


def _zeros_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    if len(input_shape) == 4:
        return True


@ttnn.register_operation(
    name="ttnn.zeros",
    validate_input_tensors=_zeros_validate_input_tensors,
    torch_function=_torch_zeros,
)
def zeros(
    input_shape: ttnn.Shape,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    Returns a new tensor filled with zeros by taking input tensor shape as reference.

    Args:
        * :attr:`input_shape`: the input shape for reference
    """
    output_tensor = ttl.tensor.zeros(input_shape, output_mem_config=memory_config)
    output_tensor = ttnn.Tensor(output_tensor)

    return output_tensor


def _torch_ones(input_shape: ttnn.Shape, **_):
    import torch

    input_shape = ttnn.from_device(input_shape)
    input_shape = ttnn.to_layout(input_shape, ttnn.ROW_MAJOR_LAYOUT)
    input_shape = ttnn.to_torch(input_shape)

    return torch.ones(input_shape)


def _ones_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    if len(input_shape) == 4:
        return True


@ttnn.register_operation(
    name="ttnn.ones",
    validate_input_tensors=_ones_validate_input_tensors,
    torch_function=_torch_ones,
)
def ones(
    input_shape: ttnn.Shape,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    Returns a new tensor filled with ones by taking input tensor shape as reference.

    Args:
        * :attr:`input_shape`: the input shape for reference
    """
    output_tensor = ttl.tensor.ones(input_shape, output_mem_config=memory_config)
    output_tensor = ttnn.Tensor(output_tensor)

    return output_tensor


def _torch_full(input_shape: ttnn.Shape, fill_value: float, **_):
    import torch

    input_shape = ttnn.from_device(input_shape)
    input_shape = ttnn.to_layout(input_shape, ttnn.ROW_MAJOR_LAYOUT)
    input_shape = ttnn.to_torch(input_shape)

    return torch.full(input_shape, fill_value=fill_value)


def _full_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    if len(input_shape) == 4:
        return True


@ttnn.register_operation(
    name="ttnn.full",
    validate_input_tensors=_full_validate_input_tensors,
    torch_function=_torch_full,
)
def full(
    input_shape: ttnn.Shape,
    fill_value: float,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    Returns a new tensor filled with fill_value by taking input tensor shape as reference.

    Args:
        * :attr:`input_shape`: the input shape for reference
        * :attr:`fill_value`: the value to be filled
    """
    output_tensor = ttl.tensor.full(input_shape, fill_value=fill_value, output_mem_config=memory_config)
    output_tensor = ttnn.Tensor(output_tensor)

    return output_tensor


__all__ = []
