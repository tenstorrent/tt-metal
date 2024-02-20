# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Union


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

    input_tensor = ttnn.to_torch(input_tensor)
    return torch.zeros_like(input_tensor)


@ttnn.register_operation(
    name="ttnn.zeros_like",
    validate_input_tensors=_zeros_like_validate_input_tensors,
    torch_function=_torch_zeros_like,
)
def zeros_like(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    zeros_like(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new tensor filled with zero by taking input tensor shape as reference.

    Args:
        * :attr:`input_tensor`: the input tensor for reference shape

    Keyword Args:
        * :attr:`memory_config`: the memory configuration for the output tensor
    """

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    output_tensor = ttnn.ttl.tensor.zeros_like(input_tensor, output_mem_config=memory_config)
    output_tensor = ttnn.reshape(output_tensor, original_shape)
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

    input_tensor = ttnn.to_torch(input_tensor)
    return torch.ones_like(input_tensor)


@ttnn.register_operation(
    name="ttnn.ones_like",
    validate_input_tensors=_ones_like_validate_input_tensors,
    torch_function=_torch_ones_like,
)
def ones_like(
    input_tensor: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""
    ones_like(input_tensor: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new tensor filled with one by taking input tensor shape as reference.

    Args:
        * :attr:`input_tensor`: the input tensor for reference shape

    Keyword Args:
        * :attr:`memory_config`: the memory configuration for the output tensor
    """

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    output_tensor = ttnn.ttl.tensor.ones_like(input_tensor, output_mem_config=memory_config)
    output_tensor = ttnn.reshape(output_tensor, original_shape)
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


def _torch_full_like(input_tensor: ttnn.Tensor, *, fill_value: float, **_):
    import torch

    input_tensor = ttnn.to_torch(input_tensor)
    return torch.full_like(input_tensor, fill_value)


@ttnn.register_operation(
    name="ttnn.full_like",
    validate_input_tensors=_full_like_validate_input_tensors,
    torch_function=_torch_full_like,
)
def full_like(
    input_tensor: ttnn.Tensor,
    *,
    fill_value: float,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    full_like(input_tensor: ttnn.Tensor, *, fill_value: float, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new tensor filled with scalar value by taking input tensor shape as reference.

    Args:
        * :attr:`input_tensor`: the input tensor for reference shape

    Keyword Args:
        * :attr:`fill_value`: the value to be filled
        * :attr:`memory_config`: the memory configuration for the output tensor
    """

    original_shape = input_tensor.shape
    input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
    output_tensor = ttnn.ttl.tensor.full_like(input_tensor, fill_value, output_mem_config=memory_config)
    output_tensor = ttnn.reshape(output_tensor, original_shape)
    return output_tensor


def _torch_zeros(input_shape: ttnn.Shape, **_):
    import torch

    input_shape = ttnn.to_torch(input_shape)
    return torch.zeros(input_shape)


def _zeros_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    ...


@ttnn.register_operation(
    name="ttnn.zeros",
    validate_input_tensors=_zeros_validate_input_tensors,
    torch_function=_torch_zeros,
)
def zeros(
    input_shape: ttnn.Shape,
    *,
    device: ttnn.Device,
    dtype: Union[ttnn.DataType, str] = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    zeros(input_shape: ttnn.Shape, *, device: ttnn.Device, dtype: Union[ttnn.DataType, str] = ttnn.bfloat16, layout: ttnn.Layout = ttnn.TILE_LAYOUT, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new tensor filled with zeros by taking input tensor shape as reference.

    Args:
        * :attr:`input_shape`: the input shape for reference

    Keyword Args:
        * :attr:`device`: the device for the output tensor
        * :attr:`dtype`: the data type for the output tensor
        * :attr:`layout`: the layout for the output tensor
        * :attr:`memory_config`: the memory configuration for the output tensor
    """
    output_tensor = ttnn.ttl.tensor.zeros(
        input_shape, data_type=dtype, layout=layout, device=device, output_mem_config=memory_config
    )
    return output_tensor


def _torch_ones(input_shape: ttnn.Shape, **_):
    import torch

    input_shape = ttnn.to_torch(input_shape)
    return torch.ones(input_shape)


def _ones_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    ...


@ttnn.register_operation(
    name="ttnn.ones",
    validate_input_tensors=_ones_validate_input_tensors,
    torch_function=_torch_ones,
)
def ones(
    input_shape: ttnn.Shape,
    *,
    device: ttnn.Device,
    dtype: Union[ttnn.DataType, str] = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    ones(input_shape: ttnn.Shape, *, device: ttnn.Device, dtype: Union[ttnn.DataType, str] = ttnn.bfloat16, layout: ttnn.Layout = ttnn.TILE_LAYOUT, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new tensor filled with ones by taking input tensor shape as reference.

    Args:
        * :attr:`input_shape`: the input shape for reference

    Keyword Args:
        * :attr:`device`: the device for the output tensor
        * :attr:`dtype`: the data type for the output tensor
        * :attr:`layout`: the layout for the output tensor
        * :attr:`memory_config`: the memory configuration for the output tensor
    """

    output_tensor = ttnn.ttl.tensor.ones(
        input_shape, data_type=dtype, layout=layout, device=device, output_mem_config=memory_config
    )
    return output_tensor


def _torch_full(input_shape: ttnn.Shape, fill_value: float, **_):
    import torch

    input_shape = ttnn.to_torch(input_shape)
    return torch.full(input_shape, fill_value=fill_value)


def _full_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    ...


@ttnn.register_operation(
    name="ttnn.full",
    validate_input_tensors=_full_validate_input_tensors,
    torch_function=_torch_full,
)
def full(
    input_shape: ttnn.Shape,
    *,
    device: ttnn.Device,
    dtype: Union[ttnn.DataType, str] = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    fill_value: float,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    full(input_shape: ttnn.Shape, *, device: ttnn.Device, dtype: Union[ttnn.DataType, str] = ttnn.bfloat16, layout: ttnn.Layout = ttnn.TILE_LAYOUT, fill_value: float, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new tensor filled with fill_value by taking input tensor shape as reference.

    Args:
        * :attr:`input_shape`: the input shape for reference

    Keyword Args:
        * :attr:`device`: the device for the output tensor
        * :attr:`dtype`: the data type for the output tensor
        * :attr:`layout`: the layout for the output tensor
        * :attr:`fill_value`: the value to be filled
        * :attr:`memory_config`: the memory configuration for the output tensor

    """

    output_tensor = ttnn.ttl.tensor.full(
        input_shape,
        fill_value=fill_value,
        device=device,
        data_type=dtype,
        layout=layout,
        output_mem_config=memory_config,
    )
    return output_tensor


__all__ = []
