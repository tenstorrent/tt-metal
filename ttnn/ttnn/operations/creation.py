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
    output_tensor = ttnn.experimental.tensor.zeros_like(input_tensor, output_mem_config=memory_config)
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
    output_tensor = ttnn.experimental.tensor.ones_like(input_tensor, output_mem_config=memory_config)
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
    output_tensor = ttnn.experimental.tensor.full_like(input_tensor, fill_value, output_mem_config=memory_config)
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
    output_tensor = ttnn.experimental.tensor.zeros(
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

    output_tensor = ttnn.experimental.tensor.ones(
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

    output_tensor = ttnn.experimental.tensor.full(
        input_shape,
        fill_value=fill_value,
        device=device,
        data_type=dtype,
        layout=layout,
        output_mem_config=memory_config,
    )
    return output_tensor


def _is_int(value):
    return isinstance(value, (int))


def _torch_arange(start: int, end: int, step: int, **_):
    import torch

    return torch.arange(start, end, step)


def _arange_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    return True


@ttnn.register_operation(
    name="ttnn.arange",
    validate_input_tensors=_arange_validate_input_tensors,
    torch_function=_torch_arange,
)
def arange(
    start: int,
    end: int,
    step: int,
    device,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    arange(start: int, end: int, step: int, device, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new 1D tensor with the incremented values in size specified by inputs start, end and step.

    Args:
        * :attr:`start`
        * :attr:`end`
        * :attr:`step`
    """
    if not _is_int(start) or not _is_int(end) or not _is_int(step):
        raise TypeError("Expected three arguments to be a int")

    output_tensor = ttnn.experimental.tensor.arange(start, end, step, device, output_mem_config=memory_config)

    return output_tensor


def _torch_empty(input_shape: ttnn.Shape, **_):
    import torch

    input_shape = ttnn.from_device(input_shape)
    input_shape = ttnn.to_layout(input_shape, ttnn.ROW_MAJOR_LAYOUT)
    input_shape = ttnn.to_torch(input_shape)

    return torch.empty(input_shape)


def _empty_validate_input_tensors(operation_name, input_shape, *args, **kwargs):
    ...


@ttnn.register_operation(
    name="ttnn.empty",
    validate_input_tensors=_empty_validate_input_tensors,
    torch_function=_torch_empty,
)
def empty(
    input_shape: ttnn.Shape,
    device: ttnn.Device,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    r"""

    empty(input_shape: ttnn.Shape, device: ttnn.Device, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Returns a new empty tensor by taking input shape as reference.

    Args:
        * :attr:`input_shape`: the input shape for reference
    """

    output_tensor = ttnn.experimental.tensor.empty(input_shape, device=device, output_mem_config=memory_config)

    return output_tensor


__all__ = []
