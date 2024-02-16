# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pathlib
from typing import Union, Tuple, Optional

from loguru import logger

import tt_lib as ttl

import ttnn


def _getitem_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32, ttnn.float32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.Tensor.__getitem__", validate_input_tensors=_getitem_validate_input_tensors)
def __getitem__(input_tensor: ttnn.Tensor, slices) -> ttnn.Tensor:
    input_rank = len(input_tensor.shape)
    input_dtype = input_tensor.dtype
    input_layout = input_tensor.layout
    if ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
        input_device = input_tensor.device
    else:
        input_device = None

    if isinstance(slices, slice):
        slices = (slices,)

    if isinstance(slices, tuple):
        if len(slices) > input_rank:
            raise RuntimeError(f"Too many slices for tensor of rank {input_rank}")

    def are_valid_device_slices(slices):
        if not isinstance(slices, tuple):
            return False

        if len(slices) != input_rank:
            return False

        def is_valid_slice(_slice, multiple_of=1):
            if not isinstance(_slice, slice):
                return False
            if _slice.start is not None and _slice.start % multiple_of != 0:
                return False
            if _slice.stop is not None and _slice.stop % multiple_of != 0:
                return False
            if _slice.step is not None and _slice.stop != 1:
                return False
            if _slice.start is not None and _slice.stop is not None:
                if (_slice.stop - _slice.start) % multiple_of != 0:
                    return False
            return True

        if len(slices) < 2:
            return False

        *batch_slices, height_slice, width_slice = slices

        for batch_slice in batch_slices:
            if not is_valid_slice(batch_slice):
                return False

        if not is_valid_slice(height_slice, ttnn.TILE_SIZE):
            return False
        if not is_valid_slice(width_slice, ttnn.TILE_SIZE):
            return False

        return True

    # TODO(arakhmati): add support for running ROW_MAJOR_LAYOUT slicing on device. The underlying op already supports it.
    if (
        ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE)
        and input_layout == ttnn.TILE_LAYOUT
        and input_rank <= 4
        and are_valid_device_slices(slices)
    ):
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

        while len(slices) != 4:
            slices = (slice(None, None, None),) + slices
        slice_start = [_slice.start if _slice.start is not None else 0 for _slice in slices]
        slice_end = [
            (_slice.stop if _slice.stop is not None else input_tensor.shape[index]) - 1
            for index, _slice in enumerate(slices)
        ]
        ttl_input_tensor = input_tensor.value
        ttl_output_tensor = ttl.tensor.unpad(ttl_input_tensor, slice_start, slice_end)

        output = ttnn.Tensor(ttl_output_tensor)
        output_shape = tuple(output.shape)[-input_rank:]
        return ttnn.reshape(output, shape=output_shape)
    """
    elif not ttnn.has_storage_type_of(input_tensor, ttnn.DEVICE_STORAGE_TYPE):
        logger.debug(
            "ttnn.Tensor.__getitem__: using torch because the tensor is on device and the slicing using unpad is not supported!"
        )
    elif input_layout != ttnn.TILE_LAYOUT:
        logger.debug(f"ttnn.Tensor.__getitem__: using torch because input layout {input_layout} is not TILE_LAYOUT!")
    elif input_rank > 4:
        logger.debug(f"ttnn.Tensor.__getitem__: using torch because input rank {input_rank} is greater than 4!")
    elif not are_valid_device_slices(slices):
        logger.debug(f"ttnn.Tensor.__getitem__: using torch because slices {slices} are not valid device slices!")
    """

    def torch_getitem(tensor, slices):
        return tensor[slices]

    output_tensor = input_tensor
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = ttl.tensor.decorate_external_operation(torch_getitem, function_name="torch.Tensor.__getitem__")(
        output_tensor, slices
    )
    if output_tensor.ndim == 0:
        raise RuntimeError("ttnn.Tensor.__getitem__: returned a scalar!")
    output_tensor = ttnn.from_torch(output_tensor, dtype=input_dtype, layout=input_layout, device=input_device)
    return output_tensor


ttnn.Tensor.__getitem__ = __getitem__


def _torch_reshape(input_tensor: ttnn.Tensor, shape: Union[ttnn.Shape, Tuple[int, ...]], **_):
    import torch

    input_tensor = to_torch(input_tensor)

    if isinstance(shape, ttnn.Shape):
        shape = tuple(shape)

    return torch.reshape(input_tensor, shape).contiguous().clone()


def _reshape_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32, ttnn.float32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(
    name="ttnn.reshape", validate_input_tensors=_reshape_validate_input_tensors, torch_function=_torch_reshape
)
def reshape(input_tensor: ttnn.Tensor, shape: Union[ttnn.Shape, Tuple[int, ...]]) -> ttnn.Tensor:
    r"""
    reshape(input_tensor: ttnn.Tensor, shape: Union[Shape, Tuple[int, ...]]) -> ttnn.Tensor

    Reshape :attr:`input_tensor` into :attr:`shape`.

    Args:
        * :attr:`input_tensor`: the input tensor
        * :attr:`shape`: the desired shape.

    Example::

        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.zeros((64, 32), dtype=torch.bfloat16)), device)
        >>> output = ttnn.reshape(tensor, (32, 64))
        >>> print(output.shape)
        ttnn.Shape([32, 64])

    """

    if isinstance(shape, tuple):
        if not (0 <= shape.count(-1) <= 1):
            raise RuntimeError("Shape cannot have more than 1 elements that is set to -1!")

        volume = math.prod(input_tensor.shape)
        new_volume = math.prod(shape)
        if new_volume < 0:
            index_of_negative_1 = shape.index(-1)
            shape = list(shape)
            shape[index_of_negative_1] = volume // (-new_volume)
            shape = tuple(shape)
        shape = ttnn.Shape(shape)

    if not isinstance(shape, ttnn.Shape):
        raise RuntimeError("Shape must be of type Shape")

    if input_tensor.shape == shape and list(input_tensor.shape) == list(shape):
        return input_tensor

    def ttnn_reshape(tensor: ttnn.Tensor, shape: ttnn.Shape) -> ttnn.Tensor:
        ttl_input_tensor = tensor.value
        return ttnn.Tensor(ttl_input_tensor.reshape(shape.value))

    layout = input_tensor.layout
    ttnn_reshape = ttl.tensor.decorate_external_operation(ttnn_reshape, function_name="ttnn.reshape")

    if input_tensor.is_contiguous():
        if ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            # Page size depends on the width, so only modify the shape if the width is the same
            if input_tensor.shape[-1] == shape[-1]:
                return ttnn_reshape(input_tensor, shape)
        else:
            return ttnn_reshape(input_tensor, shape)

    if input_tensor.layout == ttnn.TILE_LAYOUT:
        *_, new_height, new_width = tuple(shape.with_tile_padding())
        if new_height % ttnn.TILE_SIZE == 0 and new_width % ttnn.TILE_SIZE == 0:
            return ttnn_reshape(input_tensor, shape)
        else:
            raise RuntimeError(
                "Unable to reshape a tensor in TILE_LAYOUT to non-tile height and width! Please convert the tensor to ROW_MAJOR_LAYOUT first."
            )

    if (
        ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE)
        and len(input_tensor.shape) == 4
        and len(shape) == 4
        and input_tensor.dtype == ttnn.bfloat16
    ):
        ttl_input_tensor = input_tensor.value
        w, z, y, x = shape
        ttl_output_tensor = ttl.tensor.reshape(ttl_input_tensor, w, z, y, x)
        output_tensor = ttnn.Tensor(ttl_output_tensor)
        output_tensor = ttnn_reshape(output_tensor, shape)
        # Unable to handle 5D tensors!  See ttnn_optimized_functional_whisper.
        # output_tensor = _to_layout(output_tensor, layout)
        return output_tensor
    else:

        def torch_reshape(tensor, shape):
            return tensor.reshape(tuple(shape.with_tile_padding())).contiguous().clone()

        if ttnn.has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            device = input_tensor.device
        else:
            device = None

        tensor = input_tensor
        tensor = ttnn.from_device(input_tensor)
        tensor = ttnn.Tensor(tensor.value.to(ttnn.ROW_MAJOR_LAYOUT))
        tensor = ttnn.to_torch(tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_reshape, function_name="torch.reshape")(tensor, shape)
        tensor = ttnn.from_torch(tensor, dtype=input_tensor.dtype, layout=layout, device=device)
        # Unable to handle 5D tensors!  See ttnn_optimized_functional_whisper.
        # tensor = _to_layout(tensor, layout)
        tensor = ttnn_reshape(tensor, shape)

        return tensor


def _from_torch_validate_input_tensors(operation_name, tensor, *args, **kwargs):
    import torch

    ranks = (1, 2, 3, 4, 5, 6, 7, 8)
    if len(tensor.shape) not in ranks:
        raise RuntimeError(f"{operation_name}: ttnn.Tensor must be of rank {ranks}, but got {len(tensor.shape)}")
    dtypes = (torch.bfloat16, torch.float32, torch.int16, torch.int32, torch.int64, torch.float16)
    if tensor.dtype not in dtypes:
        raise RuntimeError(f"{operation_name}: ttnn.Tensor must be of type {dtypes}, but got {tensor.dtype}")
    # if not tensor.is_contiguous():
    #     raise RuntimeError(f"{operation_name}: ttnn.Tensor must be contiguous")


# TODO(arakhmati): remove this once underlying C++ code can handle non-4D shapes
def unsqueeze_to_4D(tensor):
    if len(tensor.shape) == 4:
        return tensor
    if len(tensor.shape) > 4:
        raise RuntimeError("Tensor cannot have more than 4 dimensions!")
    num_missing_dims = 4 - len(tensor.shape)
    shape = tuple(tensor.shape)
    padded_shape = tuple(tensor.shape.with_tile_padding())
    shape = (1,) * num_missing_dims + shape
    padded_shape = (1,) * num_missing_dims + padded_shape
    return ttnn.reshape(tensor, shape=ttnn.Shape(shape, padded_shape))


def squeeze(tensor, dim):
    if dim != 0:
        raise RuntimeError("Only dim=0 is supported for squeeze operation!")
    if len(tensor.shape) == 1:
        return tensor
    if len(tensor.shape) > 4:
        raise RuntimeError("Tensor cannot have more than 4 dimensions!")
    if tensor.shape[0] != 1:
        return tensor
    _, *shape = tensor.shape
    _, *full_shape = tensor.shape.with_tile_padding()
    return ttnn.reshape(tensor, shape=ttnn.Shape(shape, full_shape))


@ttnn.register_operation(name="ttnn.from_torch", validate_input_tensors=_from_torch_validate_input_tensors)
def from_torch(
    tensor: "torch.Tensor",
    dtype: Optional[ttnn.DataType] = None,
    *,
    layout: Optional[ttnn.Layout] = ttnn.ROW_MAJOR_LAYOUT,
    device: Optional[ttnn.Device] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    from_torch(tensor: torch.Tensor, dtype: Optional[DataType] = None, layout: Optional[Layout] = ROW_MAJOR_LAYOUT, device: Optional[Device] = None, memory_config: Optional[MemoryConfig] = None) -> ttnn.Tensor

    Converts the `torch.Tensor` :attr:`tensor` into a `ttnn.Tensor`.

    Args:
        * :attr:`tensor`: the torch.Tensor
        * :attr:`dtype`: the optional `ttnn` data type.
        * :attr:`layout`: the optional `ttnn` layout.
        * :attr:`device`: the optional `ttnn` device.
        * :attr:`memory_config`: the optional `ttnn` memory configuration.

    Example::

        >>> tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> print(tensor)
        Tensor([ [1.375, -1.30469, -0.714844],
            [-0.761719, 0.53125, -0.652344]], dtype=bfloat16 )
    """

    if dtype == ttnn.bfloat8_b:
        if len(tensor.shape) < 2:
            raise RuntimeError("ttnn.from_torch: bfloat8_b requires at least 2 dimensions!")
        if layout != ttnn.TILE_LAYOUT:
            raise RuntimeError("ttnn.from_torch: bfloat8_b requires TILE_LAYOUT!")
        # Tilize tensor
        tensor = ttnn.from_torch(tensor, layout=ttnn.TILE_LAYOUT)
        tensor = ttnn.Tensor(tensor.value.reshape(tensor.shape.with_tile_padding()))
        tensor = ttnn.to_torch(tensor)

    if memory_config is not None:
        if device is None:
            raise RuntimeError("device must be specified when memory_config is specified")

    def impl(tensor, dtype):
        return ttnn.Tensor(ttl.tensor.Tensor(tensor, dtype))

    tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.from_torch")(tensor, dtype)

    if layout is not None:
        tensor = ttnn.to_layout(tensor, layout)

    if device is not None:
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
        tensor = ttnn.to_device(tensor, device, memory_config=memory_config)

    return tensor


def _to_torch_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32, ttnn.float32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.to_torch", validate_input_tensors=_to_torch_validate_input_tensors)
def to_torch(tensor: ttnn.Tensor, *, torch_rank: Optional[int] = None) -> "torch.Tensor":
    """
    to_torch(tensor: ttnn.Tensor, torch_rank: Optional[int] = None) -> torch.Tensor

    Converts the `ttnn.Tensor` :attr:`tensor` into a `torch.Tensor`.

    Args:
        * :attr:`tensor`: ttnn.Tensor to be converted to torch.Tensor
        * :attr:`torch_rank`: desired rank of the torch.Tensor. Will use torch.squeeze operation to remove dimensions until the desired rank is reached. If not possible, the operation will error out.

    Example::
        >>> ttnn_tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> torch_tensor = ttnn.to_torch(ttnn_tensor)
        >>> print(torch_tensor)
        tensor([[-0.3008, -0.8438,  0.3242],
                [ 0.9023, -0.5820,  0.5312]], dtype=torch.bfloat16)
    """

    if ttnn.has_storage_type_of(tensor, ttnn.DEVICE_STORAGE_TYPE):
        tensor = ttnn.from_device(tensor)

    if tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)

    def impl(ttl_tensor):
        if ttl_tensor.storage_type() == ttnn.DEVICE_STORAGE_TYPE:
            raise RuntimeError("ttnn.Tensor cannot be on device when converting to torch.Tensor!")
        if ttl_tensor.layout() != ttnn.ROW_MAJOR_LAYOUT:
            raise RuntimeError("ttnn.Tensor has to be in ROW_MAJOR Layout to be converted to torch.Tensor")
        output = ttl_tensor.to_torch()

        if torch_rank is None:
            return output

        while len(output.shape) != torch_rank:
            if output.shape[0] != 1:
                raise RuntimeError("ttnn: Unable to squeeze to desired rank!")
            output = output.squeeze()
        return output

    ttl_tensor = tensor.value
    tensor = ttnn.Tensor(ttl_tensor.reshape(tensor.shape.with_tile_padding().value))
    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_torch")(tensor.value)


def _to_device_validate_input_tensors(operation_name, tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        tensor,
        ranks=(1, 2, 3, 4, 5),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.to_device", validate_input_tensors=_to_device_validate_input_tensors)
def to_device(tensor, device, *, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG):
    """
    to_device(tensor: ttnn.Tensor, device: ttnn.Device, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG) -> ttnn.Tensor

    Copies the `ttnn.Tensor` :attr:`tensor` to the `tt_lib.device.Device`.
    The tensor may be placed in DRAM or L1 memory.

    Currently memory_config must be of an Interleaved tensor (not sharded)

    Args:
        * :attr:`tensor`: the ttnn.Tensor
        * :attr:`device`: the ttnn.Device
        * :attr:`memory_config`: the optional MemoryConfig (DRAM_MEMORY_CONFIG or L1_MEMORY_CONFIG). Defaults to DRAM_MEMORY_CONFIG.

    Example::

        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> tensor_on_host = ttnn.from_torch(torch.randn((10, 64, 32)), dtype=ttnn.bfloat16)
        >>> tensor_on_device = ttnn.to_device(tensor_on_host, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        >>> print(tensor_on_device[0,0,:3])
        Tensor([ 0.800781, -0.455078, -0.585938], dtype=bfloat16 )
    """

    def impl(tensor, device, *, memory_config):
        ttl_tensor = tensor.value
        return ttnn.Tensor(ttl_tensor.to(device, memory_config))

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_device")(
        tensor, device, memory_config=memory_config
    )


def _from_device_validate_input_tensors(operation_name, tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        tensor,
        ranks=(1, 2, 3, 4, 5),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.from_device", validate_input_tensors=_from_device_validate_input_tensors)
def from_device(tensor):
    """
    from_device(tensor: ttnn.Tensor) -> ttnn.Tensor

    Copies the `ttnn.Tensor` :attr:`tensor` to the host.

    Args:
        * :attr:`tensor`: the ttnn.Tensor

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> tensor_on_device = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor_on_host = ttnn.from_device(tensor_on_device)
        >>> print(tensor_on_host[0,0,:3])
        Tensor([ 0.365234, 0.130859, 0.75], dtype=bfloat16 )
    """

    def impl(tensor):
        ttl_tensor = tensor.value
        return ttnn.Tensor(ttl_tensor.cpu())

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.from_device")(tensor)


def _deallocate_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(name="ttnn.deallocate", validate_input_tensors=_deallocate_validate_input_tensors)
def deallocate(tensor: ttnn.Tensor) -> None:
    """
    deallocate(tensor: ttnn.Tensor) -> None

    Releases the resources for `ttnn.Tensor` :attr:`tensor` explicitly.

    Args:
        * :attr:`tensor`: the ttnn.Tensor

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
        >>> ttnn.deallocate(tensor)
    """

    def impl(tensor):
        tensor.value.deallocate(force=True)

    ttl.tensor.decorate_external_operation(impl, function_name="ttnn.deallocate")(tensor)


def _to_memory_config_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(name="ttnn.to_memory_config", validate_input_tensors=_to_memory_config_validate_input_tensors)
def to_memory_config(tensor, memory_config: ttnn.MemoryConfig):
    """
    to_memory_config(tensor: ttnn.Tensor, memory_config: MemoryConfig) -> ttnn.Tensor

    Converts a tensor to the desired mem_config, used for converting tensors to sharded tensors or interleaved, and to convert DRAM to L1 and vice versa


    Args:
        * :attr:`tensor`: the ttnn.Tensor
        * :attr:`memory_config`: the desired MemoryConfig

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor = ttnn.to_memory_config(tensor, memory_config)
    """

    ttl_tensor = tensor.value
    if ttnn.get_memory_config(tensor) == memory_config:
        return tensor

    # to_sharded path
    if memory_config.is_sharded():
        if ttl_tensor.is_sharded():
            # reshard
            def impl(ttl_tensor, sharded_memory_config):
                ttl_tensor = ttl.tensor.sharded_to_interleaved(ttl_tensor, ttnn.DRAM_MEMORY_CONFIG)
                return ttl.tensor.interleaved_to_sharded(
                    ttl_tensor,
                    sharded_memory_config,
                )

            ttl_tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_memory_config")(
                ttl_tensor, memory_config
            )

        else:

            def impl(ttl_tensor, sharded_memory_config):
                return ttl.tensor.interleaved_to_sharded(
                    ttl_tensor,
                    sharded_memory_config,
                )

            ttl_tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_memory_config")(
                ttl_tensor, memory_config
            )
    # to_interleaved path
    else:
        if not ttl_tensor.is_sharded():
            # L1 to DRAM or DRAM to L1
            def impl(ttl_tensor, output_memory_config):
                return ttl.tensor.clone(ttl_tensor, output_memory_config)

            ttl_tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_memory_config")(
                ttl_tensor, memory_config
            )

        else:

            def impl(ttl_tensor, interleaved_memory_config):
                return ttl.tensor.sharded_to_interleaved(ttl_tensor, interleaved_memory_config)

            ttl_tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_memory_config")(
                ttl_tensor, memory_config
            )
    return ttnn.Tensor(ttl_tensor)


def _to_layout_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4, 5),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32, ttnn.float32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.to_layout", validate_input_tensors=_to_layout_validate_input_tensors)
def to_layout(tensor, layout: ttnn.Layout):
    """
    to_layout(tensor: ttnn.Tensor, layout: Layout) -> ttnn.Tensor

    Organizes the `ttnn.Tensor` :attr:`tensor` into either ROW_MAJOR_LAYOUT or TILE_LAYOUT.  When requesting ROW_MAJOR_LAYOUT
    the tensor will be returned unpadded in the last two dimensions.   When requesting TILE_LAYOUT the tensor will be automatically
    padded where the width and height become multiples of 32.
    In the case where the layout is the same, the operation simply pad or unpad the last two dimensions depending on layout requested.

    Args:
        * :attr:`tensor`: the ttnn.Tensor
        * :attr:`layout`: the layout of either ttnn.ROW_MAJOR_LAYOUT or ttnn.TILE_LAYOUT.

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
        >>> print(tensor[0,0,:3])
        Tensor([ 1.42188, -1.25, -0.398438], dtype=bfloat16 )
    """
    if tensor.layout == layout:
        return tensor

    supported_layout_mapping = {
        ttnn.ROW_MAJOR_LAYOUT: {ttnn.TILE_LAYOUT},
        ttnn.TILE_LAYOUT: {ttnn.ROW_MAJOR_LAYOUT},
    }
    supported_layouts = supported_layout_mapping[tensor.layout]
    if layout not in supported_layouts:
        raise RuntimeError(f"Unsupported layout conversion from {tensor.layout} to {layout}")

    is_on_device = ttnn.has_storage_type_of(tensor, ttl.tensor.StorageType.DEVICE)
    if is_on_device and tensor.dtype not in {ttnn.bfloat16, ttnn.bfloat8_b}:
        raise RuntimeError("ttnn.to_layout: Only bfloat16 and bfloat8_b are supported on device")

    def requires_padding_change(layout, shape):
        intended_shape = list(shape)[-2:]
        padded_shape = list(shape.with_tile_padding())[-2:]
        if layout == ttnn.ROW_MAJOR_LAYOUT and intended_shape != padded_shape:
            return True
        elif (
            layout == ttnn.TILE_LAYOUT
            and intended_shape == padded_shape
            and (
                len(intended_shape) < 2
                or intended_shape[-1] % ttnn.TILE_SIZE != 0
                or intended_shape[-2] % ttnn.TILE_SIZE != 0
            )
        ):
            return True
        else:
            return False

    if not requires_padding_change(layout, tensor.shape):
        ttl_tensor = tensor.value
        if is_on_device:
            if layout == ttnn.ROW_MAJOR_LAYOUT:
                return ttnn.Tensor(ttl.tensor.untilize(ttl_tensor))
            elif layout == ttnn.TILE_LAYOUT:
                return ttnn.Tensor(ttl.tensor.tilize(ttl_tensor, output_mem_config=ttl_tensor.memory_config()))
            else:
                raise RuntimeError(f"Unsupported layout: {layout}")
        else:
            return ttnn.Tensor(ttl_tensor.to(layout))

    # def unpad_with_pytorch(ttnn_tensor):
    #     desired_shape = list(ttnn_tensor.shape)
    #     ttl_tensor = ttnn_tensor.value
    #     if ttnn_tensor.layout != ROW_MAJOR_LAYOUT:
    #         ttl_tensor = ttl_tensor.to(ROW_MAJOR_LAYOUT)
    #     tensor = ttl_tensor.to_torch()
    #     slicing = [slice(None, desired_dim) for desired_dim in desired_shape]
    #     tensor = tensor[slicing]
    #     return _from_torch(tensor)

    intended_shape = tuple(tensor.shape)

    input_tensor = tensor
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        if is_on_device:
            *_, width = input_tensor.shape
            if width % 2 == 0:  # Can only unpad to row major tensor of even width
                input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
                intended_4D_shape = tuple(x - 1 for x in input_tensor.shape)
                ttl_input_tensor = input_tensor.value
                output_tensor = ttnn.Tensor(
                    ttl.tensor.untilize_with_unpadding(
                        ttl_input_tensor,
                        (0, 0, 0, 0),
                        intended_4D_shape,
                    )
                )
            else:
                input_tensor = ttnn.from_device(input_tensor)
                input_tensor = ttnn.unsqueeze_to_4D(input_tensor)

                def impl(input_tensor):
                    input_tensor = ttnn.Tensor(input_tensor.value.to(layout))
                    ttl_input_tensor = input_tensor.value

                    output_tensor_end = [dim - 1 for dim in input_tensor.shape]
                    output_tensor = ttnn.Tensor(ttl_input_tensor.unpad([0, 0, 0, 0], output_tensor_end))
                    return output_tensor

                output_tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_layout")(
                    input_tensor
                )
        else:
            input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
            input_tensor = ttnn.Tensor(input_tensor.value.to(layout))
            ttl_input_tensor = input_tensor.value
            output_tensor = ttnn.Tensor(ttl_input_tensor.unpad_from_tile(list(input_tensor.shape)))

        output_tensor = ttnn.reshape(output_tensor, intended_shape)
        return output_tensor
    elif layout == ttnn.TILE_LAYOUT:
        if len(tensor.shape) > 1:
            *original_batch_sizes, height, width = tensor.shape
        else:
            original_batch_sizes = []
            height = 1
            (width,) = tensor.shape

        pad_h = (ttnn.TILE_SIZE - height % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
        pad_w = (ttnn.TILE_SIZE - width % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
        padded_height = height + pad_h
        padded_width = width + pad_w
        tensor = ttnn.unsqueeze_to_4D(tensor)
        *batch_sizes, _, _ = tensor.shape

        if is_on_device:
            tensor = ttnn.Tensor(
                ttl.tensor.tilize_with_val_padding(
                    tensor.value,
                    batch_sizes + [padded_height, padded_width],
                    [0, 0, 0, 0],
                    0,
                )
            )
        else:

            def impl(tensor):
                return ttnn.Tensor(
                    tensor.value.pad(batch_sizes + [padded_height, padded_width], [0, 0, 0, 0], 0).to(layout)
                )

            tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_layout")(tensor)

        tensor = ttnn.reshape(
            tensor,
            ttnn.Shape(original_batch_sizes + [height, width], original_batch_sizes + [padded_height, padded_width]),
        )
        return tensor
    else:
        raise RuntimeError(f"Unsupported output layout: {layout}")


def _torch_identity(input_tensor):
    input_tensor = to_torch(input_tensor)
    return input_tensor.clone()


def _reallocate_validate_input_tensors(operation_name, input_tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        input_tensor,
        ranks=(1, 2, 3, 4),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=False,
    )


@ttnn.register_operation(
    name="ttnn.reallocate", validate_input_tensors=_reallocate_validate_input_tensors, torch_function=_torch_identity
)
def reallocate(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    def impl(input_tensor):
        ttl_input_tensor = input_tensor.value
        ttl_output_tensor = ttl.tensor.move(ttl_input_tensor)
        return ttnn.Tensor(ttl_output_tensor)

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.reallocate")(input_tensor)


def _load_tensor_validate_input_tensors(operation_name, file_name, *args, **kwargs):
    ...


@ttnn.register_operation(name="ttnn.load_tensor", validate_input_tensors=_load_tensor_validate_input_tensors)
def load_tensor(file_name: Union[str, pathlib.Path]) -> ttnn.Tensor:
    file_name = pathlib.Path(file_name)
    if not file_name.exists():
        raise RuntimeError(f"Unable to load the tensor from {file_name}.  The file does not exist.")
    if not file_name.is_file():
        raise RuntimeError(f"Unable to load the tensor from {file_name}.  The file is not a file.")

    def impl(file_name):
        return ttnn.Tensor(ttl.tensor.load_tensor(str(file_name)))

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.load_tensor")(file_name)


def _dump_tensor_validate_input_tensors(operation_name, _, tensor, *args, **kwargs):
    ttnn.validate_input_tensor(
        operation_name,
        tensor,
        ranks=(1, 2, 3, 4, 5, 6, 7, 8),
        dtypes=(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.uint16, ttnn.uint32),
        layouts=(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        can_be_on_device=True,
        can_be_on_cpu=True,
    )


@ttnn.register_operation(name="ttnn.dump_tensor", validate_input_tensors=_dump_tensor_validate_input_tensors)
def dump_tensor(file_name: Union[str, pathlib.Path], tensor: ttnn.Tensor) -> None:
    file_name = pathlib.Path(file_name)

    def impl(file_name, tensor):
        ttl_tensor = tensor.value
        ttl.tensor.dump_tensor(str(file_name), ttl_tensor)

    ttl.tensor.decorate_external_operation(impl, function_name="ttnn.dump_tensor")(file_name, tensor)


__all__ = []
