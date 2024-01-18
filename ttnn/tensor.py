# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pathlib
from typing import Optional, Union, Tuple

import tt_lib as ttl

from ttnn.decorators import decorate_operation

Device = ttl.device.Device


DataType = ttl.tensor.DataType
uint32 = DataType.UINT32
float32 = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
bfloat8_b = DataType.BFLOAT8_B


BufferType = ttl.tensor.BufferType
TensorMemoryLayout = ttl.tensor.TensorMemoryLayout
MemoryConfig = ttl.tensor.MemoryConfig
MathFidelity = ttl.tensor.MathFidelity
DRAM_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.DRAM)
L1_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.INTERLEAVED, BufferType.L1)


Layout = ttl.tensor.Layout
ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR
TILE_LAYOUT = Layout.TILE

StorageType = ttl.tensor.StorageType
DEVICE_STORAGE_TYPE = StorageType.DEVICE

TILE_SIZE = 32

Shape = ttl.ttnn.tensor.Shape


class Tensor(ttl.ttnn.tensor.Tensor):
    @property
    def device(self: "Tensor") -> DataType:
        if has_storage_type_of(self, DEVICE_STORAGE_TYPE):
            return self.value.device()
        else:
            raise RuntimeError("Tensor is not on device!")

    @decorate_operation()
    def __getitem__(self: "Tensor", slices) -> "Tensor":
        if self.layout != ROW_MAJOR_LAYOUT:
            raise RuntimeError("Tensor must be in ROW_MAJOR layout to use slicing!")

        def torch_getitem(tensor, slices):
            return tensor[slices].clone()

        if has_storage_type_of(self, ttl.tensor.StorageType.DEVICE):
            device = self.device
        else:
            device = None

        tensor = self
        tensor = to_torch(tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_getitem, function_name="torch.Tensor.__getitem__")(
            tensor, slices
        )
        tensor = from_torch(tensor, dtype=self.dtype, device=device)
        return tensor

    def is_contiguous(self: "Shape") -> bool:
        if self.layout == ROW_MAJOR_LAYOUT:
            return self.value.shape() == self.value.shape_without_padding()
        else:
            return False


def has_storage_type_of(tensor: Tensor, storage_type) -> bool:
    return tensor.value.storage_type() == storage_type


def _torch_reshape(input_tensor: Tensor, shape: Union[Shape, Tuple[int, ...]], **_):
    import torch

    input_tensor = to_torch(input_tensor)

    if isinstance(shape, Shape):
        shape = tuple(shape)

    return torch.reshape(input_tensor, shape).contiguous().clone()


@decorate_operation(torch_function=_torch_reshape)
def reshape(input_tensor: Tensor, shape: Union[Shape, Tuple[int, ...]]) -> Tensor:
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
        shape = Shape(shape)

    if not isinstance(shape, Shape):
        raise RuntimeError("Shape must be of type Shape")

    if input_tensor.shape == shape and list(input_tensor.shape) == list(shape):
        return input_tensor

    def ttnn_reshape(tensor, shape):
        ttl_input_tensor = tensor.value
        return Tensor(ttl_input_tensor.reshape(shape.value))

    ttnn_reshape = ttl.tensor.decorate_external_operation(ttnn_reshape, function_name="ttnn.reshape")

    if input_tensor.is_contiguous():
        if has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            # Page size depends on the width, so only modify the shape if the width is the same
            if input_tensor.shape[-1] == shape[-1]:
                return ttnn_reshape(input_tensor, shape)
        else:
            return ttnn_reshape(input_tensor, shape)

    if input_tensor.layout == TILE_LAYOUT:
        *_, new_height, new_width = tuple(shape.padded())
        if new_height % TILE_SIZE == 0 and new_width % TILE_SIZE == 0:
            return ttnn_reshape(input_tensor, shape)

    if (
        has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE)
        and len(input_tensor.shape) == 4
        and len(shape) == 4
    ):
        ttl_input_tensor = input_tensor.value
        w, z, y, x = shape
        ttl_output_tensor = ttl.tensor.reshape(ttl_input_tensor, w, z, y, x)
        output_tensor = Tensor(ttl_output_tensor)
        output_tensor = ttnn_reshape(output_tensor, shape)
        return output_tensor
    else:

        def torch_reshape(tensor, shape):
            return tensor.reshape(tuple(shape.padded())).contiguous().clone()

        if has_storage_type_of(input_tensor, ttl.tensor.StorageType.DEVICE):
            device = input_tensor.device
        else:
            device = None

        tensor = input_tensor
        tensor = from_device(input_tensor)
        tensor = Tensor(tensor.value.to(ROW_MAJOR_LAYOUT))
        tensor = to_torch(tensor)
        tensor = ttl.tensor.decorate_external_operation(torch_reshape, function_name="torch.reshape")(tensor, shape)
        tensor = from_torch(tensor, dtype=input_tensor.dtype, device=device)
        tensor = ttnn_reshape(tensor, shape)

        return tensor


# TODO(arakhmati): remove this once underlying C++ code can handle non-4D shapes
def unsqueeze_to_4D(tensor):
    if len(tensor.shape) == 4:
        return tensor
    if len(tensor.shape) > 4:
        raise RuntimeError("Tensor cannot have more than 4 dimensions!")
    num_missing_dims = 4 - len(tensor.shape)
    shape = tuple(tensor.shape)
    full_shape = tuple(tensor.shape.padded())
    shape = (1,) * num_missing_dims + shape
    full_shape = (1,) * num_missing_dims + full_shape
    return reshape(tensor, shape=Shape(shape, full_shape))


@decorate_operation()
def from_torch(
    tensor: "torch.Tensor",
    dtype: Optional[DataType] = None,
    *,
    layout: Optional[Layout] = ROW_MAJOR_LAYOUT,
    device: Optional[Device] = None,
    memory_config: Optional[MemoryConfig] = None,
) -> Tensor:
    """
    from_torch(tensor: torch.Tensor, dtype: Optional[DataType] = None) -> ttnn.Tensor

    Converts the `torch.Tensor` :attr:`tensor` into a `ttnn.Tensor`.

    Args:
        * :attr:`tensor`: the torch.Tensor
        * :attr:`dtype`: the optional `ttnn` data type.

    Example::

        >>> tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> print(tensor)
        Tensor([ [1.375, -1.30469, -0.714844],
            [-0.761719, 0.53125, -0.652344]], dtype=bfloat16 )
    """

    if memory_config is not None:
        if device is None:
            raise RuntimeError("device must be specified when memory_config is specified")

    def impl(tensor, dtype):
        return Tensor(ttl.tensor.Tensor(tensor, dtype))

    tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.from_torch")(tensor, dtype)

    if layout is not None:
        tensor = to_layout(tensor, layout)

    if device is not None:
        if memory_config is None:
            memory_config = DRAM_MEMORY_CONFIG
        tensor = to_device(tensor, device, memory_config=memory_config)

    return tensor


@decorate_operation()
def to_torch(tensor: Tensor) -> "torch.Tensor":
    """
    to_torch(tensor: ttnn.Tensor) -> torch.Tensor

    Converts the `ttnn.Tensor` :attr:`tensor` into a `torch.Tensor`.

    Args:
        * :attr:`tensor`: the ttnn.Tensor

    Example::
        >>> ttnn_tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> torch_tensor = ttnn.to_torch(ttnn_tensor)
        >>> print(torch_tensor)
        tensor([[-0.3008, -0.8438,  0.3242],
                [ 0.9023, -0.5820,  0.5312]], dtype=torch.bfloat16)
    """

    if has_storage_type_of(tensor, DEVICE_STORAGE_TYPE):
        tensor = from_device(tensor)

    if tensor.layout != ROW_MAJOR_LAYOUT:
        tensor = to_layout(tensor, ROW_MAJOR_LAYOUT)

    def impl(ttl_tensor):
        if ttl_tensor.storage_type() == DEVICE_STORAGE_TYPE:
            raise RuntimeError("ttnn.Tensor cannot be on device when converting to torch.Tensor!")
        if ttl_tensor.layout() != ROW_MAJOR_LAYOUT:
            raise RuntimeError("ttnn.Tensor has to be in ROW_MAJOR Layout to be converted to torch.Tensor")
        return ttl_tensor.to_torch()

    ttl_tensor = tensor.value
    tensor = Tensor(ttl_tensor.reshape(tensor.shape.padded().value))

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_torch")(ttl_tensor)


@decorate_operation()
def to_device(tensor, device, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG):
    """
    to_device(tensor: ttnn.Tensor, device: tt_lib.device.Device, dtype: Optional[DataType] = None) -> ttnn.Tensor

    Copies the `ttnn.Tensor` :attr:`tensor` to the `tt_lib.device.Device`.
    The tensor may be placed in DRAM or L1 memory.

    Args:
        * :attr:`tensor`: the ttnn.Tensor
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
        return Tensor(ttl_tensor.to(device, memory_config))

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_device")(
        tensor, device, memory_config=memory_config
    )


@decorate_operation()
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
        return Tensor(ttl_tensor.cpu())

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.from_device")(tensor)


@decorate_operation()
def deallocate(tensor: Tensor) -> None:
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


@decorate_operation()
def to_layout(tensor, layout: Layout):
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
        ROW_MAJOR_LAYOUT: {TILE_LAYOUT},
        TILE_LAYOUT: {ROW_MAJOR_LAYOUT},
    }
    supported_layouts = supported_layout_mapping[tensor.layout]
    if layout not in supported_layouts:
        raise RuntimeError(f"Unsupported layout conversion from {tensor.layout} to {layout}")

    is_on_device = has_storage_type_of(tensor, ttl.tensor.StorageType.DEVICE)

    def requires_padding_change(layout, shape):
        intended_shape = list(shape)[-2:]
        padded_shape = list(shape.padded())[-2:]
        if layout == ROW_MAJOR_LAYOUT and intended_shape != padded_shape:
            return True
        elif (
            layout == TILE_LAYOUT
            and intended_shape == padded_shape
            and (len(intended_shape) < 2 or intended_shape[-1] % TILE_SIZE != 0 or intended_shape[-2] % TILE_SIZE != 0)
        ):
            return True
        else:
            return False

    if not requires_padding_change(layout, tensor.shape):
        ttl_tensor = tensor.value
        if is_on_device:
            if layout == ROW_MAJOR_LAYOUT:
                return Tensor(ttl.tensor.untilize(ttl_tensor))
            elif layout == TILE_LAYOUT:
                return Tensor(ttl.tensor.tilize(ttl_tensor, output_mem_config=ttl_tensor.memory_config()))
            else:
                raise RuntimeError(f"Unsupported layout: {layout}")
        else:
            return Tensor(ttl_tensor.to(layout))

    # def unpad_with_pytorch(ttnn_tensor):
    #     desired_shape = list(ttnn_tensor.shape)
    #     ttl_tensor = ttnn_tensor.value
    #     if ttnn_tensor.layout != ROW_MAJOR_LAYOUT:
    #         ttl_tensor = ttl_tensor.to(ROW_MAJOR_LAYOUT)
    #     tensor = ttl_tensor.to_torch()
    #     slicing = [slice(None, desired_dim) for desired_dim in desired_shape]
    #     tensor = tensor[slicing]
    #     return from_torch(tensor)

    intended_shape = tuple(tensor.shape)

    input_tensor = tensor
    if layout == ROW_MAJOR_LAYOUT:
        if is_on_device:
            *_, width = input_tensor.shape
            if width % 2 == 0:  # Can only unpad to row major tensor of even width
                input_tensor = unsqueeze_to_4D(input_tensor)
                intended_4D_shape = tuple(x - 1 for x in input_tensor.shape)
                ttl_input_tensor = input_tensor.value
                output_tensor = Tensor(
                    ttl.tensor.untilize_with_unpadding(
                        ttl_input_tensor,
                        (0, 0, 0, 0),
                        intended_4D_shape,
                    )
                )
            else:
                input_tensor = from_device(input_tensor)
                input_tensor = unsqueeze_to_4D(input_tensor)
                input_tensor = Tensor(input_tensor.value.to(layout))
                ttl_input_tensor = input_tensor.value

                output_tensor_end = [dim - 1 for dim in input_tensor.shape]
                output_tensor = Tensor(ttl_input_tensor.unpad([0, 0, 0, 0], output_tensor_end))
        else:
            input_tensor = unsqueeze_to_4D(input_tensor)
            input_tensor = Tensor(input_tensor.value.to(layout))
            ttl_input_tensor = input_tensor.value
            output_tensor = Tensor(ttl_input_tensor.unpad_from_tile(list(input_tensor.shape)))

        output_tensor = reshape(output_tensor, intended_shape)
        return output_tensor
    elif layout == TILE_LAYOUT:
        if len(tensor.shape) > 1:
            *original_batch_sizes, height, width = tensor.shape
        else:
            original_batch_sizes = []
            height = 1
            (width,) = tensor.shape

        pad_h = (TILE_SIZE - height % TILE_SIZE) % TILE_SIZE
        pad_w = (TILE_SIZE - width % TILE_SIZE) % TILE_SIZE
        padded_height = height + pad_h
        padded_width = width + pad_w
        tensor = unsqueeze_to_4D(tensor)
        *batch_sizes, _, _ = tensor.shape

        ttl_input_tensor = tensor.value
        if is_on_device:
            tensor = Tensor(
                ttl.tensor.tilize_with_val_padding(
                    ttl_input_tensor,
                    batch_sizes + [padded_height, padded_width],
                    [0, 0, 0, 0],
                    0,
                )
            )
        else:
            tensor = Tensor(
                ttl_input_tensor.pad(batch_sizes + [padded_height, padded_width], [0, 0, 0, 0], 0).to(layout)
            )

        tensor = reshape(
            tensor,
            Shape(original_batch_sizes + [height, width], original_batch_sizes + [padded_height, padded_width]),
        )
        return tensor
    else:
        raise RuntimeError(f"Unsupported output layout: {layout}")


def _torch_identity(input_tensor):
    input_tensor = to_torch(input_tensor)
    return input_tensor.clone()


@decorate_operation(torch_function=_torch_identity)
def reallocate(input_tensor: Tensor) -> Tensor:
    ttl_input_tensor = input_tensor.value
    ttl_output_tensor = ttl.tensor.move(ttl_input_tensor)
    return Tensor(ttl_output_tensor)


@decorate_operation()
def load_tensor(file_name: Union[str, pathlib.Path]) -> Tensor:
    def impl(file_name):
        return Tensor(ttl.tensor.load_tensor(str(file_name)))

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.load_tensor")(file_name)


@decorate_operation()
def dump_tensor(file_name: Union[str, pathlib.Path], tensor: Tensor) -> None:
    def impl(file_name, tensor):
        ttl_tensor = tensor.value
        ttl.tensor.dump_tensor(str(file_name), ttl_tensor)

    ttl.tensor.decorate_external_operation(impl, function_name="ttnn.dump_tensor")(file_name, tensor)


__all__ = [
    "Device",
    "DataType",
    "uint32",
    "float32",
    "bfloat16",
    "bfloat8_b",
    "DRAM_MEMORY_CONFIG",
    "L1_MEMORY_CONFIG",
    "ROW_MAJOR_LAYOUT",
    "TILE_LAYOUT",
    "TILE_SIZE",
    "Tensor",
    "from_torch",
    "to_torch",
    "to_device",
    "from_device",
    "deallocate",
    "reallocate",
    "load_tensor",
    "dump_tensor",
    "to_layout",
]
