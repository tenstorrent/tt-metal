# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import io
import pathlib
from typing import Optional, Union, Tuple, List

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

        if has_storage_type_of(self, DEVICE_STORAGE_TYPE):
            tensor = self
            device = tensor.device
            tensor = from_device(tensor)
            tensor = to_torch(tensor)
            tensor = ttl.tensor.decorate_external_operation(torch_getitem, function_name="torch.Tensor.__getitem__")(
                tensor, slices
            )
            tensor = from_torch(tensor, dtype=self.dtype)
            tensor = to_device(tensor, device)
        else:
            tensor = self
            tensor = to_torch(tensor)
            tensor = ttl.tensor.decorate_external_operation(torch_getitem, function_name="torch.Tensor.__getitem__")(
                tensor, slices
            )
            tensor = from_torch(tensor, dtype=self.dtype)
        return tensor

    def is_contiguous(self: "Shape") -> bool:
        if self.layout == ROW_MAJOR_LAYOUT:
            return self.value.shape() == self.value.shape_without_padding()
        else:
            return False


def has_storage_type_of(tensor: Tensor, storage_type) -> bool:
    return tensor.value.storage_type() == storage_type


@decorate_operation()
def from_torch(
    tensor: "torch.Tensor",
    dtype: Optional[DataType] = None,
) -> Tensor:
    """
    from_torch(tensor: torch.Tensor, dtype: Optional[DataType] = None) -> Tensor

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

    def impl(tensor, dtype):
        return Tensor(ttl.tensor.Tensor(tensor, dtype))

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.from_torch")(tensor, dtype)


@decorate_operation()
def to_torch(tensor: Tensor) -> "torch.Tensor":
    def impl(ttl_tensor):
        if ttl_tensor.storage_type() == DEVICE_STORAGE_TYPE:
            raise RuntimeError("ttnn.Tensor cannot be on device when converting to torch!")
        return ttl_tensor.to_torch()

    ttl_tensor = tensor.value
    tensor = Tensor(ttl_tensor.reshape(tensor.shape.padded().value))

    return ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_torch")(ttl_tensor)


@decorate_operation()
def to_device(tensor, device, *, memory_config: MemoryConfig = DRAM_MEMORY_CONFIG):
    """
    to_device(tensor: ttnn.Tensor, device: tt_lib.device.Device, dtype: Optional[DataType] = None) -> Tensor

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
    from_device(tensor: ttnn.Tensor) -> Tensor

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
def to_layout(tensor, layout: Layout):
    """
    to_layout(tensor: ttnn.Tensor, layout: Layout) -> Tensor

    Organizes the `ttnn.Tensor` :attr:`tensor` into eiter ROW_MAJOR_LAYOUT or TILE_LAYOUT.

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
    ttl_tensor = tensor.value
    if ttl_tensor.layout() == layout:
        return tensor
    elif has_storage_type_of(tensor, DEVICE_STORAGE_TYPE):
        if layout == ROW_MAJOR_LAYOUT:
            ttl_tensor = ttl.tensor.untilize(ttl_tensor)
        elif layout == TILE_LAYOUT:
            ttl_tensor = ttl.tensor.tilize(ttl_tensor, output_mem_config=ttl_tensor.memory_config())
        else:
            raise RuntimeError(f"Unsupported layout: {layout}")
    else:

        def impl(ttl_tensor, layout):
            return ttl_tensor.to(layout)

        ttl_tensor = ttl.tensor.decorate_external_operation(impl, function_name="ttnn.to_layout")(ttl_tensor, layout)
    return Tensor(ttl_tensor)


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


def _torch_identity(input_tensor):
    import ttnn

    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_torch(input_tensor)
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
    "to_layout",
    "deallocate",
    "reallocate",
    "load_tensor",
    "dump_tensor",
]
