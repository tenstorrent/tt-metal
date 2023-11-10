# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import tt_lib as ttl


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


TILE_SIZE = 32


class Tensor:
    def __init__(self: "Tensor", ttl_tensor: ttl.tensor.Tensor):
        self._tensor: ttl.tensor.Tensor = ttl_tensor

    @property
    def shape(self: "Tensor") -> tuple:
        return self._tensor.shape()

    @property
    def dtype(self: "Tensor") -> DataType:
        return self._tensor.dtype()

    @property
    def layout(self: "Tensor") -> DataType:
        return self._tensor.layout()

    def __getitem__(self: "Tensor", slices) -> "Tensor":
        if self.layout != ROW_MAJOR_LAYOUT:
            raise RuntimeError("Tensor must be in ROW_MAJOR layout to use slicing!")

        if self._tensor.storage_type() == ttl.tensor.StorageType.DEVICE:
            tensor = self
            tensor = from_device(tensor)
            tensor = to_torch(tensor)
            ttl.tensor.log_external_operation(tensor.__getitem__, tensor, slices)
            tensor = tensor[slices]
            tensor = from_torch(tensor, dtype=self.dtype)
        else:
            tensor = self
            tensor = to_torch(tensor)
            ttl.tensor.log_external_operation(tensor.__getitem__, tensor, slices)
            tensor = tensor[slices]
            tensor = from_torch(tensor, dtype=self.dtype)
        return tensor

    def __repr__(self: "Tensor") -> str:
        return str(self._tensor)


def from_torch(
    tensor: "torch.Tensor",
    dtype: Optional[DataType] = None,
) -> Tensor:
    """
    from_torch(tensor: torch.Tensor, dtype: Optional[DataType] = None) -> Tensor

    Converts the `torch.Tensor` :attr:`tensor` into a `ttnn.Tensor`.

    Args:
        * :attr:`tensor`: the torch.Tensor
        * :attr:`dtype`: the optional `ttnn` date type.

    Example::

        >>> tensor = ttnn.from_torch(torch.randn((2,3)), dtype=ttnn.bfloat16)
        >>> print(tensor)
        Tensor([ [1.375, -1.30469, -0.714844],
            [-0.761719, 0.53125, -0.652344]], dtype=bfloat16 )
    """
    ttl.tensor.log_external_operation(from_torch, tensor)
    return Tensor(ttl.tensor.Tensor(tensor, dtype))


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
    ttl_tensor = tensor._tensor
    ttl.tensor.log_external_operation(to_torch, ttl_tensor)

    if ttl_tensor.layout() != ROW_MAJOR_LAYOUT:
        raise RuntimeError("ttnn.Tensor has to be in ROW_MAJOR Layout to be convered to torch.Tensor")

    if ttl_tensor.storage_type() == ttl.tensor.StorageType.DEVICE:
        raise RuntimeError("ttnn.Tensor cannot be on device when converting to torch!")

    return ttl_tensor.to_torch()


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
    ttl_tensor = tensor._tensor
    ttl.tensor.log_external_operation(to_device, ttl_tensor)
    return Tensor(ttl_tensor.to(device, memory_config))


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
    ttl_tensor = tensor._tensor
    ttl.tensor.log_external_operation(from_device, ttl_tensor)
    return Tensor(ttl_tensor.cpu())


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
    ttl_tensor = tensor._tensor
    if ttl_tensor.layout() == layout:
        return tensor
    elif layout == ROW_MAJOR_LAYOUT:
        ttl_tensor = ttl.tensor.untilize(ttl_tensor)
    elif layout == TILE_LAYOUT:
        ttl_tensor = ttl.tensor.tilize(ttl_tensor, output_mem_config=ttl_tensor.memory_config())
    return Tensor(ttl_tensor)


def free(tensor: Tensor):
    """
    free(tensor: ttnn.Tensor)

    Releases the resources for `ttnn.Tensor` :attr:`tensor` explicitly.

    Args:
        * :attr:`tensor`: the ttnn.Tensor

    Example::
        >>> device_id = 0
        >>> device = ttnn.open(device_id)
        >>> tensor = ttnn.to_device(ttnn.from_torch(torch.randn((10, 64, 32), dtype=torch.bfloat16)), device)
        >>> tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
        >>> ttnn.free(tensor)
    """
    tensor._tensor.deallocate(force=True)


__all__ = [
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
    "from_tensor",
    "to_tensor",
    "to_device",
    "from_device",
    "to_layout",
    "free",
]
