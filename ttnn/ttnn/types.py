# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
from enum import Enum

import tt_lib as ttl
import ttnn

DataType = ttl.tensor.DataType
uint16 = DataType.UINT16
uint32 = DataType.UINT32
float32 = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
bfloat8_b = DataType.BFLOAT8_B
bfloat4_b = DataType.BFLOAT4_B

BufferType = ttl.tensor.BufferType
TensorMemoryLayout = ttl.tensor.TensorMemoryLayout
# TODO: MemoryConfig = ttnn._ttnn.types.MemoryConfig
MemoryConfig = ttl.tensor.MemoryConfig
MathFidelity = ttl.tensor.MathFidelity
DRAM_MEMORY_CONFIG = ttnn._ttnn.types.DRAM_MEMORY_CONFIG
L1_MEMORY_CONFIG = ttnn._ttnn.types.L1_MEMORY_CONFIG
L1_BLOCK_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1)
L1_HEIGHT_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
L1_WIDTH_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.WIDTH_SHARDED, BufferType.L1)

Layout = ttl.tensor.Layout
ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR
TILE_LAYOUT = Layout.TILE

StorageType = ttl.tensor.StorageType
DEVICE_STORAGE_TYPE = StorageType.DEVICE
MULTI_DEVICE_STORAGE_TYPE = StorageType.MULTI_DEVICE

TILE_SIZE = 32

Shape = ttnn._ttnn.types.Shape
Tensor = ttl.tensor.Tensor


CoreGrid = ttnn._ttnn.types.CoreGrid

DeviceComputeKernelConfig = ttl.tensor.DeviceComputeKernelConfig
WormholeComputeKernelConfig = ttl.tensor.WormholeComputeKernelConfig
GrayskullComputeKernelConfig = ttl.tensor.GrayskullComputeKernelConfig


@dataclasses.dataclass
class CoreRange:
    start: CoreGrid
    end: CoreGrid


@dataclasses.dataclass
class DeviceGrid:
    y: int
    x: int

    @property
    def num_devices(self):
        return self.y * self.x

    def as_tuple(self):
        return (self.y, self.x)


class ShardStrategy(Enum):
    HEIGHT = 1
    WIDTH = 2
    BLOCK = 3


class ShardOrientation(Enum):
    ROW_MAJOR = 1
    COLUMN_MAJOR = 2
