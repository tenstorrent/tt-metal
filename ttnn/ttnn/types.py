# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
from enum import Enum

import ttnn

DataType = ttnn._ttnn.tensor.DataType
uint8 = DataType.UINT8
uint16 = DataType.UINT16
int32 = DataType.INT32
uint32 = DataType.UINT32
float32 = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
bfloat8_b = DataType.BFLOAT8_B
bfloat4_b = DataType.BFLOAT4_B

BufferType = ttnn._ttnn.tensor.BufferType
TensorMemoryLayout = ttnn._ttnn.tensor.TensorMemoryLayout
# TODO: MemoryConfig = ttnn._ttnn.types.MemoryConfig
MemoryConfig = ttnn._ttnn.tensor.MemoryConfig
MathFidelity = ttnn._ttnn.tensor.MathFidelity
DRAM_MEMORY_CONFIG = ttnn._ttnn.types.DRAM_MEMORY_CONFIG
L1_MEMORY_CONFIG = ttnn._ttnn.types.L1_MEMORY_CONFIG
L1_BLOCK_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.BLOCK_SHARDED, BufferType.L1)
L1_HEIGHT_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
L1_WIDTH_SHARDED_MEMORY_CONFIG = MemoryConfig(TensorMemoryLayout.WIDTH_SHARDED, BufferType.L1)

Layout = ttnn._ttnn.tensor.Layout
ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR
TILE_LAYOUT = Layout.TILE

StorageType = ttnn._ttnn.tensor.StorageType
DEVICE_STORAGE_TYPE = StorageType.DEVICE
MULTI_DEVICE_STORAGE_TYPE = StorageType.MULTI_DEVICE

TILE_SIZE = 32

Tile = ttnn._ttnn.tensor.Tile

Shape = ttnn._ttnn.types.Shape
Tensor = ttnn._ttnn.tensor.Tensor


CoreGrid = ttnn._ttnn.types.CoreGrid

DeviceComputeKernelConfig = ttnn._ttnn.operations.core.DeviceComputeKernelConfig
WormholeComputeKernelConfig = ttnn._ttnn.operations.core.WormholeComputeKernelConfig
BlackholeComputeKernelConfig = WormholeComputeKernelConfig
GrayskullComputeKernelConfig = ttnn._ttnn.operations.core.GrayskullComputeKernelConfig


@dataclasses.dataclass
class CoreRange:
    start: CoreGrid
    end: CoreGrid


@dataclasses.dataclass
class MeshShape:
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


ShardOrientation = ttnn._ttnn.tensor.ShardOrientation
ShardSpec = ttnn._ttnn.tensor.ShardSpec
CoreRangeSet = ttnn._ttnn.tensor.CoreRangeSet
CoreRange = ttnn._ttnn.tensor.CoreRange
CoreCoord = ttnn._ttnn.tensor.CoreCoord


UnaryWithParam = ttnn._ttnn.activation.UnaryWithParam
UnaryOpType = ttnn._ttnn.activation.UnaryOpType
BinaryOpType = ttnn._ttnn.operations.binary.BinaryOpType

BcastOpMath = ttnn._ttnn.types.BcastOpMath
BcastOpDim = ttnn._ttnn.types.BcastOpDim
