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
ShardShapeAlignment = ttnn._ttnn.tensor.ShardShapeAlignment
ShardDistributionStrategy = ttnn._ttnn.tensor.ShardDistributionStrategy
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

TILE_SIZE = 32

Tile = ttnn._ttnn.tensor.Tile

Shape = ttnn._ttnn.types.Shape
TensorSpec = ttnn._ttnn.tensor.TensorSpec
Tensor = ttnn._ttnn.tensor.Tensor


CoreGrid = ttnn._ttnn.types.CoreGrid
CoreType = ttnn._ttnn.types.CoreType

ThrottleLevel = ttnn._ttnn.operations.core.ThrottleLevel


DeviceComputeKernelConfig = ttnn._ttnn.operations.core.DeviceComputeKernelConfig
WormholeComputeKernelConfig = ttnn._ttnn.operations.core.WormholeComputeKernelConfig
BlackholeComputeKernelConfig = WormholeComputeKernelConfig
GrayskullComputeKernelConfig = ttnn._ttnn.operations.core.GrayskullComputeKernelConfig


@dataclasses.dataclass
class CoreRange:
    start: CoreGrid
    end: CoreGrid


class ShardStrategy(Enum):
    HEIGHT = 1
    WIDTH = 2
    BLOCK = 3


MeshShape = ttnn._ttnn.multi_device.MeshShape
MeshCoordinate = ttnn._ttnn.multi_device.MeshCoordinate
MeshCoordinateRange = ttnn._ttnn.multi_device.MeshCoordinateRange
MeshCoordinateRangeSet = ttnn._ttnn.multi_device.MeshCoordinateRangeSet
ShardOrientation = ttnn._ttnn.tensor.ShardOrientation
ShardSpec = ttnn._ttnn.tensor.ShardSpec
NdShardSpec = ttnn._ttnn.tensor.NdShardSpec
CoreRangeSet = ttnn._ttnn.tensor.CoreRangeSet
CoreRange = ttnn._ttnn.tensor.CoreRange
CoreCoord = ttnn._ttnn.tensor.CoreCoord
corerange_to_cores = ttnn._ttnn.tensor.corerange_to_cores

QueueId = ttnn._ttnn.types.QueueId

UnaryWithParam = ttnn._ttnn.activation.UnaryWithParam
UnaryOpType = ttnn._ttnn.activation.UnaryOpType
BinaryOpType = ttnn._ttnn.operations.binary.BinaryOpType

BcastOpMath = ttnn._ttnn.types.BcastOpMath
BcastOpDim = ttnn._ttnn.types.BcastOpDim

DataMovementProcessor = ttnn._ttnn.types.DataMovementProcessor
NOC = ttnn._ttnn.types.NOC
NOC_MODE = ttnn._ttnn.types.NOC_MODE

TileDescriptor = ttnn._ttnn.program_descriptor.TileDescriptor
CBFormatDescriptor = ttnn._ttnn.program_descriptor.CBFormatDescriptor
CBDescriptor = ttnn._ttnn.program_descriptor.CBDescriptor
ReaderConfigDescriptor = ttnn._ttnn.program_descriptor.ReaderConfigDescriptor
WriterConfigDescriptor = ttnn._ttnn.program_descriptor.WriterConfigDescriptor
DataMovementConfigDescriptor = ttnn._ttnn.program_descriptor.DataMovementConfigDescriptor
ComputeConfigDescriptor = ttnn._ttnn.program_descriptor.ComputeConfigDescriptor
KernelDescriptor = ttnn._ttnn.program_descriptor.KernelDescriptor
RuntimeArgs = ttnn._ttnn.program_descriptor.RuntimeArgs
RuntimeArgsColProxy = ttnn._ttnn.program_descriptor.RuntimeArgsColProxy
SemaphoreDescriptor = ttnn._ttnn.program_descriptor.SemaphoreDescriptor
ProgramDescriptor = ttnn._ttnn.program_descriptor.ProgramDescriptor
MeshProgramDescriptor = ttnn._ttnn.program_descriptor.MeshProgramDescriptor
cb_descriptor_from_sharded_tensor = ttnn._ttnn.program_descriptor.cb_descriptor_from_sharded_tensor

TensorAccessorArgs = ttnn._ttnn.tensor_accessor_args.TensorAccessorArgs
