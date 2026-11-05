# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
from enum import Enum

import ttnn

DataType = ttnn._ttnn.tensor.DataType
uint8 = DataType.UINT8
int8 = DataType.INT8
uint16 = DataType.UINT16
int32 = DataType.INT32
uint32 = DataType.UINT32
float32 = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
bfloat8_b = DataType.BFLOAT8_B
bfloat4_b = DataType.BFLOAT4_B
fp8_e4m3 = DataType.FP8_E4M3
# OCP microscaling (MX) formats, Quasar only, TILE layout only.
mxfp8_e4m3 = DataType.MXFP8_E4M3
mxfp8_e5m2 = DataType.MXFP8_E5M2
mxfp6_e2m3 = DataType.MXFP6_E2M3
mxfp6_e3m2 = DataType.MXFP6_E3M2
mxfp4 = DataType.MXFP4
mxint8 = DataType.MXINT8
mxint4 = DataType.MXINT4
mxint2 = DataType.MXINT2
MX_DTYPES = (mxfp8_e4m3, mxfp8_e5m2, mxfp6_e2m3, mxfp6_e3m2, mxfp4, mxint8, mxint4, mxint2)

# GGUF (llama.cpp) quantized weights, TILE layout [K, N], made with ttnn.from_gguf.
gguf_q8_0 = DataType.GGUF_Q8_0
gguf_q4_0 = DataType.GGUF_Q4_0
gguf_iq4_nl = DataType.GGUF_IQ4_NL
gguf_mxfp4 = DataType.GGUF_MXFP4
gguf_nvfp4 = DataType.GGUF_NVFP4
gguf_q6_k = DataType.GGUF_Q6_K
gguf_q4_k = DataType.GGUF_Q4_K
gguf_q5_k = DataType.GGUF_Q5_K
gguf_q3_k = DataType.GGUF_Q3_K
gguf_q2_k = DataType.GGUF_Q2_K
gguf_iq4_xs = DataType.GGUF_IQ4_XS
gguf_tq2_0 = DataType.GGUF_TQ2_0
gguf_q2_0 = DataType.GGUF_Q2_0
gguf_q1_0 = DataType.GGUF_Q1_0
gguf_iq2_xxs = DataType.GGUF_IQ2_XXS
gguf_iq1_s = DataType.GGUF_IQ1_S
gguf_iq2_s = DataType.GGUF_IQ2_S
gguf_iq3_s = DataType.GGUF_IQ3_S
gguf_iq3_xxs = DataType.GGUF_IQ3_XXS
gguf_iq2_xs = DataType.GGUF_IQ2_XS
gguf_iq1_m = DataType.GGUF_IQ1_M
gguf_q4_1 = DataType.GGUF_Q4_1
gguf_q5_0 = DataType.GGUF_Q5_0
gguf_q5_1 = DataType.GGUF_Q5_1
gguf_tq1_0 = DataType.GGUF_TQ1_0
GGUF_DTYPES = (
    gguf_q8_0,
    gguf_q4_0,
    gguf_iq4_nl,
    gguf_mxfp4,
    gguf_nvfp4,
    gguf_q6_k,
    gguf_q4_k,
    gguf_q5_k,
    gguf_q3_k,
    gguf_q2_k,
    gguf_iq4_xs,
    gguf_tq2_0,
    gguf_q2_0,
    gguf_q1_0,
    gguf_iq2_xxs,
    gguf_iq1_s,
    gguf_iq2_s,
    gguf_iq3_s,
    gguf_iq3_xxs,
    gguf_iq2_xs,
    gguf_iq1_m,
    gguf_q4_1,
    gguf_q5_0,
    gguf_q5_1,
    gguf_tq1_0,
)


def is_mx_dtype(dtype) -> bool:
    return dtype in MX_DTYPES


DumpTensorMode = ttnn._ttnn.tensor.DumpTensorMode

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
OverlappedTensor = ttnn._ttnn.tensor.OverlappedTensor

Shape = ttnn._ttnn.types.Shape
TensorSpec = ttnn._ttnn.tensor.TensorSpec
Tensor = ttnn._ttnn.tensor.Tensor


CoreGrid = ttnn._ttnn.types.CoreGrid
CoreType = ttnn._ttnn.types.CoreType

ThrottleLevel = ttnn._ttnn.operations.core.ThrottleLevel


DeviceComputeKernelConfig = ttnn._ttnn.operations.core.DeviceComputeKernelConfig
WormholeComputeKernelConfig = ttnn._ttnn.operations.core.WormholeComputeKernelConfig
BlackholeComputeKernelConfig = WormholeComputeKernelConfig


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
get_optimal_worker_cores_for_sharded_tensor = ttnn._ttnn.tensor.get_optimal_worker_cores_for_sharded_tensor

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
merge_program_descriptors = ttnn._ttnn.program_descriptor.merge_program_descriptors
cb_descriptor_from_sharded_tensor = ttnn._ttnn.program_descriptor.cb_descriptor_from_sharded_tensor
get_cb_address = ttnn._ttnn.program_descriptor.get_cb_address
UnpackToDestMode = ttnn._ttnn.program_descriptor.UnpackToDestMode
FaceGeometry = ttnn._ttnn.program_descriptor.FaceGeometry
compute_program_descriptor_hash = ttnn._ttnn.operations.generic.compute_program_descriptor_hash

TensorAccessorArgs = ttnn._ttnn.tensor_accessor_args.TensorAccessorArgs

# Host-side multicast helper (counterpart of kernel_lib/mcast/kernel/mcast_pipe).
attach_absent_mcast = ttnn._ttnn.mcast_host.attach_absent_mcast
TransferMode = ttnn._ttnn.mcast_host.TransferMode
Mcast = ttnn._ttnn.mcast_host.Mcast
McastCoreOrder = ttnn._ttnn.mcast_host.McastCoreOrder
McastSenderPlacement = ttnn._ttnn.mcast_host.McastSenderPlacement
McastFixedSenderConfig = ttnn._ttnn.mcast_host.McastFixedSenderConfig
McastExplicitFixedSenderConfig = ttnn._ttnn.mcast_host.McastExplicitFixedSenderConfig
McastRotatingSenderConfig = ttnn._ttnn.mcast_host.McastRotatingSenderConfig
McastConfig = ttnn._ttnn.mcast_host.McastConfig
McastDataReady = ttnn._ttnn.mcast_host.McastDataReady
