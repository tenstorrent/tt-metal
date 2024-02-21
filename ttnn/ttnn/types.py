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

Shape = ttnn._ttnn.types.Shape


Tensor = ttnn._ttnn.types.Tensor


@dataclasses.dataclass
class CoreGrid:
    y: int
    x: int

    @property
    def num_cores(self):
        return self.y * self.x


@dataclasses.dataclass
class CoreRange:
    start: CoreGrid
    end: CoreGrid


@dataclasses.dataclass
class ShardShape:
    y: int
    x: int


class ShardStrategy(Enum):
    HEIGHT = 1
    WIDTH = 2
    BLOCK = 3


class ShardOrientation(Enum):
    ROW_MAJOR = 1
    COLUMN_MAJOR = 2


DEFAULT_SHARD_ORIENTATION = ShardOrientation.ROW_MAJOR
