# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import tt_lib as ttl

from enum import Enum

Device = ttl.device.Device


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

Shape = ttl.ttnn.tensor.Shape


class Cpu:
    ...


def has_storage_type_of(tensor: "ttnn.Tensor", storage_type) -> bool:
    return tensor.value.storage_type() == storage_type


class Tensor(ttl.ttnn.tensor.Tensor):
    @property
    def device(self: "Tensor") -> DataType:
        if has_storage_type_of(self, DEVICE_STORAGE_TYPE):
            return self.value.device()
        else:
            return Cpu()

    def is_contiguous(self: "Shape") -> bool:
        if self.layout == ROW_MAJOR_LAYOUT:
            return self.value.shape() == self.value.shape_without_padding()
        else:
            return False

    def is_sharded(self) -> bool:
        return self.value.is_sharded()

    @property
    def memory_config(self) -> ttl.tensor.MemoryConfig:
        if has_storage_type_of(self, DEVICE_STORAGE_TYPE):
            return self.value.memory_config()
        else:
            raise RuntimeError("Tensor is not on device!")


class ShardStrategy(Enum):
    HEIGHT = 1
    WIDTH = 2
    BLOCK = 3


class ShardOrientation(Enum):
    ROW_MAJOR = 1
    COLUMN_MAJOR = 2


DEFAULT_SHARD_ORIENTATION = ShardOrientation.ROW_MAJOR
