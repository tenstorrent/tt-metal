# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

import tt_lib as ttl

from ttnn.types import (
    DEVICE_STORAGE_TYPE,
    MemoryConfig,
    ShardStrategy,
    ShardOrientation,
    DEFAULT_SHARD_ORIENTATION,
    TensorMemoryLayout,
    BufferType,
)


def has_storage_type_of(tensor: "ttnn.Tensor", storage_type) -> bool:
    return tensor.value.storage_type() == storage_type


def is_sharded(tensor) -> bool:
    return tensor.value.is_sharded()


def get_memory_config(tensor) -> ttl.tensor.MemoryConfig:
    if has_storage_type_of(tensor, DEVICE_STORAGE_TYPE):
        return tensor.value.memory_config()
    else:
        raise RuntimeError("Tensor is not on device!")


def has_padding(tensor):
    if len(tensor.shape) > 1:
        *_, h, w = tensor.shape
        *_, h_padded, w_padded = tensor.shape.with_tile_padding()
        return h != h_padded or w != w_padded
    return False


def create_sharded_memory_config(
    grid: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
    shard_shape: Tuple[int, int],
    strategy: ShardStrategy,
    orientation: ShardOrientation = DEFAULT_SHARD_ORIENTATION,
    halo: bool = False,
) -> MemoryConfig:
    """
    create_sharded_memory_config(grid: Tuple[int, int], shard_shape: Tuple[int, int], sharding_strategy: ShardStrategy, shard_orientation: ShardOrientation, halo: bool) -> MemoryConfig

    Creates a MemoryConfig object with a sharding spec, required for sharded ops.
    Currently sharding only supports L1 tensors.

    Args:
        * :attr:`grid`: the grid on which to distribute the sharded tensor on (writes to the cores L1s)
        * :attr:`shard_shape`: the shape in elements of a respective shard. This is a 2D shape, the upper dimension is the multiplication of dims 0 to rank-1, and the inner dimension is the last dim
        * :attr:`strategy`: the sharding strategy of either height, width or block
        * :attr:`orientation`: the order in which to traverse the cores when reading/writing shards. Defaults to ttnn.ShardOrientation.ROW_MAJOR
        * :attr:`halo`: if the shards have overlapping values. Defaults to False


    Example::
        >>> tensor = ttnn.create_sharded_memory_config((5, 8), (320,64), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, False)
    """
    if strategy == ShardStrategy.BLOCK:
        tensor_memory_layout = TensorMemoryLayout.BLOCK_SHARDED
    elif strategy == ShardStrategy.WIDTH:
        tensor_memory_layout = TensorMemoryLayout.WIDTH_SHARDED
    elif strategy == ShardStrategy.HEIGHT:
        tensor_memory_layout = TensorMemoryLayout.HEIGHT_SHARDED
    else:
        raise RuntimeError("Invalid sharding strategy")

    if orientation == ShardOrientation.ROW_MAJOR:
        shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    elif orientation == ShardOrientation.COLUMN_MAJOR:
        shard_orientation = ttl.tensor.ShardOrientation.COL_MAJOR
    else:
        raise RuntimeError("Invalid shard orientation")

    shard_grid = None
    if isinstance(grid[0], Tuple):
        grid_coord_1 = ttl.tensor.CoreCoord(grid[0][1] - 1, grid[0][0] - 1)
        grid_coord_2 = ttl.tensor.CoreCoord(grid[1][1] - 1, grid[0][0])
        shard_grid = ttl.tensor.CoreRangeSet(
            {
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), grid_coord_1),
                ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, grid[0][0]), grid_coord_2),
            }
        )
    else:
        grid_coord = ttl.tensor.CoreCoord(grid[1] - 1, grid[0] - 1)
        shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, halo)
    mem_config = MemoryConfig(tensor_memory_layout, BufferType.L1, shard_spec)
    return mem_config


__all__ = []
