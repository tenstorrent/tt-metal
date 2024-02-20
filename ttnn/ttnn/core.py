# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List, Optional
import math

import ttnn

from ttnn.types import (
    DEVICE_STORAGE_TYPE,
    MemoryConfig,
    ShardStrategy,
    ShardOrientation,
    TensorMemoryLayout,
    BufferType,
)


def has_storage_type_of(tensor: "ttnn.Tensor", storage_type) -> bool:
    return tensor.value.storage_type() == storage_type


def is_sharded(tensor) -> bool:
    return tensor.value.is_sharded()


def get_memory_config(tensor) -> ttnn.ttl.tensor.MemoryConfig:
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
    shape: Union[ttnn.Shape, Tuple[int, ...], List[int]],
    core_grid: Union[ttnn.CoreGrid, ttnn.CoreRange],
    strategy: ShardStrategy,
    orientation: Optional[ShardOrientation] = None,
    halo: bool = False,
    use_height_and_width_as_shard_shape: bool = False,
) -> MemoryConfig:
    """
    create_sharded_memory_config(shape: Union[ttnn.Shape, Tuple[int, ...], List[int]], core_grid: Union[ttnn.CoreGrid, ttnn.CoreRange], strategy: ShardStrategy, orientation: Optional[ShardOrientation] = None, halo: bool = False) -> MemoryConfig

    Creates a MemoryConfig object with a sharding spec, required for sharded ops.
    Currently sharding only supports L1 tensors.

    Args:
        * :attr:`shape`: the shape of the tensor
        * :attr:`core_grid`: the core_grid on which to distribute the sharded tensor on (writes to the cores L1s)
        * :attr:`strategy`: the sharding strategy of either height, width or block
        * :attr:`orientation`: the order in which to traverse the cores when reading/writing shards. Defaults to ttnn.ShardOrientation.ROW_MAJOR
        * :attr:`halo`: if the shards have overlapping values. Defaults to False
        * :attr:`use_height_and_width_as_shard_shape`: if True, the height and width of the tensor will be used as the shard shape. Defaults to False. If is False, the shard shape will be calculated based on the core_grid and the tensor shape where tensor shape is seen as [*height, width].


    Example::
        >>> tensor = ttnn.create_sharded_memory_config((5, 8), (320,64), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, False)
    """

    if not isinstance(shape, (list, tuple, ttnn.Shape)):
        raise RuntimeError("Invalid input shape")

    if not isinstance(core_grid, (ttnn.CoreGrid, tuple, list)):
        raise RuntimeError("Invalid core_grid type")

    if strategy == ShardStrategy.BLOCK:
        tensor_memory_layout = TensorMemoryLayout.BLOCK_SHARDED
    elif strategy == ShardStrategy.WIDTH:
        tensor_memory_layout = TensorMemoryLayout.WIDTH_SHARDED
    elif strategy == ShardStrategy.HEIGHT:
        tensor_memory_layout = TensorMemoryLayout.HEIGHT_SHARDED
    else:
        raise RuntimeError("Invalid sharding strategy")

    if orientation is None:
        shard_orientation = ttnn.ttl.tensor.ShardOrientation.ROW_MAJOR
    elif orientation == ShardOrientation.ROW_MAJOR:
        shard_orientation = ttnn.ttl.tensor.ShardOrientation.ROW_MAJOR
    elif orientation == ShardOrientation.COLUMN_MAJOR:
        shard_orientation = ttnn.ttl.tensor.ShardOrientation.COL_MAJOR
    else:
        raise RuntimeError("Invalid shard orientation")

    shard_grid = None
    if isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.ttl.tensor.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        shard_grid = ttnn.ttl.tensor.CoreRangeSet(
            {ttnn.ttl.tensor.CoreRange(ttnn.ttl.tensor.CoreCoord(0, 0), grid_coord)}
        )
    elif isinstance(core_grid, (list, tuple)):
        if len(core_grid) != 2:
            raise RuntimeError("Invalid core_grid")
        if not isinstance(core_grid[0], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        if not isinstance(core_grid[1], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")

        grid_coord_1 = ttnn.ttl.tensor.CoreCoord(core_grid[0].x - 1, core_grid[0].y - 1)
        grid_coord_2 = ttnn.ttl.tensor.CoreCoord(core_grid[1].x - 1, core_grid[0].y)
        shard_grid = ttnn.ttl.tensor.CoreRangeSet(
            {
                ttnn.ttl.tensor.CoreRange(ttnn.ttl.tensor.CoreCoord(0, 0), grid_coord_1),
                ttnn.ttl.tensor.CoreRange(ttnn.ttl.tensor.CoreCoord(0, core_grid[0].y), grid_coord_2),
            }
        )
    else:
        raise RuntimeError("Invalid core_grid type")

    *batch_sizes, height, width = shape
    batch_size = math.prod(batch_sizes)

    if use_height_and_width_as_shard_shape:
        if shard_orientation == ttnn.ttl.tensor.ShardOrientation.ROW_MAJOR:
            shard_shape = height, width
        elif shard_orientation == ttnn.ttl.tensor.ShardOrientation.COL_MAJOR:
            shard_shape = width, height
        else:
            raise RuntimeError("Invalid shard orientation")
    else:
        shard_height = batch_size * height
        shard_width = width
        if shard_orientation == ttnn.ttl.tensor.ShardOrientation.ROW_MAJOR:
            if shard_height % core_grid.y != 0:
                raise RuntimeError("Invalid sharding core_grid")
            if shard_width % core_grid.x != 0:
                raise RuntimeError("Invalid sharding core_grid")
            shard_shape = shard_height // core_grid.y, shard_width // core_grid.x

        elif shard_orientation == ttnn.ttl.tensor.ShardOrientation.COL_MAJOR:
            if shard_height % core_grid.x != 0:
                raise RuntimeError("Invalid sharding core_grid")
            if shard_width % core_grid.y != 0:
                raise RuntimeError("Invalid sharding core_grid")
            shard_shape = shard_height // core_grid.x, shard_width // core_grid.y
        else:
            raise RuntimeError("Invalid shard orientation")

    shard_spec = ttnn.ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, halo)
    memory_config = MemoryConfig(tensor_memory_layout, BufferType.L1, shard_spec)
    return memory_config


__all__ = []
