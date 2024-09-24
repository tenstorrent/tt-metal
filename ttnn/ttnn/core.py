# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List, Optional
import math

import ttnn

from ttnn.types import (
    DEVICE_STORAGE_TYPE,
    MULTI_DEVICE_STORAGE_TYPE,
    MemoryConfig,
    ShardStrategy,
    ShardOrientation,
    TensorMemoryLayout,
    BufferType,
)

set_printoptions = ttnn._ttnn.core.set_printoptions


def divup(a, b):
    return (a + b - 1) // b


def roundup(a, b):
    result = divup(a, b) * b
    return result


def has_storage_type_of(tensor: "ttnn.Tensor", storage_type) -> bool:
    return tensor.storage_type() == storage_type


def is_tensor_storage_on_device(tensor: "ttnn.Tensor") -> bool:
    return tensor.storage_type() in (DEVICE_STORAGE_TYPE, MULTI_DEVICE_STORAGE_TYPE)


def is_sharded(tensor) -> bool:
    return tensor.is_sharded()


get_memory_config = ttnn._ttnn.core.get_memory_config


def num_cores_to_corerange_set(
    target_num_cores: int,
    grid_size: ttnn.CoreCoord,
    row_wise: bool = False,
):
    """
    Create a CoreRangeSet containing the specified number of cores
    """
    return ttnn._ttnn.operations.core.num_cores_to_corerange_set(
        target_num_cores,
        grid_size,
        row_wise,
    )


def has_tile_padding(tensor, *, dim=None):
    if dim is not None:
        rank = tensor.shape.rank
        dim = dim if dim >= 0 else rank + dim
        return tensor.shape[dim] != tensor.shape.with_tile_padding()[dim]

    if len(tensor.shape) > 1:
        *_, h, w = tensor.shape
        *_, h_padded, w_padded = tensor.shape.with_tile_padding()
        return h != h_padded or w != w_padded
    return False


def dump_stack_trace_on_segfault():
    """
    Registers a handler to allow a stack trace to be logged to the console should the program fail because of a segfault.
    """
    ttnn._ttnn.core.dump_stack_trace_on_segfault()


def create_sharded_memory_config(
    shape: Union[ttnn.Shape, Tuple[int, ...], List[int]],
    core_grid: Union[ttnn.CoreGrid, ttnn.CoreRange],
    strategy: ShardStrategy,
    orientation: Optional[ShardOrientation] = None,
    halo: bool = False,
    use_height_and_width_as_shard_shape: bool = False,
) -> MemoryConfig:
    """
    Creates a MemoryConfig object with a sharding spec, required for sharded ops.

    Args:
        shape (ttnn.Shape | Tuple[int, ...] | List[int]): the shape of the tensor.
        core_grid (ttnn.CoreGrid | ttnn.CoreRange): the core_grid on which to distribute the sharded tensor on (writes to the cores L1s).
        strategy (ttnn.ShardStrategy): the sharding strategy of either height, width or block.
        orientation (ttnn.ShardOrientation, optional): the order in which to traverse the cores when reading/writing shards. Defaults to `None`.
        halo (bool, optional): if the shards have overlapping values. Defaults to `False`.
        use_height_and_width_as_shard_shape (bool, optional): if True, the height and width of the tensor will be used as the shard shape. Defaults to `False`. If is False, the shard shape will be calculated based on the core_grid and the tensor shape where tensor shape is seen as [math.prod(dims), width]

    Example:
        >>> tensor = ttnn.create_sharded_memory_config((5, 8), (320,64), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, False)

    Note:
        Currently sharding only supports L1 tensors.
    """

    if not isinstance(shape, (list, tuple, ttnn.Shape)):
        raise RuntimeError("Invalid input shape")

    if not isinstance(core_grid, (ttnn.CoreGrid, tuple, list, ttnn.CoreRangeSet)):
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
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    elif orientation == ShardOrientation.ROW_MAJOR:
        shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    elif orientation == ShardOrientation.COL_MAJOR:
        shard_orientation = ttnn.ShardOrientation.COL_MAJOR
    else:
        raise RuntimeError("Invalid shard orientation")

    shard_grid = None
    if isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    elif isinstance(core_grid, (list, tuple)):
        if len(core_grid) != 2:
            raise RuntimeError("Invalid core_grid")
        if not isinstance(core_grid[0], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        if not isinstance(core_grid[1], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")

        grid_coord_1 = ttnn.CoreCoord(core_grid[0].x - 1, core_grid[0].y - 1)
        grid_coord_2 = ttnn.CoreCoord(core_grid[1].x - 1, core_grid[0].y)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord_1),
                ttnn.CoreRange(ttnn.CoreCoord(0, core_grid[0].y), grid_coord_2),
            }
        )
    elif isinstance(core_grid, ttnn.CoreRangeSet):
        shard_grid = core_grid
        if not use_height_and_width_as_shard_shape:
            raise RuntimeError("height and width must be shard shape with CoreRangeSet")
    else:
        raise RuntimeError("Invalid core_grid type")

    *batch_sizes, height, width = shape
    batch_size = math.prod(batch_sizes)

    if use_height_and_width_as_shard_shape:
        if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
            shard_shape = height, width
        elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
            shard_shape = width, height
        else:
            raise RuntimeError("Invalid shard orientation")
    else:
        shard_height = batch_size * height
        shard_width = width
        if strategy == ShardStrategy.BLOCK:
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
                if shard_height % core_grid.y != 0:
                    raise RuntimeError("Invalid sharding core_grid")
                if shard_width % core_grid.x != 0:
                    raise RuntimeError("Invalid sharding core_grid")
                shard_shape = shard_height // core_grid.y, shard_width // core_grid.x

            elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
                if shard_height % core_grid.x != 0:
                    raise RuntimeError("Invalid sharding core_grid")
                if shard_width % core_grid.y != 0:
                    raise RuntimeError("Invalid sharding core_grid")
                shard_shape = shard_height // core_grid.x, shard_width // core_grid.y
            else:
                raise RuntimeError("Invalid shard orientation")
        elif strategy == ShardStrategy.HEIGHT:
            total_cores = core_grid.x * core_grid.y
            if shard_height % total_cores != 0:
                raise RuntimeError("Invalid sharding core_grid")
            shard_shape = shard_height // total_cores, shard_width
        elif strategy == ShardStrategy.WIDTH:
            total_cores = core_grid.x * core_grid.y
            if shard_width % total_cores != 0:
                raise RuntimeError("Invalid sharding core_grid")
            shard_shape = shard_height, shard_width // total_cores
        else:
            raise RuntimeError("Invalid sharding scheme")

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, halo)
    memory_config = MemoryConfig(tensor_memory_layout, BufferType.L1, shard_spec)
    return memory_config


# This function is based off the existing "create_sharded_memory_config". This new function has support for calculating shard shape when tensor shape is not divisible by num of cores.
# The existing function should be deprecated with this one. Not replacing right now to avoid a big change.
def create_sharded_memory_config_(
    shape: Union[ttnn.Shape, Tuple[int, ...], List[int]],
    core_grid: Union[ttnn.CoreGrid, ttnn.CoreRange],
    strategy: Union[ShardStrategy, TensorMemoryLayout],
    orientation,
    halo: bool = False,
    use_height_and_width_as_shard_shape: bool = False,
    tile_layout: bool = False,
) -> MemoryConfig:
    """
    create_sharded_memory_config(shape: Union[ttnn.Shape, Tuple[int, ...], List[int]], core_grid: Union[ttnn.CoreGrid, ttnn.CoreRange], strategy: ShardStrategy, orientation: Optional[ShardOrientation] = None, halo: bool = False, use_height_and_width_as_shard_shape: bool = False) -> MemoryConfig

    Creates a MemoryConfig object with a sharding spec, required for sharded ops.
    Currently sharding only supports L1 tensors.

    Args:
        * :attr:`shape`: the shape of the tensor
        * :attr:`core_grid`: the core_grid on which to distribute the sharded tensor on (writes to the cores L1s)
        * :attr:`strategy`: the sharding strategy of either height, width or block
        * :attr:`orientation`: the order in which to traverse the cores when reading/writing shards. Defaults to ttnn.ShardOrientation.ROW_MAJOR
        * :attr:`halo`: if the shards have overlapping values. Defaults to False
        * :attr:`use_height_and_width_as_shard_shape`: if True, the height and width of the tensor will be used as the shard shape. Defaults to False. If is False, the shard shape will be calculated based on the core_grid and the tensor shape where tensor shape is seen as [math.prod(dims), width]
        * :attr:`tile_layout`: if set to True, shard height will be set to multiple of 32. Last shard may be height padded.


    Example::
        >>> tensor = ttnn.create_sharded_memory_config((5, 8), (320,64), ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, False)
    """

    if not isinstance(shape, (list, tuple, ttnn.Shape)):
        raise RuntimeError("Invalid input shape")

    if not isinstance(core_grid, (ttnn.CoreGrid, tuple, list, ttnn.CoreRangeSet)):
        raise RuntimeError("Invalid core_grid type")

    if isinstance(strategy, ShardStrategy):
        if strategy == ShardStrategy.BLOCK:
            tensor_memory_layout = TensorMemoryLayout.BLOCK_SHARDED
        elif strategy == ShardStrategy.WIDTH:
            tensor_memory_layout = TensorMemoryLayout.WIDTH_SHARDED
        elif strategy == ShardStrategy.HEIGHT:
            tensor_memory_layout = TensorMemoryLayout.HEIGHT_SHARDED
        else:
            raise RuntimeError("Invalid sharding strategy")
    elif isinstance(strategy, TensorMemoryLayout):
        tensor_memory_layout = strategy
    else:
        raise RuntimeError("Invalid type of sharding strategy")

    if isinstance(orientation, ShardOrientation):
        if orientation is None:
            shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        elif orientation == ShardOrientation.ROW_MAJOR:
            shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
        elif orientation == ShardOrientation.COL_MAJOR:
            shard_orientation = ttnn.ShardOrientation.COL_MAJOR
        else:
            raise RuntimeError("Invalid shard orientation")
    elif isinstance(orientation, ttnn.ShardOrientation):
        shard_orientation = orientation
    else:
        raise RuntimeError("Invalid type of shard orientation")

    shard_grid = None
    if isinstance(core_grid, ttnn.CoreGrid):
        grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    elif isinstance(core_grid, (list, tuple)):
        if len(core_grid) != 2:
            raise RuntimeError("Invalid core_grid")
        if not isinstance(core_grid[0], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        if not isinstance(core_grid[1], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")

        grid_coord_1 = ttnn.CoreCoord(core_grid[0].x - 1, core_grid[0].y - 1)
        grid_coord_2 = ttnn.CoreCoord(core_grid[1].x - 1, core_grid[0].y)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord_1),
                ttnn.CoreRange(ttnn.CoreCoord(0, core_grid[0].y), grid_coord_2),
            }
        )
    elif isinstance(core_grid, ttnn.CoreRangeSet):
        shard_grid = core_grid
    else:
        raise RuntimeError("Invalid core_grid type")

    *batch_sizes, height, width = shape
    batch_size = math.prod(batch_sizes)

    if use_height_and_width_as_shard_shape:
        if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
            shard_shape = height, width
        elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
            shard_shape = width, height
        else:
            raise RuntimeError("Invalid shard orientation")
    else:
        tensor_height = batch_size * height
        tensor_width = width
        total_num_cores = shard_grid.num_cores()
        grid_size = shard_grid.bounding_box().grid_size()
        if tensor_memory_layout == TensorMemoryLayout.BLOCK_SHARDED:
            if grid_size.y * grid_size.x != total_num_cores:
                raise RuntimeError("Invalid CoreRangeSet for block sharding strategy")
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
                tensor_height_padded = roundup(tensor_height, grid_size.y * 32) if tile_layout else tensor_height
                shard_shape = divup(tensor_height_padded, grid_size.y), divup(tensor_width, grid_size.x)

            elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
                tensor_height_padded = roundup(tensor_height, grid_size.x * 32) if tile_layout else tensor_height
                shard_shape = divup(tensor_height_padded, grid_size.x), divup(tensor_width, grid_size.y)
            else:
                raise RuntimeError("Invalid shard orientation")
        elif tensor_memory_layout == TensorMemoryLayout.HEIGHT_SHARDED:
            tensor_height_padded = roundup(tensor_height, total_num_cores) if tile_layout else tensor_height
            shard_shape = divup(tensor_height_padded, total_num_cores), tensor_width
        elif tensor_memory_layout == TensorMemoryLayout.WIDTH_SHARDED:
            shard_shape = tensor_height, divup(tensor_width, total_num_cores)
        else:
            raise RuntimeError("Invalid sharding scheme")

    if tile_layout and shard_shape[0] % 32 != 0 and shard_shape[1] % 32 != 0:
        raise RuntimeError("Incorrent tensor shape")
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, halo)
    memory_config = MemoryConfig(tensor_memory_layout, BufferType.L1, shard_spec)
    return memory_config


dump_memory_config = ttnn._ttnn.tensor.dump_memory_config
load_memory_config = ttnn._ttnn.tensor.load_memory_config


__all__ = []
