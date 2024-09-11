# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from typing import Tuple, Union, Dict, Optional
import torch
import warnings
import math
import ttnn
from ttnn.operations.conv.sliding_window_op_utils import (
    calculate_shard_grid,
    roundup,
    get_output_dim as get_conv_output_dim,
)
from ttnn.operations.conv.tt_py_composite_conv import (
    SlidingWindowOpParams,
    find_closest_common_largest_divisor,
    find_closest_largest_divisor,
    find_closest_largest_divisor_with_num_padding,
)
from ttnn.device import (
    is_grayskull,
    is_wormhole_b0,
)


def _nearest_32(x):
    return math.ceil(x / 32) * 32


Conv2dConfig = ttnn._ttnn.operations.conv2d.Conv2dConfig

get_conv_padded_input_shape_and_mem_config = ttnn._ttnn.operations.conv2d.get_conv_padded_input_shape_and_mem_config
OptimizedConvParallelizationConfig = ttnn._ttnn.operations.conv2d.OptimizedConvParallelizationConfig
OptimizedConvBlockConfig = ttnn._ttnn.operations.conv2d.OptimizedConvBlockConfig


def convert_conv_weight_tensor_to_tiled_layout(conv_weight_tensor, in1_block_h, in1_block_w, output_dtype=None):
    """
    Converts convolution weights to 2d matrix tiled layout on host
    Returns a new tensor with the converted layout.

    +----------+----------------------+-----------+-------------+----------+
    | Argument | Description          | Data type | Valid range | Required |
    +==========+======================+===========+=============+==========+
    | a        | Input tensor         | Tensor    |             | Yes      |
    +----------+----------------------+-----------+-------------+----------+
    """
    return ttnn._ttnn.operations.conv2d.convert_conv_weight_tensor_to_tiled_layout(
        conv_weight_tensor, in1_block_h, in1_block_w, output_dtype
    )


def convert_conv_weight_tensor_to_special_padding_tiled_layout(
    conv_weight_tensor, in1_block_h, in1_block_w, output_dtype=None
):
    """
    Converts convolution weights to 2d matrix tiled layout on host with special block height padding
    Returns a new tensor with the converted layout.

    +----------+----------------------+-----------+-------------+----------+
    | Argument | Description          | Data type | Valid range | Required |
    +==========+======================+===========+=============+==========+
    | a        | Input tensor         | Tensor    |             | Yes      |
    +----------+----------------------+-----------+-------------+----------+
    """
    return ttnn._ttnn.operations.conv2d.convert_conv_weight_tensor_to_special_padding_tiled_layout(
        conv_weight_tensor, in1_block_h, in1_block_w, output_dtype
    )


def convert_conv_weight_tensor_to_grouped_layout(conv_weight_tensor, num_groups, output_dtype):
    """
    Converts convolution weights to grouped layout with padded zeros
    Returns a new tensor with the converted layout.

    +----------+----------------------+-----------+-------------+----------+
    | Argument | Description          | Data type | Valid range | Required |
    +==========+======================+===========+=============+==========+
    | a        | Input tensor         | Tensor    |             | Yes      |
    +----------+----------------------+-----------+-------------+----------+
    """
    return ttnn._ttnn.operations.conv2d.convert_conv_weight_tensor_to_grouped_layout(
        conv_weight_tensor, num_groups, output_dtype
    )


# internal. not user facing
class ParallelConfig:
    def __init__(
        self,
        num_cores_y: int,
        num_cores_x: int,
        num_cores_nhw: int,
        shard_scheme: ttnn.TensorMemoryLayout,
        shard_orientation: ttnn.ShardOrientation,
    ):
        # TODO: using core range set would be better
        self.grid_size = ttnn.CoreCoord(num_cores_x, num_cores_y)
        self.num_cores_nhw = num_cores_nhw
        self.shard_scheme = shard_scheme
        self.shard_orientation = shard_orientation

    def __eq__(self, other):
        if not isinstance(other, ParallelConfig):
            return NotImplemented

        return (
            self.grid_size.y == other.grid_size.y
            and self.grid_size.x == other.grid_size.x
            and self.num_cores_nhw == other.num_cores_nhw
            and self.shard_scheme == other.shard_scheme
            and self.shard_orientation == other.shard_orientation
        )

    def __ne__(self, other):
        if not isinstance(other, ParallelConfig):
            return NotImplemented
        return not (self == other)


# internal helper function. not exposed to user.
def get_shard_grid_from_core_grid(core_grid):
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
    return shard_grid


# internal helper function. not exposed to user.
def determine_parallel_config(
    is_1d_systolic,
    batch_size,
    input_channels,
    output_height,
    output_width,
    output_channels,
    device,
    config_override=None,
    is_out_tiled=True,
):
    if config_override is None:
        config_override = {}
    for k in config_override.keys():
        assert k == "grid_size" or k == "num_cores_nhw"

    conv_out_2d_matrix_height = batch_size * output_height * output_width
    # pad height to 32
    conv_out_2d_matrix_height = _nearest_32(conv_out_2d_matrix_height)

    if is_out_tiled:
        conv_out_2d_matrix_height_ntiles = (int)(conv_out_2d_matrix_height / 32)
        conv_out_2d_matrix_width_ntiles = (int)(_nearest_32(output_channels) / 32)
    else:
        conv_out_2d_matrix_height_ntiles = conv_out_2d_matrix_height
        conv_out_2d_matrix_width_ntiles = output_channels

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = (compute_with_storage_grid_size.x, compute_with_storage_grid_size.y)
    max_num_cores = device_grid_size[0] * device_grid_size[1]

    def calculate_num_cores_nhw(override):
        num_cores_nhw = (
            find_closest_largest_divisor(conv_out_2d_matrix_height_ntiles, max_num_cores)
            if is_1d_systolic
            else find_closest_largest_divisor_with_num_padding(conv_out_2d_matrix_height_ntiles, device_grid_size[0])
        )
        if override is not None and num_cores_nhw != override:
            warnings.warn(f"Overriding config: num_cores_nhw from {num_cores_nhw} to user provided config={override}")
            num_cores_nhw = override
        return num_cores_nhw

    def calculate_grid_size(num_cores_nhw, override):
        if is_1d_systolic:
            grid_size = [
                device_grid_size[0] if num_cores_nhw >= device_grid_size[0] else num_cores_nhw,
                math.ceil(num_cores_nhw / device_grid_size[0]),
            ]  # for 1d systolic array, grid size is the tightest bound of num_cores_nhw as a rectangle (x,y)
            assert (
                num_cores_nhw <= grid_size[0] * grid_size[1]
            ), "Error: For 1d systolic conv, num_cores_nhw must be <= grid size"
        else:
            grid_size = [
                num_cores_nhw,
                find_closest_common_largest_divisor(
                    conv_out_2d_matrix_width_ntiles, _nearest_32(input_channels) // 32, device_grid_size[1]
                ),
            ]
            assert (
                num_cores_nhw == grid_size[0]
            ), "Error: For 2d systolic conv, num_cores_nhw must be == # of cols in grid size"

        if override is not None and grid_size != override:
            warnings.warn(f"Overriding config: grid_size from {grid_size} to user provided config={override}")
            grid_size = override
        return grid_size

    num_cores_nhw = calculate_num_cores_nhw(config_override.get("num_cores_nhw", None))
    grid_size = calculate_grid_size(num_cores_nhw, config_override.get("grid_size", None))
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED if is_1d_systolic else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR if is_1d_systolic else ttnn.ShardOrientation.COL_MAJOR
    return ParallelConfig(grid_size[1], grid_size[0], num_cores_nhw, shard_scheme, shard_orientation)


# internal helper function. not exposed to user.
def get_grid_size_and_num_cores_nhw_from_core_grid(core_grid, height_sharded):
    if isinstance(core_grid, ttnn.CoreGrid):
        if height_sharded:
            num_cores_nhw = core_grid.x * core_grid.y
        else:
            num_cores_nhw = core_grid.x
        grid_size = core_grid
    elif isinstance(core_grid, (list, tuple)):
        if len(core_grid) != 2:
            raise RuntimeError("Invalid core_grid")
        if not isinstance(core_grid[0], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        if not isinstance(core_grid[1], ttnn.CoreGrid):
            raise RuntimeError("Invalid core_grid type")
        assert height_sharded
        num_cores_nhw = (core_grid[0].x * core_grid[0].y) + core_grid[1].x
    elif isinstance(core_grid, ttnn.CoreRangeSet):
        grid_size = core_grid.bounding_box().grid_size()
        num_cores = core_grid.num_cores()
        if height_sharded:
            num_cores_nhw = num_cores
        else:
            num_cores_nhw = grid_size.x
    else:
        raise RuntimeError("Invalid core_grid type")
    return grid_size, num_cores_nhw


# internal helper function. not exposed to user.
def create_sharded_memory_config_from_parallel_config(tensor_shape, parallel_config, tile_size):
    logger.debug(
        f"py create_sharded_memory_config_from_parallel_config: {tensor_shape}, {parallel_config.num_cores_nhw} {parallel_config.grid_size}, {tile_size}"
    )
    # tensor_shape is [N, H, W, C]
    assert len(tensor_shape) == 4
    assert tensor_shape[0] == 1 and tensor_shape[1] == 1  # todo: add support for generic non-2d shapes
    channels = tensor_shape[3]
    channels_padded = roundup(channels, tile_size)
    num_cores_nhw = parallel_config.num_cores_nhw
    num_cores_x = parallel_config.grid_size.x
    num_cores_y = parallel_config.grid_size.y
    shard_scheme = parallel_config.shard_scheme
    shard_orientation = parallel_config.shard_orientation
    is_1d_systolic = shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if is_1d_systolic:
        logical_grid_size = (num_cores_nhw, 1)
    else:
        logical_grid_size = (num_cores_x, num_cores_y)

    shard_grid, shard_layout = calculate_shard_grid((num_cores_x, num_cores_y), num_cores_nhw)
    assert shard_layout == shard_scheme
    nhw_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    nhw_padded = roundup(nhw_shape, num_cores_nhw * tile_size)
    nhw_shard = nhw_padded // num_cores_nhw
    assert channels_padded % logical_grid_size[1] == 0
    shard_shape = [nhw_shard, channels_padded // logical_grid_size[1]]
    shard_halo = False
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)
    return ttnn.MemoryConfig(shard_scheme, ttnn.BufferType.L1, shard_spec)


@ttnn.register_python_operation(name="ttnn.conv2d")
def conv2d(
    *,
    input_tensor: ttnn.Tensor,  # may or may not be sharded
    weight_tensor: ttnn.Tensor,
    device: ttnn.Device,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    input_height: int,
    input_width: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    bias_tensor: ttnn.Tensor = None,
    conv_config: Conv2dConfig = None,  # config overrides by user
    conv_op_cache={},  # basic conv object caching in python needed for intermediate refactoring. Not needed after full op refactoring in C++.
    debug=False,  # ignored
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    return ttnn._ttnn.operations.conv2d.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_tensor=bias_tensor,
        conv_config=conv_config,
    )


__all__ = []
