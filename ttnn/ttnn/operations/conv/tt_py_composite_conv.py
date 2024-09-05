# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List, Union
from ttnn.operations.conv.tt_py_op import TTPyOp
from ttnn.operations.conv.tt_py_untilize_with_halo import TTPyUntilizeWithHalo
from ttnn.operations.conv.untilize_with_halo_config_generation_and_validation import (
    trace_conv_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_conv_into_shards_and_generate_tensor_metadata,
)
from ttnn.operations.conv.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
)
from ttnn.operations.conv.sliding_window_op_utils import (
    SlidingWindowOpParams,
    SlidingWindowOpParamsWithParallelConfig,
    get_hash_from_sliding_window_op_params,
    calculate_shard_grid,
    calculate_memory_config,
)
from tt_lib.utils import (
    _nearest_32,
    _nearest_y,
    find_closest_largest_divisor,
    find_closest_largest_divisor_with_num_padding,
    divup,
)

import ttnn._ttnn.deprecated as ttl
import ttnn
import torch
import math
import warnings


def find_closest_common_largest_divisor(num1: int, num2: int, start_divisor: int):
    divisor = start_divisor
    while num1 % divisor != 0 or num2 % divisor != 0:
        divisor = divisor - 1
    return divisor


def determine_largest_subblock_size(block_height, block_width, fp32_accum=False):
    subblocks = [
        (2, 4),
        (4, 2),
        (1, 8),
        (8, 1),
        (1, 7),
        (7, 1),
        (2, 3),
        (3, 2),
        (1, 6),
        (6, 1),
        (1, 5),
        (5, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 3),
        (3, 1),
        (1, 2),
        (2, 1),
        (1, 1),
    ]
    for subblock_height, subblock_width in subblocks:
        if fp32_accum and subblock_height * subblock_width > 4:
            continue
        if block_height % subblock_height == 0 and block_width % subblock_width == 0:
            if subblock_width != block_width and subblock_height != 1:
                continue
            break
    return subblock_height, subblock_width


def compute_conv_output_height_width(input_height, input_width, sliding_window_op_params):
    stride_h = sliding_window_op_params.stride_h
    stride_w = sliding_window_op_params.stride_w
    pad_h = sliding_window_op_params.pad_h
    pad_w = sliding_window_op_params.pad_w
    filter_h = sliding_window_op_params.window_h
    filter_w = sliding_window_op_params.window_w
    output_height = ((int)((input_height - filter_h + 2 * pad_h) / stride_h)) + 1
    output_width = ((int)((input_width - filter_w + 2 * pad_w) / stride_w)) + 1
    return output_height, output_width


def determine_parallel_config(
    is_1d_systolic,
    output_channels,
    input_channels,
    sliding_window_op_params,
    device,
    config_override=None,
    is_out_tiled=True,
    transpose_mcast=True,
):
    if config_override is None:
        config_override = {}

    batch_size = sliding_window_op_params.batch_size
    input_height = sliding_window_op_params.input_h
    input_width = sliding_window_op_params.input_w
    output_height, output_width = compute_conv_output_height_width(input_height, input_width, sliding_window_op_params)
    conv_out_2d_matrix_height = batch_size * output_height * output_width
    tile_size = 32 if is_out_tiled else 1
    conv_out_2d_matrix_width_ntiles = divup(output_channels, tile_size)

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = (compute_with_storage_grid_size.x, compute_with_storage_grid_size.y)
    max_num_cores = device_grid_size[0] * device_grid_size[1]

    if "grid_size" in config_override:
        grid_size = config_override["grid_size"]
        max_num_cores = grid_size[0] * grid_size[1]
        # print(f"max_num_cores: {max_num_cores}")

    def calculate_num_cores_nhw(override):
        conv_out_2d_matrix_height_ntiles = divup(conv_out_2d_matrix_height, tile_size)
        num_cores_nhw = (
            find_closest_largest_divisor(conv_out_2d_matrix_height_ntiles, max_num_cores)
            if is_1d_systolic
            else (
                find_closest_largest_divisor_with_num_padding(conv_out_2d_matrix_height_ntiles, device_grid_size[0])
                if transpose_mcast
                else find_closest_largest_divisor_with_num_padding(
                    conv_out_2d_matrix_height_ntiles, device_grid_size[1]
                )
            )
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
            if transpose_mcast:
                grid_size = [
                    num_cores_nhw,
                    find_closest_common_largest_divisor(
                        conv_out_2d_matrix_width_ntiles,
                        _nearest_32(input_channels) // 32,
                        device_grid_size[1],
                    ),
                ]
            else:
                grid_size = [
                    find_closest_common_largest_divisor(
                        conv_out_2d_matrix_width_ntiles,
                        _nearest_32(input_channels) // 32,
                        device_grid_size[0],
                    ),
                    num_cores_nhw,
                ]
        if override is not None and grid_size != override:
            warnings.warn(f"Overriding config: grid_size from {grid_size} to user provided config={override}")
            grid_size = override
        return grid_size

    def calculate_per_core_out_matrix_height_ntiles(logical_grid_x, override):
        per_core_out_matrix_height_ntiles = divup(divup(conv_out_2d_matrix_height, logical_grid_x), tile_size)
        total_padded_height = per_core_out_matrix_height_ntiles * tile_size * logical_grid_x
        assert (
            total_padded_height - conv_out_2d_matrix_height
        ) <= per_core_out_matrix_height_ntiles * tile_size, f"total_padded_height({total_padded_height}) - original_height({conv_out_2d_matrix_height}) = {total_padded_height - conv_out_2d_matrix_height}, which exceeds the per-core shard shape height({per_core_out_matrix_height_ntiles * tile_size}).  This will result in cores doing work on padded data only which is illegal. This is a result of choosing override num_cores_nhw({num_cores_nhw}) that cannot satisfy this height after tile padding."
        if override is not None:
            assert override % tile_size == 0, "per_core_out_matrix_height must be divisible by 32 (tile height)"
            if (override // tile_size) != per_core_out_matrix_height_ntiles:
                warnings.warn(
                    f"Overriding config: per_core_out_matrix_height from {per_core_out_matrix_height_ntiles * tile_size} to user provided config={override}"
                )
                per_core_out_matrix_height_ntiles = override // tile_size
        return per_core_out_matrix_height_ntiles

    def calculate_per_core_out_matrix_width_ntiles(logical_grid_y, override):
        per_core_out_matrix_width_ntiles = conv_out_2d_matrix_width_ntiles // logical_grid_y
        if override is not None:
            assert override % 32 == 0, "per_core_weight_matrix_width must be divisible by 32 (tile width)"
            if (override // 32) != per_core_out_matrix_width_ntiles:
                warnings.warn(
                    f"Overriding config: per_core_weight_matrix_width from {per_core_out_matrix_width_ntiles * 32} to user provided config={override}"
                )
                per_core_out_matrix_width_ntiles = override // 32
        return per_core_out_matrix_width_ntiles

    num_cores_nhw = calculate_num_cores_nhw(config_override.get("num_cores_nhw", None))
    grid_size = calculate_grid_size(num_cores_nhw, config_override.get("grid_size", None))
    logical_grid_x = num_cores_nhw if is_1d_systolic else (grid_size[0] if transpose_mcast else grid_size[1])
    logical_grid_y = 1 if is_1d_systolic else (grid_size[1] if transpose_mcast else grid_size[0])
    per_core_out_matrix_height_ntiles = calculate_per_core_out_matrix_height_ntiles(
        logical_grid_x, config_override.get("per_core_out_matrix_height", None)
    )
    per_core_out_matrix_width_ntiles = calculate_per_core_out_matrix_width_ntiles(
        logical_grid_y, config_override.get("per_core_out_matrix_width", None)
    )

    logger.debug(
        f"PARALLEL CONFIG :: {is_1d_systolic} :: {input_channels} :: {output_channels} :: {sliding_window_op_params} :: {config_override} -> {num_cores_nhw} :: {grid_size} :: {per_core_out_matrix_height_ntiles} :: {per_core_out_matrix_width_ntiles}"
    )

    return ttnn.operations.conv2d.OptimizedConvParallelizationConfig(
        grid_size=grid_size,
        num_cores_nhw=num_cores_nhw,
        per_core_out_matrix_height_ntiles=per_core_out_matrix_height_ntiles,
        per_core_out_matrix_width_ntiles=per_core_out_matrix_width_ntiles,
    )


def determine_per_core_block_config(
    is_1d_systolic,
    grid_size,
    per_core_out_matrix_height_ntiles,
    per_core_out_matrix_width_ntiles,
    input_channels,
    sliding_window_op_params,
    use_shallow_conv_variant,
    padded_input_channels,
    config_override=None,
    fp32_accum=False,
    transpose_mcast=True,
):
    if config_override is None:
        config_override = {}

    act_block_h_override = 0
    if "act_block_h" in config_override:
        act_block_h_override = config_override["act_block_h"]
        assert act_block_h_override % 32 == 0, "act_block_h must be divisible by 32 (tile height)"
    act_block_h_ntiles_override = act_block_h_override // 32
    act_block_h_ntiles = (
        act_block_h_ntiles_override if act_block_h_ntiles_override > 0 else per_core_out_matrix_height_ntiles
    )
    act_block_w_ntiles = (int)(
        (
            _nearest_32(padded_input_channels * sliding_window_op_params.window_w)
            if is_1d_systolic
            else padded_input_channels
        )
        / 32
    )
    if is_1d_systolic:
        act_c_num_blocks = 1
    else:
        act_c_num_blocks = grid_size.y if transpose_mcast else grid_size.x
        assert (
            padded_input_channels % act_c_num_blocks == 0
        ), f"Cannot parallelize conv as a 2d systolic array: Input channels {padded_input_channels} must be divisible by act_c_num_blocks {act_c_num_blocks}."
    out_block_h_ntiles = per_core_out_matrix_height_ntiles
    assert out_block_h_ntiles % act_block_h_ntiles == 0, "act_block_h must evenly divide out_block_h"
    weight_block_w_ntiles = per_core_out_matrix_width_ntiles
    out_subblock_h_ntiles, out_subblock_w_ntiles = determine_largest_subblock_size(
        act_block_h_ntiles, weight_block_w_ntiles, fp32_accum
    )
    if use_shallow_conv_variant and (act_block_h_ntiles // out_subblock_h_ntiles % 2 != 0):
        assert is_1d_systolic
        # TODO: fix this temporary hack for shallow conv
        assert act_block_h_ntiles % 2 == 0
        out_subblock_h_ntiles = act_block_h_ntiles // 2
        assert out_subblock_h_ntiles * out_subblock_w_ntiles <= 8

    if "act_block_w" in config_override:
        act_block_w_override = config_override["act_block_w"]
        assert act_block_w_override % 32 == 0, "act_block_w must be divisible by 32 (tile width)"
        if (act_block_w_override // 32) != act_block_w_ntiles:
            warnings.warn(
                f"Overriding config: act_block_w from {act_block_w_ntiles * 32} to user provided config={act_block_w_override}"
            )
            act_block_w_ntiles = act_block_w_override // 32
    if "out_subblock_h" in config_override:
        assert (
            "out_subblock_w" in config_override
        ), "out_subblock_w must also be provided as override config if out_subblock_h is provided"
        out_subblock_h_override = config_override["out_subblock_h"]
        assert out_subblock_h_override % 32 == 0, "out_subblock_h must be divisible by 32 (tile height)"
        out_subblock_w_override = config_override["out_subblock_w"]
        assert out_subblock_w_override % 32 == 0, "out_subblock_w must be divisible by 32 (tile width)"
        if (out_subblock_h_override // 32) != out_subblock_h_ntiles:
            warnings.warn(
                f"Overriding config: out_subblock_h from {out_block_h_ntiles * 32} to user provided config={out_subblock_h_override}"
            )
        if (out_subblock_w_override // 32) != out_subblock_w_ntiles:
            warnings.warn(
                f"Overriding config: out_subblock_w from {out_subblock_w_ntiles * 32} to user provided config={out_subblock_w_override}"
            )
    if "out_subblock_w" in config_override:
        assert (
            "out_subblock_h" in config_override
        ), "out_subblock_h must also be provided as override config if out_subblock_w is provided"
    conv_blocking_config = ttnn.operations.conv2d.OptimizedConvBlockConfig(
        act_block_h_ntiles=act_block_h_ntiles,
        act_block_w_ntiles=act_block_w_ntiles,
        out_subblock_h_ntiles=out_subblock_h_ntiles,
        out_subblock_w_ntiles=out_subblock_w_ntiles,
    )
    return conv_blocking_config


def determine_1x1conv_as_matmul_config(
    conv_parallelization_config,
    conv_blocking_config,
    use_1d_systolic_array,
    fuse_relu,
    transpose_mcast=True,
):
    if use_1d_systolic_array:
        matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=conv_parallelization_config.grid_size,
            in0_block_w=conv_blocking_config.act_block_w_ntiles,
            out_subblock_h=conv_blocking_config.out_subblock_h_ntiles,
            out_subblock_w=conv_blocking_config.out_subblock_w_ntiles,
            per_core_M=conv_parallelization_config.per_core_out_matrix_height_ntiles,
            per_core_N=conv_parallelization_config.per_core_out_matrix_width_ntiles,
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if fuse_relu else None,
            mcast_in0=False,
        )
    else:
        grid_size_along_c = (
            conv_parallelization_config.grid_size.y if transpose_mcast else conv_parallelization_config.grid_size.x
        )
        assert (
            conv_blocking_config.act_block_w_ntiles % grid_size_along_c == 0
        ), "Expected act block width to be divisible by act channel num blocks."
        matmul_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=conv_parallelization_config.grid_size,
            in0_block_w=conv_blocking_config.act_block_w_ntiles
            // grid_size_along_c,  ##conv_parallelization_config.grid_size.y,
            out_subblock_h=conv_blocking_config.out_subblock_h_ntiles,
            out_subblock_w=conv_blocking_config.out_subblock_w_ntiles,
            per_core_M=conv_parallelization_config.per_core_out_matrix_height_ntiles,
            per_core_N=conv_parallelization_config.per_core_out_matrix_width_ntiles,
            transpose_mcast=transpose_mcast,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if fuse_relu else None,
        )
    return matmul_config
