# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections import namedtuple

import ttnn
from tt_lib.utils import _nearest_y, roundup

SlidingWindowOpParams = namedtuple(
    "SlidingWindowOpParams", "stride_h stride_w pad_h pad_w window_h window_w batch_size input_h input_w"
)
SlidingWindowOpParamsWithParallelConfig = namedtuple(
    "SlidingWindowOpParamsWithParallelConfig",
    "stride_h stride_w pad_h pad_w window_h window_w batch_size input_h input_w num_cores_w num_cores_h num_cores_nhw act_reshard_num_cores_nhw",
    defaults=[0],
)


def get_output_dim(input, window, stride=1, pad=0, dilation=1):
    return (input + (2 * pad) - dilation * (window - 1) - 1) // stride + 1


def get_hash_from_sliding_window_op_params(sliding_window_op_params: SlidingWindowOpParamsWithParallelConfig):
    return f"{sliding_window_op_params.stride_h}_{sliding_window_op_params.stride_w}_{sliding_window_op_params.pad_h}_{sliding_window_op_params.pad_w}_{sliding_window_op_params.window_h}_{sliding_window_op_params.window_w}_{sliding_window_op_params.batch_size}_{sliding_window_op_params.input_h}_{sliding_window_op_params.input_w}_{sliding_window_op_params.num_cores_w}_{sliding_window_op_params.num_cores_h}_{sliding_window_op_params.num_cores_nhw}_{sliding_window_op_params.act_reshard_num_cores_nhw}"


def get_sliding_window_op_output_nhw_shape_(
    input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
):
    output_h = ((int)((input_h + (2 * pad_h) - window_h) / stride_h)) + 1
    output_w = ((int)((input_w + (2 * pad_w) - window_w) / stride_w)) + 1
    return [input_n, output_h, output_w]


def get_sliding_window_op_input_nhw_shape(sliding_window_op_params):
    input_n = sliding_window_op_params.batch_size
    input_h = sliding_window_op_params.input_h
    input_w = sliding_window_op_params.input_w
    return [input_n, input_h, input_w]


def get_sliding_window_op_output_nhw_shape(sliding_window_op_params):
    stride_h = sliding_window_op_params.stride_h
    stride_w = sliding_window_op_params.stride_w
    pad_h = sliding_window_op_params.pad_h
    pad_w = sliding_window_op_params.pad_w
    window_h = sliding_window_op_params.window_h
    window_w = sliding_window_op_params.window_w
    input_n = sliding_window_op_params.batch_size
    input_h = sliding_window_op_params.input_h
    input_w = sliding_window_op_params.input_w
    output_h = ((int)((input_h + (2 * pad_h) - window_h) / stride_h)) + 1
    output_w = ((int)((input_w + (2 * pad_w) - window_w) / stride_w)) + 1
    return [input_n, output_h, output_w]


def get_sliding_window_op_output_shard_nhw_size(
    num_cores_nhw, input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w, is_out_tiled=True
):
    output_nhw_shape = get_sliding_window_op_output_nhw_shape_(
        input_n, input_h, input_w, stride_h, stride_w, pad_h, pad_w, window_h, window_w
    )
    if is_out_tiled:
        output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), num_cores_nhw * 32)
    else:
        output_nhw_size_to_shard_evenly = _nearest_y(np.prod(output_nhw_shape), num_cores_nhw)
    output_shard_nhw_size = (int)(output_nhw_size_to_shard_evenly / num_cores_nhw)
    return output_shard_nhw_size


def calculate_shard_grid(grid_size, num_cores_nhw, transpose_mcast=True):
    num_cores_w, num_cores_h = grid_size
    if transpose_mcast:
        shard_layout = (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED
            if (num_cores_nhw == num_cores_w and num_cores_h > 1)
            else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        )
    else:
        shard_layout = (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED
            if (num_cores_nhw == num_cores_h and num_cores_w > 1)
            else ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        )

    if shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1))
        shard_grid = ttnn.CoreRangeSet({core_range})
    else:
        if num_cores_nhw >= num_cores_w:
            num_cores_height_excluding_remainder_last_row = num_cores_nhw // num_cores_w
            assert num_cores_h >= num_cores_height_excluding_remainder_last_row
            core_range_1 = ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_cores_w - 1, num_cores_height_excluding_remainder_last_row - 1),
            )
            num_cores_last = num_cores_nhw % num_cores_w
            if num_cores_last > 0:
                assert num_cores_h == num_cores_height_excluding_remainder_last_row + 1
                core_range_2 = ttnn.CoreRange(
                    ttnn.CoreCoord(0, num_cores_height_excluding_remainder_last_row),
                    ttnn.CoreCoord(num_cores_last - 1, num_cores_height_excluding_remainder_last_row),
                )
                shard_grid = ttnn.CoreRangeSet({core_range_1, core_range_2})
            else:
                assert num_cores_h == num_cores_height_excluding_remainder_last_row
                shard_grid = ttnn.CoreRangeSet({core_range_1})
        else:
            core_range_1 = ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_cores_nhw - 1, 0),
            )
            shard_grid = ttnn.CoreRangeSet({core_range_1})
    return shard_grid, shard_layout


def calculate_memory_config(
    sliding_window_op_params, is_1d_systolic, padded_channels, calc_input=False, tile_size=1, transpose_mcast=True
):
    tensor_shape = (
        get_sliding_window_op_input_nhw_shape(sliding_window_op_params)
        if calc_input
        else get_sliding_window_op_output_nhw_shape(sliding_window_op_params)
    )
    tensor_shape.append(padded_channels)
    # tensor_shape is [N, H, W, C]
    assert len(tensor_shape) == 4
    needs_reshard = calc_input and sliding_window_op_params.act_reshard_num_cores_nhw > 0
    if needs_reshard:
        num_cores_nhw = sliding_window_op_params.act_reshard_num_cores_nhw
        if is_1d_systolic:
            num_cores_w = min(sliding_window_op_params.num_cores_w, num_cores_nhw)
            num_cores_h = (num_cores_nhw + num_cores_w - 1) // num_cores_w
        else:
            if transpose_mcast:
                num_cores_w = num_cores_nhw
                num_cores_h = sliding_window_op_params.num_cores_h
            else:
                num_cores_w = sliding_window_op_params.num_cores_h
                num_cores_h = num_cores_nhw
    else:
        num_cores_nhw = sliding_window_op_params.num_cores_nhw
        num_cores_w = sliding_window_op_params.num_cores_w
        num_cores_h = sliding_window_op_params.num_cores_h

    logical_grid_size = None
    grid_size = None
    if is_1d_systolic:
        logical_grid_size = (num_cores_nhw, 1)
        grid_size = (num_cores_w, num_cores_h)
    else:
        if transpose_mcast:
            logical_grid_size = (num_cores_w, num_cores_h)
            grid_size = (num_cores_w, num_cores_h)
        else:
            logical_grid_size = (num_cores_w, num_cores_h)
            grid_size = (num_cores_w, num_cores_h)

    shard_grid, shard_layout = calculate_shard_grid(grid_size, num_cores_nhw, transpose_mcast=transpose_mcast)
    nhw_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    nhw_padded = roundup(nhw_shape, num_cores_nhw * tile_size)
    # if (nhw_padded - nhw_shape) > 32:
    #     breakpoint()
    # assert (nhw_padded - nhw_shape) <= 32
    nhw_shard = nhw_padded // num_cores_nhw
    if is_1d_systolic or (not is_1d_systolic and transpose_mcast):
        assert padded_channels % logical_grid_size[1] == 0
        shard_shape = [nhw_shard, padded_channels // logical_grid_size[1]]
    else:
        assert padded_channels % logical_grid_size[0] == 0
        shard_shape = [nhw_shard, padded_channels // logical_grid_size[0]]
    shard_orientation = (
        ttnn.ShardOrientation.ROW_MAJOR
        if is_1d_systolic
        else (ttnn.ShardOrientation.COL_MAJOR if transpose_mcast else ttnn.ShardOrientation.ROW_MAJOR)
    )
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, False)
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED if is_1d_systolic else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    return ttnn.MemoryConfig(shard_scheme, ttnn.BufferType.L1, shard_spec)
