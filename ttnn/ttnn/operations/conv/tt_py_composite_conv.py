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


class TTPyCompositeConvSetupDepracated(TTPyOp):
    config_keys = [
        "num_cores_nhw",
        "grid_size",
        "per_core_out_matrix_height",
        "per_core_weight_matrix_width",
        "act_block_h",
        "act_block_w",
        "out_block_w",
        "out_subblock_h",
        "out_subblock_w",
        "act_reshard_num_cores_nhw",
    ]

    def __init__(
        self,
        sliding_window_op_params: Union[SlidingWindowOpParams, SlidingWindowOpParamsWithParallelConfig],
        weight: ttnn.Tensor,  # should user send TT tensor as weight tensor
        output_channels,
        input_channels,
        device,
        is_1d_systolic,
        reader_patterns_cache,
        bias: ttnn.Tensor = None,
        conv_blocking_and_parallelization_config_override=None,
        fuse_relu=False,
        weights_dtype=None,
        output_dtype=None,
        math_fidelity=None,
        move_utwh_output=False,
        using_parameters_cache=False,
        move_weights_to_device=True,
        use_shallow_conv_variant=False,
        transpose_mcast=True,  ## default is the original CM
        enable_auto_formatting=False,
        deallocate_activation=True,
        padded_input_channels=None,
        compute_kernel_config=None,
        output_layout=ttnn.TILE_LAYOUT,
        use_dram_for_matmul=False,
    ):
        fp32_accum = (
            compute_kernel_config
            and isinstance(compute_kernel_config, ttnn.WormholeComputeKernelConfig)
            and compute_kernel_config.fp32_dest_acc_en
        )
        self.use_dram_for_matmul = use_dram_for_matmul

        if padded_input_channels is None:
            self.padded_input_channels = _nearest_32(input_channels)
        else:
            self.padded_input_channels = padded_input_channels
        self.enable_auto_formatting = enable_auto_formatting
        self.deallocate_activation = deallocate_activation
        self.use_shallow_conv_variant = use_shallow_conv_variant
        self.transpose_mcast = transpose_mcast
        self.untilize_out = output_layout == ttnn.ROW_MAJOR_LAYOUT
        if reader_patterns_cache is None:
            reader_patterns_cache = {}

        if "conv" not in reader_patterns_cache:
            reader_patterns_cache["conv"] = {}
        if "halo" not in reader_patterns_cache:
            reader_patterns_cache["halo"] = {}

        for key in reader_patterns_cache:
            assert (
                key == "conv" or key == "halo" or key == "max_pool"
            ), f"reader_patterns_cache should have 1 of the following keys only - conv, max_pool or halo. Found key - {key}"
        if conv_blocking_and_parallelization_config_override is None:
            conv_blocking_and_parallelization_config_override = {}
        for key in conv_blocking_and_parallelization_config_override:
            assert (
                key in TTPyCompositeConv.config_keys
            ), f"Error: unsupported config key: {key}. Supported config keys are: {TTPyCompositeConv.config_keys}"
        batch_size = sliding_window_op_params.batch_size
        input_height = sliding_window_op_params.input_h
        input_width = sliding_window_op_params.input_w
        output_height, output_width = compute_conv_output_height_width(
            input_height, input_width, sliding_window_op_params
        )
        self.conv_output_shape = [batch_size, output_height, output_width, output_channels]
        self.input_tensor_shape = [batch_size, input_height, input_width, input_channels]
        self.is_1d_systolic = is_1d_systolic
        if is_1d_systolic:
            assert transpose_mcast, "With 1D systolic array, please set transpose_mcast=True"
        self.device = device
        # determine conv op parallelization and blocking config
        self.opt_conv_parall_conf_auto = determine_parallel_config(
            is_1d_systolic,
            output_channels,
            input_channels,
            sliding_window_op_params,
            device,
            config_override=conv_blocking_and_parallelization_config_override,
            transpose_mcast=transpose_mcast,
        )
        self.per_core_out_matrix_height = self.opt_conv_parall_conf_auto.per_core_out_matrix_height_ntiles * 32
        self.per_core_out_matrix_width = self.opt_conv_parall_conf_auto.per_core_out_matrix_width_ntiles * 32
        self.parallel_config = self.opt_conv_parall_conf_auto

        self.opt_conv_block_conf_auto = determine_per_core_block_config(
            is_1d_systolic,
            self.opt_conv_parall_conf_auto.grid_size,
            self.opt_conv_parall_conf_auto.per_core_out_matrix_height_ntiles,
            self.opt_conv_parall_conf_auto.per_core_out_matrix_width_ntiles,
            input_channels,
            sliding_window_op_params,
            use_shallow_conv_variant,
            self.padded_input_channels,
            config_override=conv_blocking_and_parallelization_config_override,
            fp32_accum=fp32_accum,
            transpose_mcast=transpose_mcast,
        )

        if not is_1d_systolic:  # 2D conv
            output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
        else:
            output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

        stride_h = sliding_window_op_params.stride_h
        stride_w = sliding_window_op_params.stride_w
        pad_h = sliding_window_op_params.pad_h
        pad_w = sliding_window_op_params.pad_w
        filter_height = sliding_window_op_params.window_h
        filter_width = sliding_window_op_params.window_w
        self.matmul_config = None
        self.use_matmul_for_1x1_conv = False
        if (
            filter_height == filter_width
            and filter_height == 1
            and stride_h == stride_w
            and stride_h == 1
            and pad_h == pad_w
            and pad_h == 0
        ):
            self.use_matmul_for_1x1_conv = True
            self.matmul_config = determine_1x1conv_as_matmul_config(
                self.opt_conv_parall_conf_auto,
                self.opt_conv_block_conf_auto,
                is_1d_systolic,
                fuse_relu,
                transpose_mcast,
            )
        if isinstance(sliding_window_op_params, SlidingWindowOpParams):
            # populate parallelization params in sliding_window_op_params
            sliding_window_op_params = SlidingWindowOpParamsWithParallelConfig(
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                window_h=filter_height,
                window_w=filter_width,
                batch_size=batch_size,
                input_h=input_height,
                input_w=input_width,
                num_cores_h=self.opt_conv_parall_conf_auto.grid_size.y,
                num_cores_w=self.opt_conv_parall_conf_auto.grid_size.x,
                num_cores_nhw=self.opt_conv_parall_conf_auto.num_cores_nhw,
            )

        if "act_reshard_num_cores_nhw" in conv_blocking_and_parallelization_config_override:
            sliding_window_op_params = sliding_window_op_params._replace(
                act_reshard_num_cores_nhw=conv_blocking_and_parallelization_config_override["act_reshard_num_cores_nhw"]
            )

        self.sliding_window_op_params = sliding_window_op_params
        self.move_utwh_output = move_utwh_output
        self.deallocate_input = deallocate_activation
        self.kernel_size = [filter_height, filter_width]
        self.strides = [stride_h, stride_w]

        sliding_window_op_params_hash = get_hash_from_sliding_window_op_params(sliding_window_op_params)

        # TODO: consolidate conv_params and sliding_window_op_params
        # K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
        conv_params = [
            output_channels,
            input_channels,
            filter_height,
            filter_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            1,
            1,
        ]
        conv_reader_indices = None
        if not self.use_matmul_for_1x1_conv:
            # set_op_configs populates reader_patterns_cache["conv"][sliding_window_op_params_hash] with conv_reader_indices sharded tensor
            self.set_op_configs(
                self.device,
                sliding_window_op_params_hash,
                sliding_window_op_params,
                conv_params,
                not is_1d_systolic,
                reader_patterns_cache["conv"],
                move_weights_to_device=move_weights_to_device,
            )
            assert sliding_window_op_params_hash in reader_patterns_cache["conv"]
            conv_reader_indices = reader_patterns_cache["conv"][sliding_window_op_params_hash]

        self.set_op_weights_biases(
            weight,
            conv_params,
            self.device,
            self.opt_conv_block_conf_auto.act_block_w_ntiles,
            self.opt_conv_block_conf_auto.out_subblock_w_ntiles,
            self.opt_conv_parall_conf_auto,
            self.opt_conv_block_conf_auto,
            fuse_relu,
            output_mem_config,
            output_dtype,
            math_fidelity,
            conv_reader_indices,
            bias=bias,
            weights_dtype=weights_dtype,
            using_parameters_cache=using_parameters_cache,
            use_shallow_conv_variant=use_shallow_conv_variant,
            padded_input_channels=self.padded_input_channels,
            move_weights_to_device=move_weights_to_device,
            compute_kernel_config=compute_kernel_config,
        )

        if not self.use_matmul_for_1x1_conv:
            # create untilize with halo op
            self.tt_py_untilize_with_halo_op = TTPyUntilizeWithHalo(
                device,
                self.sliding_window_op_params,
                reader_patterns_cache["halo"],
                transpose_mcast=self.transpose_mcast,
            )
        self.grid_size = (
            self.sliding_window_op_params.num_cores_w,
            self.sliding_window_op_params.num_cores_h,
        )
        self.input_sharded_memory_config = calculate_memory_config(
            self.sliding_window_op_params,
            self.is_1d_systolic,
            self.padded_input_channels,
            calc_input=True,
            tile_size=32,
            transpose_mcast=transpose_mcast,
        )
        self.output_sharded_memory_config = calculate_memory_config(
            self.sliding_window_op_params,
            self.is_1d_systolic,
            _nearest_32(output_channels),
            calc_input=False,
            tile_size=32,
            transpose_mcast=transpose_mcast,
        )

    # override abstract methods from base class TTPyOp
    def set_op_configs(
        self,
        device,
        sliding_window_op_params_hash,
        sliding_window_op_params,
        conv_params,
        conv_is_2d,
        conv_reader_patterns_cache,
        move_weights_to_device,
    ):
        # TODO: Need way of hashing sliding_window_op_params
        if sliding_window_op_params_hash not in conv_reader_patterns_cache:
            # TODO: Need to clean up sliding_window_op_params and conv_params (they are basically the same)
            stride_h = sliding_window_op_params.stride_h
            stride_w = sliding_window_op_params.stride_w
            pad_h = sliding_window_op_params.pad_h
            pad_w = sliding_window_op_params.pad_w
            filter_h = sliding_window_op_params.window_h
            filter_w = sliding_window_op_params.window_w
            batch_size = sliding_window_op_params.batch_size
            input_h = sliding_window_op_params.input_h
            input_w = sliding_window_op_params.input_w
            # TODO: Had to add this (should this be shard grid?)
            num_cores_w = sliding_window_op_params.num_cores_w
            num_cores_h = sliding_window_op_params.num_cores_h
            num_cores_nhw = sliding_window_op_params.num_cores_nhw

            input_nchw_shape = [batch_size, 1, input_h, input_w]
            conv_input_volume = batch_size * input_h * input_w
            conv_output_h = ((int)((input_h + (2 * pad_h) - filter_h) / stride_h)) + 1
            conv_output_w = ((int)((input_w + (2 * pad_w) - filter_w) / stride_w)) + 1
            conv_output_volume = batch_size * conv_output_h * conv_output_w

            input_size_to_shard_evenly = _nearest_y(conv_input_volume, num_cores_nhw * 32)
            untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
            output_size_to_shard_evenly = _nearest_y(conv_output_volume, num_cores_nhw * 32)
            conv_output_shard_height = (int)(output_size_to_shard_evenly / num_cores_nhw)

            input_padded_width = input_w + 2 * pad_w

            # TODO: We should remove C from input_nchw_shape since none of the specs depend on it
            # TODO: Pass sliding_window_op_params instead of conv_param?
            pad_metadata, data_top_left_indices = trace_conv_to_generate_data_top_left_indices_and_pad_metadata(
                conv_params, input_nchw_shape
            )

            req_conv_input_shard_start_end, tensor_metadata = decompose_conv_into_shards_and_generate_tensor_metadata(
                data_top_left_indices,
                pad_metadata,
                input_padded_width,
                conv_output_shard_height,
                untilize_with_halo_input_shard_height,
                num_cores_nhw,
                filter_h,
                filter_w,
            )

            sliding_window_op_sharded_input_top_left_indices = (
                generate_sliding_window_op_sharded_input_top_left_indices(
                    data_top_left_indices,
                    req_conv_input_shard_start_end,
                    pad_tile=True,
                    pad_last_core=True,
                )
            )

            indices_torch_dtype = torch.int16
            indices_tt_dtype = ttnn.uint16
            # For 2d convs, each core in a column or row share the same specs
            if conv_is_2d:
                if self.transpose_mcast:
                    sliding_window_op_sharded_input_top_left_indices *= num_cores_h  ## replicate along columns
                else:
                    sliding_window_op_sharded_input_top_left_indices *= num_cores_w  ## replicate along rows

            # Create sharded tensor on device for conv_reader_indices
            conv_reader_indices_torch_tensor = torch.tensor(
                [[sliding_window_op_sharded_input_top_left_indices]], dtype=indices_torch_dtype
            )

            conv_reader_indices_tt_tensor = ttnn.Tensor(
                conv_reader_indices_torch_tensor,
                indices_tt_dtype,
            )
            shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(num_cores_w - 1, num_cores_h - 1),
                    )
                }
            )
            shard_orientation = (
                ttnn.ShardOrientation.ROW_MAJOR if self.transpose_mcast else ttnn.ShardOrientation.COL_MAJOR
            )
            shard_spec = ttnn.ShardSpec(shard_grid, [1, conv_output_shard_height], shard_orientation, False)
            mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1_SMALL,
                shard_spec,
            )
            conv_reader_indices_sharded_tensor = (
                conv_reader_indices_tt_tensor.to(device, mem_config)
                if move_weights_to_device
                else conv_reader_indices_tt_tensor
            )

            conv_reader_patterns_cache[sliding_window_op_params_hash] = conv_reader_indices_sharded_tensor

    # TODO: Maybe need to have this be more general to settting up conv
    def set_op_weights_biases(
        self,
        weight: ttnn.Tensor,
        conv_params,
        device,
        weight_block_h_ntiles,
        weight_block_w_ntiles,
        opt_conv_parall_conf,
        opt_conv_block_conf,
        fuse_relu,
        output_mem_config,
        output_dtype,
        math_fidelity,
        conv_reader_indices,
        weights_dtype,
        bias,
        using_parameters_cache,
        use_shallow_conv_variant,
        padded_input_channels,
        move_weights_to_device=False,
        compute_kernel_config=None,
    ):
        assert len(conv_params) == 10
        K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

        assert dilation == 1 and groups == 1
        assert padded_input_channels >= C
        if not using_parameters_cache:
            weights_shape = [K, C, R, S]
            weights_channels_padded_shape = [
                _nearest_32(K),
                padded_input_channels,
                R,
                S,
            ]
            if weights_dtype is None:
                weights_dtype = weight.get_dtype()
            weights_untiled_dtype = weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
            assert weight.get_layout() == ttnn.ROW_MAJOR_LAYOUT
            assert weight.get_dtype() == weights_untiled_dtype
            assert weight.get_legacy_shape() == weights_shape
            weight_untiled = weight.pad(weights_channels_padded_shape, (0, 0, 0, 0), 0)
            # for conv op, pad the weights to block shape
            if self.is_1d_systolic:
                weight_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_special_padding_tiled_layout(
                    weight_untiled,
                    weight_block_h_ntiles,
                    weight_block_w_ntiles,
                    output_dtype=weights_dtype,
                )
            else:
                weight_tiled_ = ttl.tensor.convert_conv_weight_tensor_to_tiled_layout(
                    weight_untiled,
                    weight_block_h_ntiles,
                    weight_block_w_ntiles,
                    output_dtype=weights_dtype,
                )
            weight_on_device = weight_tiled_.to(device) if move_weights_to_device else weight_tiled_
            bias_on_device = None
            self.weight = weight_on_device
            if bias is not None:
                bias_shape = [1, 1, 1, K]
                assert bias.get_layout() == ttnn.ROW_MAJOR_LAYOUT
                assert bias.get_dtype() == weights_untiled_dtype
                assert bias.get_legacy_shape() == bias_shape

                bias_channels_padded_shape = [1, 1, 32, _nearest_y(K, weight_block_w_ntiles * 32)]
                bias_untiled = bias.pad(bias_channels_padded_shape, (0, 0, 0, 0), 0)

                bias_untiled = bias_untiled.to_torch()
                bias_ = ttnn.from_torch(bias_untiled, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT)
                bias_on_device = bias_.to(device) if move_weights_to_device else bias_
            self.bias = bias_on_device
        else:
            self.weight = weight
            self.bias = bias
            weight_on_device = weight
            bias_on_device = bias

        def downsample_if_needed(activation, deallocate_activation):
            use_downsample = False
            for kernel_size, stride in zip(self.kernel_size, self.strides):
                if kernel_size == 1 and stride > 1:
                    use_downsample = True
            if use_downsample:
                ttnn.dump_device_memory_state(device, "before_ds_")
                ds_out = ttnn.downsample(activation, [*self.input_tensor_shape[:-1], *self.strides])
                if deallocate_activation:
                    activation.deallocate()
            else:
                ds_out = activation
            return ds_out

        def conv1x1_as_matmul(activation):
            activation = downsample_if_needed(activation, self.deallocate_activation)
            output = ttnn.linear(
                activation,
                weight_on_device,
                bias=bias_on_device,
                program_config=self.matmul_config,
                memory_config=activation.memory_config() if output_mem_config is None else output_mem_config,
                dtype=output_dtype,
                compute_kernel_config=compute_kernel_config,
            )
            if self.deallocate_activation:
                activation.deallocate()
            return output

    def get_parallelization_config(self):
        return self.opt_conv_parall_conf_auto

    def get_blocking_config(self):
        return self.opt_conv_block_conf_auto

    def get_num_cores_nhw(self):
        return self.sliding_window_op_params.num_cores_nhw

    def get_parallel_config(self):
        return self.parallel_config

    # TODO: with this api, we get incorrect output
    def copy_input_to_device_with_sharded_api(self, conv_input: ttnn.Tensor):
        assert conv_input.get_legacy_shape() == self.input_tensor_shape
        num_cores_nhw = self.sliding_window_op_params.num_cores_nhw
        num_cores_w = self.sliding_window_op_params.num_cores_w
        num_cores_h = self.sliding_window_op_params.num_cores_h
        input_channels = self.input_tensor_shape[3]
        input_size_to_shard_evenly = _nearest_y(
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2],
            num_cores_nhw * 32,
        )
        untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
        conv_input = conv_input.reshape(
            1,
            1,
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2],
            self.input_tensor_shape[3],
        )
        conv_input = conv_input.pad([1, 1, input_size_to_shard_evenly, self.input_tensor_shape[3]], (0, 0, 0, 0), 0.0)
        if self.input_tensor_shape[0] >= 32:
            # Convert activation RM to tile layout
            conv_input = conv_input.to(ttnn.TILE_LAYOUT)

        shard_grid, shard_layout = calculate_shard_grid((num_cores_w, num_cores_h), self.ncores_nhw)
        assert self.is_1d_systolic == (shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
        shard_orientation = (
            ttnn.ShardOrientation.ROW_MAJOR
            if self.is_1d_systolic
            else (ttnn.ShardOrientation.COL_MAJOR if self.transpose_mcast else ttnn.ShardOrientation.ROW_MAJOR)
        )
        shard_shape = [
            untilize_with_halo_input_shard_height,
            (
                input_channels
                if self.is_1d_systolic
                else (
                    (
                        (int)(input_channels / self.opt_conv_parall_conf_auto.grid_size.y)
                        if self.transpose_mcase
                        else (int)(input_channels / self.opt_conv_parall_conf_auto.grid_size.x)
                    ),
                )
            ),
        ]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, False)
        mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1)
        conv_input_on_device = conv_input.to(self.device, mem_config, shard_spec)
        return conv_input_on_device

    def conv_input_interleaved_to_sharded(self, conv_input_on_device: ttnn.Tensor):
        num_cores_nhw = self.sliding_window_op_params.num_cores_nhw
        num_cores_w = self.sliding_window_op_params.num_cores_w
        num_cores_h = self.sliding_window_op_params.num_cores_h

        input_channels = self.input_tensor_shape[3]
        padded_input_channels = conv_input_on_device.get_legacy_shape()[3]
        assert padded_input_channels >= input_channels
        assert padded_input_channels == self.padded_input_channels
        grid_size = (num_cores_w, num_cores_h)

        input_size_to_shard_evenly = _nearest_y(
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2],
            num_cores_nhw * 32,
        )
        untilize_with_halo_input_shard_height = (int)(input_size_to_shard_evenly / num_cores_nhw)
        # Convert interleaved to sharded
        if self.is_1d_systolic:
            conv_input_on_device = ttnn.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    untilize_with_halo_input_shard_height,
                    padded_input_channels,
                ],  # act_block_w_datums may include reads of multiple pixels in window
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        else:
            grid_size_along_c = (
                self.opt_conv_parall_conf_auto.grid_size.y
                if self.transpose_mcast
                else self.opt_conv_parall_conf_auto.grid_size.x
            )
            assert (
                padded_input_channels % grid_size_along_c == 0
            ), f"Expected padded_input_channels to be divisible by grid_size_along_c. Found padded_input_channels={padded_input_channels} and grid_size_along_c={grid_size_along_c}"
            assert (
                not self.use_shallow_conv_variant
            ), "Do not support shallow depth convs with 2d systolic variant. Run with use_1d_systolic_array=True or unset use_shallow_conv_variant. Default value of use_shallow_conv_variant is False."
            conv_input_on_device = ttnn.interleaved_to_sharded(
                conv_input_on_device,
                grid_size,
                [
                    untilize_with_halo_input_shard_height,
                    (int)(padded_input_channels / grid_size_along_c),
                ],  # act_block_w_datums may include reads of multiple pixels in window
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.COL_MAJOR if self.transpose_mcast else ttnn.ShardOrientation.ROW_MAJOR,
            )

        return conv_input_on_device

    def copy_input_to_device(self, conv_input: ttnn.Tensor, layout=ttnn.TILE_LAYOUT):
        interleaved_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        assert conv_input.get_legacy_shape() == self.input_tensor_shape
        # Reshape 4d to 2d
        conv_input = conv_input.reshape(
            1,
            1,
            self.input_tensor_shape[0] * self.input_tensor_shape[1] * self.input_tensor_shape[2],
            self.input_tensor_shape[3],
        )

        if conv_input.storage_type() != ttnn.StorageType.DEVICE:
            padded_input_channels = self.padded_input_channels
            channels_padded_shape = [
                conv_input.get_legacy_shape()[0],
                conv_input.get_legacy_shape()[1],
                conv_input.get_legacy_shape()[2],
                padded_input_channels,
            ]
            conv_input = conv_input.pad(channels_padded_shape, (0, 0, 0, 0), 0)
            conv_input_on_device = conv_input.to(self.device, interleaved_mem_config)
        else:
            conv_input_on_device = conv_input
        assert conv_input_on_device.get_legacy_shape()[3] % 16 == 0
        if not self.use_shallow_conv_variant:
            input_padded_shape = ttnn.pad_to_tile_shape(
                conv_input_on_device.get_legacy_shape(), False, False, True, True
            )
            assert self.padded_input_channels == input_padded_shape[3]
            if conv_input.get_legacy_shape() != input_padded_shape:
                conv_input_on_device = ttnn.format_input_tensor(
                    conv_input_on_device,
                    self.device,
                    input_padded_shape,
                    0.0,
                    ttnn.TILE_LAYOUT,
                    interleaved_mem_config,
                )
            else:
                conv_input_on_device = ttnn.tilize(
                    conv_input_on_device, memory_config=interleaved_mem_config, use_multicore=True
                )
        return self.conv_input_interleaved_to_sharded(conv_input_on_device)

    def conv_output_sharded_to_interleaved(self, conv_output_on_device):
        interleaved_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        # Convert sharded output to tiled interleaved
        return ttnn.sharded_to_interleaved(conv_output_on_device, interleaved_mem_config)

    def copy_output_from_device(self, conv_output_on_device):
        interleaved_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        # Convert sharded output to tiled interleaved
        conv_output_on_device = self.conv_output_sharded_to_interleaved(conv_output_on_device)

        # convert tiled output to RM
        if conv_output_on_device.get_legacy_shape() != conv_output_on_device.shape_without_padding():
            conv_output_on_device = ttnn.format_output_tensor(
                conv_output_on_device,
                conv_output_on_device.shape_without_padding(),
                self.device,
                ttnn.ROW_MAJOR_LAYOUT,
                interleaved_mem_config,
            )
        elif not self.untilize_out:
            assert conv_output_on_device.get_layout() == ttnn.TILE_LAYOUT
            conv_output_on_device = ttnn.untilize(
                conv_output_on_device, memory_config=interleaved_mem_config, use_multicore=True
            )

        conv_output_on_device = conv_output_on_device.reshape(
            self.conv_output_shape[0],
            self.conv_output_shape[1],
            self.conv_output_shape[2],
            conv_output_on_device.shape_without_padding()[3],
        )

        # Copy to host
        cpu_tensor = conv_output_on_device.cpu()
        if cpu_tensor.shape_without_padding()[3] != self.conv_output_shape[3]:
            cpu_tensor = cpu_tensor[:, :, :, : self.conv_output_shape[3]]
        return cpu_tensor

    # TODO: with this api, we get TT_ASSERT @ tt_metal/impl/dispatch/command_queue.cpp:790: dev_page_id < num_pages and dev_page_id >= 0
    def copy_output_from_device_with_sharded_api(self, conv_output_on_device):
        conv_output = conv_output_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

        conv_output = conv_output.reshape(
            self.conv_output_shape[0],
            self.conv_output_shape[1],
            self.conv_output_shape[2],
            self.conv_output_shape[3],
        )

        return conv_output
