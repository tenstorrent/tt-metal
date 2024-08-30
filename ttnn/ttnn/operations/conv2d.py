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
    TTPyCompositeConv,
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


class Conv2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        dtype: ttnn.DataType = None,
        *,
        device: ttnn.Device,
        use_1d_systolic_array: bool,
        batch_size: int,
        input_height: int,
        input_width: int,
        reader_patterns_cache: Optional[Dict],
        weight: ttnn.Tensor,
        bias: ttnn.Tensor = None,
        math_fidelity: ttnn.MathFidelity = None,
        weights_dtype: ttnn.DataType = None,
        activation: str = None,
        conv_blocking_and_parallelization_config_override: Dict = None,
        reallocate_halo_output: bool = False,
        using_parameters_cache: bool = False,
        move_weights_to_device: bool = True,
        use_shallow_conv_variant: bool = False,
        transpose_mcast: bool = True,
        enable_auto_formatting: bool = False,
        deallocate_activation: bool = False,
        padded_input_channels: Optional[int] = None,
        compute_kernel_config: Union[ttnn.GrayskullComputeKernelConfig, ttnn.WormholeComputeKernelConfig] = None,
        use_dram_for_matmul: bool = False,
        output_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ):
        assert (
            padding_mode == "zeros"
        ), f"Only convs with padding_mode=zeroes supported. Found padding_mode set to {padding_mode}."
        if isinstance(kernel_size, int):
            window_h = kernel_size
            window_w = kernel_size
        else:
            window_h, window_w = kernel_size

        if isinstance(stride, int):
            stride_h = stride
            stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h = padding
            pad_w = padding
        else:
            pad_h, pad_w = padding

        if isinstance(dilation, int):
            dilation_h = dilation
            dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation

        assert dilation_h == 1, f"Only convs with dilation == 1 supported. Found dilation_h={dilation_h}"
        assert dilation_w == 1, f"Only convs with dilation == 1 supported. Found dilation_w={dilation_w}"
        assert groups == 1, "Only convs with groups == 1 supported"
        sliding_window_op_params = SlidingWindowOpParams(
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            window_h=window_h,
            window_w=window_w,
            batch_size=batch_size,
            input_h=input_height,
            input_w=input_width,
        )
        fuse_relu = False
        if activation is not None:
            activation = activation.lower()
            assert activation == "relu", f"Only support relu fusion with conv. Got activation={activation}."
            fuse_relu = True
        self.conv = TTPyCompositeConv(
            sliding_window_op_params,
            weight,
            out_channels,
            in_channels,
            device,
            use_1d_systolic_array,
            reader_patterns_cache,
            bias=bias,
            conv_blocking_and_parallelization_config_override=conv_blocking_and_parallelization_config_override,
            fuse_relu=fuse_relu,
            output_dtype=dtype,
            weights_dtype=weights_dtype,
            math_fidelity=math_fidelity,
            move_utwh_output=reallocate_halo_output,
            using_parameters_cache=using_parameters_cache,
            move_weights_to_device=move_weights_to_device,
            use_shallow_conv_variant=use_shallow_conv_variant,
            transpose_mcast=transpose_mcast,
            enable_auto_formatting=enable_auto_formatting,
            deallocate_activation=deallocate_activation,
            padded_input_channels=padded_input_channels,
            compute_kernel_config=compute_kernel_config,
            use_dram_for_matmul=use_dram_for_matmul,
            output_layout=output_layout,
        )
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = (input_height + (2 * pad_h) - dilation_h * (window_h - 1) - 1) // stride_h + 1
        self.output_width = (input_width + (2 * pad_w) - dilation_w * (window_w - 1) - 1) // stride_w + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.original_weight = ttnn.to_torch(weight)
        self.original_bias = None if bias is None else ttnn.to_torch(bias)
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups

    def _golden_function_conv2d(self, activations):
        import torch

        # inputs in [1, 1, NWH, C] format, reshape to N, H, W, C
        if self.in_channels < 32:
            activations = activations[:, :, :, : self.in_channels]
        activations = activations.reshape(self.batch_size, self.input_height, self.input_width, self.in_channels)
        # permute to N, C, H, W
        activations = activations.permute(0, 3, 1, 2)

        torch_out = torch.nn.functional.conv2d(
            activations.float(),
            self.original_weight.float(),
            self.original_bias.squeeze().float() if self.original_bias is not None else None,
            stride=(self.stride_h, self.stride_w),
            padding=(self.pad_h, self.pad_w),
            dilation=(self.dilation_h, self.dilation_w),
            groups=self.groups,
        )
        torch_out = torch_out.permute(0, 2, 3, 1)
        torch_out = torch_out.reshape(1, 1, -1, self.out_channels)
        return torch_out

    @ttnn.register_python_operation(
        name="ttnn.Conv2d.__call__",
        is_method=True,
        golden_function=_golden_function_conv2d,
    )
    def __call__(self, activation: ttnn.Tensor):
        return self.conv(activation)

    def _golden_function_copy_input(self, input):
        return input

    @ttnn.register_python_operation(
        name="ttnn.Conv2d.copy_input_to_device",
        golden_function=_golden_function_copy_input,
        is_method=True,
    )
    def copy_input_to_device(self, input: ttnn.Tensor):
        return self.conv.copy_input_to_device(input)

    def _golden_function_copy_output(self, output):
        return output

    @ttnn.register_python_operation(
        name="ttnn.Conv2d.copy_output_from_device",
        golden_function=_golden_function_copy_output,
        is_method=True,
    )
    def copy_output_from_device(self, output: ttnn.Tensor):
        return self.conv.copy_output_from_device(output)

    def get_parallel_config(self):
        return self.conv.get_parallel_config()


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
    debug=False,
) -> Tuple[ttnn.Tensor, int, int, ttnn.Tensor, ttnn.Tensor]:
    run_new_conv = True
    if debug:
        deallocate_act_debug_mode = conv_config.deallocate_activation
        conv_config.deallocate_activation = False
    if run_new_conv:
        (
            output_tensor_new,
            output_height_new,
            output_width_new,
            weight_tensor_on_dev_new,
            bias_tensor_on_dev_new,
        ) = ttnn._ttnn.operations.conv2d.conv2d(
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
        if not debug:
            return (
                output_tensor_new,
                output_height_new,
                output_width_new,
                weight_tensor_on_dev_new,
                bias_tensor_on_dev_new,
            )
    if run_new_conv:
        print("DEBUG MODE ENABLED. WILL RUN OLD PATH AND COMPARE WEIGHT, BIAS & OUTPUT TENSORS.")
        assert (
            not conv_config.deallocate_activation
        )  # cannot run old path if activation was deallocated in the new path above
    output_height = ((int)((input_height - kernel_size[0] + 2 * padding[0]) / stride[0])) + 1
    output_width = ((int)((input_width - kernel_size[1] + 2 * padding[1]) / stride[1])) + 1
    conv_config.deallocate_activation = deallocate_act_debug_mode
    if "reader_patterns_cache" not in conv_op_cache:
        conv_op_cache["reader_patterns_cache"] = {}
    weight_is_on_device = ttnn.is_tensor_storage_on_device(weight_tensor)
    if bias_tensor is not None:
        bias_is_on_device = ttnn.is_tensor_storage_on_device(bias_tensor)
        assert (
            weight_is_on_device == bias_is_on_device
        ), "Both weight and bias tensors both must be pre-processed if one of them is pre-processed."

    # Input processing. TODO: Cache input processing decisions
    if conv_config is None:
        conv_config = Conv2dConfig()
    config_shard_grid = None
    # breakpoint()
    if conv_config.core_grid is not None:
        config_shard_grid = get_shard_grid_from_core_grid(conv_config.core_grid)

    needs_reshard = False
    input_memory_config = ttnn.get_memory_config(input_tensor)
    if ttnn.is_sharded(input_tensor):
        input_shard_scheme = input_memory_config.memory_layout
        input_shard_orientation = input_memory_config.shard_spec.orientation
        input_shard_grid = input_memory_config.shard_spec.grid
        if not (
            input_shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            or input_shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        ):
            needs_reshard = True
        if (
            input_shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED
            and input_shard_orientation != ttnn.ShardOrientation.COL_MAJOR
        ):
            needs_reshard = True
        if (
            input_shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and input_shard_orientation != ttnn.ShardOrientation.ROW_MAJOR
        ):
            needs_reshard = True
        if config_shard_grid is not None:
            if config_shard_grid != input_shard_grid:
                needs_reshard = True
        if conv_config.shard_layout is not None:
            if input_shard_scheme != conv_config.shard_layout:
                needs_reshard = True
    else:
        needs_reshard = True
    parallel_config = None
    if conv_config.reshard_if_not_optimal or needs_reshard:
        optimal_parallel_config = determine_parallel_config(
            conv_config.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            device,
        )
    if needs_reshard:
        if conv_config.shard_layout is None:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if conv_config.core_grid is None:
            parallel_config = optimal_parallel_config
        else:
            assert config_shard_grid is not None
            grid_size, num_cores_nhw = get_grid_size_and_num_cores_nhw_from_core_grid(
                conv_config.core_grid, conv_config.conv_config.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            )

            shard_orientation = (
                ttnn.ShardOrientation.ROW_MAJOR
                if conv_config.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                else ttnn.ShardOrientation.COL_MAJOR
            )
            parallel_config = ParallelConfig(
                grid_size.y, grid_size.x, num_cores_nhw, conv_config.shard_layout, shard_orientation
            )
    else:
        assert ttnn.is_sharded(input_tensor)
        grid_size, num_cores_nhw = get_grid_size_and_num_cores_nhw_from_core_grid(
            input_memory_config.shard_spec.grid,
            input_memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        parallel_config = ParallelConfig(
            grid_size.y,
            grid_size.x,
            num_cores_nhw,
            input_memory_config.memory_layout,
            input_memory_config.shard_spec.orientation,
        )

    if conv_config.reshard_if_not_optimal:
        if parallel_config != optimal_parallel_config:
            parallel_config = optimal_parallel_config
            needs_reshard = True
    if needs_reshard:
        input_is_on_device = ttnn.is_tensor_storage_on_device(input_tensor)
        # not sure if reshard op works for all cases
        # copying to l1 interleaved first
        # input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
        if input_tensor.shape[0] != 1 or input_tensor.shape[1] != 1:
            # reshape to [1, 1, N*H*W, C]
            input_tensor = ttnn.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))
        input_num_cores_nhw = parallel_config.num_cores_nhw
        input_tensor_height_snapped_to_tile = roundup(input_tensor.shape[2], input_num_cores_nhw * 32)
        assert input_tensor_height_snapped_to_tile >= input_tensor.shape[2]
        input_tensor_width_snapped_to_channels_alignment = roundup(
            input_tensor.shape[3], conv_config.input_channels_alignment
        )
        assert input_tensor_width_snapped_to_channels_alignment >= input_tensor.shape[3]
        if not input_is_on_device and (
            input_tensor_height_snapped_to_tile != input_tensor.shape[2]
            or input_tensor_width_snapped_to_channels_alignment != input_tensor.shape[3]
        ):
            if input_is_on_device:
                input_tensor = ttnn.pad(
                    input_tensor,
                    padding=(
                        (0, 0),
                        (0, 0),
                        (0, input_tensor_height_snapped_to_tile - input_tensor.shape[2]),
                        (0, input_tensor_width_snapped_to_channels_alignment - input_tensor.shape[3]),
                    ),
                    value=0,
                )
            else:
                import torch

                input_tensor = ttnn.to_torch(input_tensor)
                input_tensor = torch.nn.functional.pad(
                    input_tensor,
                    (
                        0,
                        input_tensor_width_snapped_to_channels_alignment - input_tensor.shape[3],
                        0,
                        input_tensor_height_snapped_to_tile - input_tensor.shape[2],
                        0,
                        0,
                    ),
                )
                input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

        input_tensor_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            input_tensor.shape, parallel_config, tile_size=32
        )
        if input_is_on_device:
            input_tensor_before_tm = input_tensor
            input_tensor = ttnn.to_memory_config(input_tensor, input_tensor_sharded_memory_config)
            if conv_config.deallocate_activation:
                ttnn.deallocate(input_tensor_before_tm)
        else:
            input_tensor = ttnn.to_device(input_tensor, device=device, memory_config=input_tensor_sharded_memory_config)
        # since we resharded/moved the input tensor, we can deallocate it after halo op within composite conv
        conv_config.deallocate_activation = True
    is_1x1_conv = kernel_size == (1, 1) and stride[0] == stride[1] and stride[0] == 1 and padding == (0, 0)
    if is_1x1_conv and input_tensor.layout != ttnn.TILE_LAYOUT:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, dtype=conv_config.dtype)
    input_is_on_device = ttnn.is_tensor_storage_on_device(input_tensor)
    assert input_is_on_device
    if weight_tensor in conv_op_cache:
        assert weight_is_on_device
        # Run conv
        conv = conv_op_cache[weight_tensor]
        assert conv.conv.weight == weight_tensor
        assert conv.conv.bias == bias_tensor
    else:
        # Following code will be removed after op refactoring
        block_and_parallel_config_override = {}
        if conv_config.act_block_h_override > 0:
            block_and_parallel_config_override["act_block_h"] = conv_config.act_block_h_override
        assert parallel_config is not None
        block_and_parallel_config_override["grid_size"] = [parallel_config.grid_size.x, parallel_config.grid_size.y]
        block_and_parallel_config_override["num_cores_nhw"] = parallel_config.num_cores_nhw
        if is_grayskull(device=device):
            compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=conv_config.math_fidelity,
                math_approx_mode=conv_config.math_approx_mode_enabled,
            )
        elif is_wormhole_b0(device=device):
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=conv_config.math_fidelity,
                math_approx_mode=conv_config.math_approx_mode_enabled,
                fp32_dest_acc_en=conv_config.fp32_dest_acc_enabled,
                packer_l1_acc=conv_config.packer_l1_accum_enabled,
            )
        else:
            assert False, f"Unsupported device: {device}"
        # Build conv op object
        conv = ttnn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            dtype=conv_config.dtype,
            device=device,
            use_1d_systolic_array=parallel_config.shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            weight=weight_tensor,
            bias=bias_tensor,
            math_fidelity=conv_config.math_fidelity,
            weights_dtype=conv_config.weights_dtype,
            conv_blocking_and_parallelization_config_override=block_and_parallel_config_override,
            compute_kernel_config=compute_kernel_config,
            activation=conv_config.activation if conv_config.activation != "" else None,
            using_parameters_cache=weight_is_on_device,
            reader_patterns_cache=conv_op_cache["reader_patterns_cache"],
            deallocate_activation=conv_config.deallocate_activation,
            padded_input_channels=input_tensor.shape[3],
            reallocate_halo_output=conv_config.reallocate_halo_output,
            use_shallow_conv_variant=conv_config.input_channels_alignment == 16,
        )
        # Cache conv by weight tensor
        conv_op_cache[conv.conv.weight] = conv
    # Run conv
    output_tensor = conv(input_tensor)
    if run_new_conv:
        import torch

        assert output_height == output_height_new
        assert output_width == output_width_new
        assert conv.conv.weight.layout == weight_tensor_on_dev_new.layout
        assert conv.conv.bias.layout == bias_tensor_on_dev_new.layout
        weight_t_cpu_golden = ttnn.to_torch(conv.conv.weight)
        bias_t_cpu_golden = ttnn.to_torch(conv.conv.bias)
        bias_t_cpu_golden = bias_t_cpu_golden[:, :, 0:1, :]
        weight_t_cpu = ttnn.to_torch(weight_tensor_on_dev_new)
        bias_t_cpu = ttnn.to_torch(bias_tensor_on_dev_new)
        output_t_cpu_golden = ttnn.to_torch(output_tensor)
        output_t_cpu = ttnn.to_torch(output_tensor_new)
        assert torch.all(torch.eq(weight_t_cpu_golden, weight_t_cpu))
        assert torch.all(torch.eq(bias_t_cpu_golden, bias_t_cpu))
        # breakpoint()
        # assert torch.all(torch.eq(output_t_cpu_golden, output_t_cpu))
        # breakpoint()
        print("Returning output tensor from new path")
        return (output_tensor_new, output_height, output_width, weight_tensor_on_dev_new, bias_tensor_on_dev_new)
    return (output_tensor, output_height, output_width, conv.conv.weight, conv.conv.bias)


__all__ = []
