# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Union

import ttnn

from tt_lib.utils import find_closest_largest_divisor
import math


def _golden_function(input_tensor: ttnn.Tensor, dim: int, **_):
    import torch

    return torch.softmax(input_tensor, dim)


ttnn.attach_golden_function(
    ttnn.softmax,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.softmax_in_place,
    golden_function=_golden_function,
)


def _golden_function(input_tensor: ttnn.Tensor, scalar: float, attention_mask=None, **_):
    import torch

    input_tensor = input_tensor.float()
    input_tensor = input_tensor * scalar
    if attention_mask is not None:
        input_tensor = input_tensor + attention_mask
    return torch.softmax(input_tensor, dim=-1)


ttnn.attach_golden_function(
    ttnn.scale_mask_softmax_in_place,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.scale_mask_softmax,
    golden_function=_golden_function,
)

ttnn.attach_golden_function(
    ttnn.scale_causal_mask_hw_dims_softmax_in_place,
    golden_function=_golden_function,
)


SoftmaxProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxProgramConfig
SoftmaxDefaultProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxDefaultProgramConfig
SoftmaxShardedMultiCoreProgramConfig = ttnn._ttnn.operations.normalization.SoftmaxShardedMultiCoreProgramConfig


def _golden_function(
    input_tensor: ttnn.Tensor,
    *,
    epsilon=1e-12,
    residual_input_tensor=None,
    weight=None,
    bias=None,
    **_,
):
    import torch

    if residual_input_tensor is not None:
        input_tensor += residual_input_tensor

    if weight is not None:
        if len(weight.shape) >= 2:
            weight = weight.squeeze()
        weight = weight.to(input_tensor.dtype)

    if bias is not None:
        if len(bias.shape) >= 2:
            bias = bias.squeeze()
        bias = bias.to(input_tensor.dtype)

    return torch.nn.functional.layer_norm(input_tensor, (input_tensor.shape[-1],), weight, bias, eps=epsilon)


ttnn.attach_golden_function(ttnn.layer_norm, golden_function=_golden_function)


def _golden_function(input_tensor: ttnn.Tensor, weight=None, *, epsilon=1e-12, **_):
    import torch

    variance = input_tensor.to(torch.float32).pow(2).mean(-1, keepdim=True)
    input_tensor = input_tensor * torch.rsqrt(variance + epsilon)

    if weight.dtype in [torch.float16, torch.bfloat16]:
        input_tensor = input_tensor.to(weight.dtype)

    return weight * input_tensor


ttnn.attach_golden_function(ttnn.rms_norm, golden_function=_golden_function)

LayerNormProgramConfig = ttnn._ttnn.operations.normalization.LayerNormProgramConfig
LayerNormDefaultProgramConfig = ttnn._ttnn.operations.normalization.LayerNormDefaultProgramConfig
LayerNormShardedMultiCoreProgramConfig = ttnn._ttnn.operations.normalization.LayerNormShardedMultiCoreProgramConfig


# group norm helper function
def determine_expected_group_norm_sharded_config_and_grid_size(
    *, device, num_channels, num_groups, input_nhw, is_height_sharded, is_row_major=False
):
    assert num_channels % num_groups == 0
    assert num_channels % 32 == 0  # TODO: remove this later
    group_size = num_channels // num_groups
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = [compute_with_storage_grid_size.x, compute_with_storage_grid_size.y]
    if is_row_major:
        device_grid_size = [compute_with_storage_grid_size.y, compute_with_storage_grid_size.x]

    max_num_cores = device_grid_size[0] * device_grid_size[1]
    input_nhw_paddedto32 = math.ceil(input_nhw / 32) * 32
    num_cores_nhw = find_closest_largest_divisor(
        input_nhw_paddedto32 // 32, max_num_cores if is_height_sharded else device_grid_size[0]
    )
    if is_height_sharded:
        num_cores_channels = 1
    else:
        num_cores_channels = device_grid_size[1]
        # num_channels_tiles = num_channels // 16
        num_channels_tiles = num_channels // 8
        while (num_channels_tiles % num_cores_channels != 0) or (
            ((num_channels // num_cores_channels) % group_size) != 0
        ):
            num_cores_channels -= 1
            assert num_cores_channels > 0
    input_nhw_padded_to_ncores = math.ceil(input_nhw / (num_cores_nhw * 32)) * (num_cores_nhw * 32)
    gn_in_channels_per_core = num_channels // num_cores_channels
    # assert gn_in_channels_per_core % 16 == 0
    assert gn_in_channels_per_core % 8 == 0
    gn_nhw_per_core = input_nhw_padded_to_ncores // num_cores_nhw
    if is_height_sharded:
        grid_size = [
            device_grid_size[0] if num_cores_nhw >= device_grid_size[0] else num_cores_nhw,
            math.ceil(num_cores_nhw / device_grid_size[0]),
        ]  # for 1d systolic array, grid size is the tightest bound of num_cores_nhw as a rectangle (x,y)
        assert (
            num_cores_nhw <= grid_size[0] * grid_size[1]
        ), "Error: For height sharding, num_cores_nhw must be <= grid size"
    else:
        grid_size = [num_cores_channels, num_cores_nhw] if is_row_major else [num_cores_nhw, num_cores_channels]
    shard_shape = (
        (1, 1, gn_nhw_per_core, gn_in_channels_per_core)
        if is_row_major
        else (1, 1, gn_in_channels_per_core, gn_nhw_per_core)
    )
    shard_strategy = ttnn.ShardStrategy.HEIGHT if is_height_sharded else ttnn.ShardStrategy.BLOCK
    shard_orientation = (
        ttnn.ShardOrientation.ROW_MAJOR if is_height_sharded or is_row_major else ttnn.ShardOrientation.COL_MAJOR
    )
    return ttnn.create_sharded_memory_config(
        shard_shape,
        ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        shard_strategy,
        shard_orientation,
        halo=False,
        use_height_and_width_as_shard_shape=True,
    ), ttnn.CoreGrid(y=grid_size[1], x=grid_size[0])


def create_group_norm_weight_bias_rm(input_tensor, num_channels, num_groups):
    import torch

    def find_ceil_divisible_by_32(n):
        return ((n + 31) // 32) * 32

    values_per_chunk = num_channels // num_groups
    zeros_to_insert = find_ceil_divisible_by_32(values_per_chunk) - values_per_chunk
    input_tensor = input_tensor.view(-1, values_per_chunk)
    input_tensor = torch.nn.functional.pad(input_tensor, (0, zeros_to_insert))
    input_tensor = input_tensor.flatten()
    input_tensor = input_tensor[: num_channels + zeros_to_insert * (num_channels // values_per_chunk)]
    return input_tensor.reshape(1, 1, -1, 32)


def find_max_tile_span(W, group_size, tile_width):
    current_position = 0
    max_tile_span = 0

    while current_position < W:
        group_end = current_position + group_size
        start_tile = current_position // tile_width
        end_tile = (group_end - 1) // tile_width
        current_tile_span = end_tile - start_tile + 1
        max_tile_span = max(max_tile_span, current_tile_span)
        current_position = group_end
    return max_tile_span


def create_group_norm_input_mask(num_channel, num_groups, num_cores_across_channel):
    import torch

    block_wt = find_max_tile_span(num_channel, num_channel // num_groups, 32)
    input_mask_tensor = torch.zeros((1, num_groups, 32, int(32 * block_wt)), dtype=torch.bfloat16)

    num_groups_per_core = num_groups // num_cores_across_channel
    num_cols_per_group = num_channel // num_groups

    start_strides = []
    for _ in range(num_cores_across_channel):
        row_offset = 0
        start_strides.append(0)
        for _ in range(num_groups_per_core - 1):
            if row_offset + (num_cols_per_group % 32) == 32:
                row_offset = 0
            elif row_offset + (num_cols_per_group % 32) > 32:
                row_offset = (num_cols_per_group % 32) + row_offset - 32
            else:
                row_offset += num_cols_per_group % 32
            start_strides.append(row_offset)
        end_strides = [i + num_cols_per_group for i in start_strides]

    for group in range(num_groups):
        start_stride = start_strides[group]
        end_stride = end_strides[group]
        end_stride = min(end_stride, input_mask_tensor.shape[3])
        input_mask_tensor[:, group, :, start_stride:end_stride] = 1

    return input_mask_tensor


def get_group_norm_cores_accross_channel(memory_layout, core_grid):
    if memory_layout == ttnn.types.TensorMemoryLayout.BLOCK_SHARDED:
        num_cores_across_channel = core_grid.y
    elif memory_layout == ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED:
        num_cores_across_channel = 1
    else:
        num_cores_across_channel = core_grid.x * core_grid.y

    return num_cores_across_channel


def _golden_function(
    input_tensor: ttnn.Tensor,
    *,
    num_groups,
    epsilon=1e-05,
    weight=None,
    bias=None,
    memory_config=None,
    core_grid=None,
    input_mask=None,
    **kwargs,
):
    import torch

    num_channels = input_tensor.shape[-1]
    num_cores_across_channel = get_group_norm_cores_accross_channel(memory_config.memory_layout, core_grid)
    weight = weight.reshape((num_cores_across_channel, -1))
    weight = weight[:, : num_channels // num_cores_across_channel].flatten()
    if bias is not None:
        bias = bias.reshape((num_cores_across_channel, -1))
        bias = bias[:, : num_channels // num_cores_across_channel].flatten()

    input_tensor = input_tensor.permute(0, 3, 1, 2)
    output = torch.nn.functional.group_norm(input_tensor.float(), num_groups, weight.float(), bias.float(), eps=epsilon)
    output = output.permute(0, 2, 3, 1)
    return output


def _postprocess_golden_function_outputs(output, args, kwargs):
    input_tensor = args[0]
    output = ttnn.reshape(output, input_tensor.shape)
    return output


ttnn.attach_golden_function(
    ttnn.group_norm,
    golden_function=_golden_function,
    postprocess_golden_function_outputs=_postprocess_golden_function_outputs,
)

__all__ = []
